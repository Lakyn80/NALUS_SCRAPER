#!/usr/bin/env python3
"""Slice 4: isolated A/B BGE-M3 + BM25 indexes for chunking_ab_pilot_300_v1 (parser v8).

HARD STOP after index construction + integrity validation.
Does NOT run FAST, CE-7, or touch production/pilot_600 collections.

Resume: by default continues after pause — skips chunk_ids already present in
BOTH Qdrant and BM25. Use --force-recreate only for an intentional wipe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
import urllib.parse
import urllib.request
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.audit import PARSER_VERSION  # noqa: E402
from app.rag.legal_v2.ingest import index_builder as ib  # noqa: E402
from app.rag.legal_v2.ingest.chunkers import chunk_document_for_experiment  # noqa: E402
from app.rag.legal_v2.ingest.chunkers.contextual_packed_v1 import (  # noqa: E402
    ContextualPackedConfigV1,
)
from app.rag.legal_v2.ingest.chunkers.names import (  # noqa: E402
    CHUNKER_A_CURRENT,
    CHUNKER_B_CONTEXTUAL_PACKED_V1,
)
from app.rag.legal_v2.ingest.indexing import (  # noqa: E402
    LEGAL_V2_PROFILE,
    payload_for_child_chunk,
)
from app.rag.legal_v2.parser import parse_legal_document  # noqa: E402
from app.rag.retrieval.bge_m3_embedder import BgeM3Embedder  # noqa: E402
from app.rag.retrieval.production_profile import ProductionRetrievalConfig  # noqa: E402
from scripts.legal_v2.run_chunking_ab_pilot_300_chunk_qa import _policy_hash  # noqa: E402

EXPECTED_PARSER = "legal-decision-parser.cz-courts.v8"
EXPECTED_PARSER_COMMIT = "f9c83727c6f141d2c6e54f38217b6918361f4dab"
EXPECTED_INVENTORY_ID = "chunking_ab_pilot_300_v1"
EXPECTED_INVENTORY_HASH = (
    "89233b9fe9b06eda8dea00abd99a48aa54940e616aa88c00860ced4ae49c011b"
)
EXPECTED_DOC_COUNT = 300
EXPECTED_A_CHUNKS = 6162
EXPECTED_B_CHUNKS = 4168
EXPECTED_B_POLICY_HASH = (
    "8fa196c58a9c537d311af6849582481ac195324c4f358634e81fcecb8f3f5898"
)
EXPECTED_EMBEDDING_DIM = 1024
EXPECTED_DISTANCE = "Cosine"

COLLECTION_A = "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_a_current_300"
COLLECTION_B = "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_b_contextual_300"
BM25_ID_A = "nalus_legal_paragraph_bm25_v2_chunk_ab_v8_a_current_300"
BM25_ID_B = "nalus_legal_paragraph_bm25_v2_chunk_ab_v8_b_contextual_300"
PROTECTED = {
    "nalus_legal_paragraph_chunks_v2_pilot_600",
    "nalus_legal_paragraph_chunks_v2",
}

DEFAULT_INVENTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "chunking_ab_pilot_300_v1"
    / "inventory_manifest.json"
)
DEFAULT_QA = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "chunking_ab_pilot_300_v1"
    / "chunk_qa_v8"
    / "chunk_qa_summary.json"
)
DEFAULT_OUT = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "chunking_ab_pilot_300_v1"
    / "slice4_indexes_v8"
)


class _BatchedBgeM3Embedder:
    """Experiment-only wrapper: keep production defaults, larger encode batches."""

    def __init__(self, inner: BgeM3Embedder, *, encode_batch_size: int = 32) -> None:
        self._inner = inner
        self._encode_batch_size = max(1, int(encode_batch_size))

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._inner.load()
        encoded = self._inner._model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=self._encode_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = []
        for vector in encoded:
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            vectors.append([float(value) for value in vector])
        for index, vector in enumerate(vectors):
            if len(vector) != LEGAL_V2_PROFILE.embedding_dimension:
                raise RuntimeError(
                    f"BGE-M3 dimension mismatch at {index}: "
                    f"{len(vector)} != {LEGAL_V2_PROFILE.embedding_dimension}"
                )
        return vectors


class _ExperimentBgeM3Embedder:
    """Slice4-only embedder: allows cuda without changing production BgeM3Embedder."""

    def __init__(
        self,
        *,
        model_path: str,
        device: str,
        encode_batch_size: int = 32,
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._encode_batch_size = max(1, int(encode_batch_size))
        self._model: Any | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer  # type: ignore[import]

        self._model = SentenceTransformer(
            self._model_path,
            device=self._device,
            local_files_only=True,
            trust_remote_code=False,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self.load()
        assert self._model is not None
        encoded = self._model.encode(
            texts,
            batch_size=self._encode_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = []
        for vector in encoded:
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            vectors.append([float(value) for value in vector])
        for index, vector in enumerate(vectors):
            if len(vector) != LEGAL_V2_PROFILE.embedding_dimension:
                raise RuntimeError(
                    f"BGE-M3 dimension mismatch at {index}: "
                    f"{len(vector)} != {LEGAL_V2_PROFILE.embedding_dimension}"
                )
        return vectors


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_head(explicit: str | None = None) -> str:
    if explicit and explicit.strip() and explicit.strip().lower() != "unknown":
        return explicit.strip()
    for key in ("LEGAL_V2_GIT_COMMIT", "GIT_COMMIT", "SOURCE_COMMIT"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ordered_id_hash(ids: list[str]) -> str:
    return _sha256_text("\n".join(ids))


def _set_hash(ids: set[str]) -> str:
    return _ordered_id_hash(sorted(ids))


def _fetch_full_text(api: str, ecli: str) -> str:
    url = f"{api.rstrip('/')}/api/rag/documents/{urllib.parse.quote(ecli, safe='')}"
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    text = payload.get("full_text") or payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"empty text for {ecli}")
    return text


def _resolve_model_path() -> str:
    env = (os.environ.get("EMBEDDING_MODEL_NAME") or "").strip()
    if env and Path(env).exists():
        return env
    for cand in (
        "/root/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/"
        "5617a9f61b028005a4858fdac845db406aefb181",
        "/app/models/BAAI/bge-m3",
        "BAAI/bge-m3",
    ):
        if cand == "BAAI/bge-m3" or Path(cand).exists():
            return cand
    raise SystemExit("EXPERIMENT_BLOCKED: cannot resolve BGE-M3 model path")


def _model_revision(model_path: str) -> str:
    path = Path(model_path)
    if path.name and len(path.name) >= 12 and all(
        c in "0123456789abcdef" for c in path.name.lower()
    ):
        return path.name
    return "unresolved"


def _gate(
    *,
    inventory: dict[str, Any],
    qa: dict[str, Any],
    b_policy_hash: str,
) -> None:
    if PARSER_VERSION != EXPECTED_PARSER:
        raise SystemExit(f"EXPERIMENT_BLOCKED: parser={PARSER_VERSION}")
    if inventory.get("inventory_id") != EXPECTED_INVENTORY_ID:
        raise SystemExit("EXPERIMENT_BLOCKED: inventory_id mismatch")
    if inventory.get("inventory_hash_sha256") != EXPECTED_INVENTORY_HASH:
        raise SystemExit("EXPERIMENT_BLOCKED: inventory hash mismatch")
    if int(inventory.get("document_count") or -1) != EXPECTED_DOC_COUNT:
        raise SystemExit("EXPERIMENT_BLOCKED: inventory document_count != 300")
    if qa.get("classification") != "CHUNK_QA_PASS_V8":
        raise SystemExit(
            f"EXPERIMENT_BLOCKED: QA classification={qa.get('classification')}"
        )
    if qa.get("SLICE_4_SAFE_TO_START") != "YES":
        raise SystemExit("EXPERIMENT_BLOCKED: SLICE_4_SAFE_TO_START != YES")
    ig = qa.get("integrity_gates") or {}
    loss_a = ig.get("confirmed_text_loss_a")
    loss_b = ig.get("confirmed_text_loss_b")
    if loss_a is None or loss_b is None:
        raise SystemExit("EXPERIMENT_BLOCKED: confirmed_text_loss fields missing from QA")
    if int(loss_a) != 0 or int(loss_b) != 0:
        raise SystemExit("EXPERIMENT_BLOCKED: confirmed text loss != 0")
    if b_policy_hash != EXPECTED_B_POLICY_HASH:
        raise SystemExit(f"EXPERIMENT_BLOCKED: B policy hash {b_policy_hash}")
    a_chunks = (qa.get("A") or {}).get("total_child_chunks")
    b_chunks = (qa.get("B") or {}).get("total_child_chunks")
    if a_chunks is None or int(a_chunks) != EXPECTED_A_CHUNKS:
        raise SystemExit("EXPERIMENT_BLOCKED: QA A chunk count != 6162")
    if b_chunks is None or int(b_chunks) != EXPECTED_B_CHUNKS:
        raise SystemExit("EXPERIMENT_BLOCKED: QA B chunk count != 4168")
    if CHUNKER_A_CURRENT != "legal_v2_hierarchical_chunker_v1":
        raise SystemExit("EXPERIMENT_BLOCKED: A chunker version changed")
    if CHUNKER_B_CONTEXTUAL_PACKED_V1 != "legal_contextual_packed_v1":
        raise SystemExit("EXPERIMENT_BLOCKED: B chunker version changed")


def _build_side_chunks(
    *,
    ordered: list[str],
    meta_by_ecli: dict[str, dict[str, Any]],
    api: str,
    chunker_version: str,
) -> tuple[list[dict[str, Any]], list[str], set[str]]:
    payloads: list[dict[str, Any]] = []
    ordered_ids: list[str] = []
    eclis: set[str] = set()
    collection = COLLECTION_A if chunker_version == CHUNKER_A_CURRENT else COLLECTION_B
    bm25_id = BM25_ID_A if chunker_version == CHUNKER_A_CURRENT else BM25_ID_B
    for index, ecli in enumerate(ordered, start=1):
        print(f"[{chunker_version}] [{index}/{len(ordered)}] chunk {ecli}", flush=True)
        text = _fetch_full_text(api, ecli)
        meta = meta_by_ecli.get(ecli) or {}
        court = str(meta.get("court") or "")
        parsed = parse_legal_document(
            document_id=ecli,
            text=text,
            metadata={"court": court, "ecli": ecli},
        )
        result = chunk_document_for_experiment(parsed, chunker_version=chunker_version)
        for child in result.child_chunks:
            enriched = replace(
                child,
                metadata={
                    **dict(child.metadata or {}),
                    "ecli": ecli,
                    "court": court or None,
                    "parser_version": PARSER_VERSION,
                    "chunker_version": chunker_version,
                    "inventory_id": EXPECTED_INVENTORY_ID,
                    "inventory_hash_sha256": EXPECTED_INVENTORY_HASH,
                    "ingest_run_id": f"chunking_ab_slice4_v8_{chunker_version}",
                    "source": meta.get("source") or "api_full_text",
                    "case_reference": meta.get("case_reference"),
                    "decision_date": meta.get("decision_date"),
                    "language": "cs",
                },
            )
            payload = payload_for_child_chunk(
                enriched,
                qdrant_collection=collection,
                bm25_index_id=bm25_id,
            )
            payload["inventory_id"] = EXPECTED_INVENTORY_ID
            payload["inventory_hash_sha256"] = EXPECTED_INVENTORY_HASH
            payloads.append(payload)
            ordered_ids.append(str(payload["chunk_id"]))
            eclis.add(ecli)
    return payloads, ordered_ids, eclis


def _make_build_config(
    *,
    collection: str,
    bm25_id: str,
    bm25_path: Path,
    output_dir: Path,
    batch_size: int,
    document_batch_size: int,
    force_recreate: bool,
) -> ib.LegalV2BuildConfig:
    if collection in PROTECTED:
        raise SystemExit(f"EXPERIMENT_BLOCKED: refusing protected collection {collection}")
    return ib.LegalV2BuildConfig(
        collection_name=collection,
        bm25_index_id=bm25_id,
        bm25_path=bm25_path,
        output_dir=output_dir,
        recreate_collection=force_recreate,
        overwrite_bm25=force_recreate,
        resume=False,
        allow_existing_collection=not force_recreate,
        batch_size=batch_size,
        document_batch_size=document_batch_size,
        source_selection={
            "inventory_id": EXPECTED_INVENTORY_ID,
            "experiment": "chunking_ab_pilot_300_v1_slice4_v8",
        },
    )


def _bm25_chunk_ids(path: Path) -> set[str]:
    with sqlite3.connect(path) as connection:
        return {str(r[0]) for r in connection.execute("SELECT chunk_id FROM bm25_chunks")}


def _resolve_already_indexed(
    *,
    qdrant_client: Any,
    collection: str,
    bm25_path: Path,
    source_ids: set[str],
    force_recreate: bool,
) -> set[str]:
    """Return chunk IDs already present in BOTH Qdrant and BM25 (safe to skip)."""
    if force_recreate:
        return set()
    collections = {c.name for c in qdrant_client.get_collections().collections}
    q_ids: set[str] = set()
    if collection in collections:
        q_ids = ib._qdrant_payload_chunk_ids(qdrant_client, collection)
    b_ids: set[str] = set()
    if bm25_path.exists():
        b_ids = _bm25_chunk_ids(bm25_path)
    if not q_ids and not b_ids:
        return set()
    if q_ids != b_ids:
        only_q = sorted(q_ids - b_ids)[:5]
        only_b = sorted(b_ids - q_ids)[:5]
        raise SystemExit(
            "RESUME_BLOCKED: Qdrant/BM25 identity mismatch "
            f"(qdrant={len(q_ids)} bm25={len(b_ids)} "
            f"only_qdrant_sample={only_q} only_bm25_sample={only_b}). "
            "Refuse to continue; repair indexes or pass --force-recreate."
        )
    extras = q_ids - source_ids
    if extras:
        raise SystemExit(
            "RESUME_BLOCKED: index contains chunk_ids not in source set "
            f"(extras={len(extras)} sample={sorted(extras)[:5]}). "
            "Pass --force-recreate for a clean rebuild."
        )
    return q_ids


def _index_payloads(
    *,
    payloads: list[dict[str, Any]],
    config: ib.LegalV2BuildConfig,
    embedder: Any,
    qdrant_client: Any,
    force_recreate: bool,
) -> dict[str, Any]:
    config.validate()
    source_ids = {str(p["chunk_id"]) for p in payloads}
    already = _resolve_already_indexed(
        qdrant_client=qdrant_client,
        collection=config.collection_name,
        bm25_path=config.bm25_path,
        source_ids=source_ids,
        force_recreate=force_recreate,
    )
    ib._prepare_collection(qdrant_client, config)
    ib._prepare_bm25_sidecar(
        config.bm25_path,
        overwrite=force_recreate,
        resume=(not force_recreate and config.bm25_path.exists()),
    )
    if force_recreate:
        already = set()
    remaining = [p for p in payloads if str(p["chunk_id"]) not in already]
    print(
        json.dumps(
            {
                "event": "chunk_ab_slice4_resume",
                "collection": config.collection_name,
                "already_indexed": len(already),
                "remaining": len(remaining),
                "total": len(payloads),
                "force_recreate": force_recreate,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    chunk_ids: set[str] = set(already)
    upsert_batches = 0
    upsert_points = 0
    finite_ok = True
    nan_or_inf = 0
    norm_samples: list[float] = []
    for start in range(0, len(remaining), config.batch_size):
        batch = remaining[start : start + config.batch_size]
        ib._validate_payload_identity(batch, existing_chunk_ids=chunk_ids)
        texts = [p["text"] for p in batch]
        vectors = embedder.embed_texts(texts)
        ib._validate_vectors(vectors, LEGAL_V2_PROFILE.embedding_dimension)
        for vec in vectors:
            if any((not math.isfinite(float(x))) for x in vec):
                finite_ok = False
                nan_or_inf += 1
            if len(norm_samples) < 32:
                norm_samples.append(math.sqrt(sum(float(x) * float(x) for x in vec)))
        ib._upsert_payloads(
            qdrant_client,
            collection_name=config.collection_name,
            payloads=batch,
            vectors=vectors,
            batch_size=config.batch_size,
        )
        ib._append_bm25_payloads(batch, config.bm25_path)
        chunk_ids.update(str(p["chunk_id"]) for p in batch)
        upsert_batches += 1
        upsert_points += len(batch)
        print(
            json.dumps(
                {
                    "event": "chunk_ab_slice4_batch",
                    "collection": config.collection_name,
                    "done": len(chunk_ids),
                    "upserted_this_run": upsert_points,
                    "total": len(payloads),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    ib._validate_qdrant_identity_by_ids(
        qdrant_client,
        collection_name=config.collection_name,
        expected=source_ids,
        require_exact_match=True,
    )
    ib._validate_bm25_identity_by_ids(source_ids, config.bm25_path, require_exact_match=True)
    return {
        "chunk_ids": chunk_ids,
        "chunk_count": len(chunk_ids),
        "already_indexed": len(already),
        "qdrant_upsert_batches": upsert_batches,
        "qdrant_upsert_points": upsert_points,
        "finite_ok": finite_ok,
        "nan_or_inf_vectors": nan_or_inf,
        "norm_samples": norm_samples,
    }


def _qdrant_ecli_set(client: Any, collection: str) -> set[str]:
    eclis: set[str] = set()
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in batch:
            payload = getattr(point, "payload", None) or {}
            ecli = payload.get("ecli") or payload.get("document_id")
            if ecli:
                eclis.add(str(ecli))
        if offset is None:
            break
    return eclis


def _bm25_ecli_set(path: Path) -> set[str]:
    eclis: set[str] = set()
    with sqlite3.connect(path) as connection:
        for row in connection.execute("SELECT metadata FROM bm25_chunks"):
            meta = json.loads(row[0])
            ecli = meta.get("ecli") or meta.get("document_id")
            if ecli:
                eclis.add(str(ecli))
    return eclis


def _payload_sample(
    client: Any,
    collection: str,
    inventory_docs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    wanted: list[str] = []
    by_bucket: dict[str, list[str]] = {}
    for doc in inventory_docs:
        ecli = str(doc.get("ecli") or "")
        bucket = str(doc.get("length_bucket") or "unknown")
        by_bucket.setdefault(bucket, []).append(ecli)
        if doc.get("selection_reason") == "mandatory_golden_or_hn":
            wanted.append(ecli)
    for bucket in ("short", "medium", "long", "very_long"):
        pool = sorted(by_bucket.get(bucket) or [])
        if pool:
            wanted.append(pool[0])
    wanted = list(dict.fromkeys([w for w in wanted if w]))[:12]
    samples: list[dict[str, Any]] = []
    for ecli in wanted:
        offset = None
        found = None
        while found is None:
            points, offset = client.scroll(
                collection_name=collection,
                limit=64,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                if str(payload.get("ecli") or payload.get("document_id")) == ecli:
                    found = payload
                    break
            if offset is None:
                break
        if found:
            samples.append(
                {
                    "ecli": ecli,
                    "chunk_id": found.get("chunk_id"),
                    "section_type": found.get("section_type"),
                    "chunker_version": found.get("chunker_version"),
                    "parser_version": found.get("parser_version"),
                    "has_paragraph_ids": bool(found.get("paragraph_ids")),
                    "has_text": bool(found.get("text")),
                    "court": found.get("court"),
                    "inventory_id": found.get("inventory_id"),
                }
            )
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--qa-summary", type=Path, default=DEFAULT_QA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--qdrant-url", default=os.environ.get("QDRANT_URL") or "http://qdrant:6333"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--document-batch-size", type=int, default=16)
    parser.add_argument("--git-commit", default="")
    parser.add_argument(
        "--chunks-only",
        action="store_true",
        help="Only regenerate/verify A/B chunk counts; do not embed/index.",
    )
    parser.add_argument("--side", choices=("A", "B", "both"), default="both")
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help=(
            "DANGEROUS: delete target A/B collections + BM25 and rebuild from zero. "
            "Default is always resume (skip already-indexed chunk_ids)."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Embedding device. cuda requires GPU image + --gpus (Slice4 experiment only).",
    )
    args = parser.parse_args()

    git_commit = _git_head(args.git_commit)
    if not git_commit or git_commit == "unknown":
        raise SystemExit("EXPERIMENT_BLOCKED: git_commit unresolved (pass --git-commit)")

    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    qa = json.loads(args.qa_summary.read_text(encoding="utf-8"))
    b_hash = _policy_hash(ContextualPackedConfigV1())
    _gate(inventory=inventory, qa=qa, b_policy_hash=b_hash)

    ordered = list(inventory["ordered_eclis"])
    if len(ordered) != EXPECTED_DOC_COUNT or len(set(ordered)) != EXPECTED_DOC_COUNT:
        raise SystemExit("EXPERIMENT_BLOCKED: ordered ECLI integrity failed")
    meta_by_ecli = {
        str(d["ecli"]): d for d in (inventory.get("documents") or []) if d.get("ecli")
    }
    inventory_eclis = set(ordered)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    bm25_dir = PROJECT_ROOT / "storage" / "rag" / "bm25"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    bm25_path_a = bm25_dir / f"{BM25_ID_A}.sqlite"
    bm25_path_b = bm25_dir / f"{BM25_ID_B}.sqlite"

    model_path = _resolve_model_path()
    model_revision = _model_revision(model_path)
    if LEGAL_V2_PROFILE.embedding_dimension != EXPECTED_EMBEDDING_DIM:
        raise SystemExit("INDEX_CONFIG_MISMATCH: embedding dimension")
    if LEGAL_V2_PROFILE.embedding_model != "BAAI/bge-m3":
        raise SystemExit(
            f"INDEX_CONFIG_MISMATCH: embedding model {LEGAL_V2_PROFILE.embedding_model}"
        )

    sides: list[str] = []
    if args.side in ("A", "both"):
        sides.append("A")
    if args.side in ("B", "both"):
        sides.append("B")

    built: dict[str, Any] = {}
    for side in sides:
        chunker = CHUNKER_A_CURRENT if side == "A" else CHUNKER_B_CONTEXTUAL_PACKED_V1
        expected = EXPECTED_A_CHUNKS if side == "A" else EXPECTED_B_CHUNKS
        payloads, ordered_ids, eclis = _build_side_chunks(
            ordered=ordered,
            meta_by_ecli=meta_by_ecli,
            api=args.api,
            chunker_version=chunker,
        )
        if len(payloads) != expected:
            raise SystemExit(
                f"EXPERIMENT_BLOCKED: {side} chunk count {len(payloads)} != {expected}"
            )
        if eclis != inventory_eclis:
            raise SystemExit(f"EXPERIMENT_BLOCKED: {side} ECLI set mismatch")
        if len(ordered_ids) != len(set(ordered_ids)):
            raise SystemExit(f"EXPERIMENT_BLOCKED: {side} duplicate chunk IDs")
        built[side] = {
            "payloads": payloads,
            "ordered_ids": ordered_ids,
            "eclis": eclis,
            "chunker_version": chunker,
            "source_chunk_count": len(payloads),
            "ordered_chunk_id_hash": _ordered_id_hash(ordered_ids),
            "chunk_id_set_hash": _set_hash(set(ordered_ids)),
        }
        print(
            f"SIDE {side} chunks={len(payloads)} hash={built[side]['ordered_chunk_id_hash'][:16]}",
            flush=True,
        )

    if args.chunks_only:
        def _side_public(side: dict[str, Any]) -> dict[str, Any]:
            return {
                "chunker_version": side.get("chunker_version"),
                "source_chunk_count": side.get("source_chunk_count"),
                "ordered_chunk_id_hash": side.get("ordered_chunk_id_hash"),
                "chunk_id_set_hash": side.get("chunk_id_set_hash"),
                "unique_ecli_count": len(side.get("eclis") or []),
            }

        report = {
            "classification": "CHUNKS_REPRODUCED",
            "git_commit": git_commit,
            "parser_version": PARSER_VERSION,
            "A": _side_public(built.get("A") or {}),
            "B": _side_public(built.get("B") or {}),
        }
        (out_dir / "chunks_only_repro.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("CHUNKS_ONLY_OK", flush=True)
        return 0

    from qdrant_client import QdrantClient  # type: ignore[import-not-found]

    qdrant = QdrantClient(url=args.qdrant_url, timeout=120)
    existing = {c.name for c in qdrant.get_collections().collections}
    for name in (COLLECTION_A, COLLECTION_B):
        if name in PROTECTED:
            raise SystemExit(f"EXPERIMENT_BLOCKED: protected {name}")
        if name in existing and args.force_recreate:
            print(f"WARN --force-recreate: wiping existing collection {name}", flush=True)
        elif name in existing:
            print(f"RESUME: existing collection kept ({name})", flush=True)

    device = str(args.device).lower()
    if device == "cuda":
        try:
            import torch  # type: ignore[import]

            if not torch.cuda.is_available():
                raise SystemExit(
                    "EXPERIMENT_BLOCKED: --device cuda but torch.cuda.is_available() is False"
                )
            print(
                json.dumps(
                    {
                        "event": "chunk_ab_slice4_cuda",
                        "device_name": torch.cuda.get_device_name(0),
                        "cuda_version": getattr(torch.version, "cuda", None),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except ImportError as exc:
            raise SystemExit(
                "EXPERIMENT_BLOCKED: --device cuda requires torch with CUDA"
            ) from exc
        embedder = _ExperimentBgeM3Embedder(
            model_path=model_path,
            device="cuda",
            encode_batch_size=max(args.batch_size, 16),
        )
    else:
        prod_config = ProductionRetrievalConfig(
            profile=LEGAL_V2_PROFILE,
            qdrant_collection=COLLECTION_A,
            bm25_sidecar_path=bm25_path_a,
            bm25_index_id=BM25_ID_A,
            model_path=model_path,
            local_files_only=True,
            trust_remote_code=False,
            device="cpu",
            candidate_multiplier=1,
            min_candidate_count=1,
            max_candidate_count=1,
            lexical_filter_enabled=False,
        )
        embedder = _BatchedBgeM3Embedder(
            BgeM3Embedder(prod_config), encode_batch_size=max(args.batch_size, 16)
        )

    index_results: dict[str, Any] = {}
    for side in sides:
        chunker = built[side]["chunker_version"]
        collection = COLLECTION_A if side == "A" else COLLECTION_B
        bm25_id = BM25_ID_A if side == "A" else BM25_ID_B
        bm25_path = bm25_path_a if side == "A" else bm25_path_b
        config = _make_build_config(
            collection=collection,
            bm25_id=bm25_id,
            bm25_path=bm25_path,
            output_dir=out_dir / f"build_{side.lower()}",
            batch_size=args.batch_size,
            document_batch_size=args.document_batch_size,
            force_recreate=bool(args.force_recreate),
        )
        print(f"INDEXING {side} -> {collection}", flush=True)
        summary = _index_payloads(
            payloads=built[side]["payloads"],
            config=config,
            embedder=embedder,
            qdrant_client=qdrant,
            force_recreate=bool(args.force_recreate),
        )
        dense_ids = ib._qdrant_payload_chunk_ids(qdrant, collection)
        bm25_ids = _bm25_chunk_ids(bm25_path)
        source_ids = set(built[side]["ordered_ids"])
        dense_eclis = _qdrant_ecli_set(qdrant, collection)
        bm25_eclis = _bm25_ecli_set(bm25_path)
        index_results[side] = {
            "collection": collection,
            "bm25_index_id": bm25_id,
            "bm25_path": str(bm25_path.as_posix()),
            "source_chunk_count": built[side]["source_chunk_count"],
            "dense_vector_count": len(dense_ids),
            "bm25_logical_unit_count": len(bm25_ids),
            "already_indexed_at_start": summary.get("already_indexed"),
            "upserted_this_run": summary.get("qdrant_upsert_points"),
            "source_chunk_id_hash": built[side]["ordered_chunk_id_hash"],
            "dense_identity_hash": _set_hash(dense_ids),
            "bm25_identity_hash": _set_hash(bm25_ids),
            "source_eq_dense": source_ids == dense_ids,
            "source_eq_bm25": source_ids == bm25_ids,
            "dense_eq_bm25": dense_ids == bm25_ids,
            "unique_ecli_count": len(dense_eclis),
            "dense_eclis_eq_inventory": dense_eclis == inventory_eclis,
            "bm25_eclis_eq_inventory": bm25_eclis == inventory_eclis,
            "duplicate_source_ids": len(built[side]["ordered_ids"]) - len(source_ids),
            "finite_ok": summary["finite_ok"],
            "nan_or_inf_vectors": summary["nan_or_inf_vectors"],
            "norm_samples_mean": (
                sum(summary["norm_samples"]) / len(summary["norm_samples"])
                if summary["norm_samples"]
                else None
            ),
            "norm_within_tolerance": all(
                abs(n - 1.0) < 0.05 for n in summary["norm_samples"]
            )
            if summary["norm_samples"]
            else False,
            "payload_samples": _payload_sample(
                qdrant, collection, list(inventory.get("documents") or [])
            ),
        }

    ok = True
    reasons: list[str] = []
    for side in sides:
        r = index_results[side]
        expected = EXPECTED_A_CHUNKS if side == "A" else EXPECTED_B_CHUNKS
        if r["source_chunk_count"] != expected:
            ok = False
            reasons.append(f"{side}_count")
        if not (
            r["source_eq_dense"]
            and r["source_eq_bm25"]
            and r["dense_eq_bm25"]
            and r["dense_eclis_eq_inventory"]
            and r["bm25_eclis_eq_inventory"]
            and r["finite_ok"]
            and r["duplicate_source_ids"] == 0
        ):
            ok = False
            reasons.append(f"{side}_integrity")
    classification = (
        "INDEX_AB_READY"
        if ok and set(sides) == {"A", "B"}
        else ("INDEX_INTEGRITY_FAILED" if reasons else "INDEX_EXPERIMENT_BLOCKED")
    )
    fast_safe = "YES" if classification == "INDEX_AB_READY" else "NO"

    manifest = {
        "generated_at": _utc_now(),
        "classification": classification,
        "FAST_AB_SAFE_TO_START": fast_safe,
        "reasons": reasons,
        "git_commit": git_commit,
        "parser_version": PARSER_VERSION,
        "parser_commit": EXPECTED_PARSER_COMMIT,
        "inventory_id": EXPECTED_INVENTORY_ID,
        "inventory_hash_sha256": EXPECTED_INVENTORY_HASH,
        "document_count": EXPECTED_DOC_COUNT,
        "inventory_ecli_set_hash": _set_hash(inventory_eclis),
        "a_implementation": "app.rag.legal_v2.ingest.chunking.build_hierarchical_chunks",
        "a_chunker_version": CHUNKER_A_CURRENT,
        "b_implementation": (
            "app.rag.legal_v2.ingest.chunkers.contextual_packed_v1."
            "build_contextual_packed_chunks_v1"
        ),
        "b_chunker_version": CHUNKER_B_CONTEXTUAL_PACKED_V1,
        "b_policy_hash_sha256": b_hash,
        "embedding": {
            "model_name": LEGAL_V2_PROFILE.embedding_model,
            "model_path": model_path,
            "resolved_revision": model_revision,
            "dimension": LEGAL_V2_PROFILE.embedding_dimension,
            "normalization": True,
            "distance_metric": EXPECTED_DISTANCE,
            "device": str(args.device).lower(),
            "local_files_only": True,
            "batch_size": args.batch_size,
        },
        "force_recreate": bool(args.force_recreate),
        "resume_default": not bool(args.force_recreate),
        "corpus_level_ce_truncation_diagnostic": {
            "A": 0.8536,
            "B": 0.7922,
            "note": "Corpus-level only; not CE-7 selected-passage truncation.",
        },
        "A": {k: v for k, v in built.get("A", {}).items() if k != "payloads"}
        | index_results.get("A", {}),
        "B": {k: v for k, v in built.get("B", {}).items() if k != "payloads"}
        | index_results.get("B", {}),
        "no_fast_benchmark": True,
        "no_ce_scoring": True,
        "no_production_overwrite": True,
        "full_75k_untouched": True,
        "parser_unchanged": True,
        "chunkers_unchanged": True,
        "queryspec_unchanged": True,
        "bm25_rrf_config_unchanged": True,
        "golden_labels_unchanged": True,
    }
    for side in ("A", "B"):
        if side in manifest and isinstance(manifest[side], dict):
            manifest[side].pop("eclis", None)
            manifest[side].pop("ordered_ids", None)

    manifest_path = out_dir / "slice4_index_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md = [
        "# Slice 4 isolated A/B indexes (parser v8)",
        "",
        f"- classification: `{classification}`",
        f"- FAST_AB_SAFE_TO_START: `{fast_safe}`",
        f"- git_commit: `{git_commit}`",
        f"- parser_version: `{PARSER_VERSION}`",
        f"- force_recreate: `{bool(args.force_recreate)}`",
        f"- resume_default: `{not bool(args.force_recreate)}`",
        "",
        "## Collections",
        "",
        f"- A Qdrant: `{COLLECTION_A}`",
        f"- B Qdrant: `{COLLECTION_B}`",
        f"- A BM25: `{bm25_path_a.as_posix()}`",
        f"- B BM25: `{bm25_path_b.as_posix()}`",
        "",
        "## Counts",
        "",
    ]
    for side in sides:
        r = index_results[side]
        md.append(
            f"- {side}: source={r['source_chunk_count']} dense={r['dense_vector_count']} "
            f"bm25={r['bm25_logical_unit_count']} ecli={r['unique_ecli_count']} "
            f"aligned={r['source_eq_dense'] and r['source_eq_bm25']} "
            f"resumed_from={r.get('already_indexed_at_start')} "
            f"upserted_this_run={r.get('upserted_this_run')}"
        )
    md.extend(["", "STOP: FAST / CE-7 / production switch not started."])
    (out_dir / "slice4_index_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"WROTE {manifest_path}", flush=True)
    print(f"CLASSIFICATION {classification}", flush=True)
    print(f"FAST_AB_SAFE_TO_START {fast_safe}", flush=True)
    return 0 if classification == "INDEX_AB_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
