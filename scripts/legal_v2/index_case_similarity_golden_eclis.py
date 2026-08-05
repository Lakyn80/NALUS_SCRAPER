#!/usr/bin/env python3
"""Additive upsert of Case Similarity Golden v1 judgments under verified ECLI.

Does NOT recreate or wipe the target collection. Indexes only documents that
have a verified ECLI in case_similarity_document_identity_v1.json.

Sources (offline):
- reviewed-pool / supplemental text via load_case_similarity_corpus()
- production document_id / payload identity = verified ECLI
- source_document_id = doc-* review ID
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.benchmark.case_similarity_identity import (  # noqa: E402
    load_case_similarity_identity_map,
)
from app.rag.legal_v2.benchmark.corpus import load_case_similarity_corpus  # noqa: E402
from app.rag.legal_v2.identity import (  # noqa: E402
    IDENTITY_STATUS_VERIFIED,
    normalize_ecli,
    validate_decision_identity,
)
from app.rag.legal_v2.ingest.adapters import LegalSourceDocument  # noqa: E402
from app.rag.legal_v2.ingest.index_builder import (  # noqa: E402
    LegalV2BuildConfig,
    build_legal_v2_index,
)
from app.rag.legal_v2.ingest.indexing import LEGAL_V2_PROFILE  # noqa: E402
from app.rag.legal_v2.retriever import legal_v2_retriever_config_from_env  # noqa: E402
from app.rag.retrieval.bge_m3_embedder import BgeM3Embedder  # noqa: E402
from app.rag.retrieval.production_profile import ProductionRetrievalConfig  # noqa: E402

DEFAULT_COLLECTION = "nalus_legal_paragraph_chunks_v2_pilot_600"
DEFAULT_OUT = PROJECT_ROOT / "artifacts" / "legal_v2" / "case_similarity_golden_v1_pilot" / "ecli_index_build"


def _plain_text_from_bundle(bundle: Any) -> str:
    lines: list[str] = []
    for block in getattr(bundle, "blocks", []) or []:
        text = str(getattr(block, "raw_text", "") or "").strip()
        if text:
            lines.append(text)
    return "\n\n".join(lines)


def build_source_documents() -> list[LegalSourceDocument]:
    identity = load_case_similarity_identity_map()
    corpus = load_case_similarity_corpus()
    docs: list[LegalSourceDocument] = []
    for ref in corpus.documents:
        row = identity.get(ref.document_id)
        if not row or row.get("identity_status") != IDENTITY_STATUS_VERIFIED:
            continue
        ecli = validate_decision_identity(
            ecli=row.get("ecli"),
            canonical_document_id=row.get("canonical_document_id"),
        )
        bundle = corpus.bundles.get(ref.document_id)
        if bundle is None:
            raise KeyError(f"missing corpus bundle for {ref.document_id}")
        text = _plain_text_from_bundle(bundle)
        if not text.strip():
            raise ValueError(f"empty text for {ref.document_id} / {ecli}")
        metadata = {
            "ecli": ecli,
            "canonical_document_id": ecli,
            "source_document_id": ref.document_id,
            "case_number": ref.case_number or row.get("case_number"),
            "case_reference": ref.case_number or row.get("case_number"),
            "court": ref.court or row.get("court"),
            "decision_date": ref.decision_date or row.get("decision_date"),
            "decision_type": ref.decision_type,
            "language": "cs",
            "jurisdiction": "CZ",
        }
        docs.append(
            LegalSourceDocument(
                document_id=ecli,
                source="case_similarity_golden_v1",
                text=text,
                metadata=metadata,
                origin_path=f"case_similarity_corpus:{ref.document_id}",
            )
        )
    docs.sort(key=lambda item: item.document_id)
    return docs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--qdrant-collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--recreate-v2-collection",
        action="store_true",
        help="Dangerous: recreate the target collection. Default is additive upsert only.",
    )
    args = parser.parse_args(argv)

    documents = build_source_documents()
    if args.limit is not None:
        documents = documents[: args.limit]
    summary = {
        "document_count": len(documents),
        "eclis": [doc.document_id for doc in documents],
        "collection": args.qdrant_collection,
        "dry_run": bool(args.dry_run),
        "recreate": bool(args.recreate_v2_collection),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "planned_documents.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "planned", **summary}, ensure_ascii=False))
    if args.dry_run:
        return 0
    if args.recreate_v2_collection:
        raise SystemExit(
            "Refusing --recreate-v2-collection for golden ECLI upsert. "
            "Omit the flag to perform an additive upsert."
        )

    os.environ["NALUS_LEGAL_V2_QDRANT_COLLECTION"] = args.qdrant_collection
    retriever_config = legal_v2_retriever_config_from_env()
    from qdrant_client import QdrantClient  # type: ignore[import-not-found]

    bm25_index_id = os.getenv(
        "NALUS_LEGAL_V2_BM25_INDEX_ID",
        f"{args.qdrant_collection}_bm25",
    )
    bm25_path = Path(
        os.getenv(
            "NALUS_LEGAL_V2_BM25_SIDECAR_PATH",
            str(Path("storage") / "rag" / "bm25" / f"{bm25_index_id}.sqlite"),
        )
    )
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    # Additive upsert into an existing collection must not reuse a stale full-build checkpoint.
    resume = False
    checkpoint = args.output_dir / "legal_v2_execute_checkpoint.json"
    if checkpoint.exists():
        checkpoint.unlink()
    prod_config = ProductionRetrievalConfig(
        profile=LEGAL_V2_PROFILE,
        qdrant_collection=args.qdrant_collection,
        bm25_sidecar_path=bm25_path,
        bm25_index_id=bm25_index_id,
        model_path=retriever_config.model_path,
        local_files_only=True,
        trust_remote_code=False,
        device="cpu",
        candidate_multiplier=1,
        min_candidate_count=1,
        max_candidate_count=1,
        lexical_filter_enabled=False,
    )
    manifest = build_legal_v2_index(
        documents=documents,
        embedder=BgeM3Embedder(prod_config),
        qdrant_client=QdrantClient(url=args.qdrant_url, timeout=120),
        config=LegalV2BuildConfig(
            collection_name=args.qdrant_collection,
            bm25_index_id=bm25_index_id,
            bm25_path=bm25_path,
            output_dir=args.output_dir,
            recreate_collection=False,
            overwrite_bm25=False,
            resume=resume,
            allow_existing_collection=True,
            batch_size=32,
            document_batch_size=8,
        ),
    )
    print(json.dumps({"event": "indexed", "manifest": manifest.to_dict()}, ensure_ascii=False))
    return 0 if manifest.validation_status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
