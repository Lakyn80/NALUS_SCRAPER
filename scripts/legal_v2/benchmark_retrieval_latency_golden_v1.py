#!/usr/bin/env python3
"""Latency/cost tier benchmark: FAST A vs B+ColBERT vs CE B (quality already known).

Measures warm wall-clock latency with CUDA synchronize on GPU paths.
Does not change canonical FAST/CE profiles. No quality re-scoring.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.benchmark.case_similarity_golden import (  # noqa: E402
    DEFAULT_PILOT_DATASET,
    load_case_similarity_golden_jsonl,
)
from app.rag.legal_v2.identity import is_valid_ecli, normalize_ecli  # noqa: E402
from app.rag.legal_v2.indexing import LEGAL_V2_PROFILE  # noqa: E402
from app.rag.legal_v2.query_spec import build_query_spec_v2  # noqa: E402
from app.rag.legal_v2.rerank.config import CrossEncoderConfig  # noqa: E402
from app.rag.legal_v2.rerank.selectors.names import DIVERSIFIED_STAGE1_EVIDENCE_V1  # noqa: E402
from app.rag.legal_v2.rerank.service import CrossEncoderRerankingService  # noqa: E402
from app.rag.legal_v2.retrieve.colbert import (  # noqa: E402
    DEFAULT_COLBERT_MODEL,
    DEFAULT_INDEX_NAME,
    ColbertConfig,
    ColbertRetriever,
    PyLateColbertBackend,
)
from app.rag.legal_v2.retrieve.colbert_hybrid import retrieve_hybrid_plus_colbert  # noqa: E402
from app.rag.legal_v2.retrieve.retrieval_profiles import (  # noqa: E402
    CE_CANONICAL_BM25_INDEX_ID,
    CE_CANONICAL_QDRANT_COLLECTION,
    FAST_CANONICAL_BM25_INDEX_ID,
    FAST_CANONICAL_QDRANT_COLLECTION,
)
from app.rag.legal_v2.retriever import (  # noqa: E402
    LegalV2RetrieverConfig,
    build_live_legal_v2_retriever,
)
from app.rag.retrieval.bge_m3_embedder import BgeM3Embedder  # noqa: E402
from app.rag.retrieval.production_profile import ProductionRetrievalConfig  # noqa: E402

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "chunking_ab_pilot_300_v1"
    / "latency_tier_v1"
)
DEFAULT_INDEX_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "chunking_ab_pilot_300_v1"
    / "colbert_v1"
    / "index"
)
DEFAULT_MAPPING = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "chunking_ab_pilot_300_v1"
    / "colbert_v1"
    / "colbert_chunk_mapping.jsonl"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "chunking_ab_pilot_300_v1"
    / "colbert_v1"
    / "colbert_index_manifest.json"
)

# Known quality from prior golden evals (not remeasured here).
KNOWN_QUALITY = {
    "FAST_A": {"hit_at_10": 0.95, "mrr": 0.607, "hit_at_5": 0.80},
    "B_PLUS_COLBERT": {"hit_at_10": 0.95, "mrr": 0.625, "hit_at_5": 0.85},
    "CE_B": {"hit_at_10": 1.00, "mrr": 0.975, "hit_at_5": 1.00},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", type=Path, default=DEFAULT_PILOT_DATASET)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    p.add_argument("--dense-candidate-chunks", type=int, default=80)
    p.add_argument("--bm25-candidate-chunks", type=int, default=80)
    p.add_argument("--colbert-candidate-chunks", type=int, default=80)
    p.add_argument("--fused-candidate-chunks", type=int, default=120)
    p.add_argument("--candidate-documents", type=int, default=40)
    p.add_argument("--ce-candidate-documents", type=int, default=30)
    p.add_argument("--ce-passages-per-document", type=int, default=7)
    p.add_argument("--ce-evidence-pool-limit", type=int, default=40)
    p.add_argument("--ce-batch-size", type=int, default=8)
    p.add_argument("--ce-max-length", type=int, default=512)
    p.add_argument("--ce-model", default="BAAI/bge-reranker-v2-m3")
    p.add_argument("--ce-device", default="cuda")
    p.add_argument("--embedding-device", default="cpu")
    p.add_argument("--colbert-device", default="cuda")
    p.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_DIR)
    p.add_argument("--mapping-path", type=Path, default=DEFAULT_MAPPING)
    p.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--model", default=DEFAULT_COLBERT_MODEL)
    p.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--allow-download", action="store_true")
    p.add_argument(
        "--fast-bm25-sidecar-path",
        type=Path,
        default=Path(
            os.getenv(
                "NALUS_LATENCY_FAST_BM25_SIDECAR",
                "/app/storage/rag/bm25/"
                "nalus_legal_paragraph_bm25_v2_chunk_ab_v8_a_current_300.sqlite",
            )
        ),
    )
    p.add_argument(
        "--ce-bm25-sidecar-path",
        type=Path,
        default=Path(
            os.getenv(
                "NALUS_LATENCY_CE_BM25_SIDECAR",
                "/app/storage/rag/bm25/"
                "nalus_legal_paragraph_bm25_v2_chunk_ab_v8_b_contextual_300.sqlite",
            )
        ),
    )
    return p.parse_args(argv)


def _git_meta() -> dict[str, Any]:
    def _run(args: list[str]) -> str:
        try:
            return (
                subprocess.check_output(args, cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL)
                .decode("utf-8")
                .strip()
            )
        except Exception:  # noqa: BLE001
            return "unknown"

    return {
        "git_head": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "branch", "--show-current"]),
        "dirty": bool(_run(["git", "status", "--porcelain"])),
    }


def _cuda_sync_if_needed(device: str) -> None:
    cleaned = str(device or "").strip().lower()
    if not cleaned.startswith("cuda"):
        return
    try:
        import torch
    except Exception:  # noqa: BLE001
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def _agg(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "min": None, "max": None, "n": 0}
    ordered = sorted(values)
    return {
        "mean": float(statistics.mean(ordered)),
        "p50": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "n": len(ordered),
    }


def _embedder_config(config: LegalV2RetrieverConfig, *, device: str) -> ProductionRetrievalConfig:
    return ProductionRetrievalConfig(
        profile=LEGAL_V2_PROFILE,
        qdrant_collection=config.qdrant_collection,
        bm25_sidecar_path=config.bm25_sidecar_path,
        bm25_index_id=config.bm25_index_id,
        model_path=config.model_path,
        local_files_only=True,
        trust_remote_code=False,
        device=device,
        candidate_multiplier=1,
        min_candidate_count=1,
        max_candidate_count=max(
            config.dense_candidate_chunks,
            config.bm25_candidate_chunks,
        ),
        lexical_filter_enabled=False,
    )


def _build_retriever(
    *,
    client: Any,
    collection: str,
    bm25_path: Path,
    bm25_id: str,
    args: argparse.Namespace,
) -> Any:
    cfg = LegalV2RetrieverConfig(
        qdrant_collection=collection,
        bm25_sidecar_path=bm25_path,
        bm25_index_id=bm25_id,
        dense_candidate_chunks=args.dense_candidate_chunks,
        bm25_candidate_chunks=args.bm25_candidate_chunks,
        fused_candidate_chunks=args.fused_candidate_chunks,
        candidate_documents=max(args.candidate_documents, args.ce_candidate_documents),
        model_path=os.getenv("EMBEDDING_MODEL_NAME", "/app/models/BAAI/bge-m3"),
    )
    embedder = BgeM3Embedder(_embedder_config(cfg, device=str(args.embedding_device)))
    return build_live_legal_v2_retriever(client, embedder, cfg)


def _stage1_docs_for_ce(documents: list[Any], *, limit: int, evidence_limit: int) -> list[Any]:
    from app.rag.legal_v2.retrieve.case_similarity_search import (
        Stage1DocumentResult,
        Stage1Passage,
    )

    out: list[Stage1DocumentResult] = []
    for index, doc in enumerate(list(documents)[: max(0, limit)], start=1):
        raw_id = str(getattr(doc, "document_id", "") or "")
        meta = dict(getattr(doc, "metadata", None) or {})
        ecli_raw = str(meta.get("ecli") or raw_id)
        if not ecli_raw or not is_valid_ecli(ecli_raw):
            continue
        ecli = normalize_ecli(ecli_raw)
        chunk_evidence = [
            dict(item)
            for item in list(getattr(doc, "chunk_evidence", None) or [])[
                : max(0, evidence_limit)
            ]
            if isinstance(item, dict)
        ]
        passages: list[Stage1Passage] = []
        for item in chunk_evidence:
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            passages.append(
                Stage1Passage(
                    text=text,
                    chunk_id=str(item.get("chunk_id") or f"p-{len(passages)}"),
                    section=item.get("section"),
                    page=item.get("page"),
                    score=item.get("rrf_score"),
                    dense_rank=item.get("dense_rank"),
                    bm25_rank=item.get("bm25_rank"),
                    rrf_rank=item.get("rrf_rank"),
                    retrieval_channels=tuple(item.get("retrieval_channels") or ()),
                    chunk_position=item.get("chunk_position"),
                )
            )
        if not passages:
            for paragraph in list(getattr(doc, "paragraphs", None) or [])[:evidence_limit]:
                text = str(
                    getattr(paragraph, "normalized_text", None)
                    or getattr(paragraph, "original_text", None)
                    or ""
                ).strip()
                if not text:
                    continue
                passages.append(
                    Stage1Passage(
                        text=text,
                        chunk_id=str(
                            getattr(paragraph, "paragraph_id", "") or f"p-{len(passages)}"
                        ),
                    )
                )
        out.append(
            Stage1DocumentResult(
                rank=index,
                document_id=ecli,
                canonical_document_id=ecli,
                ecli=ecli,
                court=meta.get("court"),
                case_number=meta.get("case_number"),
                decision_date=meta.get("decision_date"),
                document_type=meta.get("document_type"),
                score=float(getattr(doc, "score", 0.0) or 0.0),
                relevant_passages=passages,
                dense_rank=getattr(doc, "dense_rank", None),
                bm25_rank=getattr(doc, "bm25_rank", None),
                rrf_score=getattr(doc, "rrf_score", None),
                metadata=meta,
                stage1_rank=index,
                stage1_score=float(getattr(doc, "score", 0.0) or 0.0),
                chunk_evidence=chunk_evidence,
            )
        )
    return out


def _cost_proxies(wall_ms: float, *, uses_gpu: bool, pair_count: int | None = None) -> dict[str, Any]:
    wall_s = float(wall_ms) / 1000.0
    return {
        "wall_s": wall_s,
        "gpu_sec_est": wall_s if uses_gpu else 0.0,
        "cpu_sec_est": 0.0 if uses_gpu else wall_s,
        "pair_count": pair_count,
    }


def _verdict(summary: dict[str, Any]) -> dict[str, Any]:
    fast = summary["FAST_A"]["wall_ms"]
    bal = summary["B_PLUS_COLBERT"]["wall_ms"]
    ce = summary["CE_B"]["wall_ms"]
    fast_p50 = fast.get("p50")
    bal_p50 = bal.get("p50")
    ce_p50 = ce.get("p50")
    reasons: list[str] = []
    if None in (fast_p50, bal_p50, ce_p50):
        return {
            "LATENCY_TIER_VERDICT": "INCONCLUSIVE",
            "reasons": ["missing p50 latency"],
        }
    bal_vs_fast = float(bal_p50) / max(float(fast_p50), 1e-9)
    bal_vs_ce = float(bal_p50) / max(float(ce_p50), 1e-9)
    reasons.append(f"B+ColBERT p50 / FAST p50 = {bal_vs_fast:.2f}x")
    reasons.append(f"B+ColBERT p50 / CE p50 = {bal_vs_ce:.2f}x")
    reasons.append("known quality vs FAST: +0.018 MRR, +0.05 Hit@5, Hit@10 tie")

    # Keep as BALANCED if clearly cheaper than CE and not catastrophically slower than FAST.
    # Archive if latency is close to CE (no room for a middle tier) or >> FAST without CE savings.
    if bal_vs_ce <= 0.55 and bal_vs_fast <= 4.0:
        label = "KEEP_COLBERT_AS_BALANCED"
        reasons.append("materially faster than CE while not extreme vs FAST")
    elif bal_vs_ce >= 0.85:
        label = "ARCHIVE_COLBERT"
        reasons.append("latency too close to CE — no distinct middle tier")
    elif bal_vs_fast >= 6.0 and bal_vs_ce > 0.55:
        label = "ARCHIVE_COLBERT"
        reasons.append("too slow vs FAST without enough savings vs CE")
    else:
        label = "INCONCLUSIVE"
        reasons.append("latency sits in a gray zone; need product judgment")
    return {
        "LATENCY_TIER_VERDICT": label,
        "bal_vs_fast_p50": bal_vs_fast,
        "bal_vs_ce_p50": bal_vs_ce,
        "reasons": reasons,
        "note": (
            "Quality already fixed from prior goldens. "
            "No profile activation in this step."
        ),
    }


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.index_path.exists():
        raise SystemExit(f"ColBERT index missing: {args.index_path}")
    if not args.mapping_path.exists():
        raise SystemExit(f"ColBERT mapping missing: {args.mapping_path}")
    if not args.fast_bm25_sidecar_path.exists():
        raise SystemExit(f"FAST BM25 missing: {args.fast_bm25_sidecar_path}")
    if not args.ce_bm25_sidecar_path.exists():
        raise SystemExit(f"CE BM25 missing: {args.ce_bm25_sidecar_path}")
    manifest = {}
    if args.manifest_path.exists():
        manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    if manifest and not bool(manifest.get("COLBERT_INDEX_READY")):
        raise SystemExit("COLBERT_INDEX_READY is false")

    items = load_case_similarity_golden_jsonl(args.benchmark)
    if len(items) != 20:
        raise SystemExit(f"expected 20 golden rows, found {len(items)}")

    try:
        from qdrant_client import QdrantClient  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing qdrant_client") from exc

    client = QdrantClient(url=args.qdrant_url, timeout=60)
    model_path = os.getenv("EMBEDDING_MODEL_NAME", "/app/models/BAAI/bge-m3")

    # Shared BGE embedder path: build separate retrievers for A and B indexes.
    fast_retriever = _build_retriever(
        client=client,
        collection=FAST_CANONICAL_QDRANT_COLLECTION,
        bm25_path=args.fast_bm25_sidecar_path,
        bm25_id=FAST_CANONICAL_BM25_INDEX_ID,
        args=args,
    )
    # Reuse same args but B collection — need separate embedder instance OK for fairness.
    ce_stage1_retriever = _build_retriever(
        client=client,
        collection=CE_CANONICAL_QDRANT_COLLECTION,
        bm25_path=args.ce_bm25_sidecar_path,
        bm25_id=CE_CANONICAL_BM25_INDEX_ID,
        args=args,
    )
    hybrid_b_retriever = ce_stage1_retriever  # same B corpus

    colbert_config = ColbertConfig(
        model_name=args.model,
        index_path=args.index_path,
        index_name=args.index_name,
        device=args.colbert_device,
        top_k=max(int(args.colbert_candidate_chunks), 10),
        batch_size=int(args.batch_size),
        concurrency_limit=1,
        mapping_path=args.mapping_path,
        allow_download=bool(args.allow_download),
    )
    backend = PyLateColbertBackend(colbert_config)
    colbert_retriever = ColbertRetriever(colbert_config, backend=backend)
    await backend.initialize()

    ce_service = CrossEncoderRerankingService(
        CrossEncoderConfig(
            enabled=True,
            model_id=args.ce_model,
            candidate_documents=args.ce_candidate_documents,
            passages_per_document=args.ce_passages_per_document,
            batch_size=args.ce_batch_size,
            device=args.ce_device,
            max_length=args.ce_max_length,
            allow_download=bool(args.allow_download),
            local_files_only=not bool(args.allow_download),
            aggregation="max",
            experiment_mode="ce_bge_v2m3_p7_diverse_v1",
            passage_selector=DIVERSIFIED_STAGE1_EVIDENCE_V1,
            evidence_pool_limit=args.ce_evidence_pool_limit,
        )
    )
    await asyncio.to_thread(ce_service._get_provider().load)

    warmup_query = items[0].query
    warmup_spec = build_query_spec_v2(warmup_query)

    # --- warmups (discard) ---
    print("WARMUP FAST_A", flush=True)
    await asyncio.to_thread(fast_retriever.retrieve, warmup_spec)

    print("WARMUP B_PLUS_COLBERT", flush=True)
    _cuda_sync_if_needed(args.colbert_device)
    await retrieve_hybrid_plus_colbert(
        hybrid_retriever=hybrid_b_retriever,
        colbert_retriever=colbert_retriever,
        query_spec=warmup_spec,
        colbert_candidate_chunks=int(args.colbert_candidate_chunks),
        fused_candidate_chunks=int(args.fused_candidate_chunks),
        candidate_documents=int(args.candidate_documents),
        rrf_k=int(LEGAL_V2_PROFILE.rrf_k),
    )
    _cuda_sync_if_needed(args.colbert_device)

    print("WARMUP CE_B", flush=True)
    _cuda_sync_if_needed(args.ce_device)
    stage1 = await asyncio.to_thread(ce_stage1_retriever.retrieve, warmup_spec)
    stage1_docs = _stage1_docs_for_ce(
        stage1.documents,
        limit=args.ce_candidate_documents,
        evidence_limit=args.ce_evidence_pool_limit,
    )
    await asyncio.to_thread(ce_service.rerank, warmup_query, stage1_docs, require_success=True)
    _cuda_sync_if_needed(args.ce_device)

    per_mode: dict[str, list[dict[str, Any]]] = {
        "FAST_A": [],
        "B_PLUS_COLBERT": [],
        "CE_B": [],
    }

    try:
        for item in items:
            qid = item.benchmark_id
            query = item.query
            spec = build_query_spec_v2(query)

            # FAST A (CPU embedding path — no CUDA sync required)
            t0 = time.perf_counter()
            fast_res = await asyncio.to_thread(fast_retriever.retrieve, spec)
            wall_fast = (time.perf_counter() - t0) * 1000.0
            d_fast = dict(fast_res.diagnostics or {})
            per_mode["FAST_A"].append(
                {
                    "query_id": qid,
                    "wall_ms": wall_fast,
                    "cuda_synchronized": False,
                    "device": str(args.embedding_device),
                    "dense_latency_ms": d_fast.get("dense_latency_ms"),
                    "bm25_latency_ms": d_fast.get("bm25_latency_ms"),
                    "rrf_latency_ms": d_fast.get("rrf_latency_ms"),
                    "total_retrieval_latency_ms": d_fast.get("total_retrieval_latency_ms"),
                    "cost": _cost_proxies(wall_fast, uses_gpu=False),
                }
            )

            # B+ColBERT (GPU ColBERT — CUDA sync)
            _cuda_sync_if_needed(args.colbert_device)
            t1 = time.perf_counter()
            hyb = await retrieve_hybrid_plus_colbert(
                hybrid_retriever=hybrid_b_retriever,
                colbert_retriever=colbert_retriever,
                query_spec=spec,
                colbert_candidate_chunks=int(args.colbert_candidate_chunks),
                fused_candidate_chunks=int(args.fused_candidate_chunks),
                candidate_documents=int(args.candidate_documents),
                rrf_k=int(LEGAL_V2_PROFILE.rrf_k),
            )
            _cuda_sync_if_needed(args.colbert_device)
            wall_hyb = (time.perf_counter() - t1) * 1000.0
            d_hyb = dict(hyb.diagnostics or {})
            per_mode["B_PLUS_COLBERT"].append(
                {
                    "query_id": qid,
                    "wall_ms": wall_hyb,
                    "cuda_synchronized": True,
                    "device_embedding": str(args.embedding_device),
                    "device_colbert": str(args.colbert_device),
                    "dense_latency_ms": d_hyb.get("dense_latency_ms"),
                    "bm25_latency_ms": d_hyb.get("bm25_latency_ms"),
                    "colbert_latency_ms": d_hyb.get("colbert_latency_ms"),
                    "rrf_latency_ms": d_hyb.get("rrf_latency_ms"),
                    "total_retrieval_latency_ms": d_hyb.get("total_retrieval_latency_ms"),
                    "cost": _cost_proxies(wall_hyb, uses_gpu=True),
                }
            )

            # CE B (GPU CE — CUDA sync); Stage1 B + CE-7
            _cuda_sync_if_needed(args.ce_device)
            t2 = time.perf_counter()
            stage1 = await asyncio.to_thread(ce_stage1_retriever.retrieve, spec)
            stage1_docs = _stage1_docs_for_ce(
                stage1.documents,
                limit=args.ce_candidate_documents,
                evidence_limit=args.ce_evidence_pool_limit,
            )
            reranked = await asyncio.to_thread(
                ce_service.rerank, query, stage1_docs, require_success=True
            )
            _cuda_sync_if_needed(args.ce_device)
            wall_ce = (time.perf_counter() - t2) * 1000.0
            d_s1 = dict(stage1.diagnostics or {})
            d_ce = getattr(reranked, "diagnostics", None)
            ce_ms = getattr(d_ce, "rerank_latency_ms", None) if d_ce is not None else None
            pair_count = getattr(d_ce, "pair_count", None) if d_ce is not None else None
            per_mode["CE_B"].append(
                {
                    "query_id": qid,
                    "wall_ms": wall_ce,
                    "cuda_synchronized": True,
                    "device_embedding": str(args.embedding_device),
                    "device_ce": str(args.ce_device),
                    "dense_latency_ms": d_s1.get("dense_latency_ms"),
                    "bm25_latency_ms": d_s1.get("bm25_latency_ms"),
                    "rrf_latency_ms": d_s1.get("rrf_latency_ms"),
                    "total_retrieval_latency_ms": d_s1.get("total_retrieval_latency_ms"),
                    "ce_latency_ms": ce_ms,
                    "pair_count": pair_count,
                    "cost": _cost_proxies(wall_ce, uses_gpu=True, pair_count=pair_count),
                }
            )
            print(
                f"QUERY {qid} FAST={wall_fast:.1f}ms HYB={wall_hyb:.1f}ms CE={wall_ce:.1f}ms",
                flush=True,
            )
    finally:
        await backend.close()

    summary: dict[str, Any] = {}
    for mode, rows in per_mode.items():
        walls = [float(r["wall_ms"]) for r in rows]
        summary[mode] = {
            "wall_ms": _agg(walls),
            "cuda_synchronized": bool(rows[0]["cuda_synchronized"]) if rows else False,
            "mean_dense_latency_ms": _mean_opt([r.get("dense_latency_ms") for r in rows]),
            "mean_bm25_latency_ms": _mean_opt([r.get("bm25_latency_ms") for r in rows]),
            "mean_colbert_latency_ms": _mean_opt([r.get("colbert_latency_ms") for r in rows]),
            "mean_rrf_latency_ms": _mean_opt([r.get("rrf_latency_ms") for r in rows]),
            "mean_ce_latency_ms": _mean_opt([r.get("ce_latency_ms") for r in rows]),
            "mean_pair_count": _mean_opt([r.get("pair_count") for r in rows]),
            "mean_gpu_sec_est": _mean_opt(
                [(r.get("cost") or {}).get("gpu_sec_est") for r in rows]
            ),
            "mean_cpu_sec_est": _mean_opt(
                [(r.get("cost") or {}).get("cpu_sec_est") for r in rows]
            ),
            "known_quality": KNOWN_QUALITY.get(mode),
        }

    verdict = _verdict(summary)
    command = " ".join(
        ["python", "scripts/legal_v2/benchmark_retrieval_latency_golden_v1.py", *sys.argv[1:]]
    )
    payload = {
        "schema": "latency_tier_benchmark.v1",
        "benchmark": {
            "dataset": str(args.benchmark),
            "query_count": len(items),
            "modes": ["FAST_A", "B_PLUS_COLBERT", "CE_B"],
            "depths": {
                "dense": args.dense_candidate_chunks,
                "bm25": args.bm25_candidate_chunks,
                "colbert": args.colbert_candidate_chunks,
                "fused": args.fused_candidate_chunks,
                "rrf_k": int(LEGAL_V2_PROFILE.rrf_k),
            },
            "ce": {
                "model": args.ce_model,
                "candidate_documents": args.ce_candidate_documents,
                "passages_per_document": args.ce_passages_per_document,
                "passage_selector": DIVERSIFIED_STAGE1_EVIDENCE_V1,
                "evidence_pool_limit": args.ce_evidence_pool_limit,
                "batch_size": args.ce_batch_size,
                "device": args.ce_device,
                "max_length": args.ce_max_length,
            },
            "devices": {
                "embedding": args.embedding_device,
                "colbert": args.colbert_device,
                "ce": args.ce_device,
            },
            "embedding_model_path": model_path,
            "command": command,
            "git": _git_meta(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "COLBERT_INDEX_READY": bool(manifest.get("COLBERT_INDEX_READY", True)),
        },
        "known_quality": KNOWN_QUALITY,
        "summary": summary,
        "per_query": per_mode,
        "verdict": verdict,
    }

    json_path = output_dir / "LATENCY_TIER_RESULTS.json"
    md_path = output_dir / "LATENCY_TIER_RESULTS.md"
    html_path = output_dir / "LATENCY_TIER_RESULTS.html"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_md(payload), encoding="utf-8")
    html_path.write_text(_render_html(payload), encoding="utf-8")
    print(f"WROTE {json_path}", flush=True)
    print(f"WROTE {md_path}", flush=True)
    print(f"WROTE {html_path}", flush=True)
    print(f"LATENCY TIER VERDICT: {verdict['LATENCY_TIER_VERDICT']}", flush=True)
    for mode in ("FAST_A", "B_PLUS_COLBERT", "CE_B"):
        w = summary[mode]["wall_ms"]
        print(
            f"{mode} wall_ms mean={w['mean']:.1f} p50={w['p50']:.1f} p95={w['p95']:.1f}",
            flush=True,
        )
    return 0


def _mean_opt(values: list[Any]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return float(statistics.mean(nums))


def _fmt(v: Any, digits: int = 1) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _render_md(payload: dict[str, Any]) -> str:
    v = payload["verdict"]
    s = payload["summary"]
    lines = [
        "# Latency Tier Benchmark",
        "",
        f"## LATENCY TIER VERDICT: {v['LATENCY_TIER_VERDICT']}",
        "",
        f"- bal/fast p50: `{_fmt(v.get('bal_vs_fast_p50'), 2)}x`",
        f"- bal/ce p50: `{_fmt(v.get('bal_vs_ce_p50'), 2)}x`",
        "",
        "| Mode | p50 wall_ms | p95 wall_ms | mean wall_ms | cuda_sync | known Hit@10 | known MRR |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for mode in ("FAST_A", "B_PLUS_COLBERT", "CE_B"):
        row = s[mode]
        w = row["wall_ms"]
        q = row.get("known_quality") or {}
        lines.append(
            f"| {mode} | {_fmt(w['p50'])} | {_fmt(w['p95'])} | {_fmt(w['mean'])} | "
            f"{row['cuda_synchronized']} | {_fmt(q.get('hit_at_10'), 2)} | {_fmt(q.get('mrr'), 3)} |"
        )
    lines.extend(["", "## Reasons", ""])
    for r in v.get("reasons") or []:
        lines.append(f"- {r}")
    lines.extend(["", f"Note: {v.get('note')}", ""])
    return "\n".join(lines)


def _render_html(payload: dict[str, Any]) -> str:
    v = payload["verdict"]
    s = payload["summary"]
    esc = html.escape
    rows = []
    for mode in ("FAST_A", "B_PLUS_COLBERT", "CE_B"):
        row = s[mode]
        w = row["wall_ms"]
        q = row.get("known_quality") or {}
        rows.append(
            "<tr>"
            f"<td>{esc(mode)}</td>"
            f"<td>{esc(_fmt(w['p50']))}</td>"
            f"<td>{esc(_fmt(w['p95']))}</td>"
            f"<td>{esc(_fmt(w['mean']))}</td>"
            f"<td>{esc(str(row['cuda_synchronized']))}</td>"
            f"<td>{esc(_fmt(q.get('hit_at_10'), 2))}</td>"
            f"<td>{esc(_fmt(q.get('mrr'), 3))}</td>"
            "</tr>"
        )
    reasons = "".join(f"<li>{esc(r)}</li>" for r in (v.get("reasons") or []))
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Latency Tier Results</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#f7f7f5;color:#1a1a1a}}
.verdict{{font-size:1.4rem;font-weight:700;padding:12px 16px;background:#fff;border-left:6px solid #0a7;margin-bottom:18px}}
table{{border-collapse:collapse;width:100%;background:#fff;margin:12px 0 24px}}
th,td{{border:1px solid #ddd;padding:8px;text-align:left}}
th{{background:#eee}}
</style></head><body>
<h1>Latency Tier Benchmark</h1>
<div class="verdict">LATENCY TIER VERDICT: {esc(v['LATENCY_TIER_VERDICT'])}</div>
<table><thead><tr>
<th>Mode</th><th>p50 wall_ms</th><th>p95 wall_ms</th><th>mean</th><th>cuda_sync</th><th>Hit@10</th><th>MRR</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table>
<ul>{reasons}</ul>
<p>{esc(str(v.get('note') or ''))}</p>
</body></html>
"""


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
