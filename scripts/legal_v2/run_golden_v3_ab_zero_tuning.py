#!/usr/bin/env python3
"""Zero-tuning A/B: Dense vs BM25 vs Hybrid on Golden v3 graded DEV.

Uses frozen DEV qrels. Does not tune parameters. Does not touch TEST.
Prefer one hybrid retrieve per query, then derive three document rankings
from the same dense/bm25/fused candidate pools (fair candidate depths).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.benchmark.case_similarity_golden_v3 import (  # noqa: E402
    EXPECTED_DEV_COUNT,
    load_case_similarity_golden_v3_jsonl,
)
from app.rag.legal_v2.benchmark.case_similarity_graded_eval import (  # noqa: E402
    GradedAggregateMetrics,
    GradedQueryEvalResult,
    QrelEntry,
    aggregate_graded_metrics,
    evaluate_graded_query,
    first_highly_relevant_rank,
    is_binary_relevant,
)
from app.rag.legal_v2.identity import is_valid_ecli, normalize_ecli  # noqa: E402
from app.rag.legal_v2.indexing import LEGAL_V2_PROFILE  # noqa: E402
from app.rag.legal_v2.query_spec import build_query_spec_v2  # noqa: E402
from app.rag.legal_v2.retrieve.retriever import (  # noqa: E402
    LegalV2RetrieverConfig,
    aggregate_legal_v2_documents,
    build_live_legal_v2_retriever,
)
from app.rag.retrieval.bge_m3_embedder import BgeM3Embedder  # noqa: E402
from app.rag.retrieval.production_profile import ProductionRetrievalConfig  # noqa: E402

FREEZE_LABEL = "DEV_QRELS_FROZEN_WITH_AGENT_LOW_GRADE_TAIL"
DEFAULT_COLLECTION = "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_a_current_full"
DEFAULT_BM25_ID = "nalus_legal_paragraph_bm25_v2_chunk_ab_v8_a_current_full"
DEFAULT_BM25_PATH = Path(
    "/app/storage/rag/bm25/nalus_legal_paragraph_bm25_v2_chunk_ab_v8_a_current_full.sqlite"
)
DEFAULT_BENCHMARK = PROJECT_ROOT / "benchmarks" / "legal_v2" / "case_similarity_golden_v3_graded.jsonl"
DEFAULT_QRELS = PROJECT_ROOT / "artifacts" / "legal_v2" / "golden_v3_graded" / "qrels_dev_reviewed.jsonl"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts" / "legal_v2" / "golden_v3_graded" / "ab_zero_tuning"
TOP_DOCS = 50
NDCG_TIE_EPS = 1e-9
HYBRID_CLEAR_GAIN = 0.005
DENSE_CLEAR_BM25_MARGIN = 0.05


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://nalus-scraper-qdrant-1:6333"),
    )
    parser.add_argument("--qdrant-collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--bm25-sidecar-path", type=Path, default=DEFAULT_BM25_PATH)
    parser.add_argument("--bm25-index-id", default=DEFAULT_BM25_ID)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--top-docs", type=int, default=TOP_DOCS)
    return parser.parse_args(argv)


def _embedder_config(config: LegalV2RetrieverConfig) -> ProductionRetrievalConfig:
    return ProductionRetrievalConfig(
        profile=LEGAL_V2_PROFILE,
        qdrant_collection=config.qdrant_collection,
        bm25_sidecar_path=config.bm25_sidecar_path,
        bm25_index_id=config.bm25_index_id,
        model_path=config.model_path,
        local_files_only=True,
        trust_remote_code=False,
        device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        candidate_multiplier=1,
        min_candidate_count=1,
        max_candidate_count=max(config.dense_candidate_chunks, config.bm25_candidate_chunks),
        lexical_filter_enabled=False,
    )


def _load_qrels(path: Path) -> dict[str, list[QrelEntry]]:
    by_query: dict[str, list[QrelEntry]] = defaultdict(list)
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
        query_id = str(row.get("query_id") or "").strip()
        document_id = str(row.get("document_id") or "").strip()
        if not query_id or not document_id:
            raise ValueError(f"{path}:{line_no}: missing query_id/document_id")
        grade = int(row["grade"])
        judgment_state = str(row.get("judgment_state") or "graded").strip() or "graded"
        by_query[query_id].append(
            QrelEntry(
                query_id=query_id,
                document_id=normalize_ecli(document_id) if is_valid_ecli(document_id) else document_id,
                grade=grade,
                judgment_state=judgment_state,
                review_reason=str(row.get("review_reason") or ""),
            )
        )
    return dict(by_query)


def _doc_ecli(doc: Any) -> str:
    meta = dict(getattr(doc, "metadata", None) or {})
    raw = str(meta.get("ecli") or getattr(doc, "document_id", "") or "").strip()
    if not raw:
        return ""
    if is_valid_ecli(raw):
        return normalize_ecli(raw)
    return raw


def _ranked_from_docs(docs: list[Any], *, limit: int) -> tuple[list[str], list[float | None]]:
    ranked: list[str] = []
    scores: list[float | None] = []
    for doc in docs:
        ecli = _doc_ecli(doc)
        if not ecli or ecli in ranked:
            continue
        ranked.append(ecli)
        score = getattr(doc, "score", None)
        rrf = getattr(doc, "rrf_score", None)
        if score is not None:
            scores.append(float(score))
        elif rrf is not None:
            scores.append(float(rrf))
        else:
            scores.append(None)
        if len(ranked) >= limit:
            break
    return ranked, scores


def _relevant_docs_at_k(ranked: list[str], qrels: list[QrelEntry], *, k: int) -> int:
    qmap = {
        (normalize_ecli(e.document_id) if is_valid_ecli(e.document_id) else e.document_id): e
        for e in qrels
    }
    hits = 0
    for doc_id in ranked[:k]:
        key = normalize_ecli(doc_id) if is_valid_ecli(doc_id) else doc_id
        entry = qmap.get(key)
        if entry and is_binary_relevant(entry.grade):
            hits += 1
    return hits


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return round(ordered[index], 3)


def _latency_block(values: list[float], *, total_wall_ms: float | None = None) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean_ms": round(statistics.fmean(values), 3) if values else None,
        "p50_ms": _percentile(values, 0.5),
        "p95_ms": _percentile(values, 0.95),
        "total_ms": round(sum(values), 3) if values else None,
        "total_wall_ms": round(total_wall_ms, 3) if total_wall_ms is not None else None,
    }


def _metrics_dict(metrics: GradedAggregateMetrics) -> dict[str, Any]:
    return {
        "total_queries": metrics.total_queries,
        "ndcg_at_5": metrics.ndcg_at_5,
        "ndcg_at_10": metrics.ndcg_at_10,
        "ndcg_at_20": metrics.ndcg_at_20,
        "precision_at_5": metrics.precision_at_5,
        "precision_at_10": metrics.precision_at_10,
        "recall_at_10": metrics.recall_at_10,
        "recall_at_20": metrics.recall_at_20,
        "recall_at_50": metrics.recall_at_50,
        "mrr_highly_relevant": metrics.mrr_highly_relevant,
        "success_at_10_highly_relevant": metrics.success_at_10_highly_relevant,
    }


def _winner(dense_ndcg: float | None, bm25_ndcg: float | None, hybrid_ndcg: float | None) -> str:
    scores = {
        "DENSE_WIN": dense_ndcg if dense_ndcg is not None else float("-inf"),
        "BM25_WIN": bm25_ndcg if bm25_ndcg is not None else float("-inf"),
        "HYBRID_WIN": hybrid_ndcg if hybrid_ndcg is not None else float("-inf"),
    }
    best = max(scores.values())
    leaders = [name for name, value in scores.items() if abs(value - best) <= NDCG_TIE_EPS]
    if len(leaders) != 1:
        return "TIE"
    return leaders[0]


def _decide_verdict(
    *,
    dense_ndcg: float,
    bm25_ndcg: float,
    hybrid_ndcg: float,
    win_counts: dict[str, int],
    bm25_helps_hybrid_count: int,
    bm25_hurts_dense_count: int,
    dense_beats_bm25_count: int,
    query_count: int,
) -> str:
    """Documented decision rules for SUMMARY.md."""
    hybrid_gain_vs_dense = hybrid_ndcg - dense_ndcg
    hybrid_gain_vs_bm25 = hybrid_ndcg - bm25_ndcg
    dense_vs_bm25 = dense_ndcg - bm25_ndcg

    if bm25_ndcg >= dense_ndcg + DENSE_CLEAR_BM25_MARGIN and bm25_ndcg >= hybrid_ndcg + HYBRID_CLEAR_GAIN:
        return "KEEP_BM25"

    if (
        hybrid_ndcg >= dense_ndcg + HYBRID_CLEAR_GAIN
        and hybrid_ndcg >= bm25_ndcg + HYBRID_CLEAR_GAIN
    ):
        return "KEEP_HYBRID"
    if (
        hybrid_ndcg >= dense_ndcg - NDCG_TIE_EPS
        and hybrid_ndcg >= bm25_ndcg - NDCG_TIE_EPS
        and bm25_helps_hybrid_count >= max(8, query_count // 4)
        and bm25_hurts_dense_count <= max(6, query_count // 5)
        and win_counts.get("HYBRID_WIN", 0) >= win_counts.get("DENSE_WIN", 0)
    ):
        return "KEEP_HYBRID"

    if (
        dense_ndcg >= hybrid_ndcg - NDCG_TIE_EPS
        and hybrid_gain_vs_dense < HYBRID_CLEAR_GAIN
        and dense_vs_bm25 >= DENSE_CLEAR_BM25_MARGIN
    ):
        return "KEEP_DENSE"
    if (
        dense_ndcg >= bm25_ndcg - NDCG_TIE_EPS
        and dense_ndcg >= hybrid_ndcg - NDCG_TIE_EPS
        and hybrid_gain_vs_dense < HYBRID_CLEAR_GAIN
        and dense_beats_bm25_count >= max(10, query_count // 3)
        and bm25_hurts_dense_count >= bm25_helps_hybrid_count
    ):
        return "KEEP_DENSE"

    if abs(hybrid_gain_vs_dense) < HYBRID_CLEAR_GAIN and abs(hybrid_gain_vs_bm25) < HYBRID_CLEAR_GAIN:
        return "REQUIRES_TARGETED_DEV_TUNING"
    if bm25_helps_hybrid_count > 0 and bm25_hurts_dense_count > 0 and abs(hybrid_gain_vs_dense) < 0.02:
        return "REQUIRES_TARGETED_DEV_TUNING"
    return "REQUIRES_TARGETED_DEV_TUNING"


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _build_summary_md(
    *,
    freeze_label: str,
    metrics: dict[str, dict[str, Any]],
    latency: dict[str, Any],
    win_counts: dict[str, int],
    flag_lists: dict[str, list[str]],
    verdict: str,
    config_snapshot: dict[str, Any],
    query_count: int,
) -> str:
    lines: list[str] = []
    lines.append("# Golden v3 graded — zero-tuning Dense vs BM25 vs Hybrid (DEV)")
    lines.append("")
    lines.append(f"- Freeze label: `{freeze_label}`")
    lines.append("- Scope: **DEV only** (40 queries). TEST untouched.")
    lines.append("- Zero-tuning: production defaults only; no parameter search.")
    lines.append("- No ColBERT / cross-encoder / reranking.")
    lines.append(
        f"- Collection: `{config_snapshot['qdrant_collection']}`; "
        f"BM25 index: `{config_snapshot['bm25_index_id']}`"
    )
    lines.append(
        f"- Candidates: dense={config_snapshot['dense_candidate_chunks']}, "
        f"bm25={config_snapshot['bm25_candidate_chunks']}, "
        f"fused={config_snapshot['fused_candidate_chunks']}, "
        f"docs={config_snapshot['candidate_documents']}, "
        f"rrf_k={config_snapshot['rrf_k']}, "
        f"bm25_k1={config_snapshot['bm25_k1']}, bm25_b={config_snapshot['bm25_b']}"
    )
    lines.append(f"- Queries evaluated: {query_count}")
    lines.append("")
    lines.append("## Aggregate metrics (mean over DEV)")
    lines.append("")
    lines.append(
        "| System | nDCG@10 | nDCG@5 | nDCG@20 | P@5 | P@10 | R@10 | R@20 | R@50 | "
        "MRR_highly | Success@10_highly |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in ("dense", "bm25", "hybrid"):
        m = metrics[name]
        lines.append(
            f"| {name.upper()} | {_fmt(m['ndcg_at_10'])} | {_fmt(m['ndcg_at_5'])} | "
            f"{_fmt(m['ndcg_at_20'])} | {_fmt(m['precision_at_5'])} | {_fmt(m['precision_at_10'])} | "
            f"{_fmt(m['recall_at_10'])} | {_fmt(m['recall_at_20'])} | {_fmt(m['recall_at_50'])} | "
            f"{_fmt(m['mrr_highly_relevant'])} | {_fmt(m['success_at_10_highly_relevant'])} |"
        )
    lines.append("")
    lines.append("## Latency")
    lines.append("")
    lines.append("| System | mean_ms | p50_ms | p95_ms | total_ms | note |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for name, note in (
        ("dense", "dense channel from hybrid diagnostics"),
        ("bm25", "bm25 channel from hybrid diagnostics"),
        ("hybrid", "total retrieval wall per query"),
    ):
        block = latency["per_system"][name]
        lines.append(
            f"| {name.upper()} | {_fmt(block.get('mean_ms'), 1)} | {_fmt(block.get('p50_ms'), 1)} | "
            f"{_fmt(block.get('p95_ms'), 1)} | {_fmt(block.get('total_ms'), 1)} | {note} |"
        )
    lines.append(
        f"- Overall run wall clock: {_fmt(latency.get('overall_wall_ms'), 1)} ms"
    )
    lines.append("")
    lines.append("## Win counts (by nDCG@10)")
    lines.append("")
    for key in ("DENSE_WIN", "BM25_WIN", "HYBRID_WIN", "TIE"):
        lines.append(f"- {key}: {win_counts.get(key, 0)}")
    lines.append("")
    lines.append("## Interpretation flags")
    lines.append("")
    lines.append(
        f"- BM25 helps Hybrid (`hybrid_ndcg > dense_ndcg`): "
        f"**{len(flag_lists['bm25_helps_hybrid'])}** — {', '.join(flag_lists['bm25_helps_hybrid']) or 'none'}"
    )
    lines.append(
        f"- BM25 hurts Dense after fusion (`hybrid_ndcg < dense_ndcg`): "
        f"**{len(flag_lists['bm25_hurts_dense'])}** — {', '.join(flag_lists['bm25_hurts_dense']) or 'none'}"
    )
    lines.append(
        f"- Dense clearly beats BM25 (`dense_ndcg > bm25_ndcg + 0.05`): "
        f"**{len(flag_lists['dense_beats_bm25'])}** — {', '.join(flag_lists['dense_beats_bm25']) or 'none'}"
    )
    lines.append(
        f"- BM25 lexical useful vs Dense (`bm25_ndcg > dense_ndcg`): "
        f"**{len(flag_lists['bm25_lexical_useful'])}** — {', '.join(flag_lists['bm25_lexical_useful']) or 'none'}"
    )
    lines.append("")
    lines.append("## Verdict rules")
    lines.append("")
    lines.append(
        "- `KEEP_DENSE` if Dense best nDCG@10 and Hybrid does not clearly improve "
        "(hybrid <= dense or gain < 0.005) and Dense clearly ahead of BM25."
    )
    lines.append("- `KEEP_BM25` if BM25 best by clear margin.")
    lines.append(
        "- `KEEP_HYBRID` if Hybrid best nDCG@10 by >= 0.005 over both, or Hybrid best and "
        "helps on many queries without large regressions."
    )
    lines.append(
        "- `REQUIRES_TARGETED_DEV_TUNING` if Hybrid and Dense trade wins with mixed signals / "
        "no clear winner / Hybrid helps some but hurts many."
    )
    lines.append("")
    lines.append("## Interpretation note")
    lines.append("")
    lines.append(
        "This measures **semantic case similarity** retrieval quality on graded DEV qrels. "
        "Do **not** conclude BM25 is globally useless — lexical anchors can still matter for "
        "identifiers, rare phrases, or other task types outside this benchmark."
    )
    lines.append("")
    lines.append("## Final verdict")
    lines.append("")
    lines.append(f"**{verdict}**")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_started = time.perf_counter()
    items = load_case_similarity_golden_v3_jsonl(args.benchmark)
    dev_items = [item for item in items if item.split == "dev"]
    if len(dev_items) != EXPECTED_DEV_COUNT:
        raise SystemExit(
            f"Expected {EXPECTED_DEV_COUNT} DEV queries, got {len(dev_items)} from {args.benchmark}"
        )
    qrels_by_query = _load_qrels(args.qrels)
    missing_qrels = [item.query_id for item in dev_items if item.query_id not in qrels_by_query]
    if missing_qrels:
        raise SystemExit(f"Missing qrels for DEV queries: {missing_qrels[:10]}")

    if not args.bm25_sidecar_path.exists():
        raise SystemExit(f"BM25 sidecar missing: {args.bm25_sidecar_path}")

    try:
        from qdrant_client import QdrantClient
    except ModuleNotFoundError as exc:
        raise SystemExit("Run inside the API container (qdrant_client required).") from exc

    client = QdrantClient(url=args.qdrant_url, timeout=120)
    collections = {c.name for c in client.get_collections().collections}
    if args.qdrant_collection not in collections:
        raise SystemExit(
            f"Collection {args.qdrant_collection!r} not found at {args.qdrant_url}. "
            f"Available count={len(collections)}"
        )

    model_path = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "/root/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/"
        "5617a9f61b028005a4858fdac845db406aefb181",
    )
    config = LegalV2RetrieverConfig(
        qdrant_collection=args.qdrant_collection,
        bm25_sidecar_path=args.bm25_sidecar_path,
        bm25_index_id=args.bm25_index_id,
        model_path=model_path,
        dense_candidate_chunks=80,
        bm25_candidate_chunks=80,
        fused_candidate_chunks=120,
        candidate_documents=args.top_docs,
        bm25_enabled=True,
    )
    retriever = build_live_legal_v2_retriever(
        client, BgeM3Embedder(_embedder_config(config)), config
    )

    config_snapshot = {
        "freeze_label": FREEZE_LABEL,
        "scope": "DEV_only",
        "test_touched": False,
        "zero_tuning": True,
        "qdrant_url": args.qdrant_url,
        "qdrant_collection": args.qdrant_collection,
        "bm25_index_id": args.bm25_index_id,
        "bm25_sidecar_path": str(args.bm25_sidecar_path),
        "dense_candidate_chunks": config.dense_candidate_chunks,
        "bm25_candidate_chunks": config.bm25_candidate_chunks,
        "fused_candidate_chunks": config.fused_candidate_chunks,
        "candidate_documents": config.candidate_documents,
        "rrf_k": LEGAL_V2_PROFILE.rrf_k,
        "bm25_k1": LEGAL_V2_PROFILE.bm25_k1,
        "bm25_b": LEGAL_V2_PROFILE.bm25_b,
        "reranking": False,
        "colbert": False,
        "cross_encoder": False,
        "model_path": model_path,
        "benchmark": str(args.benchmark),
        "qrels": str(args.qrels),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    dense_eval: list[GradedQueryEvalResult] = []
    bm25_eval: list[GradedQueryEvalResult] = []
    hybrid_eval: list[GradedQueryEvalResult] = []
    dense_latencies: list[float] = []
    bm25_latencies: list[float] = []
    hybrid_latencies: list[float] = []
    per_query_rows: list[dict[str, Any]] = []
    dense_result_rows: list[dict[str, Any]] = []
    bm25_result_rows: list[dict[str, Any]] = []
    hybrid_result_rows: list[dict[str, Any]] = []
    win_counts = {"DENSE_WIN": 0, "BM25_WIN": 0, "HYBRID_WIN": 0, "TIE": 0}
    flag_lists: dict[str, list[str]] = {
        "bm25_helps_hybrid": [],
        "bm25_hurts_dense": [],
        "dense_beats_bm25": [],
        "bm25_lexical_useful": [],
    }

    total = len(dev_items)
    for index, item in enumerate(dev_items, start=1):
        query_spec = build_query_spec_v2(item.query_text)
        qrels = qrels_by_query[item.query_id]
        legacy_primary = normalize_ecli(
            item.legacy_primary_ecli or item.legacy_primary_document_id
        )

        started = time.perf_counter()
        retrieval = retriever.retrieve(query_spec)
        wall_ms = (time.perf_counter() - started) * 1000.0
        diag = dict(retrieval.diagnostics or {})
        dense_ms = float(diag.get("dense_latency_ms") or 0.0)
        bm25_ms = float(diag.get("bm25_latency_ms") or 0.0)
        hybrid_ms = float(diag.get("total_retrieval_latency_ms") or wall_ms)
        dense_latencies.append(dense_ms)
        bm25_latencies.append(bm25_ms)
        hybrid_latencies.append(hybrid_ms)

        dense_docs = aggregate_legal_v2_documents(
            list(retrieval.dense_results),
            dense=list(retrieval.dense_results),
            bm25=[],
            query_spec=query_spec,
            limit=args.top_docs,
        )
        bm25_docs = aggregate_legal_v2_documents(
            list(retrieval.bm25_results),
            dense=[],
            bm25=list(retrieval.bm25_results),
            query_spec=query_spec,
            limit=args.top_docs,
        )
        hybrid_docs = list(retrieval.documents)[: args.top_docs]

        dense_ranked, dense_scores = _ranked_from_docs(dense_docs, limit=args.top_docs)
        bm25_ranked, bm25_scores = _ranked_from_docs(bm25_docs, limit=args.top_docs)
        hybrid_ranked, hybrid_scores = _ranked_from_docs(hybrid_docs, limit=args.top_docs)

        dense_result = evaluate_graded_query(
            query_id=item.query_id,
            ranked_document_ids=dense_ranked,
            qrel_entries=qrels,
            legacy_primary_document_id=legacy_primary,
            legacy_query=item.query_text,
            legacy_query_style=item.query_type,
        )
        bm25_result = evaluate_graded_query(
            query_id=item.query_id,
            ranked_document_ids=bm25_ranked,
            qrel_entries=qrels,
            legacy_primary_document_id=legacy_primary,
            legacy_query=item.query_text,
            legacy_query_style=item.query_type,
        )
        hybrid_result = evaluate_graded_query(
            query_id=item.query_id,
            ranked_document_ids=hybrid_ranked,
            qrel_entries=qrels,
            legacy_primary_document_id=legacy_primary,
            legacy_query=item.query_text,
            legacy_query_style=item.query_type,
        )
        dense_eval.append(dense_result)
        bm25_eval.append(bm25_result)
        hybrid_eval.append(hybrid_result)

        dense_ndcg = float(dense_result.ndcg_at_10 or 0.0)
        bm25_ndcg = float(bm25_result.ndcg_at_10 or 0.0)
        hybrid_ndcg = float(hybrid_result.ndcg_at_10 or 0.0)
        winner = _winner(dense_result.ndcg_at_10, bm25_result.ndcg_at_10, hybrid_result.ndcg_at_10)
        win_counts[winner] = win_counts.get(winner, 0) + 1

        bm25_helps = hybrid_ndcg > dense_ndcg
        bm25_hurts = hybrid_ndcg < dense_ndcg
        dense_beats = dense_ndcg > bm25_ndcg + DENSE_CLEAR_BM25_MARGIN
        bm25_useful = bm25_ndcg > dense_ndcg
        if bm25_helps:
            flag_lists["bm25_helps_hybrid"].append(item.query_id)
        if bm25_hurts:
            flag_lists["bm25_hurts_dense"].append(item.query_id)
        if dense_beats:
            flag_lists["dense_beats_bm25"].append(item.query_id)
        if bm25_useful:
            flag_lists["bm25_lexical_useful"].append(item.query_id)

        dense_g3 = first_highly_relevant_rank(dense_ranked, dense_result.qrels)
        bm25_g3 = first_highly_relevant_rank(bm25_ranked, bm25_result.qrels)
        hybrid_g3 = first_highly_relevant_rank(hybrid_ranked, hybrid_result.qrels)

        per_query_rows.append(
            {
                "query_id": item.query_id,
                "dense_ndcg_at_10": dense_result.ndcg_at_10,
                "bm25_ndcg_at_10": bm25_result.ndcg_at_10,
                "hybrid_ndcg_at_10": hybrid_result.ndcg_at_10,
                "dense_first_grade3_rank": dense_g3,
                "bm25_first_grade3_rank": bm25_g3,
                "hybrid_first_grade3_rank": hybrid_g3,
                "dense_relevant_docs_at_10": _relevant_docs_at_k(dense_ranked, qrels, k=10),
                "bm25_relevant_docs_at_10": _relevant_docs_at_k(bm25_ranked, qrels, k=10),
                "hybrid_relevant_docs_at_10": _relevant_docs_at_k(hybrid_ranked, qrels, k=10),
                "winner": winner,
                "bm25_helps_hybrid": bm25_helps,
                "bm25_hurts_dense": bm25_hurts,
                "dense_beats_bm25": dense_beats,
                "bm25_lexical_useful": bm25_useful,
            }
        )

        def _system_row(
            *,
            system: str,
            ranked: list[str],
            scores: list[float | None],
            latency_ms: float,
            eval_result: GradedQueryEvalResult,
        ) -> dict[str, Any]:
            return {
                "query_id": item.query_id,
                "system": system,
                "ranked_document_ids": ranked,
                "scores": scores,
                "latency_ms": round(latency_ms, 3),
                "metrics": {
                    "ndcg_at_5": eval_result.ndcg_at_5,
                    "ndcg_at_10": eval_result.ndcg_at_10,
                    "ndcg_at_20": eval_result.ndcg_at_20,
                    "precision_at_5": eval_result.precision_at_5,
                    "precision_at_10": eval_result.precision_at_10,
                    "recall_at_10": eval_result.recall_at_10,
                    "recall_at_20": eval_result.recall_at_20,
                    "recall_at_50": eval_result.recall_at_50,
                    "mrr_highly_relevant": eval_result.mrr_highly_relevant,
                    "success_at_10_highly_relevant": eval_result.success_at_10_highly_relevant,
                },
            }

        dense_result_rows.append(
            _system_row(
                system="dense",
                ranked=dense_ranked,
                scores=dense_scores,
                latency_ms=dense_ms,
                eval_result=dense_result,
            )
        )
        bm25_result_rows.append(
            _system_row(
                system="bm25",
                ranked=bm25_ranked,
                scores=bm25_scores,
                latency_ms=bm25_ms,
                eval_result=bm25_result,
            )
        )
        hybrid_result_rows.append(
            _system_row(
                system="hybrid",
                ranked=hybrid_ranked,
                scores=hybrid_scores,
                latency_ms=hybrid_ms,
                eval_result=hybrid_result,
            )
        )

        print(
            f"[{index}/{total}] {item.query_id} "
            f"dense={_fmt(dense_result.ndcg_at_10)} "
            f"bm25={_fmt(bm25_result.ndcg_at_10)} "
            f"hybrid={_fmt(hybrid_result.ndcg_at_10)} "
            f"winner={winner} wall_ms={wall_ms:.0f}",
            flush=True,
        )

    overall_wall_ms = (time.perf_counter() - run_started) * 1000.0
    dense_metrics = _metrics_dict(aggregate_graded_metrics(dense_eval))
    bm25_metrics = _metrics_dict(aggregate_graded_metrics(bm25_eval))
    hybrid_metrics = _metrics_dict(aggregate_graded_metrics(hybrid_eval))
    metrics_payload = {
        "freeze_label": FREEZE_LABEL,
        "scope": "DEV_only",
        "query_count": total,
        "zero_tuning": True,
        "config": config_snapshot,
        "systems": {
            "dense": dense_metrics,
            "bm25": bm25_metrics,
            "hybrid": hybrid_metrics,
        },
        "win_counts": win_counts,
        "flags": {
            "bm25_helps_hybrid_count": len(flag_lists["bm25_helps_hybrid"]),
            "bm25_hurts_dense_count": len(flag_lists["bm25_hurts_dense"]),
            "dense_beats_bm25_count": len(flag_lists["dense_beats_bm25"]),
            "bm25_lexical_useful_count": len(flag_lists["bm25_lexical_useful"]),
            "bm25_helps_hybrid": flag_lists["bm25_helps_hybrid"],
            "bm25_hurts_dense": flag_lists["bm25_hurts_dense"],
            "dense_beats_bm25": flag_lists["dense_beats_bm25"],
            "bm25_lexical_useful": flag_lists["bm25_lexical_useful"],
        },
    }

    dense_ndcg10 = float(dense_metrics["ndcg_at_10"] or 0.0)
    bm25_ndcg10 = float(bm25_metrics["ndcg_at_10"] or 0.0)
    hybrid_ndcg10 = float(hybrid_metrics["ndcg_at_10"] or 0.0)
    verdict = _decide_verdict(
        dense_ndcg=dense_ndcg10,
        bm25_ndcg=bm25_ndcg10,
        hybrid_ndcg=hybrid_ndcg10,
        win_counts=win_counts,
        bm25_helps_hybrid_count=len(flag_lists["bm25_helps_hybrid"]),
        bm25_hurts_dense_count=len(flag_lists["bm25_hurts_dense"]),
        dense_beats_bm25_count=len(flag_lists["dense_beats_bm25"]),
        query_count=total,
    )
    metrics_payload["verdict"] = verdict

    latency_payload = {
        "freeze_label": FREEZE_LABEL,
        "method": "single_hybrid_retrieve_per_query",
        "overall_wall_ms": round(overall_wall_ms, 3),
        "per_system": {
            "dense": _latency_block(dense_latencies),
            "bm25": _latency_block(bm25_latencies),
            "hybrid": _latency_block(hybrid_latencies, total_wall_ms=overall_wall_ms),
        },
        "notes": {
            "dense": "dense_latency_ms from LegalV2HybridRetriever diagnostics",
            "bm25": "bm25_latency_ms from LegalV2HybridRetriever diagnostics",
            "hybrid": "total_retrieval_latency_ms from LegalV2HybridRetriever diagnostics",
            "fairness": (
                "One dense encode + one BM25 search + one RRF per query; "
                "document rankings aggregated separately from the same candidate pools."
            ),
        },
    }

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (args.artifacts_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.artifacts_dir / "latency.json").write_text(
        json.dumps(latency_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(args.artifacts_dir / "per_query.jsonl", per_query_rows)
    _write_jsonl(args.artifacts_dir / "dense_results.jsonl", dense_result_rows)
    _write_jsonl(args.artifacts_dir / "bm25_results.jsonl", bm25_result_rows)
    _write_jsonl(args.artifacts_dir / "hybrid_results.jsonl", hybrid_result_rows)
    summary = _build_summary_md(
        freeze_label=FREEZE_LABEL,
        metrics=metrics_payload["systems"],
        latency=latency_payload,
        win_counts=win_counts,
        flag_lists=flag_lists,
        verdict=verdict,
        config_snapshot=config_snapshot,
        query_count=total,
    )
    (args.artifacts_dir / "SUMMARY.md").write_text(summary, encoding="utf-8")

    print(
        json.dumps(
            {
                "verdict": verdict,
                "ndcg_at_10": {
                    "dense": dense_ndcg10,
                    "bm25": bm25_ndcg10,
                    "hybrid": hybrid_ndcg10,
                },
                "win_counts": win_counts,
                "artifacts_dir": str(args.artifacts_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
