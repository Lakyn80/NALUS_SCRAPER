#!/usr/bin/env python3
"""DEV-only weighted RRF fusion over frozen Dense/BM25 document rankings.

Research-only evaluator. Does not modify production routing, BM25, Dense,
qrels, or TEST. Reuses integrity-audited zero-tuning ranked document lists.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.benchmark.case_similarity_graded_eval import (  # noqa: E402
    QrelEntry,
    aggregate_graded_metrics,
    evaluate_graded_query,
)
from app.rag.legal_v2.identity import is_valid_ecli, normalize_ecli  # noqa: E402
from app.rag.legal_v2.indexing import LEGAL_V2_PROFILE  # noqa: E402

FREEZE_LABEL = "DEV_QRELS_FROZEN_WITH_AGENT_LOW_GRADE_TAIL"
BM25_BASELINE_NDCG10 = 0.5027
MEANINGFUL_GAIN = 0.01
DENSE_CLEAR_MARGIN = 0.05
TOP_DOCS = 50
WEIGHT_PAIRS: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (0.9, 0.1),
    (0.8, 0.2),
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),
)

DEFAULT_AB = PROJECT_ROOT / "artifacts" / "legal_v2" / "golden_v3_graded" / "ab_zero_tuning"
DEFAULT_QRELS = (
    PROJECT_ROOT / "artifacts" / "legal_v2" / "golden_v3_graded" / "qrels_dev_reviewed.jsonl"
)
DEFAULT_OUT = PROJECT_ROOT / "artifacts" / "legal_v2" / "golden_v3_graded" / "weighted_fusion_dev"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ab-dir", type=Path, default=DEFAULT_AB)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-docs", type=int, default=TOP_DOCS)
    parser.add_argument("--rrf-k", type=int, default=int(LEGAL_V2_PROFILE.rrf_k))
    return parser.parse_args(argv)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def _canon(doc_id: str) -> str:
    text = str(doc_id or "").strip()
    if not text:
        return ""
    return normalize_ecli(text) if is_valid_ecli(text) else text


def _load_qrels(path: Path) -> dict[str, list[QrelEntry]]:
    by_query: dict[str, list[QrelEntry]] = defaultdict(list)
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        query_id = str(row.get("query_id") or "").strip()
        document_id = _canon(str(row.get("document_id") or "").strip())
        if not query_id or not document_id:
            raise ValueError(f"{path}:{line_no}: missing query_id/document_id")
        by_query[query_id].append(
            QrelEntry(
                query_id=query_id,
                document_id=document_id,
                grade=int(row["grade"]),
                judgment_state=str(row.get("judgment_state") or "graded").strip() or "graded",
                review_reason=str(row.get("review_reason") or ""),
            )
        )
    return dict(by_query)


def _rank_map(ranked_ids: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for index, doc_id in enumerate(ranked_ids, start=1):
        key = _canon(doc_id)
        if not key or key in out:
            continue
        out[key] = index
    return out


def weighted_rrf_fuse(
    *,
    bm25_ranked: list[str],
    dense_ranked: list[str],
    w_bm25: float,
    w_dense: float,
    rrf_k: int,
    top_docs: int,
) -> list[str]:
    """Document-level weighted RRF. Research-only; not production routing."""
    bm25_ranks = _rank_map(bm25_ranked)
    dense_ranks = _rank_map(dense_ranked)
    scores: dict[str, float] = {}
    for doc_id, rank in bm25_ranks.items():
        if w_bm25 > 0:
            scores[doc_id] = scores.get(doc_id, 0.0) + w_bm25 / (rrf_k + rank)
    for doc_id, rank in dense_ranks.items():
        if w_dense > 0:
            scores[doc_id] = scores.get(doc_id, 0.0) + w_dense / (rrf_k + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [doc_id for doc_id, _ in ordered[:top_docs]]


def _metrics_dict(result: Any) -> dict[str, float | None]:
    return {
        "ndcg_at_5": result.ndcg_at_5,
        "ndcg_at_10": result.ndcg_at_10,
        "ndcg_at_20": result.ndcg_at_20,
        "precision_at_10": result.precision_at_10,
        "recall_at_10": result.recall_at_10,
        "recall_at_20": result.recall_at_20,
        "recall_at_50": result.recall_at_50,
        "mrr_highly_relevant": result.mrr_highly_relevant,
        "success_at_10_highly_relevant": (
            1.0 if result.success_at_10_highly_relevant else 0.0
        ),
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dense_rows = {row["query_id"]: row for row in _load_jsonl(args.ab_dir / "dense_results.jsonl")}
    bm25_rows = {row["query_id"]: row for row in _load_jsonl(args.ab_dir / "bm25_results.jsonl")}
    ab_per_query = {
        row["query_id"]: row for row in _load_jsonl(args.ab_dir / "per_query.jsonl")
    }
    if set(dense_rows) != set(bm25_rows):
        raise SystemExit("Dense/BM25 result query_id sets differ")
    query_ids = sorted(dense_rows.keys())
    if len(query_ids) != 40:
        raise SystemExit(f"Expected 40 DEV queries, got {len(query_ids)}")

    qrels_by_query = _load_qrels(args.qrels)
    missing = [qid for qid in query_ids if qid not in qrels_by_query]
    if missing:
        raise SystemExit(f"Missing qrels for: {missing[:5]}")

    dense_clear_queries = sorted(
        qid
        for qid, row in ab_per_query.items()
        if bool(row.get("dense_beats_bm25"))
        or (
            float(row.get("dense_ndcg_at_10") or 0.0)
            > float(row.get("bm25_ndcg_at_10") or 0.0) + DENSE_CLEAR_MARGIN
        )
    )
    bm25_strong_queries = sorted(
        qid
        for qid, row in ab_per_query.items()
        if bool(row.get("bm25_lexical_useful"))
        or float(row.get("bm25_ndcg_at_10") or 0.0)
        > float(row.get("dense_ndcg_at_10") or 0.0)
    )

    per_query_rows: list[dict[str, Any]] = []
    per_weight: list[dict[str, Any]] = []
    systems_aggregate: dict[str, dict[str, Any]] = {}

    for w_bm25, w_dense in WEIGHT_PAIRS:
        label = f"bm25_{w_bm25:.2f}_dense_{w_dense:.2f}"
        eval_results = []
        improved: list[tuple[str, float]] = []
        degraded: list[tuple[str, float]] = []
        deltas: list[float] = []
        dense_clear_deltas: list[dict[str, Any]] = []
        bm25_strong_deltas: list[dict[str, Any]] = []

        for query_id in query_ids:
            dense_ranked = list(dense_rows[query_id]["ranked_document_ids"])
            bm25_ranked = list(bm25_rows[query_id]["ranked_document_ids"])
            fused = weighted_rrf_fuse(
                bm25_ranked=bm25_ranked,
                dense_ranked=dense_ranked,
                w_bm25=w_bm25,
                w_dense=w_dense,
                rrf_k=args.rrf_k,
                top_docs=args.top_docs,
            )
            bm25_only = weighted_rrf_fuse(
                bm25_ranked=bm25_ranked,
                dense_ranked=dense_ranked,
                w_bm25=1.0,
                w_dense=0.0,
                rrf_k=args.rrf_k,
                top_docs=args.top_docs,
            )
            # Prefer baseline BM25 ranking identity for w=1.0:0.0
            if w_bm25 == 1.0 and w_dense == 0.0:
                fused = [_canon(d) for d in bm25_ranked if _canon(d)][: args.top_docs]

            qrels = qrels_by_query[query_id]
            fused_eval = evaluate_graded_query(
                query_id=query_id,
                ranked_document_ids=fused,
                qrel_entries=qrels,
                legacy_primary_document_id="",
            )
            bm25_eval = evaluate_graded_query(
                query_id=query_id,
                ranked_document_ids=bm25_only
                if not (w_bm25 == 1.0 and w_dense == 0.0)
                else fused,
                qrel_entries=qrels,
                legacy_primary_document_id="",
            )
            # Always compare against pure BM25 ranking from artifacts
            bm25_baseline_eval = evaluate_graded_query(
                query_id=query_id,
                ranked_document_ids=[_canon(d) for d in bm25_ranked if _canon(d)][
                    : args.top_docs
                ],
                qrel_entries=qrels,
                legacy_primary_document_id="",
            )
            eval_results.append(fused_eval)

            fused_ndcg = float(fused_eval.ndcg_at_10 or 0.0)
            bm25_ndcg = float(bm25_baseline_eval.ndcg_at_10 or 0.0)
            delta = fused_ndcg - bm25_ndcg
            deltas.append(delta)
            if delta > 1e-12:
                improved.append((query_id, delta))
            elif delta < -1e-12:
                degraded.append((query_id, delta))

            dense_ndcg = float(
                evaluate_graded_query(
                    query_id=query_id,
                    ranked_document_ids=[_canon(d) for d in dense_ranked if _canon(d)][
                        : args.top_docs
                    ],
                    qrel_entries=qrels,
                    legacy_primary_document_id="",
                ).ndcg_at_10
                or 0.0
            )
            row = {
                "query_id": query_id,
                "weight_label": label,
                "w_bm25": w_bm25,
                "w_dense": w_dense,
                "fused_ndcg_at_10": fused_eval.ndcg_at_10,
                "bm25_ndcg_at_10": bm25_ndcg,
                "dense_ndcg_at_10": dense_ndcg,
                "delta_vs_bm25_ndcg_at_10": delta,
                "fused_metrics": _metrics_dict(fused_eval),
                "ranked_document_ids": fused,
            }
            per_query_rows.append(row)

            if query_id in dense_clear_queries:
                dense_clear_deltas.append(
                    {
                        "query_id": query_id,
                        "dense_ndcg_at_10": dense_ndcg,
                        "bm25_ndcg_at_10": bm25_ndcg,
                        "fused_ndcg_at_10": fused_ndcg,
                        "delta_vs_bm25": delta,
                        "recovered": fused_ndcg > bm25_ndcg + 1e-12,
                    }
                )
            if query_id in bm25_strong_queries:
                bm25_strong_deltas.append(
                    {
                        "query_id": query_id,
                        "delta_vs_bm25": delta,
                        "degraded": delta < -1e-12,
                    }
                )

            _ = bm25_eval  # kept for symmetry / future debug

        agg = aggregate_graded_metrics(eval_results)
        mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
        worst = min(degraded, key=lambda x: x[1]) if degraded else None
        best = max(improved, key=lambda x: x[1]) if improved else None
        recall20 = float(agg.recall_at_20 or 0.0)

        weight_payload = {
            "weight_label": label,
            "w_bm25": w_bm25,
            "w_dense": w_dense,
            "metrics": {
                "ndcg_at_5": agg.ndcg_at_5,
                "ndcg_at_10": agg.ndcg_at_10,
                "ndcg_at_20": agg.ndcg_at_20,
                "precision_at_10": agg.precision_at_10,
                "recall_at_10": agg.recall_at_10,
                "recall_at_20": agg.recall_at_20,
                "recall_at_50": agg.recall_at_50,
                "mrr_highly_relevant": agg.mrr_highly_relevant,
                "success_at_10_highly_relevant": agg.success_at_10_highly_relevant,
            },
            "vs_bm25": {
                "queries_improved": len(improved),
                "queries_degraded": len(degraded),
                "queries_unchanged": len(query_ids) - len(improved) - len(degraded),
                "mean_ndcg_at_10_delta": mean_delta,
                "best_improvement": (
                    {"query_id": best[0], "delta": best[1]} if best else None
                ),
                "worst_degradation": (
                    {"query_id": worst[0], "delta": worst[1]} if worst else None
                ),
            },
            "dense_clear_subset": {
                "query_ids": dense_clear_queries,
                "count": len(dense_clear_queries),
                "recovered_count": sum(
                    1 for item in dense_clear_deltas if item["recovered"]
                ),
                "mean_delta_vs_bm25": (
                    sum(item["delta_vs_bm25"] for item in dense_clear_deltas)
                    / len(dense_clear_deltas)
                    if dense_clear_deltas
                    else None
                ),
                "per_query": dense_clear_deltas,
            },
            "bm25_strong_subset": {
                "query_ids": bm25_strong_queries,
                "count": len(bm25_strong_queries),
                "degraded_count": sum(
                    1 for item in bm25_strong_deltas if item["degraded"]
                ),
                "mean_delta_vs_bm25": (
                    sum(item["delta_vs_bm25"] for item in bm25_strong_deltas)
                    / len(bm25_strong_deltas)
                    if bm25_strong_deltas
                    else None
                ),
            },
            "recall_at_20": recall20,
        }
        per_weight.append(weight_payload)
        systems_aggregate[label] = weight_payload

    # Decision
    bm25_row = next(row for row in per_weight if row["w_bm25"] == 1.0 and row["w_dense"] == 0.0)
    bm25_ndcg = float(bm25_row["metrics"]["ndcg_at_10"] or 0.0)
    bm25_r20 = float(bm25_row["metrics"]["recall_at_20"] or 0.0)

    candidates = [
        row
        for row in per_weight
        if not (row["w_bm25"] == 1.0 and row["w_dense"] == 0.0)
    ]
    best_weighted = max(
        candidates,
        key=lambda row: float(row["metrics"]["ndcg_at_10"] or 0.0),
    )
    best_ndcg = float(best_weighted["metrics"]["ndcg_at_10"] or 0.0)
    best_delta = best_ndcg - bm25_ndcg
    best_r20 = float(best_weighted["metrics"]["recall_at_20"] or 0.0)
    major_r20_regression = (bm25_r20 - best_r20) >= 0.05

    if best_delta >= MEANINGFUL_GAIN and not major_r20_regression:
        verdict = "KEEP_WEIGHTED_HYBRID"
        selected = {
            "w_bm25": best_weighted["w_bm25"],
            "w_dense": best_weighted["w_dense"],
            "ndcg_at_10": best_ndcg,
            "absolute_delta_vs_bm25": best_delta,
            "recall_at_20": best_r20,
        }
    else:
        verdict = "KEEP_BM25"
        selected = None

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "freeze_label": FREEZE_LABEL,
        "scope": "DEV_only",
        "test_touched": False,
        "research_only": True,
        "production_routing_modified": False,
        "bm25_tuned": False,
        "dense_tuned": False,
        "rrf_k": args.rrf_k,
        "top_docs": args.top_docs,
        "source_ab_dir": str(args.ab_dir),
        "reported_bm25_baseline_ndcg_at_10": BM25_BASELINE_NDCG10,
        "recomputed_bm25_ndcg_at_10": bm25_ndcg,
        "decision_threshold_absolute_ndcg_at_10": MEANINGFUL_GAIN,
        "weights": WEIGHT_PAIRS,
        "dense_clear_queries": dense_clear_queries,
        "bm25_strong_queries": bm25_strong_queries,
        "systems": {
            row["weight_label"]: {
                "w_bm25": row["w_bm25"],
                "w_dense": row["w_dense"],
                "metrics": row["metrics"],
                "vs_bm25": row["vs_bm25"],
            }
            for row in per_weight
        },
        "best_weighted": {
            "weight_label": best_weighted["weight_label"],
            "w_bm25": best_weighted["w_bm25"],
            "w_dense": best_weighted["w_dense"],
            "ndcg_at_10": best_ndcg,
            "absolute_delta_vs_bm25": best_delta,
            "recall_at_20": best_r20,
            "major_recall_at_20_regression": major_r20_regression,
        },
        "selected": selected,
        "verdict": verdict,
    }

    (args.artifacts_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.artifacts_dir / "per_weight.json").write_text(
        json.dumps(per_weight, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.artifacts_dir / "per_query.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in per_query_rows) + "\n",
        encoding="utf-8",
    )

    lines: list[str] = []
    lines.append("# Golden v3 DEV — weighted RRF fusion (BM25-dominant)")
    lines.append("")
    lines.append("- Scope: **DEV only** (40 queries). TEST untouched.")
    lines.append("- Research-only document-level weighted RRF over frozen A/B rankings.")
    lines.append("- Production routing / BM25 / Dense parameters unchanged.")
    lines.append(f"- rrf_k={args.rrf_k}; top_docs={args.top_docs}")
    lines.append(
        f"- Recomputed BM25 nDCG@10={_fmt(bm25_ndcg)} "
        f"(reported baseline {BM25_BASELINE_NDCG10})"
    )
    lines.append("")
    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append(
        "| BM25:Dense | nDCG@10 | Δ vs BM25 | nDCG@5 | nDCG@20 | P@10 | R@10 | R@20 | "
        "R@50 | MRR_highly | Success@10_highly | improved | degraded |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in per_weight:
        m = row["metrics"]
        vs = row["vs_bm25"]
        delta = float(m["ndcg_at_10"] or 0.0) - bm25_ndcg
        lines.append(
            f"| {row['w_bm25']:.2f}:{row['w_dense']:.2f} | {_fmt(m['ndcg_at_10'])} | "
            f"{delta:+.4f} | {_fmt(m['ndcg_at_5'])} | {_fmt(m['ndcg_at_20'])} | "
            f"{_fmt(m['precision_at_10'])} | {_fmt(m['recall_at_10'])} | "
            f"{_fmt(m['recall_at_20'])} | {_fmt(m['recall_at_50'])} | "
            f"{_fmt(m['mrr_highly_relevant'])} | {_fmt(m['success_at_10_highly_relevant'])} | "
            f"{vs['queries_improved']} | {vs['queries_degraded']} |"
        )
    lines.append("")
    lines.append("## Dense-clear subset (~queries where Dense beat BM25 by >0.05)")
    lines.append("")
    lines.append(f"- Query count: {len(dense_clear_queries)}")
    lines.append(f"- IDs: {', '.join(dense_clear_queries)}")
    lines.append("")
    for row in per_weight:
        if row["w_dense"] == 0.0:
            continue
        sub = row["dense_clear_subset"]
        lines.append(
            f"- {row['w_bm25']:.2f}:{row['w_dense']:.2f} recovered "
            f"{sub['recovered_count']}/{sub['count']} "
            f"(mean Δ vs BM25 on subset={_fmt(sub['mean_delta_vs_bm25'])})"
        )
    lines.append("")
    lines.append("## BM25-strong subset damage")
    lines.append("")
    lines.append(f"- Query count: {len(bm25_strong_queries)}")
    for row in per_weight:
        if row["w_dense"] == 0.0:
            continue
        sub = row["bm25_strong_subset"]
        lines.append(
            f"- {row['w_bm25']:.2f}:{row['w_dense']:.2f} degraded "
            f"{sub['degraded_count']}/{sub['count']} "
            f"(mean Δ={_fmt(sub['mean_delta_vs_bm25'])})"
        )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(
        f"- Meaningful win threshold: absolute nDCG@10 ≥ +{MEANINGFUL_GAIN:.2f} "
        "vs BM25 without major Recall@20 regression (≥0.05 absolute drop)."
    )
    lines.append(
        f"- Best weighted: {best_weighted['w_bm25']:.2f}:{best_weighted['w_dense']:.2f} "
        f"nDCG@10={_fmt(best_ndcg)} (Δ={best_delta:+.4f}), "
        f"R@20={_fmt(best_r20)} (BM25 R@20={_fmt(bm25_r20)})"
    )
    if selected:
        lines.append(
            f"- Selected weighted hybrid: BM25={selected['w_bm25']:.2f}, "
            f"Dense={selected['w_dense']:.2f}, nDCG@10={_fmt(selected['ndcg_at_10'])}, "
            f"Δ={selected['absolute_delta_vs_bm25']:+.4f}"
        )
    lines.append("")
    lines.append(f"**Final verdict: {verdict}**")
    lines.append("")
    (args.artifacts_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "verdict": verdict,
                "bm25_ndcg_at_10": bm25_ndcg,
                "best_weighted": metrics_payload["best_weighted"],
                "selected": selected,
                "artifacts_dir": str(args.artifacts_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
