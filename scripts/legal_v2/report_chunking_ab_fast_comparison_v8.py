#!/usr/bin/env python3
"""Build user-facing FAST A/B comparison reports for chunking_ab Slice 4 indexes.

Reads two evaluate_case_similarity_golden_v1.py --profile fast run directories and
writes MD / JSON / HTML under --output-dir. Does not run retrieval or CE.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.rag.legal_v2.benchmark.case_similarity_run_comparison import (  # noqa: E402
    CaseSimilarityRunComparisonError,
    compare_case_similarity_runs,
    load_run_config,
    load_run_results,
)
from app.rag.legal_v2.benchmark.case_similarity_eval import (  # noqa: E402
    CaseSimilarityQueryEvalResult,
    aggregate_case_similarity_metrics,
)

VARIANT_A = {
    "label": "A",
    "name": "current hierarchical",
    "chunker": "legal_v2_hierarchical_chunker_v1",
    "collection": "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_a_current_300",
    "bm25_index_id": "nalus_legal_paragraph_bm25_v2_chunk_ab_v8_a_current_300",
}
VARIANT_B = {
    "label": "B",
    "name": "contextual packed v1",
    "chunker": "legal_contextual_packed_v1",
    "collection": "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_b_contextual_300",
    "bm25_index_id": "nalus_legal_paragraph_bm25_v2_chunk_ab_v8_b_contextual_300",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-a", type=Path, required=True)
    p.add_argument("--run-b", type=Path, required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "legal_v2"
            / "chunking_ab_pilot_300_v1"
            / "fast_ab_results"
        ),
    )
    p.add_argument("--command-a", default="")
    p.add_argument("--command-b", default="")
    return p.parse_args(argv)


def _git_meta() -> dict[str, Any]:
    def _run(args: list[str]) -> str:
        try:
            return subprocess.check_output(
                args, cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:  # noqa: BLE001
            return ""

    return {
        "branch": _run(["git", "branch", "--show-current"]),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(_run(["git", "status", "--porcelain"])),
    }


def _index_by_qid(
    rows: list[CaseSimilarityQueryEvalResult],
) -> dict[str, CaseSimilarityQueryEvalResult]:
    return {r.query_id: r for r in rows}


def _mean_relevant_rank(rows: list[CaseSimilarityQueryEvalResult]) -> float | None:
    ranks = [r.best_positive_rank for r in rows if r.best_positive_rank is not None]
    if not ranks:
        return None
    return float(mean(ranks))


def _metrics_from_rows(rows: list[CaseSimilarityQueryEvalResult]) -> dict[str, Any]:
    # Match evaluate_case_similarity_golden_v1.py / aggregate_case_similarity_metrics.
    agg = aggregate_case_similarity_metrics(rows)
    evaluable = [
        row
        for row in rows
        if row.corpus_compatible
        and row.failure_type != "retrieval_error"
        and not row.error
    ]
    return {
        "evaluable_queries": agg.evaluable_positive_retrieval_queries,
        "hit_at_1": agg.hit_at_1,
        "hit_at_3": agg.hit_at_3,
        "hit_at_5": agg.hit_at_5,
        "hit_at_10": agg.hit_at_10,
        "mrr": agg.mrr,
        "mean_relevant_rank": _mean_relevant_rank(evaluable),
        "hit_at_1_count": sum(1 for r in evaluable if r.hit_at_1),
        "hit_at_3_count": sum(1 for r in evaluable if r.hit_at_3),
        "hit_at_5_count": sum(1 for r in evaluable if r.hit_at_5),
        "hit_at_10_count": sum(1 for r in evaluable if r.hit_at_10),
        "no_positive_in_top_10": agg.no_positive_in_top_10,
        "hard_negative_outrank_count": agg.hard_negative_outrank_count,
        "hard_negative_outrank_rate": agg.hard_negative_outrank_rate,
        "retrieval_execution_failures": agg.retrieval_execution_failures,
        "accepted_alternative_wins": agg.accepted_alternative_wins,
    }


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(b) - float(a)


def _winner_higher(a: float | None, b: float | None, *, tol: float = 1e-12) -> str:
    if a is None and b is None:
        return "TIE"
    if a is None:
        return "B"
    if b is None:
        return "A"
    if abs(a - b) <= tol:
        return "TIE"
    return "B" if b > a else "A"


def _winner_lower(a: float | None, b: float | None, *, tol: float = 1e-12) -> str:
    if a is None and b is None:
        return "TIE"
    if a is None:
        return "B"
    if b is None:
        return "A"
    if abs(a - b) <= tol:
        return "TIE"
    return "B" if b < a else "A"


def _rank_delta(rank_a: int | None, rank_b: int | None) -> int | None:
    """Positive => B better (lower rank). Missing treated as 11 (outside TOP10)."""
    a = 11 if rank_a is None else rank_a
    b = 11 if rank_b is None else rank_b
    return a - b


def _query_winner(rank_a: int | None, rank_b: int | None) -> str:
    d = _rank_delta(rank_a, rank_b)
    assert d is not None
    if d > 0:
        return "B"
    if d < 0:
        return "A"
    return "TIE"


def _top_brief(row: CaseSimilarityQueryEvalResult, limit: int = 5) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in row.retrieved_results[:limit]:
        out.append(
            {
                "rank": item.rank,
                "ecli": item.ecli or item.document_id,
                "score": item.score,
                "fusion_score": item.fusion_score,
            }
        )
    return out


def _assign_overall_verdict(
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
    *,
    gained_hit1: list[str],
    lost_hit1: list[str],
    improvements: list[dict[str, Any]],
    regressions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Derive A/B/TIE from primary FAST metrics (data-driven)."""
    score = 0.0
    reasons: list[str] = []

    # Weighted primary gates (higher-is-better unless noted).
    weighted = [
        ("hit_at_1", 3.0, True),
        ("hit_at_10", 2.0, True),
        ("mrr", 2.5, True),
        ("hit_at_3", 1.0, True),
        ("hit_at_5", 1.0, True),
        ("mean_relevant_rank", 1.5, False),
        ("hard_negative_outrank_rate", 1.5, False),
    ]
    for key, weight, higher_better in weighted:
        a = metrics_a.get(key)
        b = metrics_b.get(key)
        if a is None or b is None:
            continue
        if abs(float(a) - float(b)) < 1e-12:
            continue
        b_better = (b > a) if higher_better else (b < a)
        delta = abs(float(b) - float(a))
        # Require a material margin for fractional rates.
        material = delta >= (0.05 if key.startswith("hit_") or key.endswith("_rate") else 0.01)
        if not material and key != "mean_relevant_rank":
            # still count small MRR/mean rank differences lightly
            weight = weight * 0.35
        points = weight if b_better else -weight
        score += points
        direction = "B" if b_better else "A"
        reasons.append(
            f"{key}: {direction} better ({a:.4f} → {b:.4f})"
            if isinstance(a, float)
            else f"{key}: {direction} better"
        )

    # Hit@1 transitions as tie-break signal.
    if len(gained_hit1) != len(lost_hit1):
        if len(gained_hit1) > len(lost_hit1):
            score += 1.0
            reasons.append(
                f"Hit@1 transitions favor B (gained {gained_hit1}, lost {lost_hit1})"
            )
        else:
            score -= 1.0
            reasons.append(
                f"Hit@1 transitions favor A (gained {gained_hit1}, lost {lost_hit1})"
            )

    if abs(score) < 0.75:
        winner = "TIE"
        reason = (
            "No material advantage across primary FAST metrics "
            f"(score={score:.2f}). " + ("; ".join(reasons[:3]) if reasons else "metrics match")
        )
    elif score > 0:
        winner = "B"
        reason = "B wins on weighted FAST metrics: " + "; ".join(reasons[:4])
    else:
        winner = "A"
        reason = "A wins on weighted FAST metrics: " + "; ".join(reasons[:4])

    # CE readiness: complete eval, no retrieval failures, Hit@10 not collapsing.
    failures = int(metrics_a.get("retrieval_execution_failures") or 0) + int(
        metrics_b.get("retrieval_execution_failures") or 0
    )
    hit10_a = metrics_a.get("hit_at_10")
    hit10_b = metrics_b.get("hit_at_10")
    hit10_ok = (
        hit10_a is not None
        and hit10_b is not None
        and min(float(hit10_a), float(hit10_b)) >= 0.7
    )
    evaluable_ok = (
        int(metrics_a.get("evaluable_queries") or 0) == 20
        and int(metrics_b.get("evaluable_queries") or 0) == 20
    )
    # CE next is safe when FAST completed cleanly with stable recall floor.
    # A clear A/B winner is preferred but not required (TIE can still proceed).
    safe = failures == 0 and hit10_ok and evaluable_ok

    return {
        "winner": winner,
        "reason": reason,
        "score": score,
        "metric_reasons": reasons,
        "safe_for_ce_next": bool(safe),
        "ce_readiness_notes": [
            f"retrieval_failures_total={failures}",
            f"hit_at_10_floor_ok={hit10_ok}",
            f"evaluable_a={metrics_a.get('evaluable_queries')}",
            f"evaluable_b={metrics_b.get('evaluable_queries')}",
            f"material_improvements={len(improvements)}",
            f"material_regressions={len(regressions)}",
            f"verdict_clear={winner != 'TIE'}",
        ],
    }


def build_payload(
    *,
    run_a: Path,
    run_b: Path,
    command_a: str,
    command_b: str,
) -> dict[str, Any]:
    rows_a = load_run_results(run_a)
    rows_b = load_run_results(run_b)
    cfg_a = load_run_config(run_a)
    cfg_b = load_run_config(run_b)
    by_a = _index_by_qid(rows_a)
    by_b = _index_by_qid(rows_b)
    if set(by_a) != set(by_b):
        raise CaseSimilarityRunComparisonError(
            f"query id mismatch: only_a={sorted(set(by_a)-set(by_b))} "
            f"only_b={sorted(set(by_b)-set(by_a))}"
        )

    # Reuse canonical rank-diff machinery (A=before, B=after).
    rank_diff = compare_case_similarity_runs(before_dir=run_a, after_dir=run_b)

    metrics_a = _metrics_from_rows(rows_a)
    metrics_b = _metrics_from_rows(rows_b)
    delta = {
        key: _delta(metrics_a.get(key), metrics_b.get(key))  # type: ignore[arg-type]
        for key in (
            "hit_at_1",
            "hit_at_3",
            "hit_at_5",
            "hit_at_10",
            "mrr",
            "mean_relevant_rank",
            "hard_negative_outrank_rate",
        )
    }
    metric_winners = {
        "hit_at_1": _winner_higher(metrics_a["hit_at_1"], metrics_b["hit_at_1"]),
        "hit_at_3": _winner_higher(metrics_a["hit_at_3"], metrics_b["hit_at_3"]),
        "hit_at_5": _winner_higher(metrics_a["hit_at_5"], metrics_b["hit_at_5"]),
        "hit_at_10": _winner_higher(metrics_a["hit_at_10"], metrics_b["hit_at_10"]),
        "mrr": _winner_higher(metrics_a["mrr"], metrics_b["mrr"]),
        "mean_relevant_rank": _winner_lower(
            metrics_a["mean_relevant_rank"], metrics_b["mean_relevant_rank"]
        ),
        "hard_negative_outrank_rate": _winner_lower(
            metrics_a["hard_negative_outrank_rate"],
            metrics_b["hard_negative_outrank_rate"],
        ),
    }

    queries: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    highlights: list[str] = []

    for qid in sorted(by_a):
        a = by_a[qid]
        b = by_b[qid]
        rank_a = a.best_positive_rank
        rank_b = b.best_positive_rank
        d_rank = _rank_delta(rank_a, rank_b)
        winner = _query_winner(rank_a, rank_b)
        flags: list[str] = []
        if a.hit_at_1 and not b.hit_at_1:
            flags.append("hit1_regression_b")
        if b.hit_at_1 and not a.hit_at_1:
            flags.append("hit1_gain_b")
        if a.hit_at_10 and not b.hit_at_10:
            flags.append("dropped_from_top10_b")
        if b.hit_at_10 and not a.hit_at_10:
            flags.append("entered_top10_b")
        if d_rank is not None and d_rank >= 3:
            flags.append("material_improvement_b")
        if d_rank is not None and d_rank <= -3:
            flags.append("material_regression_b")

        entry = {
            "query_id": qid,
            "query_text": a.query,
            "query_style": a.query_style,
            "difficulty": a.difficulty,
            "expected_primary_ecli": a.expected_primary_ecli,
            "accepted_alternative_eclis": list(a.accepted_alternative_document_ids),
            "rank_a": rank_a,
            "rank_b": rank_b,
            "rank_delta_a_minus_b": d_rank,
            "winner": winner,
            "hit_at_1_a": a.hit_at_1,
            "hit_at_1_b": b.hit_at_1,
            "hit_at_10_a": a.hit_at_10,
            "hit_at_10_b": b.hit_at_10,
            "reciprocal_rank_a": a.reciprocal_rank,
            "reciprocal_rank_b": b.reciprocal_rank,
            "top_a": _top_brief(a),
            "top_b": _top_brief(b),
            "flags": flags,
            "diagnostics_note": (
                "Per-query dense/BM25 channel contribution is not persisted in "
                "canonical FAST eval artifacts; scores below are Stage-1 document "
                "score / fusion_score when present."
            ),
        }
        queries.append(entry)

        if d_rank is not None and d_rank > 0:
            improvements.append(
                {
                    "query_id": qid,
                    "rank_a": rank_a,
                    "rank_b": rank_b,
                    "positions_gained": d_rank,
                    "summary": (
                        f"{qid}: A rank {rank_a if rank_a is not None else '>10'} "
                        f"-> B rank {rank_b if rank_b is not None else '>10'} "
                        f"(+{d_rank} positions)"
                    ),
                }
            )
        elif d_rank is not None and d_rank < 0:
            regressions.append(
                {
                    "query_id": qid,
                    "rank_a": rank_a,
                    "rank_b": rank_b,
                    "positions_lost": -d_rank,
                    "summary": (
                        f"{qid}: A rank {rank_a if rank_a is not None else '>10'} "
                        f"-> B rank {rank_b if rank_b is not None else '>10'} "
                        f"({d_rank} positions)"
                    ),
                }
            )

        for f in flags:
            highlights.append(f"{qid}:{f}")

    improvements.sort(key=lambda x: x["positions_gained"], reverse=True)
    regressions.sort(key=lambda x: x["positions_lost"], reverse=True)

    summary = _assign_overall_verdict(
        metrics_a,
        metrics_b,
        gained_hit1=list(rank_diff["hit1_transitions"]["gained_hit1"]),
        lost_hit1=list(rank_diff["hit1_transitions"]["lost_hit1"]),
        improvements=improvements,
        regressions=regressions,
    )

    git = _git_meta()
    return {
        "schema": "chunking_ab_fast_comparison.v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark": {
            "profile": "fast",
            "dataset": "benchmarks/legal_v2/case_similarity_golden_v1_pilot.jsonl",
            "query_count": len(queries),
            "variant_a": VARIANT_A,
            "variant_b": VARIANT_B,
            "run_a_dir": str(run_a),
            "run_b_dir": str(run_b),
            "run_a_config": {
                "run_id": cfg_a.get("run_id"),
                "collection": cfg_a.get("target_collection"),
                "bm25_index_id": cfg_a.get("bm25_index_id"),
                "profile": cfg_a.get("profile"),
                "execution_command": cfg_a.get("execution_command"),
            },
            "run_b_config": {
                "run_id": cfg_b.get("run_id"),
                "collection": cfg_b.get("target_collection"),
                "bm25_index_id": cfg_b.get("bm25_index_id"),
                "profile": cfg_b.get("profile"),
                "execution_command": cfg_b.get("execution_command"),
            },
            "command_a": command_a or cfg_a.get("execution_command"),
            "command_b": command_b or cfg_b.get("execution_command"),
            "git": git,
            "ce_scoring": False,
        },
        "summary": summary,
        "metrics": {
            "A": metrics_a,
            "B": metrics_b,
            "delta_b_minus_a": delta,
            "winners": metric_winners,
        },
        "hit1_transitions": rank_diff["hit1_transitions"],
        "canonical_rank_diff_verdict": rank_diff.get("verdict"),
        "queries": queries,
        "improvements": improvements,
        "regressions": regressions,
        "highlights": highlights,
        "retrieval_diagnostics": {
            "available": False,
            "note": (
                "Canonical FAST evaluator stores document-level Stage-1 score and "
                "optional fusion_score per hit, but does not persist separate "
                "dense-vs-BM25 contribution traces per query."
            ),
        },
    }


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.3f}"


def _fmt_num(v: float | None) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "n/a"
    return f"{v:.4f}" if abs(v) < 10 else f"{v:.3f}"


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    s = payload["summary"]
    m = payload["metrics"]
    b = payload["benchmark"]
    lines: list[str] = [
        "# FAST A/B Comparison — chunking_ab_pilot_300_v1 (parser v8)",
        "",
        f"**FAST A/B VERDICT: {s['winner']} WINS**"
        if s["winner"] in {"A", "B"}
        else "**FAST A/B VERDICT: TIE**",
        "",
        s["reason"],
        "",
        f"- Safe for CE next: `{s['safe_for_ce_next']}`",
        f"- Generated: `{payload['generated_at']}`",
        f"- Branch: `{b['git'].get('branch')}` @ `{b['git'].get('git_commit')}`",
        f"- Profile: `{b['profile']}` (CE scoring: **off**)",
        f"- Queries: `{b['query_count']}`",
        "",
        "## Executive summary",
        "",
        "### B improves" if s["winner"] != "A" else "### Notes favoring B",
        "",
    ]
    b_better = [k for k, w in m["winners"].items() if w == "B"]
    a_better = [k for k, w in m["winners"].items() if w == "A"]
    if b_better:
        for k in b_better:
            lines.append(
                f"- `{k}`: A={_fmt_num(m['A'].get(k))} → B={_fmt_num(m['B'].get(k))} "
                f"(Δ={_fmt_num(m['delta_b_minus_a'].get(k))})"
            )
    else:
        lines.append("- (none)")
    lines.extend(["", "### A remains better" if a_better else "### Notes favoring A", ""])
    if a_better:
        for k in a_better:
            lines.append(
                f"- `{k}`: A={_fmt_num(m['A'].get(k))} → B={_fmt_num(m['B'].get(k))} "
                f"(Δ={_fmt_num(m['delta_b_minus_a'].get(k))})"
            )
    else:
        lines.append("- (none)")

    lines.extend(
        [
            "",
            "### Hit@1 transitions (B relative to A)",
            "",
            f"- gained Hit@1: `{payload['hit1_transitions']['gained_hit1']}`",
            f"- lost Hit@1: `{payload['hit1_transitions']['lost_hit1']}`",
            "",
            "## Indexes",
            "",
            f"- A collection: `{b['variant_a']['collection']}`",
            f"- A BM25: `{b['variant_a']['bm25_index_id']}`",
            f"- B collection: `{b['variant_b']['collection']}`",
            f"- B BM25: `{b['variant_b']['bm25_index_id']}`",
            f"- Run A: `{b['run_a_dir']}`",
            f"- Run B: `{b['run_b_dir']}`",
            "",
            "## Commands",
            "",
            "```text",
            str(b.get("command_a") or ""),
            "```",
            "",
            "```text",
            str(b.get("command_b") or ""),
            "```",
            "",
            "## Summary metrics",
            "",
            "| Metric | Variant A | Variant B | Delta B-A | Winner |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for key, label in [
        ("hit_at_1", "Hit@1"),
        ("hit_at_3", "Hit@3"),
        ("hit_at_5", "Hit@5"),
        ("hit_at_10", "Hit@10"),
        ("mrr", "MRR"),
        ("mean_relevant_rank", "Mean relevant rank"),
        ("hard_negative_outrank_rate", "HN outrank rate"),
    ]:
        lines.append(
            f"| {label} | {_fmt_num(m['A'].get(key))} | {_fmt_num(m['B'].get(key))} | "
            f"{_fmt_num(m['delta_b_minus_a'].get(key))} | {m['winners'].get(key)} |"
        )

    lines.extend(
        [
            "",
            f"- Hit@1 counts: A `{m['A']['hit_at_1_count']}/{m['A']['evaluable_queries']}` · "
            f"B `{m['B']['hit_at_1_count']}/{m['B']['evaluable_queries']}`",
            f"- No positive in TOP10: A `{m['A']['no_positive_in_top_10']}` · "
            f"B `{m['B']['no_positive_in_top_10']}`",
            "",
            "## Largest improvements (B vs A)",
            "",
        ]
    )
    if payload["improvements"]:
        for row in payload["improvements"][:10]:
            lines.append(f"- {row['summary']}")
    else:
        lines.append("- (none)")

    lines.extend(["", "## Largest regressions (B vs A)", ""])
    if payload["regressions"]:
        for row in payload["regressions"][:10]:
            lines.append(f"- {row['summary']}")
    else:
        lines.append("- (none)")

    lines.extend(
        [
            "",
            "## Query-by-query",
            "",
            "| Query | Expected ECLI | Rank A | Rank B | Δ (A−B) | Winner | Flags |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for q in payload["queries"]:
        flags = ",".join(q["flags"]) if q["flags"] else ""
        lines.append(
            f"| `{q['query_id']}` | `{q['expected_primary_ecli']}` | "
            f"{q['rank_a'] if q['rank_a'] is not None else '>10'} | "
            f"{q['rank_b'] if q['rank_b'] is not None else '>10'} | "
            f"{q['rank_delta_a_minus_b']} | {q['winner']} | {flags} |"
        )

    lines.extend(["", "## Query details", ""])
    for q in payload["queries"]:
        lines.extend(
            [
                f"### {q['query_id']} — winner `{q['winner']}`",
                "",
                f"- style/difficulty: `{q['query_style']}` / `{q['difficulty']}`",
                f"- expected: `{q['expected_primary_ecli']}`",
                f"- ranks: A=`{q['rank_a']}` B=`{q['rank_b']}` Δ=`{q['rank_delta_a_minus_b']}`",
                f"- Hit@1: A=`{q['hit_at_1_a']}` B=`{q['hit_at_1_b']}`",
                "",
                "<details><summary>Query text</summary>",
                "",
                q["query_text"],
                "",
                "</details>",
                "",
                "Top A:",
                "",
                "```json",
                json.dumps(q["top_a"], ensure_ascii=False, indent=2),
                "```",
                "",
                "Top B:",
                "",
                "```json",
                json.dumps(q["top_b"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Retrieval diagnostics",
            "",
            payload["retrieval_diagnostics"]["note"],
            "",
            "---",
            "",
            "CE scoring was **not** run. Slice 4 indexes were not modified.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_html(payload: dict[str, Any], path: Path) -> None:
    s = payload["summary"]
    m = payload["metrics"]
    b = payload["benchmark"]
    verdict = s["winner"] if s["winner"] in {"A", "B"} else "TIE"
    verdict_class = {"A": "win-a", "B": "win-b", "TIE": "tie"}[verdict]

    def esc(x: Any) -> str:
        return html.escape("" if x is None else str(x))

    metric_rows = ""
    for key, label in [
        ("hit_at_1", "Hit@1"),
        ("hit_at_3", "Hit@3"),
        ("hit_at_5", "Hit@5"),
        ("hit_at_10", "Hit@10"),
        ("mrr", "MRR"),
        ("mean_relevant_rank", "Mean relevant rank"),
        ("hard_negative_outrank_rate", "HN outrank rate"),
    ]:
        w = m["winners"].get(key, "TIE")
        metric_rows += (
            f"<tr><td>{esc(label)}</td>"
            f"<td class='num'>{esc(_fmt_num(m['A'].get(key)))}</td>"
            f"<td class='num'>{esc(_fmt_num(m['B'].get(key)))}</td>"
            f"<td class='num'>{esc(_fmt_num(m['delta_b_minus_a'].get(key)))}</td>"
            f"<td class='w-{esc(w).lower()}'>{esc(w)}</td></tr>"
        )

    imp_html = "".join(f"<li>{esc(x['summary'])}</li>" for x in payload["improvements"][:12]) or "<li>(none)</li>"
    reg_html = "".join(f"<li>{esc(x['summary'])}</li>" for x in payload["regressions"][:12]) or "<li>(none)</li>"

    q_rows = ""
    for q in payload["queries"]:
        cls = ""
        if "material_improvement_b" in q["flags"] or "hit1_gain_b" in q["flags"]:
            cls = "imp"
        elif "material_regression_b" in q["flags"] or "hit1_regression_b" in q["flags"] or "dropped_from_top10_b" in q["flags"]:
            cls = "reg"
        q_rows += (
            f"<tr class='{cls}'>"
            f"<td><code>{esc(q['query_id'])}</code></td>"
            f"<td class='wrap'>{esc(q['query_text'][:140])}{'…' if len(q['query_text'])>140 else ''}</td>"
            f"<td><code>{esc(q['expected_primary_ecli'])}</code></td>"
            f"<td class='num'>{esc(q['rank_a'] if q['rank_a'] is not None else '>10')}</td>"
            f"<td class='num'>{esc(q['rank_b'] if q['rank_b'] is not None else '>10')}</td>"
            f"<td class='num'>{esc(q['rank_delta_a_minus_b'])}</td>"
            f"<td>{esc(q['winner'])}</td>"
            f"<td>{esc(','.join(q['flags']))}</td>"
            f"</tr>"
        )

    cards = ""
    for key, label in [
        ("hit_at_1", "Hit@1"),
        ("hit_at_10", "Hit@10"),
        ("mrr", "MRR"),
        ("mean_relevant_rank", "Mean rank"),
    ]:
        cards += (
            f"<div class='card'><div class='label'>{esc(label)}</div>"
            f"<div class='vals'>A {_fmt_num(m['A'].get(key))} · B {_fmt_num(m['B'].get(key))}</div>"
            f"<div class='delta'>Δ {_fmt_num(m['delta_b_minus_a'].get(key))} · {esc(m['winners'].get(key))}</div></div>"
        )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>FAST A/B Comparison — {esc(verdict)}</title>
<style>
body {{ font-family: Segoe UI, system-ui, sans-serif; margin: 0; background: #f6f7f9; color: #1b1f24; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
.verdict {{ padding: 20px 24px; border-radius: 12px; color: #fff; margin-bottom: 20px; }}
.verdict.win-b {{ background: #0f766e; }}
.verdict.win-a {{ background: #1d4ed8; }}
.verdict.tie {{ background: #475569; }}
.verdict h1 {{ margin: 0 0 8px; font-size: 28px; }}
.reason {{ opacity: .95; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap: 12px; margin: 16px 0 24px; }}
.card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px 14px; }}
.card .label {{ font-size: 12px; text-transform: uppercase; color: #64748b; }}
.card .vals {{ font-size: 18px; font-weight: 600; margin-top: 4px; }}
.card .delta {{ font-size: 13px; color: #334155; margin-top: 4px; }}
section {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px 18px; margin-bottom: 16px; }}
h2 {{ margin-top: 0; font-size: 18px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #eef2f7; padding: 8px 6px; text-align: left; vertical-align: top; }}
th {{ font-size: 12px; color: #64748b; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.w-b {{ color: #0f766e; font-weight: 600; }}
.w-a {{ color: #1d4ed8; font-weight: 600; }}
.w-tie {{ color: #64748b; }}
tr.imp {{ background: #ecfdf5; }}
tr.reg {{ background: #fef2f2; }}
.cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.imp-box {{ background: #ecfdf5; border-radius: 8px; padding: 10px 12px; }}
.reg-box {{ background: #fef2f2; border-radius: 8px; padding: 10px 12px; }}
code {{ font-size: 12px; }}
.wrap {{ max-width: 280px; }}
.meta {{ color: #64748b; font-size: 13px; }}
pre {{ white-space: pre-wrap; background: #f8fafc; padding: 10px; border-radius: 8px; overflow: auto; }}
@media (max-width: 800px) {{ .cols {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<main>
  <div class="verdict {verdict_class}">
    <h1>FAST A/B VERDICT: {esc(verdict)}{" WINS" if verdict in {"A","B"} else ""}</h1>
    <div class="reason">{esc(s["reason"])}</div>
    <div class="meta" style="margin-top:10px;color:#e2e8f0">
      Safe for CE next: {esc(s["safe_for_ce_next"])} · {esc(payload["generated_at"])} ·
      {esc(b["git"].get("branch"))} @ {esc((b["git"].get("git_commit") or "")[:12])}
    </div>
  </div>

  <div class="cards">{cards}</div>

  <section>
    <h2>Summary metrics</h2>
    <table>
      <thead><tr><th>Metric</th><th>A</th><th>B</th><th>Δ B−A</th><th>Winner</th></tr></thead>
      <tbody>{metric_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>Improvements / regressions</h2>
    <div class="cols">
      <div class="imp-box"><strong>B improvements</strong><ul>{imp_html}</ul></div>
      <div class="reg-box"><strong>B regressions</strong><ul>{reg_html}</ul></div>
    </div>
  </section>

  <section>
    <h2>Query-by-query</h2>
    <table>
      <thead>
        <tr>
          <th>Query</th><th>Text</th><th>Expected</th>
          <th>Rank A</th><th>Rank B</th><th>Δ</th><th>Winner</th><th>Flags</th>
        </tr>
      </thead>
      <tbody>{q_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>Reproducibility</h2>
    <div class="meta">
      Profile: fast · CE: off · Queries: {esc(b["query_count"])}<br/>
      A: <code>{esc(b["variant_a"]["collection"])}</code><br/>
      B: <code>{esc(b["variant_b"]["collection"])}</code>
    </div>
    <h3>Command A</h3>
    <pre>{esc(b.get("command_a"))}</pre>
    <h3>Command B</h3>
    <pre>{esc(b.get("command_b"))}</pre>
  </section>
</main>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    payload = build_payload(
        run_a=args.run_a,
        run_b=args.run_b,
        command_a=args.command_a,
        command_b=args.command_b,
    )
    json_path = out / "FAST_AB_COMPARISON.json"
    md_path = out / "FAST_AB_COMPARISON.md"
    html_path = out / "FAST_AB_COMPARISON.html"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(payload, md_path)
    write_html(payload, html_path)
    print(f"WROTE {json_path}")
    print(f"WROTE {md_path}")
    print(f"WROTE {html_path}")
    print(f"VERDICT {payload['summary']['winner']}")
    print(f"SAFE_FOR_CE {payload['summary']['safe_for_ce_next']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
