#!/usr/bin/env python3
"""Build CE A/B + FAST→CE comparison reports for chunking_ab Slice 4.

Reads:
  - CE run dirs from evaluate_case_similarity_golden_v1.py --profile fast_ce
  - optional FAST run dirs for FAST→CE deltas

Writes CE_AB_COMPARISON.{md,json,html}. Does not run retrieval/CE.
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

from app.rag.legal_v2.benchmark.case_similarity_eval import (  # noqa: E402
    CaseSimilarityQueryEvalResult,
    aggregate_case_similarity_metrics,
)
from app.rag.legal_v2.benchmark.case_similarity_run_comparison import (  # noqa: E402
    CaseSimilarityRunComparisonError,
    compare_case_similarity_runs,
    load_run_config,
    load_run_results,
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

CE_CANONICAL = {
    "profile": "fast_ce",
    "experiment_name": "ce_bge_v2m3_p7_diverse_v1",
    "model": "BAAI/bge-reranker-v2-m3",
    "candidate_documents": 30,
    "passages_per_document": 7,
    "passage_selector": "diversified_stage1_evidence_v1",
    "evidence_pool_limit": 40,
    "batch_size": 8,
    "device": "cpu",
    "max_length": 512,
    "allow_download": False,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-a-ce", type=Path, required=True)
    p.add_argument("--run-b-ce", type=Path, required=True)
    p.add_argument("--run-a-fast", type=Path, required=True)
    p.add_argument("--run-b-fast", type=Path, required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "legal_v2"
            / "chunking_ab_pilot_300_v1"
            / "ce_ab_results"
        ),
    )
    p.add_argument("--command-a", default="")
    p.add_argument("--command-b", default="")
    p.add_argument("--fast-verdict", default="A")
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
                "reranker_score": item.reranker_score,
            }
        )
    return out


def _fmt_num(v: float | None) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "n/a"
    return f"{v:.4f}" if abs(v) < 10 else f"{v:.3f}"


def _assign_overall_verdict(
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
    *,
    gained_hit1: list[str],
    lost_hit1: list[str],
) -> dict[str, Any]:
    score = 0.0
    reasons: list[str] = []
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
        material = delta >= (0.05 if key.startswith("hit_") or key.endswith("_rate") else 0.01)
        if not material and key != "mean_relevant_rank":
            weight = weight * 0.35
        points = weight if b_better else -weight
        score += points
        direction = "B" if b_better else "A"
        reasons.append(f"{key}: {direction} better ({a:.4f} → {b:.4f})")

    if len(gained_hit1) != len(lost_hit1):
        if len(gained_hit1) > len(lost_hit1):
            score += 1.0
            reasons.append(f"Hit@1 transitions favor B (gained {gained_hit1}, lost {lost_hit1})")
        else:
            score -= 1.0
            reasons.append(f"Hit@1 transitions favor A (gained {gained_hit1}, lost {lost_hit1})")

    if abs(score) < 0.75:
        winner = "TIE"
        reason = (
            "No material advantage across primary CE metrics "
            f"(score={score:.2f}). "
            + ("; ".join(reasons[:3]) if reasons else "metrics match")
        )
    elif score > 0:
        winner = "B"
        reason = "B wins on weighted CE metrics: " + "; ".join(reasons[:4])
    else:
        winner = "A"
        reason = "A wins on weighted CE metrics: " + "; ".join(reasons[:4])

    failures = int(metrics_a.get("retrieval_execution_failures") or 0) + int(
        metrics_b.get("retrieval_execution_failures") or 0
    )
    hit10_ok = (
        metrics_a.get("hit_at_10") is not None
        and metrics_b.get("hit_at_10") is not None
        and min(float(metrics_a["hit_at_10"]), float(metrics_b["hit_at_10"])) >= 0.7
    )
    evaluable_ok = (
        int(metrics_a.get("evaluable_queries") or 0) == 20
        and int(metrics_b.get("evaluable_queries") or 0) == 20
    )
    return {
        "winner": winner,
        "reason": reason,
        "score": score,
        "metric_reasons": reasons,
        "complete": bool(failures == 0 and hit10_ok and evaluable_ok),
    }


def _side_fast_to_ce(
    *,
    side: str,
    fast_rows: list[CaseSimilarityQueryEvalResult],
    ce_rows: list[CaseSimilarityQueryEvalResult],
    metrics_fast: dict[str, Any],
    metrics_ce: dict[str, Any],
) -> dict[str, Any]:
    by_f = _index_by_qid(fast_rows)
    by_c = _index_by_qid(ce_rows)
    gained: list[str] = []
    lost: list[str] = []
    improved: list[dict[str, Any]] = []
    worsened: list[dict[str, Any]] = []
    for qid in sorted(by_f):
        f = by_f[qid]
        c = by_c[qid]
        if (not f.hit_at_1) and c.hit_at_1:
            gained.append(qid)
        if f.hit_at_1 and (not c.hit_at_1):
            lost.append(qid)
        d = _rank_delta(f.best_positive_rank, c.best_positive_rank)
        if d is not None and d > 0:
            improved.append(
                {
                    "query_id": qid,
                    "rank_fast": f.best_positive_rank,
                    "rank_ce": c.best_positive_rank,
                    "positions_gained": d,
                    "summary": (
                        f"{qid}: FAST {f.best_positive_rank if f.best_positive_rank is not None else '>10'} "
                        f"-> CE {c.best_positive_rank if c.best_positive_rank is not None else '>10'} "
                        f"(+{d})"
                    ),
                }
            )
        elif d is not None and d < 0:
            worsened.append(
                {
                    "query_id": qid,
                    "rank_fast": f.best_positive_rank,
                    "rank_ce": c.best_positive_rank,
                    "positions_lost": -d,
                    "summary": (
                        f"{qid}: FAST {f.best_positive_rank if f.best_positive_rank is not None else '>10'} "
                        f"-> CE {c.best_positive_rank if c.best_positive_rank is not None else '>10'} "
                        f"({d})"
                    ),
                }
            )
    improved.sort(key=lambda x: x["positions_gained"], reverse=True)
    worsened.sort(key=lambda x: x["positions_lost"], reverse=True)
    delta = {
        key: _delta(metrics_fast.get(key), metrics_ce.get(key))  # type: ignore[arg-type]
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
    # CE helped if MRR/Hit@1 improved and HN did not worsen materially.
    help_score = 0.0
    notes: list[str] = []
    for key, weight, higher_better in [
        ("hit_at_1", 2.0, True),
        ("mrr", 2.0, True),
        ("hit_at_10", 1.5, True),
        ("hard_negative_outrank_rate", 1.5, False),
    ]:
        dlt = delta.get(key)
        if dlt is None:
            continue
        if abs(dlt) < 1e-12:
            continue
        better = (dlt > 0) if higher_better else (dlt < 0)
        help_score += weight if better else -weight
        notes.append(f"{key} Δ={dlt:+.4f}")
    if help_score > 0.5:
        helped = True
        helped_label = "YES"
    elif help_score < -0.5:
        helped = False
        helped_label = "NO"
    else:
        helped = None
        helped_label = "MIXED/NEUTRAL"
    return {
        "side": side,
        "helped": helped,
        "helped_label": helped_label,
        "help_score": help_score,
        "notes": notes,
        "metrics_fast": metrics_fast,
        "metrics_ce": metrics_ce,
        "delta_ce_minus_fast": delta,
        "hit1_gained": gained,
        "hit1_lost": lost,
        "improvements": improved,
        "regressions": worsened,
    }


def _overall_chunking_verdict(
    *,
    fast_winner: str,
    ce_winner: str,
    ce_summary: dict[str, Any],
    fast_to_ce_a: dict[str, Any],
    fast_to_ce_b: dict[str, Any],
) -> dict[str, Any]:
    if ce_winner == fast_winner and ce_winner in {"A", "B"}:
        overall = ce_winner
        reason = (
            f"FAST and CE agree on {ce_winner}. "
            f"CE A helped={fast_to_ce_a['helped_label']}; "
            f"CE B helped={fast_to_ce_b['helped_label']}."
        )
    elif ce_winner in {"A", "B"} and fast_winner == "TIE":
        overall = ce_winner
        reason = f"FAST was TIE; CE selects {ce_winner}."
    elif fast_winner in {"A", "B"} and ce_winner == "TIE":
        overall = fast_winner
        reason = (
            f"CE is TIE; retain FAST winner {fast_winner} as overall chunking verdict."
        )
    elif ce_winner in {"A", "B"} and fast_winner in {"A", "B"} and ce_winner != fast_winner:
        # Prefer CE for final ranking quality, but flag flip.
        overall = ce_winner
        reason = (
            f"CE flips FAST winner ({fast_winner} → {ce_winner}). "
            f"Prefer CE ranking for overall chunking verdict. "
            f"{ce_summary['reason']}"
        )
    else:
        overall = "TIE"
        reason = "FAST and CE both inconclusive; overall chunking verdict is TIE."
    return {
        "fast_verdict": fast_winner,
        "ce_verdict": ce_winner,
        "overall_chunking_verdict": overall,
        "reason": reason,
    }


def build_payload(
    *,
    run_a_ce: Path,
    run_b_ce: Path,
    run_a_fast: Path,
    run_b_fast: Path,
    command_a: str,
    command_b: str,
    fast_verdict: str,
) -> dict[str, Any]:
    rows_a_ce = load_run_results(run_a_ce)
    rows_b_ce = load_run_results(run_b_ce)
    rows_a_fast = load_run_results(run_a_fast)
    rows_b_fast = load_run_results(run_b_fast)
    cfg_a = load_run_config(run_a_ce)
    cfg_b = load_run_config(run_b_ce)
    by_a = _index_by_qid(rows_a_ce)
    by_b = _index_by_qid(rows_b_ce)
    if set(by_a) != set(by_b):
        raise CaseSimilarityRunComparisonError(
            f"query id mismatch: only_a={sorted(set(by_a)-set(by_b))} "
            f"only_b={sorted(set(by_b)-set(by_a))}"
        )

    rank_diff = compare_case_similarity_runs(before_dir=run_a_ce, after_dir=run_b_ce)
    metrics_a = _metrics_from_rows(rows_a_ce)
    metrics_b = _metrics_from_rows(rows_b_ce)
    metrics_a_fast = _metrics_from_rows(rows_a_fast)
    metrics_b_fast = _metrics_from_rows(rows_b_fast)

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
    by_a_fast = _index_by_qid(rows_a_fast)
    by_b_fast = _index_by_qid(rows_b_fast)

    for qid in sorted(by_a):
        a = by_a[qid]
        b = by_b[qid]
        af = by_a_fast[qid]
        bf = by_b_fast[qid]
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
        if b.hard_negative_before_positive and not a.hard_negative_before_positive:
            flags.append("hn_regression_b")
        if a.hard_negative_before_positive and not b.hard_negative_before_positive:
            flags.append("hn_improved_b")
        # CE changed FAST relative winner for this query?
        fast_winner = _query_winner(af.best_positive_rank, bf.best_positive_rank)
        if winner != fast_winner:
            flags.append(f"ce_flipped_query_winner_{fast_winner}_to_{winner}")

        entry = {
            "query_id": qid,
            "query_text": a.query,
            "query_style": a.query_style,
            "difficulty": a.difficulty,
            "expected_primary_ecli": a.expected_primary_ecli,
            "rank_a_ce": rank_a,
            "rank_b_ce": rank_b,
            "rank_a_fast": af.best_positive_rank,
            "rank_b_fast": bf.best_positive_rank,
            "rank_delta_a_minus_b_ce": d_rank,
            "winner_ce": winner,
            "winner_fast": fast_winner,
            "hit_at_1_a_ce": a.hit_at_1,
            "hit_at_1_b_ce": b.hit_at_1,
            "hit_at_1_a_fast": af.hit_at_1,
            "hit_at_1_b_fast": bf.hit_at_1,
            "hit_at_10_a_ce": a.hit_at_10,
            "hit_at_10_b_ce": b.hit_at_10,
            "reciprocal_rank_a_ce": a.reciprocal_rank,
            "reciprocal_rank_b_ce": b.reciprocal_rank,
            "top_a_ce": _top_brief(a),
            "top_b_ce": _top_brief(b),
            "flags": flags,
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
                        f"{qid}: A CE {rank_a if rank_a is not None else '>10'} "
                        f"-> B CE {rank_b if rank_b is not None else '>10'} "
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
                        f"{qid}: A CE {rank_a if rank_a is not None else '>10'} "
                        f"-> B CE {rank_b if rank_b is not None else '>10'} "
                        f"({d_rank} positions)"
                    ),
                }
            )

    improvements.sort(key=lambda x: x["positions_gained"], reverse=True)
    regressions.sort(key=lambda x: x["positions_lost"], reverse=True)

    summary = _assign_overall_verdict(
        metrics_a,
        metrics_b,
        gained_hit1=list(rank_diff["hit1_transitions"]["gained_hit1"]),
        lost_hit1=list(rank_diff["hit1_transitions"]["lost_hit1"]),
    )
    fast_to_ce_a = _side_fast_to_ce(
        side="A",
        fast_rows=rows_a_fast,
        ce_rows=rows_a_ce,
        metrics_fast=metrics_a_fast,
        metrics_ce=metrics_a,
    )
    fast_to_ce_b = _side_fast_to_ce(
        side="B",
        fast_rows=rows_b_fast,
        ce_rows=rows_b_ce,
        metrics_fast=metrics_b_fast,
        metrics_ce=metrics_b,
    )
    overall = _overall_chunking_verdict(
        fast_winner=fast_verdict,
        ce_winner=summary["winner"],
        ce_summary=summary,
        fast_to_ce_a=fast_to_ce_a,
        fast_to_ce_b=fast_to_ce_b,
    )

    return {
        "schema": "chunking_ab_ce_comparison.v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark": {
            "profile": "fast_ce",
            "ce": CE_CANONICAL,
            "dataset": "benchmarks/legal_v2/case_similarity_golden_v1_pilot.jsonl",
            "query_count": len(queries),
            "variant_a": VARIANT_A,
            "variant_b": VARIANT_B,
            "run_a_ce_dir": str(run_a_ce),
            "run_b_ce_dir": str(run_b_ce),
            "run_a_fast_dir": str(run_a_fast),
            "run_b_fast_dir": str(run_b_fast),
            "run_a_config": {
                "run_id": cfg_a.get("run_id"),
                "collection": cfg_a.get("target_collection"),
                "bm25_index_id": cfg_a.get("bm25_index_id"),
                "profile": cfg_a.get("profile"),
                "ce_passages_per_document": cfg_a.get("ce_passages_per_document"),
                "ce_passage_selector": cfg_a.get("ce_passage_selector"),
                "ce_experiment_name": cfg_a.get("ce_experiment_name"),
                "execution_command": cfg_a.get("execution_command"),
            },
            "run_b_config": {
                "run_id": cfg_b.get("run_id"),
                "collection": cfg_b.get("target_collection"),
                "bm25_index_id": cfg_b.get("bm25_index_id"),
                "profile": cfg_b.get("profile"),
                "ce_passages_per_document": cfg_b.get("ce_passages_per_document"),
                "ce_passage_selector": cfg_b.get("ce_passage_selector"),
                "ce_experiment_name": cfg_b.get("ce_experiment_name"),
                "execution_command": cfg_b.get("execution_command"),
            },
            "command_a": command_a or cfg_a.get("execution_command"),
            "command_b": command_b or cfg_b.get("execution_command"),
            "git": _git_meta(),
        },
        "summary": {
            **summary,
            "fast_verdict": overall["fast_verdict"],
            "ce_verdict": overall["ce_verdict"],
            "overall_chunking_verdict": overall["overall_chunking_verdict"],
            "overall_reason": overall["reason"],
        },
        "metrics": {
            "A_ce": metrics_a,
            "B_ce": metrics_b,
            "A_fast": metrics_a_fast,
            "B_fast": metrics_b_fast,
            "delta_b_minus_a_ce": delta,
            "winners_ce": metric_winners,
        },
        "hit1_transitions_ce": rank_diff["hit1_transitions"],
        "fast_to_ce": {"A": fast_to_ce_a, "B": fast_to_ce_b},
        "queries": queries,
        "improvements": improvements,
        "regressions": regressions,
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    s = payload["summary"]
    m = payload["metrics"]
    b = payload["benchmark"]
    verdict = s["ce_verdict"]
    lines = [
        "# CE A/B Comparison — chunking_ab_pilot_300_v1 (parser v8)",
        "",
        f"**CE A/B VERDICT: {verdict} WINS**"
        if verdict in {"A", "B"}
        else "**CE A/B VERDICT: TIE**",
        "",
        f"**FAST VERDICT: {s['fast_verdict']}**",
        f"**OVERALL CHUNKING VERDICT: {s['overall_chunking_verdict']}**",
        "",
        s["reason"],
        "",
        s["overall_reason"],
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Branch: `{b['git'].get('branch')}` @ `{b['git'].get('git_commit')}`",
        f"- CE experiment: `{b['ce']['experiment_name']}`",
        f"- CE model: `{b['ce']['model']}` · passages=`{b['ce']['passages_per_document']}` · "
        f"selector=`{b['ce']['passage_selector']}` · candidates=`{b['ce']['candidate_documents']}`",
        f"- Queries: `{b['query_count']}`",
        "",
        "## CE summary metrics",
        "",
        "| Metric | A CE | B CE | Delta B−A | Winner |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
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
            f"| {label} | {_fmt_num(m['A_ce'].get(key))} | {_fmt_num(m['B_ce'].get(key))} | "
            f"{_fmt_num(m['delta_b_minus_a_ce'].get(key))} | {m['winners_ce'].get(key)} |"
        )

    lines.extend(
        [
            "",
            "### Hit@1 transitions (B CE relative to A CE)",
            "",
            f"- gained: `{payload['hit1_transitions_ce']['gained_hit1']}`",
            f"- lost: `{payload['hit1_transitions_ce']['lost_hit1']}`",
            "",
            "## FAST → CE",
            "",
        ]
    )
    for side in ("A", "B"):
        ft = payload["fast_to_ce"][side]
        lines.extend(
            [
                f"### Variant {side}",
                "",
                f"- CE helped: **{ft['helped_label']}** (score={ft['help_score']:.2f})",
                f"- Hit@1 gained: `{ft['hit1_gained']}`",
                f"- Hit@1 lost: `{ft['hit1_lost']}`",
                "",
                "| Metric | FAST | CE | Δ CE−FAST |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for key, label in [
            ("hit_at_1", "Hit@1"),
            ("hit_at_3", "Hit@3"),
            ("hit_at_5", "Hit@5"),
            ("hit_at_10", "Hit@10"),
            ("mrr", "MRR"),
            ("mean_relevant_rank", "Mean relevant rank"),
            ("hard_negative_outrank_rate", "HN outrank"),
        ]:
            lines.append(
                f"| {label} | {_fmt_num(ft['metrics_fast'].get(key))} | "
                f"{_fmt_num(ft['metrics_ce'].get(key))} | "
                f"{_fmt_num(ft['delta_ce_minus_fast'].get(key))} |"
            )
        lines.extend(["", "CE fixes:", ""])
        if ft["improvements"]:
            for row in ft["improvements"][:8]:
                lines.append(f"- {row['summary']}")
        else:
            lines.append("- (none)")
        lines.extend(["", "CE regressions:", ""])
        if ft["regressions"]:
            for row in ft["regressions"][:8]:
                lines.append(f"- {row['summary']}")
        else:
            lines.append("- (none)")
        lines.append("")

    lines.extend(["## Largest CE A/B improvements (B vs A)", ""])
    if payload["improvements"]:
        for row in payload["improvements"][:10]:
            lines.append(f"- {row['summary']}")
    else:
        lines.append("- (none)")
    lines.extend(["", "## Largest CE A/B regressions (B vs A)", ""])
    if payload["regressions"]:
        for row in payload["regressions"][:10]:
            lines.append(f"- {row['summary']}")
    else:
        lines.append("- (none)")

    lines.extend(
        [
            "",
            "## Query-by-query (CE)",
            "",
            "| Query | Expected | Rank A CE | Rank B CE | Δ | Winner CE | Winner FAST | Flags |",
            "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for q in payload["queries"]:
        flags = ",".join(q["flags"]) if q["flags"] else ""
        lines.append(
            f"| `{q['query_id']}` | `{q['expected_primary_ecli']}` | "
            f"{q['rank_a_ce'] if q['rank_a_ce'] is not None else '>10'} | "
            f"{q['rank_b_ce'] if q['rank_b_ce'] is not None else '>10'} | "
            f"{q['rank_delta_a_minus_b_ce']} | {q['winner_ce']} | {q['winner_fast']} | {flags} |"
        )

    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- A collection: `{b['variant_a']['collection']}`",
            f"- B collection: `{b['variant_b']['collection']}`",
            "",
            "```text",
            str(b.get("command_a") or ""),
            "```",
            "",
            "```text",
            str(b.get("command_b") or ""),
            "```",
            "",
            "HARD STOP after CE A/B. No further Slice / chunker optimization in this step.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_html(payload: dict[str, Any], path: Path) -> None:
    s = payload["summary"]
    m = payload["metrics"]
    b = payload["benchmark"]
    verdict = s["ce_verdict"] if s["ce_verdict"] in {"A", "B"} else "TIE"
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
        w = m["winners_ce"].get(key, "TIE")
        metric_rows += (
            f"<tr><td>{esc(label)}</td>"
            f"<td class='num'>{esc(_fmt_num(m['A_ce'].get(key)))}</td>"
            f"<td class='num'>{esc(_fmt_num(m['B_ce'].get(key)))}</td>"
            f"<td class='num'>{esc(_fmt_num(m['delta_b_minus_a_ce'].get(key)))}</td>"
            f"<td class='w-{esc(w).lower()}'>{esc(w)}</td></tr>"
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
            f"<div class='vals'>A {_fmt_num(m['A_ce'].get(key))} · B {_fmt_num(m['B_ce'].get(key))}</div>"
            f"<div class='delta'>Δ {_fmt_num(m['delta_b_minus_a_ce'].get(key))} · {esc(m['winners_ce'].get(key))}</div></div>"
        )

    ft_blocks = ""
    for side in ("A", "B"):
        ft = payload["fast_to_ce"][side]
        imp = "".join(f"<li>{esc(x['summary'])}</li>" for x in ft["improvements"][:6]) or "<li>(none)</li>"
        reg = "".join(f"<li>{esc(x['summary'])}</li>" for x in ft["regressions"][:6]) or "<li>(none)</li>"
        ft_blocks += f"""
        <div class="ft">
          <h3>FAST → CE ({esc(side)}): {esc(ft['helped_label'])}</h3>
          <p>Hit@1 gained {esc(ft['hit1_gained'])} · lost {esc(ft['hit1_lost'])}</p>
          <div class="cols">
            <div class="imp-box"><strong>CE fixes</strong><ul>{imp}</ul></div>
            <div class="reg-box"><strong>CE regressions</strong><ul>{reg}</ul></div>
          </div>
        </div>
        """

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
            f"<td class='wrap'>{esc(q['query_text'][:120])}{'…' if len(q['query_text'])>120 else ''}</td>"
            f"<td class='num'>{esc(q['rank_a_ce'] if q['rank_a_ce'] is not None else '>10')}</td>"
            f"<td class='num'>{esc(q['rank_b_ce'] if q['rank_b_ce'] is not None else '>10')}</td>"
            f"<td class='num'>{esc(q['rank_delta_a_minus_b_ce'])}</td>"
            f"<td>{esc(q['winner_ce'])}</td>"
            f"<td>{esc(q['winner_fast'])}</td>"
            f"<td>{esc(','.join(q['flags']))}</td>"
            f"</tr>"
        )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>CE A/B Comparison — {esc(verdict)}</title>
<style>
body {{ font-family: Segoe UI, system-ui, sans-serif; margin: 0; background: #f6f7f9; color: #1b1f24; }}
main {{ max-width: 1120px; margin: 0 auto; padding: 24px; }}
.verdict {{ padding: 20px 24px; border-radius: 12px; color: #fff; margin-bottom: 20px; }}
.verdict.win-b {{ background: #0f766e; }}
.verdict.win-a {{ background: #1d4ed8; }}
.verdict.tie {{ background: #475569; }}
.verdict h1 {{ margin: 0 0 8px; font-size: 28px; }}
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
.wrap {{ max-width: 260px; }}
.meta {{ color: #64748b; font-size: 13px; }}
pre {{ white-space: pre-wrap; background: #f8fafc; padding: 10px; border-radius: 8px; overflow: auto; }}
.overall {{ background: #fff7ed; border: 1px solid #fed7aa; border-radius: 10px; padding: 12px 14px; margin-bottom: 16px; }}
@media (max-width: 800px) {{ .cols {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<main>
  <div class="verdict {verdict_class}">
    <h1>CE A/B VERDICT: {esc(verdict)}{" WINS" if verdict in {"A","B"} else ""}</h1>
    <div>{esc(s["reason"])}</div>
    <div class="meta" style="margin-top:10px;color:#e2e8f0">
      {esc(payload["generated_at"])} · {esc(b["git"].get("branch"))} @ {esc((b["git"].get("git_commit") or "")[:12])}
      · {esc(b["ce"]["experiment_name"])}
    </div>
  </div>
  <div class="overall">
    <strong>FAST VERDICT:</strong> {esc(s["fast_verdict"])}<br/>
    <strong>OVERALL CHUNKING VERDICT:</strong> {esc(s["overall_chunking_verdict"])}<br/>
    <span class="meta">{esc(s["overall_reason"])}</span>
  </div>
  <div class="cards">{cards}</div>
  <section>
    <h2>CE summary metrics</h2>
    <table>
      <thead><tr><th>Metric</th><th>A CE</th><th>B CE</th><th>Δ B−A</th><th>Winner</th></tr></thead>
      <tbody>{metric_rows}</tbody>
    </table>
  </section>
  <section>
    <h2>FAST → CE</h2>
    {ft_blocks}
  </section>
  <section>
    <h2>CE A/B improvements / regressions</h2>
    <div class="cols">
      <div class="imp-box"><strong>B improvements</strong><ul>{imp_html}</ul></div>
      <div class="reg-box"><strong>B regressions</strong><ul>{reg_html}</ul></div>
    </div>
  </section>
  <section>
    <h2>Query-by-query</h2>
    <table>
      <thead>
        <tr><th>Query</th><th>Text</th><th>A CE</th><th>B CE</th><th>Δ</th><th>CE</th><th>FAST</th><th>Flags</th></tr>
      </thead>
      <tbody>{q_rows}</tbody>
    </table>
  </section>
  <section>
    <h2>Reproducibility</h2>
    <div class="meta">CE model {esc(b["ce"]["model"])} · p={esc(b["ce"]["passages_per_document"])} · {esc(b["ce"]["passage_selector"])}</div>
    <h3>Command A</h3><pre>{esc(b.get("command_a"))}</pre>
    <h3>Command B</h3><pre>{esc(b.get("command_b"))}</pre>
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
        run_a_ce=args.run_a_ce,
        run_b_ce=args.run_b_ce,
        run_a_fast=args.run_a_fast,
        run_b_fast=args.run_b_fast,
        command_a=args.command_a,
        command_b=args.command_b,
        fast_verdict=args.fast_verdict,
    )
    json_path = out / "CE_AB_COMPARISON.json"
    md_path = out / "CE_AB_COMPARISON.md"
    html_path = out / "CE_AB_COMPARISON.html"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, md_path)
    write_html(payload, html_path)
    print(f"WROTE {json_path}")
    print(f"WROTE {md_path}")
    print(f"WROTE {html_path}")
    print(f"CE_VERDICT {payload['summary']['ce_verdict']}")
    print(f"OVERALL_CHUNKING_VERDICT {payload['summary']['overall_chunking_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
