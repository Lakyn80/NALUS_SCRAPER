"""Read-only rank-diff audit between two Case Similarity Golden evaluation runs."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from app.rag.legal_v2.benchmark.case_similarity_eval import (
    CaseSimilarityAggregateMetrics,
    CaseSimilarityQueryEvalResult,
    aggregate_case_similarity_metrics,
)
from app.rag.legal_v2.identity import ecli_key

RESULTS_FILENAME = "case_similarity_retrieval_results.jsonl"
REPORT_FILENAME = "case_similarity_retrieval_report.md"
CONFIG_FILENAME = "retrieval_run_config.json"

_METRIC_LINE_RE = re.compile(
    r"^-\s+(?P<label>Hit@1|Hit@3|Hit@5|Hit@10|MRR|"
    r"evaluable_positive_retrieval_queries|retrieval_execution_failures|"
    r"hard_negative_outrank_count|hard_negative_outrank_rate[^:]*):\s*`(?P<value>[^`]+)`",
    re.MULTILINE,
)


class CaseSimilarityRunComparisonError(ValueError):
    """Raised when two runs cannot be compared safely."""


@dataclass(frozen=True)
class ParsedStoredMetrics:
    hit_at_1: float | None = None
    hit_at_3: float | None = None
    hit_at_5: float | None = None
    hit_at_10: float | None = None
    mrr: float | None = None
    evaluable_positive_retrieval_queries: int | None = None
    retrieval_execution_failures: int | None = None
    hard_negative_outrank_count: int | None = None
    hard_negative_outrank_rate: float | None = None


@dataclass(frozen=True)
class MetricsSnapshot:
    evaluable: int
    retrieval_failures: int
    hit_at_1: float | None
    hit_at_3: float | None
    hit_at_5: float | None
    hit_at_10: float | None
    mrr: float | None
    hit_at_1_count: int
    hit_at_3_count: int
    hit_at_5_count: int
    hit_at_10_count: int
    hard_negative_outrank_count: int
    hard_negative_outrank_rate: float | None
    hard_negative_evaluable_query_count: int


@dataclass(frozen=True)
class QueryRankDiff:
    benchmark_id: str
    expected_primary_ecli: str | None
    accepted_alternative_eclis: list[str]
    before_top1_ecli: str | None
    after_top1_ecli: str | None
    before_primary_rank: int | None
    after_primary_rank: int | None
    before_best_alternative_rank: int | None
    after_best_alternative_rank: int | None
    before_effective_rank: int | None
    after_effective_rank: int | None
    before_reciprocal_rank: float
    after_reciprocal_rank: float
    primary_rank_delta: int | None
    effective_rank_delta: int | None
    top1_changed: bool
    hit1_before: bool
    hit1_after: bool
    hit3_before: bool
    hit3_after: bool
    hit5_before: bool
    hit5_after: bool
    hit10_before: bool
    hit10_after: bool
    classification: str
    after_top1_role: str


def load_run_results(run_dir: Path) -> list[CaseSimilarityQueryEvalResult]:
    path = Path(run_dir) / RESULTS_FILENAME
    if not path.is_file():
        raise CaseSimilarityRunComparisonError(f"missing required results file: {path}")
    rows: list[CaseSimilarityQueryEvalResult] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise CaseSimilarityRunComparisonError(
                    f"invalid JSONL at {path}:{line_no}: {exc}"
                ) from exc
            rows.append(CaseSimilarityQueryEvalResult.model_validate(payload))
    if not rows:
        raise CaseSimilarityRunComparisonError(f"empty results file: {path}")
    return rows


def load_run_config(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / CONFIG_FILENAME
    if not path.is_file():
        raise CaseSimilarityRunComparisonError(f"missing required config file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_stored_report_metrics(run_dir: Path) -> ParsedStoredMetrics:
    path = Path(run_dir) / REPORT_FILENAME
    if not path.is_file():
        raise CaseSimilarityRunComparisonError(f"missing required report file: {path}")
    text = path.read_text(encoding="utf-8")
    values: dict[str, str] = {}
    for match in _METRIC_LINE_RE.finditer(text):
        label = match.group("label")
        if label.startswith("hard_negative_outrank_rate"):
            values["hard_negative_outrank_rate"] = match.group("value")
        else:
            values[label] = match.group("value")

    def _float(key: str) -> float | None:
        raw = values.get(key)
        return float(raw) if raw is not None else None

    def _int(key: str) -> int | None:
        raw = values.get(key)
        return int(raw) if raw is not None else None

    return ParsedStoredMetrics(
        hit_at_1=_float("Hit@1"),
        hit_at_3=_float("Hit@3"),
        hit_at_5=_float("Hit@5"),
        hit_at_10=_float("Hit@10"),
        mrr=_float("MRR"),
        evaluable_positive_retrieval_queries=_int("evaluable_positive_retrieval_queries"),
        retrieval_execution_failures=_int("retrieval_execution_failures"),
        hard_negative_outrank_count=_int("hard_negative_outrank_count"),
        hard_negative_outrank_rate=_float("hard_negative_outrank_rate"),
    )


def _index_by_query_id(
    rows: Sequence[CaseSimilarityQueryEvalResult],
) -> dict[str, CaseSimilarityQueryEvalResult]:
    indexed: dict[str, CaseSimilarityQueryEvalResult] = {}
    duplicates: list[str] = []
    for row in rows:
        if row.query_id in indexed:
            duplicates.append(row.query_id)
            continue
        indexed[row.query_id] = row
    if duplicates:
        raise CaseSimilarityRunComparisonError(
            "duplicate benchmark IDs in run: " + ", ".join(sorted(set(duplicates)))
        )
    return indexed


def _sorted_ecli_list(values: Sequence[str] | None) -> list[str]:
    items = [str(value) for value in (values or []) if str(value or "").strip()]
    return sorted(items, key=lambda text: ecli_key(text) if text.upper().startswith("ECLI:") else text)


def _top1(row: CaseSimilarityQueryEvalResult) -> str | None:
    if row.retrieved_document_ids:
        return row.retrieved_document_ids[0]
    if row.retrieved_eclis:
        return row.retrieved_eclis[0]
    return None


def _rank_delta(before: int | None, after: int | None) -> int | None:
    if before is None or after is None:
        return None
    return before - after


def _crossed_into(limit: int, before: int | None, after: int | None) -> bool:
    before_in = before is not None and before <= limit
    after_in = after is not None and after <= limit
    return (not before_in) and after_in


def _crossed_out(limit: int, before: int | None, after: int | None) -> bool:
    before_in = before is not None and before <= limit
    after_in = after is not None and after <= limit
    return before_in and (not after_in)


def classify_rank_change(
    *,
    before_effective_rank: int | None,
    after_effective_rank: int | None,
    before_top1: str | None,
    after_top1: str | None,
) -> str:
    if before_effective_rank == after_effective_rank and before_top1 == after_top1:
        return "unchanged"
    if _crossed_into(1, before_effective_rank, after_effective_rank) or _crossed_into(
        3, before_effective_rank, after_effective_rank
    ) or _crossed_into(10, before_effective_rank, after_effective_rank):
        return "material_improvement"
    if _crossed_out(1, before_effective_rank, after_effective_rank) or _crossed_out(
        3, before_effective_rank, after_effective_rank
    ) or _crossed_out(10, before_effective_rank, after_effective_rank):
        return "material_degradation"
    if before_effective_rank is None and after_effective_rank is not None:
        return "material_improvement"
    if after_effective_rank is None and before_effective_rank is not None:
        return "material_degradation"
    if (
        before_effective_rank is not None
        and after_effective_rank is not None
        and after_effective_rank < before_effective_rank
    ):
        return "minor_improvement"
    if (
        before_effective_rank is not None
        and after_effective_rank is not None
        and after_effective_rank > before_effective_rank
    ):
        return "minor_degradation"
    if before_top1 != after_top1:
        return "unchanged_rank_top1_changed"
    return "unchanged"


def _top1_role(row: CaseSimilarityQueryEvalResult, top1: str | None) -> str:
    if top1 is None:
        return "missing"
    top_key = ecli_key(top1) if top1.upper().startswith("ECLI:") else top1
    primary = row.expected_primary_ecli or row.expected_primary_document_id
    if primary and (
        ecli_key(primary) if str(primary).upper().startswith("ECLI:") else primary
    ) == top_key:
        return "expected_primary"
    for alt in row.accepted_alternative_document_ids:
        alt_key = ecli_key(alt) if str(alt).upper().startswith("ECLI:") else alt
        if alt_key == top_key:
            return "accepted_alternative"
    for hard in row.hard_negative_document_ids:
        hard_key = ecli_key(hard) if str(hard).upper().startswith("ECLI:") else hard
        if hard_key == top_key:
            return "hard_negative"
    return "unrelated_document"


def metrics_snapshot_from_rows(
    rows: Sequence[CaseSimilarityQueryEvalResult],
) -> tuple[MetricsSnapshot, CaseSimilarityAggregateMetrics]:
    aggregate = aggregate_case_similarity_metrics(rows)
    evaluable = [
        row
        for row in rows
        if row.corpus_compatible and row.failure_type != "retrieval_error" and not row.error
    ]
    snapshot = MetricsSnapshot(
        evaluable=aggregate.evaluable_positive_retrieval_queries,
        retrieval_failures=aggregate.retrieval_execution_failures,
        hit_at_1=aggregate.hit_at_1,
        hit_at_3=aggregate.hit_at_3,
        hit_at_5=aggregate.hit_at_5,
        hit_at_10=aggregate.hit_at_10,
        mrr=aggregate.mrr,
        hit_at_1_count=sum(1 for row in evaluable if row.hit_at_1),
        hit_at_3_count=sum(1 for row in evaluable if row.hit_at_3),
        hit_at_5_count=sum(1 for row in evaluable if row.hit_at_5),
        hit_at_10_count=sum(1 for row in evaluable if row.hit_at_10),
        hard_negative_outrank_count=aggregate.hard_negative_outrank_count,
        hard_negative_outrank_rate=aggregate.hard_negative_outrank_rate,
        hard_negative_evaluable_query_count=aggregate.hard_negative_evaluable_query_count,
    )
    return snapshot, aggregate


def _float_close(left: float | None, right: float | None, *, tol: float = 1e-12) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return abs(left - right) <= tol


def compare_stored_versus_recomputed(
    *,
    stored: ParsedStoredMetrics,
    recomputed: MetricsSnapshot,
) -> list[str]:
    mismatches: list[str] = []
    checks: list[tuple[str, Any, Any]] = [
        ("Hit@1", stored.hit_at_1, recomputed.hit_at_1),
        ("Hit@3", stored.hit_at_3, recomputed.hit_at_3),
        ("Hit@5", stored.hit_at_5, recomputed.hit_at_5),
        ("Hit@10", stored.hit_at_10, recomputed.hit_at_10),
        ("MRR", stored.mrr, recomputed.mrr),
        (
            "evaluable_positive_retrieval_queries",
            stored.evaluable_positive_retrieval_queries,
            recomputed.evaluable,
        ),
        (
            "retrieval_execution_failures",
            stored.retrieval_execution_failures,
            recomputed.retrieval_failures,
        ),
        (
            "hard_negative_outrank_count",
            stored.hard_negative_outrank_count,
            recomputed.hard_negative_outrank_count,
        ),
        (
            "hard_negative_outrank_rate",
            stored.hard_negative_outrank_rate,
            recomputed.hard_negative_outrank_rate,
        ),
    ]
    for label, left, right in checks:
        if left is None:
            continue
        if isinstance(left, float) or isinstance(right, float):
            if not _float_close(float(left) if left is not None else None, float(right) if right is not None else None):
                mismatches.append(f"{label}: stored={left} recomputed={right}")
        elif left != right:
            mismatches.append(f"{label}: stored={left} recomputed={right}")
    return mismatches


def validate_run_compatibility(
    before: Mapping[str, CaseSimilarityQueryEvalResult],
    after: Mapping[str, CaseSimilarityQueryEvalResult],
) -> None:
    before_ids = set(before)
    after_ids = set(after)
    if before_ids != after_ids:
        only_before = sorted(before_ids - after_ids)
        only_after = sorted(after_ids - before_ids)
        raise CaseSimilarityRunComparisonError(
            "benchmark ID sets differ; "
            f"only_before={only_before}; only_after={only_after}"
        )
    primary_mismatches: list[str] = []
    alternative_mismatches: list[str] = []
    for query_id in sorted(before_ids):
        left = before[query_id]
        right = after[query_id]
        left_primary = left.expected_primary_ecli or left.expected_primary_document_id
        right_primary = right.expected_primary_ecli or right.expected_primary_document_id
        if (left_primary or "") != (right_primary or ""):
            primary_mismatches.append(query_id)
        if _sorted_ecli_list(left.accepted_alternative_document_ids) != _sorted_ecli_list(
            right.accepted_alternative_document_ids
        ):
            alternative_mismatches.append(query_id)
    if primary_mismatches:
        raise CaseSimilarityRunComparisonError(
            "expected primary ECLIs differ for: " + ", ".join(primary_mismatches)
        )
    if alternative_mismatches:
        raise CaseSimilarityRunComparisonError(
            "accepted alternative sets differ for: " + ", ".join(alternative_mismatches)
        )


def build_rank_diffs(
    before: Mapping[str, CaseSimilarityQueryEvalResult],
    after: Mapping[str, CaseSimilarityQueryEvalResult],
) -> list[QueryRankDiff]:
    diffs: list[QueryRankDiff] = []
    for query_id in sorted(before):
        left = before[query_id]
        right = after[query_id]
        before_top1 = _top1(left)
        after_top1 = _top1(right)
        diffs.append(
            QueryRankDiff(
                benchmark_id=query_id,
                expected_primary_ecli=right.expected_primary_ecli
                or right.expected_primary_document_id,
                accepted_alternative_eclis=_sorted_ecli_list(
                    right.accepted_alternative_document_ids
                ),
                before_top1_ecli=before_top1,
                after_top1_ecli=after_top1,
                before_primary_rank=left.primary_rank,
                after_primary_rank=right.primary_rank,
                before_best_alternative_rank=left.best_accepted_alternative_rank,
                after_best_alternative_rank=right.best_accepted_alternative_rank,
                before_effective_rank=left.best_positive_rank,
                after_effective_rank=right.best_positive_rank,
                before_reciprocal_rank=float(left.reciprocal_rank),
                after_reciprocal_rank=float(right.reciprocal_rank),
                primary_rank_delta=_rank_delta(left.primary_rank, right.primary_rank),
                effective_rank_delta=_rank_delta(
                    left.best_positive_rank, right.best_positive_rank
                ),
                top1_changed=before_top1 != after_top1,
                hit1_before=bool(left.hit_at_1),
                hit1_after=bool(right.hit_at_1),
                hit3_before=bool(left.hit_at_3),
                hit3_after=bool(right.hit_at_3),
                hit5_before=bool(left.hit_at_5),
                hit5_after=bool(right.hit_at_5),
                hit10_before=bool(left.hit_at_10),
                hit10_after=bool(right.hit_at_10),
                classification=classify_rank_change(
                    before_effective_rank=left.best_positive_rank,
                    after_effective_rank=right.best_positive_rank,
                    before_top1=before_top1,
                    after_top1=after_top1,
                ),
                after_top1_role=_top1_role(right, after_top1),
            )
        )
    return diffs


def hit1_transition_groups(
    diffs: Sequence[QueryRankDiff],
) -> dict[str, list[str]]:
    return {
        "gained_hit1": [row.benchmark_id for row in diffs if (not row.hit1_before) and row.hit1_after],
        "lost_hit1": [row.benchmark_id for row in diffs if row.hit1_before and (not row.hit1_after)],
        "retained_hit1": [row.benchmark_id for row in diffs if row.hit1_before and row.hit1_after],
        "retained_non_hit1": [
            row.benchmark_id for row in diffs if (not row.hit1_before) and (not row.hit1_after)
        ],
    }


def assign_verdict(
    *,
    groups: Mapping[str, Sequence[str]],
    compatibility_error: str | None = None,
    metric_mismatches: Sequence[str] | None = None,
    semantic_explanation: str | None = None,
) -> str:
    if compatibility_error:
        return "RUN_INCOMPATIBILITY"
    if metric_mismatches:
        return "REPORT_INCONSISTENCY"
    if semantic_explanation:
        return "METRIC_SEMANTICS_EXPLANATION"
    gained = list(groups.get("gained_hit1") or [])
    lost = list(groups.get("lost_hit1") or [])
    if not lost:
        return "NO_RANK1_REGRESSION"
    if gained and lost:
        return "OFFSET_RANK1_REGRESSION"
    return "OFFSET_RANK1_REGRESSION"


def compare_case_similarity_runs(
    *,
    before_dir: Path,
    after_dir: Path,
) -> dict[str, Any]:
    before_rows = load_run_results(before_dir)
    after_rows = load_run_results(after_dir)
    before_cfg = load_run_config(before_dir)
    after_cfg = load_run_config(after_dir)
    before_indexed = _index_by_query_id(before_rows)
    after_indexed = _index_by_query_id(after_rows)
    validate_run_compatibility(before_indexed, after_indexed)

    before_metrics, _ = metrics_snapshot_from_rows(before_rows)
    after_metrics, _ = metrics_snapshot_from_rows(after_rows)
    before_stored = parse_stored_report_metrics(before_dir)
    after_stored = parse_stored_report_metrics(after_dir)
    before_mismatches = compare_stored_versus_recomputed(
        stored=before_stored, recomputed=before_metrics
    )
    after_mismatches = compare_stored_versus_recomputed(
        stored=after_stored, recomputed=after_metrics
    )
    if before_mismatches or after_mismatches:
        raise CaseSimilarityRunComparisonError(
            "stored-versus-recomputed metric mismatch: "
            + json.dumps(
                {"before": before_mismatches, "after": after_mismatches},
                ensure_ascii=False,
            )
        )

    diffs = build_rank_diffs(before_indexed, after_indexed)
    groups = hit1_transition_groups(diffs)
    verdict = assign_verdict(groups=groups)

    schema_notes: list[str] = []
    before_keys = sorted(before_rows[0].model_dump().keys())
    after_keys = sorted(after_rows[0].model_dump().keys())
    if before_keys != after_keys:
        schema_notes.append(
            "query result field sets differ: "
            f"only_before={sorted(set(before_keys) - set(after_keys))}; "
            f"only_after={sorted(set(after_keys) - set(before_keys))}"
        )

    hit1_before_count = before_metrics.hit_at_1_count
    hit1_after_count = after_metrics.hit_at_1_count
    arithmetic = {
        "hit1_before_count": hit1_before_count,
        "hit1_after_count": hit1_after_count,
        "gained_hit1_count": len(groups["gained_hit1"]),
        "lost_hit1_count": len(groups["lost_hit1"]),
        "formula": "hit1_after = hit1_before + gained_hit1 - lost_hit1",
        "check": hit1_after_count
        == hit1_before_count + len(groups["gained_hit1"]) - len(groups["lost_hit1"]),
        "hit_at_1_before_fraction": f"{hit1_before_count}/{before_metrics.evaluable}",
        "hit_at_1_after_fraction": f"{hit1_after_count}/{after_metrics.evaluable}",
    }

    material_improvements = [
        row.benchmark_id for row in diffs if row.classification == "material_improvement"
    ]
    material_degradations = [
        row.benchmark_id for row in diffs if row.classification == "material_degradation"
    ]

    return {
        "schema": "case_similarity_rank_diff.v1",
        "before_dir": str(Path(before_dir)),
        "after_dir": str(Path(after_dir)),
        "verdict": verdict,
        "evaluator_semantics": {
            "hit_at_k_uses": "best_positive_rank = min(primary_rank, best_accepted_alternative_rank)",
            "hit_at_1_requires_primary_at_rank_1": False,
            "accepted_alternative_counts_for_hit_at_1": True,
            "reciprocal_rank_uses": "1 / best_positive_rank (0 if missing)",
            "ranks_scoped_to_top_k": 10,
            "missing_rank_means": "not present in stored TOP 10 retrieved list",
            "blocked_hard_negative_rows_included_in_hit_k": True,
            "hard_negative_outrank_denominator": "hard_negative_evaluable queries only",
        },
        "compatibility": {
            "same_benchmark_ids": True,
            "same_expected_primaries": True,
            "same_accepted_alternatives": True,
            "before_benchmark_sha256": before_cfg.get("benchmark_sha256"),
            "after_benchmark_sha256": after_cfg.get("benchmark_sha256"),
            "before_code_commit": before_cfg.get("code_commit"),
            "after_code_commit": after_cfg.get("code_commit"),
            "schema_notes": schema_notes,
            "retrieval_knobs_equal": {
                key: before_cfg.get(key) == after_cfg.get(key)
                for key in (
                    "target_collection",
                    "embedding_model",
                    "bm25_index_id",
                    "dense_candidate_chunks",
                    "bm25_candidate_chunks",
                    "fused_candidate_chunks",
                    "candidate_documents",
                    "rrf_k",
                    "fusion",
                    "reranker",
                    "aggregation",
                )
            },
        },
        "metrics": {
            "before": asdict(before_metrics),
            "after": asdict(after_metrics),
            "stored_before": asdict(before_stored),
            "stored_after": asdict(after_stored),
            "stored_versus_recomputed_ok": True,
        },
        "hit1_transitions": groups,
        "hit1_arithmetic": arithmetic,
        "material_improvements": material_improvements,
        "material_degradations": material_degradations,
        "rows": [asdict(row) for row in diffs],
    }


def write_comparison_outputs(payload: Mapping[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "case_similarity_rank_diff.json"
    csv_path = output_dir / "case_similarity_rank_diff.csv"
    md_path = output_dir / "case_similarity_rank_diff_report.md"

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    rows = list(payload.get("rows") or [])
    fieldnames = list(rows[0].keys()) if rows else [
        "benchmark_id",
        "expected_primary_ecli",
        "classification",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = {}
            for key, value in row.items():
                if isinstance(value, list):
                    serialized[key] = "|".join(str(item) for item in value)
                elif value is None:
                    serialized[key] = ""
                else:
                    serialized[key] = value
            writer.writerow(serialized)

    md_path.write_text(_render_markdown_report(payload), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": md_path}


def _render_markdown_report(payload: Mapping[str, Any]) -> str:
    metrics = payload["metrics"]
    before = metrics["before"]
    after = metrics["after"]
    groups = payload["hit1_transitions"]
    arithmetic = payload["hit1_arithmetic"]
    lines = [
        "# Case Similarity Rank Diff Audit",
        "",
        f"**Verdict:** `{payload['verdict']}`",
        "",
        "## Answer first",
        "",
        (
            f"Hit@1 stayed `{before['hit_at_1']}` "
            f"({arithmetic['hit_at_1_before_fraction']}) because "
            f"`gained_hit1={groups['gained_hit1']}` and "
            f"`lost_hit1={groups['lost_hit1']}`."
        ),
        "",
        (
            f"Arithmetic check: "
            f"{arithmetic['hit1_before_count']} + {arithmetic['gained_hit1_count']} "
            f"- {arithmetic['lost_hit1_count']} = {arithmetic['hit1_after_count']} "
            f"(ok={arithmetic['check']})."
        ),
        "",
        "## Metrics",
        "",
        "| metric | before | after |",
        "|---|---:|---:|",
        f"| evaluable | {before['evaluable']} | {after['evaluable']} |",
        f"| retrieval_failures | {before['retrieval_failures']} | {after['retrieval_failures']} |",
        f"| Hit@1 | {before['hit_at_1']} ({before['hit_at_1_count']}/{before['evaluable']}) | {after['hit_at_1']} ({after['hit_at_1_count']}/{after['evaluable']}) |",
        f"| Hit@3 | {before['hit_at_3']} | {after['hit_at_3']} |",
        f"| Hit@5 | {before['hit_at_5']} | {after['hit_at_5']} |",
        f"| Hit@10 | {before['hit_at_10']} | {after['hit_at_10']} |",
        f"| MRR | {before['mrr']} | {after['mrr']} |",
        f"| HN outrank rate | {before['hard_negative_outrank_rate']} | {after['hard_negative_outrank_rate']} |",
        "",
        "## Hit@1 transitions",
        "",
        f"- gained_hit1: {', '.join(groups['gained_hit1']) or '—'}",
        f"- lost_hit1: {', '.join(groups['lost_hit1']) or '—'}",
        f"- retained_hit1: {', '.join(groups['retained_hit1']) or '—'}",
        f"- retained_non_hit1: {', '.join(groups['retained_non_hit1']) or '—'}",
        "",
        "## Material movements",
        "",
        f"- material_improvements: {', '.join(payload['material_improvements']) or '—'}",
        f"- material_degradations: {', '.join(payload['material_degradations']) or '—'}",
        "",
        "## Per-query effective ranks",
        "",
        "| id | primary | eff before | eff after | top1 before | top1 after | H@1 | H@10 | class |",
        "|---|---|---:|---:|---|---|---|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {id} | `{primary}` | {eb} | {ea} | `{tb}` | `{ta}` | {h1b}->{h1a} | {h10b}->{h10a} | {cls} |".format(
                id=row["benchmark_id"],
                primary=row["expected_primary_ecli"],
                eb=row["before_effective_rank"],
                ea=row["after_effective_rank"],
                tb=row["before_top1_ecli"],
                ta=row["after_top1_ecli"],
                h1b=row["hit1_before"],
                h1a=row["hit1_after"],
                h10b=row["hit10_before"],
                h10a=row["hit10_after"],
                cls=row["classification"],
            )
        )
    lines.extend(
        [
            "",
            "## Evaluator semantics",
            "",
            f"- Hit@K / MRR use `{payload['evaluator_semantics']['hit_at_k_uses']}`.",
            (
                "- Accepted alternatives "
                f"{'do' if payload['evaluator_semantics']['accepted_alternative_counts_for_hit_at_1'] else 'do not'} "
                "count for Hit@1."
            ),
            f"- Stored ranks are scoped to TOP `{payload['evaluator_semantics']['ranks_scoped_to_top_k']}` "
            f"({payload['evaluator_semantics']['missing_rank_means']}).",
            "",
        ]
    )
    return "\n".join(lines)


def compare_and_write(
    *,
    before_dir: Path,
    after_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    payload = compare_case_similarity_runs(before_dir=before_dir, after_dir=after_dir)
    write_comparison_outputs(payload, output_dir)
    return payload
