"""Tests for Case Similarity run comparison / rank-diff audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.rag.legal_v2.benchmark.case_similarity_eval import (
    CaseSimilarityQueryEvalResult,
    evaluate_ranked_documents,
)
from app.rag.legal_v2.benchmark.case_similarity_run_comparison import (
    CaseSimilarityRunComparisonError,
    assign_verdict,
    build_rank_diffs,
    classify_rank_change,
    compare_and_write,
    compare_case_similarity_runs,
    hit1_transition_groups,
    metrics_snapshot_from_rows,
    write_comparison_outputs,
)


def _row(
    *,
    query_id: str,
    primary: str,
    ranked: list[str],
    alternatives: list[str] | None = None,
    hard_negatives: list[str] | None = None,
) -> CaseSimilarityQueryEvalResult:
    return evaluate_ranked_documents(
        query_id=query_id,
        query=f"query text for {query_id}",
        query_style="client_narrative",
        difficulty="easy",
        expected_primary_document_id=primary,
        expected_primary_ecli=primary,
        accepted_alternative_document_ids=list(alternatives or []),
        hard_negative_document_ids=list(hard_negatives or []),
        hard_negative_evaluable=True,
        hard_negative_blocker=None,
        ranked_document_ids=ranked,
    )


def _write_run(
    path: Path,
    rows: list[CaseSimilarityQueryEvalResult],
    *,
    metrics_override: dict | None = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "case_similarity_retrieval_results.jsonl").write_text(
        "\n".join(row.model_dump_json() for row in rows) + "\n",
        encoding="utf-8",
    )
    snapshot, _ = metrics_snapshot_from_rows(rows)
    metrics = {
        "Hit@1": snapshot.hit_at_1,
        "Hit@3": snapshot.hit_at_3,
        "Hit@5": snapshot.hit_at_5,
        "Hit@10": snapshot.hit_at_10,
        "MRR": snapshot.mrr,
        "evaluable_positive_retrieval_queries": snapshot.evaluable,
        "retrieval_execution_failures": snapshot.retrieval_failures,
        "hard_negative_outrank_count": snapshot.hard_negative_outrank_count,
        "hard_negative_outrank_rate": snapshot.hard_negative_outrank_rate,
    }
    if metrics_override:
        metrics.update(metrics_override)
    report_lines = [
        "# Case Similarity Golden v1 — Retrieval Baseline Report",
        "",
        "## Aggregate positive-retrieval metrics",
        "",
        f"- evaluable_positive_retrieval_queries: `{metrics['evaluable_positive_retrieval_queries']}`",
        f"- retrieval_execution_failures: `{metrics['retrieval_execution_failures']}`",
        f"- Hit@1: `{metrics['Hit@1']}`",
        f"- Hit@3: `{metrics['Hit@3']}`",
        f"- Hit@5: `{metrics['Hit@5']}`",
        f"- Hit@10: `{metrics['Hit@10']}`",
        f"- MRR: `{metrics['MRR']}`",
        "",
        "## Hard-negative metrics",
        "",
        f"- hard_negative_outrank_count: `{metrics['hard_negative_outrank_count']}`",
        (
            "- hard_negative_outrank_rate (evaluable denominator only): "
            f"`{metrics['hard_negative_outrank_rate']}`"
        ),
        "",
    ]
    (path / "case_similarity_retrieval_report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )
    config = {
        "benchmark_sha256": "abc",
        "code_commit": "deadbeef",
        "target_collection": "coll",
        "embedding_model": "model",
        "bm25_index_id": "bm25",
        "dense_candidate_chunks": 80,
        "bm25_candidate_chunks": 80,
        "fused_candidate_chunks": 120,
        "candidate_documents": 40,
        "rrf_k": 60,
        "fusion": "rrf",
        "reranker": None,
        "aggregation": "agg",
    }
    (path / "retrieval_run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_identical_runs_produce_no_rank_changes(tmp_path: Path) -> None:
    rows = [
        _row(
            query_id="q1",
            primary="ECLI:CZ:US:2024:1.US.1.24.1",
            ranked=["ECLI:CZ:US:2024:1.US.1.24.1", "ECLI:CZ:US:2024:1.US.2.24.1"],
        ),
        _row(
            query_id="q2",
            primary="ECLI:CZ:US:2024:1.US.3.24.1",
            ranked=["ECLI:CZ:US:2024:1.US.9.24.1", "ECLI:CZ:US:2024:1.US.3.24.1"],
        ),
    ]
    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_run(before, rows)
    _write_run(after, rows)
    payload = compare_case_similarity_runs(before_dir=before, after_dir=after)
    assert payload["verdict"] == "NO_RANK1_REGRESSION"
    assert all(row["classification"] == "unchanged" for row in payload["rows"])
    assert payload["hit1_transitions"]["gained_hit1"] == []
    assert payload["hit1_transitions"]["lost_hit1"] == []


def test_gained_and_lost_hit1_reconcile_to_unchanged_aggregate(tmp_path: Path) -> None:
    before_rows = [
        _row(
            query_id="gained",
            primary="ECLI:CZ:US:2025:1.US.1.25.1",
            ranked=["ECLI:CZ:US:2025:1.US.9.25.1", "ECLI:CZ:US:2025:1.US.8.25.1"],
        ),
        _row(
            query_id="lost",
            primary="ECLI:CZ:VSPH:2026:2.Cmo.1.2026.1",
            ranked=[
                "ECLI:CZ:VSPH:2026:2.Cmo.1.2026.1",
                "ECLI:CZ:NS:2025:22.CDO.1.2024.1",
            ],
        ),
    ]
    after_rows = [
        _row(
            query_id="gained",
            primary="ECLI:CZ:US:2025:1.US.1.25.1",
            ranked=[
                "ECLI:CZ:US:2025:1.US.1.25.1",
                "ECLI:CZ:US:2025:1.US.9.25.1",
            ],
        ),
        _row(
            query_id="lost",
            primary="ECLI:CZ:VSPH:2026:2.Cmo.1.2026.1",
            ranked=[
                "ECLI:CZ:NS:2025:22.CDO.1.2024.1",
                "ECLI:CZ:VSPH:2026:2.Cmo.1.2026.1",
            ],
        ),
    ]
    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_run(before, before_rows)
    _write_run(after, after_rows)
    payload = compare_case_similarity_runs(before_dir=before, after_dir=after)
    assert payload["verdict"] == "OFFSET_RANK1_REGRESSION"
    assert payload["hit1_transitions"]["gained_hit1"] == ["gained"]
    assert payload["hit1_transitions"]["lost_hit1"] == ["lost"]
    assert payload["metrics"]["before"]["hit_at_1"] == 0.5
    assert payload["metrics"]["after"]["hit_at_1"] == 0.5
    assert payload["hit1_arithmetic"]["check"] is True


def test_accepted_alternative_rank_1_counts_for_hit1() -> None:
    row = _row(
        query_id="alt",
        primary="ECLI:CZ:US:2024:1.US.1.24.1",
        alternatives=["ECLI:CZ:US:2024:1.US.2.24.1"],
        ranked=["ECLI:CZ:US:2024:1.US.2.24.1", "ECLI:CZ:US:2024:1.US.1.24.1"],
    )
    assert row.primary_rank == 2
    assert row.best_accepted_alternative_rank == 1
    assert row.best_positive_rank == 1
    assert row.hit_at_1 is True
    assert row.reciprocal_rank == 1.0


def test_missing_rank_is_null_in_diff() -> None:
    before = {
        "q": _row(
            query_id="q",
            primary="ECLI:CZ:US:2024:1.US.1.24.1",
            ranked=["ECLI:CZ:US:2024:1.US.9.24.1"],
        )
    }
    after = {
        "q": _row(
            query_id="q",
            primary="ECLI:CZ:US:2024:1.US.1.24.1",
            ranked=["ECLI:CZ:US:2024:1.US.1.24.1"],
        )
    }
    diff = build_rank_diffs(before, after)[0]
    assert diff.before_primary_rank is None
    assert diff.before_effective_rank is None
    assert diff.after_primary_rank == 1
    assert diff.classification == "material_improvement"


def test_classification_material_thresholds() -> None:
    assert (
        classify_rank_change(
            before_effective_rank=None,
            after_effective_rank=1,
            before_top1="a",
            after_top1="b",
        )
        == "material_improvement"
    )
    assert (
        classify_rank_change(
            before_effective_rank=1,
            after_effective_rank=2,
            before_top1="a",
            after_top1="b",
        )
        == "material_degradation"
    )
    assert (
        classify_rank_change(
            before_effective_rank=8,
            after_effective_rank=7,
            before_top1="a",
            after_top1="a",
        )
        == "minor_improvement"
    )


def test_same_hit10_lower_mrr_is_visible(tmp_path: Path) -> None:
    before_rows = [
        _row(
            query_id="q1",
            primary="ECLI:CZ:US:2024:1.US.1.24.1",
            ranked=["ECLI:CZ:US:2024:1.US.1.24.1"],
        ),
        _row(
            query_id="q2",
            primary="ECLI:CZ:US:2024:1.US.2.24.1",
            ranked=[
                "ECLI:CZ:US:2024:1.US.9.24.1",
                "ECLI:CZ:US:2024:1.US.2.24.1",
            ],
        ),
    ]
    after_rows = [
        _row(
            query_id="q1",
            primary="ECLI:CZ:US:2024:1.US.1.24.1",
            ranked=["ECLI:CZ:US:2024:1.US.1.24.1"],
        ),
        _row(
            query_id="q2",
            primary="ECLI:CZ:US:2024:1.US.2.24.1",
            ranked=[
                "ECLI:CZ:US:2024:1.US.9.24.1",
                "ECLI:CZ:US:2024:1.US.8.24.1",
                "ECLI:CZ:US:2024:1.US.2.24.1",
            ],
        ),
    ]
    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_run(before, before_rows)
    _write_run(after, after_rows)
    payload = compare_case_similarity_runs(before_dir=before, after_dir=after)
    assert payload["metrics"]["before"]["hit_at_10"] == 1.0
    assert payload["metrics"]["after"]["hit_at_10"] == 1.0
    assert payload["metrics"]["after"]["mrr"] < payload["metrics"]["before"]["mrr"]


def test_different_benchmark_ids_fail(tmp_path: Path) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_run(
        before,
        [_row(query_id="a", primary="ECLI:CZ:US:2024:1.US.1.24.1", ranked=["ECLI:CZ:US:2024:1.US.1.24.1"])],
    )
    _write_run(
        after,
        [_row(query_id="b", primary="ECLI:CZ:US:2024:1.US.1.24.1", ranked=["ECLI:CZ:US:2024:1.US.1.24.1"])],
    )
    with pytest.raises(CaseSimilarityRunComparisonError, match="benchmark ID sets differ"):
        compare_case_similarity_runs(before_dir=before, after_dir=after)


def test_different_expected_primary_fails(tmp_path: Path) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_run(
        before,
        [_row(query_id="q", primary="ECLI:CZ:US:2024:1.US.1.24.1", ranked=["ECLI:CZ:US:2024:1.US.1.24.1"])],
    )
    _write_run(
        after,
        [_row(query_id="q", primary="ECLI:CZ:US:2024:1.US.2.24.1", ranked=["ECLI:CZ:US:2024:1.US.2.24.1"])],
    )
    with pytest.raises(CaseSimilarityRunComparisonError, match="expected primary"):
        compare_case_similarity_runs(before_dir=before, after_dir=after)


def test_different_alternatives_fail(tmp_path: Path) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_run(
        before,
        [
            _row(
                query_id="q",
                primary="ECLI:CZ:US:2024:1.US.1.24.1",
                alternatives=["ECLI:CZ:US:2024:1.US.2.24.1"],
                ranked=["ECLI:CZ:US:2024:1.US.1.24.1"],
            )
        ],
    )
    _write_run(
        after,
        [
            _row(
                query_id="q",
                primary="ECLI:CZ:US:2024:1.US.1.24.1",
                alternatives=["ECLI:CZ:US:2024:1.US.3.24.1"],
                ranked=["ECLI:CZ:US:2024:1.US.1.24.1"],
            )
        ],
    )
    with pytest.raises(CaseSimilarityRunComparisonError, match="accepted alternative"):
        compare_case_similarity_runs(before_dir=before, after_dir=after)


def test_stored_metric_mismatch_detected(tmp_path: Path) -> None:
    rows = [
        _row(
            query_id="q",
            primary="ECLI:CZ:US:2024:1.US.1.24.1",
            ranked=["ECLI:CZ:US:2024:1.US.1.24.1"],
        )
    ]
    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_run(before, rows, metrics_override={"Hit@1": 0.42})
    _write_run(after, rows)
    with pytest.raises(CaseSimilarityRunComparisonError, match="stored-versus-recomputed"):
        compare_case_similarity_runs(before_dir=before, after_dir=after)


def test_duplicate_benchmark_ids_fail(tmp_path: Path) -> None:
    row = _row(
        query_id="dup",
        primary="ECLI:CZ:US:2024:1.US.1.24.1",
        ranked=["ECLI:CZ:US:2024:1.US.1.24.1"],
    )
    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_run(before, [row, row])
    _write_run(after, [row])
    with pytest.raises(CaseSimilarityRunComparisonError, match="duplicate benchmark IDs"):
        compare_case_similarity_runs(before_dir=before, after_dir=after)


def test_deterministic_json_csv_ordering(tmp_path: Path) -> None:
    rows = [
        _row(
            query_id="nalus-cs-pilot-002",
            primary="ECLI:CZ:US:2024:1.US.2.24.1",
            ranked=["ECLI:CZ:US:2024:1.US.2.24.1"],
        ),
        _row(
            query_id="nalus-cs-pilot-001",
            primary="ECLI:CZ:US:2024:1.US.1.24.1",
            ranked=["ECLI:CZ:US:2024:1.US.1.24.1"],
        ),
    ]
    before = tmp_path / "before"
    after = tmp_path / "after"
    out = tmp_path / "out"
    _write_run(before, rows)
    _write_run(after, rows)
    payload = compare_and_write(before_dir=before, after_dir=after, output_dir=out)
    ids = [row["benchmark_id"] for row in payload["rows"]]
    assert ids == sorted(ids)
    csv_text = (out / "case_similarity_rank_diff.csv").read_text(encoding="utf-8")
    assert "nalus-cs-pilot-001" in csv_text.splitlines()[1]
    assert "nalus-cs-pilot-002" in csv_text.splitlines()[2]
    # Historical run artifacts remain untouched.
    before_jsonl = (before / "case_similarity_retrieval_results.jsonl").read_text(encoding="utf-8")
    assert "nalus-cs-pilot-002" in before_jsonl


def test_assign_verdict_helpers() -> None:
    assert (
        assign_verdict(groups={"gained_hit1": [], "lost_hit1": []})
        == "NO_RANK1_REGRESSION"
    )
    assert (
        assign_verdict(groups={"gained_hit1": ["a"], "lost_hit1": ["b"]})
        == "OFFSET_RANK1_REGRESSION"
    )
    assert (
        assign_verdict(
            groups={"gained_hit1": [], "lost_hit1": []},
            metric_mismatches=["Hit@1"],
        )
        == "REPORT_INCONSISTENCY"
    )


def test_hit1_transition_groups_cover_all_queries() -> None:
    before = {
        "a": _row(
            query_id="a",
            primary="ECLI:CZ:US:2024:1.US.1.24.1",
            ranked=["ECLI:CZ:US:2024:1.US.1.24.1"],
        ),
        "b": _row(
            query_id="b",
            primary="ECLI:CZ:US:2024:1.US.2.24.1",
            ranked=["ECLI:CZ:US:2024:1.US.9.24.1"],
        ),
    }
    after = {
        "a": _row(
            query_id="a",
            primary="ECLI:CZ:US:2024:1.US.1.24.1",
            ranked=["ECLI:CZ:US:2024:1.US.9.24.1", "ECLI:CZ:US:2024:1.US.1.24.1"],
        ),
        "b": _row(
            query_id="b",
            primary="ECLI:CZ:US:2024:1.US.2.24.1",
            ranked=["ECLI:CZ:US:2024:1.US.2.24.1"],
        ),
    }
    diffs = build_rank_diffs(before, after)
    groups = hit1_transition_groups(diffs)
    assert groups["lost_hit1"] == ["a"]
    assert groups["gained_hit1"] == ["b"]
