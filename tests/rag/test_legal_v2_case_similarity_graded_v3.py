"""Tests for Legal v2 golden v3 graded multi-relevance benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.rag.legal_v2.benchmark.case_similarity_graded_eval import (
    QrelEntry,
    aggregate_graded_metrics,
    build_qrel_map,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    evaluate_graded_query,
    infer_relevance_from_retrieval_rank,
    is_binary_relevant,
    JUDGMENT_EXPLICIT_NOT_RELEVANT,
    JUDGMENT_GRADED,
)
from app.rag.legal_v2.benchmark.case_similarity_golden_v2 import (
    EXPECTED_DEV_COUNT,
    EXPECTED_QUERY_COUNT,
    EXPECTED_TEST_COUNT,
    load_case_similarity_golden_v2_jsonl,
)
from app.rag.legal_v2.benchmark.case_similarity_golden_v3 import (
    BINARY_RELEVANCE_THRESHOLD,
    DEFAULT_V3_DATASET,
    load_case_similarity_golden_v3_jsonl,
    validate_v3_split_counts,
)
from app.rag.legal_v2.benchmark.case_similarity_query_audit import (
    CURATED_QUERY_REWRITES,
    audit_and_rewrite_v2_query,
    classify_query,
    rewrite_query,
)
from scripts.legal_v2.build_case_similarity_golden_v3_graded import _build_v3_items


V2_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "legal_v2" / "case_similarity_golden_v2_full_corpus.jsonl"


def test_v3_split_preserved_from_v2() -> None:
    v2_items = load_case_similarity_golden_v2_jsonl(V2_PATH)
    v3_items, _ = _build_v3_items(v2_items, client=None, collection="")
    validate_v3_split_counts(v3_items)
    v2_splits = {item.query_id: item.split for item in v2_items}
    for item in v3_items:
        assert item.split == v2_splits[item.query_id]


def test_multiple_grade_3_judgments_one_query() -> None:
    qrels = build_qrel_map(
        [
            QrelEntry("q1", "ECLI:CZ:US:2024:1.US.1.1", 3),
            QrelEntry("q1", "ECLI:CZ:US:2024:1.US.2.1", 3),
            QrelEntry("q1", "ECLI:CZ:US:2024:1.US.3.1", 1),
        ]
    )
    ranked = [
        "ECLI:CZ:US:2024:1.US.2.1",
        "ECLI:CZ:US:2024:1.US.9.9.1",
        "ECLI:CZ:US:2024:1.US.1.1",
    ]
    ndcg, mode = compute_ndcg_at_k(ranked, qrels, k=3)
    assert ndcg > 0
    result = evaluate_graded_query(
        query_id="q1",
        ranked_document_ids=ranked,
        qrel_entries=list(qrels.values()),
        legacy_primary_document_id="ECLI:CZ:US:2024:1.US.9.9.1",
    )
    assert result.judged_highly_relevant_count == 2
    assert result.mrr_highly_relevant == 1.0
    assert result.success_at_10_highly_relevant is True


def test_unjudged_distinct_from_explicit_grade_0() -> None:
    qrels = build_qrel_map(
        [
            QrelEntry("q1", "ECLI:CZ:US:2024:1.US.1.1", 0, judgment_state=JUDGMENT_EXPLICIT_NOT_RELEVANT),
            QrelEntry("q1", "ECLI:CZ:US:2024:1.US.2.1", 3, judgment_state=JUDGMENT_GRADED),
        ]
    )
    ranked = ["ECLI:CZ:US:2024:1.US.9.9.1", "ECLI:CZ:US:2024:1.US.2.1"]
    result = evaluate_graded_query(
        query_id="q1",
        ranked_document_ids=ranked,
        qrel_entries=list(qrels.values()),
        legacy_primary_document_id="ECLI:CZ:US:2024:1.US.9.9.1",
    )
    assert result.explicit_not_relevant_count == 1
    assert result.unjudged_in_pool_count == 1
    assert "pooled_eval_unjudged" in " ".join(result.notes)


def test_ndcg_calculation_known_ordering() -> None:
    qrels = build_qrel_map([QrelEntry("q1", "doc-a", 3), QrelEntry("q1", "doc-b", 2)])
    perfect, _ = compute_ndcg_at_k(["doc-a", "doc-b"], qrels, k=2)
    imperfect, _ = compute_ndcg_at_k(["doc-b", "doc-a"], qrels, k=2)
    assert perfect == pytest.approx(1.0)
    assert imperfect < perfect


def test_recall_binary_threshold_grade_gte_2() -> None:
    qrels = build_qrel_map(
        [
            QrelEntry("q1", "doc-a", 3),
            QrelEntry("q1", "doc-b", 2),
            QrelEntry("q1", "doc-c", 1),
        ]
    )
    recall = compute_recall_at_k(["doc-a", "doc-x"], qrels, k=10)
    assert recall == pytest.approx(0.5)
    assert is_binary_relevant(2) is True
    assert is_binary_relevant(1) is False
    assert BINARY_RELEVANCE_THRESHOLD == 2


def test_precision_ignores_partial_relevance() -> None:
    qrels = build_qrel_map([QrelEntry("q1", "doc-a", 1), QrelEntry("q1", "doc-b", 2)])
    precision = compute_precision_at_k(["doc-a", "doc-b"], qrels, k=2)
    assert precision == pytest.approx(0.5)


def test_mrr_highly_relevant_and_legacy_primary_diagnostic() -> None:
    qrels = build_qrel_map([QrelEntry("q1", "doc-a", 3)])
    result = evaluate_graded_query(
        query_id="q1",
        ranked_document_ids=["doc-x", "doc-a"],
        qrel_entries=list(qrels.values()),
        legacy_primary_document_id="doc-x",
    )
    assert result.mrr_highly_relevant == pytest.approx(0.5)
    assert result.legacy_primary_rank == 1
    assert result.legacy_primary_hit_at_1 is True


def test_no_automatic_relevance_from_retrieval_rank() -> None:
    with pytest.raises(RuntimeError, match="forbidden"):
        infer_relevance_from_retrieval_rank("doc", rank=1)


def test_query_audit_rewrites_auto_generated_template() -> None:
    text = (
        "Zajímá mě, jak Ústavní soud posoudil tuto situaci: test otázka. "
        "Chci podobnou judikaturu bez uvádění konkrétní spisové značky."
    )
    status, flags = classify_query("nalus-cs-v2-099", text)
    assert status == "needs_edit"
    assert "boilerplate_closing_phrase" in flags
    rewritten, applied = rewrite_query("nalus-cs-v2-014", text, legal_area="formal_rejection")
    assert applied is True
    assert "advokátem" in rewritten


def test_sanity_check_query_001_rewrite_covers_defective_complaint_theme() -> None:
    rewrite = CURATED_QUERY_REWRITES["nalus-cs-v2-001"]
    lowered = rewrite.casefold()
    assert "náležitost" in lowered or "vad" in lowered
    assert "vyloučení" in lowered or "soudc" in lowered


def test_curated_rewrites_cover_all_v2_query_ids() -> None:
    v2_items = load_case_similarity_golden_v2_jsonl(V2_PATH)
    assert len(v2_items) == EXPECTED_QUERY_COUNT
    for item in v2_items:
        assert item.query_id in CURATED_QUERY_REWRITES


def test_build_v3_benchmark_file_roundtrip(tmp_path: Path) -> None:
    v2_items = load_case_similarity_golden_v2_jsonl(V2_PATH)
    v3_items, reviews = _build_v3_items(v2_items, client=None, collection="")
    out = tmp_path / "v3.jsonl"
    from app.rag.legal_v2.benchmark.case_similarity_golden_v3 import write_case_similarity_golden_v3_jsonl

    write_case_similarity_golden_v3_jsonl(out, v3_items)
    loaded = load_case_similarity_golden_v3_jsonl(out)
    assert len(loaded) == EXPECTED_QUERY_COUNT
    assert sum(1 for row in reviews if row["review_status"] == "edited") >= 50
    dev = sum(1 for item in loaded if item.split == "dev")
    test = sum(1 for item in loaded if item.split == "test")
    assert dev == EXPECTED_DEV_COUNT
    assert test == EXPECTED_TEST_COUNT


def test_aggregate_graded_metrics() -> None:
    qrels = [QrelEntry("q1", "doc-a", 3)]
    row = evaluate_graded_query(
        query_id="q1",
        ranked_document_ids=["doc-a"],
        qrel_entries=qrels,
        legacy_primary_document_id="doc-a",
    )
    agg = aggregate_graded_metrics([row])
    assert agg.total_queries == 1
    assert agg.ndcg_at_10 == pytest.approx(1.0)
