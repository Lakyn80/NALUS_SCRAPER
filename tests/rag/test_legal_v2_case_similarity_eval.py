from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.rag.legal_v2.benchmark.case_similarity_eval import (
    FAILURE_EXPECTED_DOCUMENT_MISSING,
    FAILURE_HARD_NEGATIVE_ABOVE_POSITIVE,
    FAILURE_POSITIVE_NOT_RETRIEVED,
    aggregate_case_similarity_metrics,
    dedupe_document_ids,
    evaluate_ranked_documents,
)
from app.rag.legal_v2.benchmark.case_similarity_golden import (
    HARD_NEGATIVE_BLOCKER_INSUFFICIENT_SAME_DOMAIN_CORPUS,
    CaseSimilarityGoldenItem,
    CaseSimilarityProvenance,
    AnswerEvidenceItem,
    HardNegativeRationale,
    load_case_similarity_golden_jsonl,
    validate_case_similarity_dataset,
)
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PILOT_PATH = PROJECT_ROOT / "benchmarks" / "legal_v2" / "case_similarity_golden_v1_pilot.jsonl"


def _minimal_item(**overrides):
    base = {
        "benchmark_id": "nalus-cs-pilot-test",
        "query": (
            "Potřebuji podobný rozsudek o náhradě škody po předběžném opatření a o tom, "
            "jestli soudy správně posoudily příčinnou souvislost mezi omezením dispozice "
            "a neuskutečněným prodejem. Hledám srovnatelné případy s dovoláním a ústavní "
            "stížností. Chci dokument, který řeší stejný právní problém, ne jen obecný "
            "odkaz na náhradu škody. Potřebuji i odůvodnění k zamítnutí nároku a jasné "
            "vysvětlení, proč nebyla shledána dostatečná souvislost mezi omezením a ztrátou."
        ),
        "query_style": "client_narrative",
        "difficulty": "medium",
        "source_document_id": "doc-aaaaaaaaaaaaaaaa",
        "expected_document_ids": ["doc-aaaaaaaaaaaaaaaa"],
        "hard_negative_document_ids": ["doc-bbbbbbbbbbbbbbbb"],
        "supporting_block_ids": [
            "doc-aaaaaaaaaaaaaaaa:p:00001:aaaaaaaaaaaaaaaaaaaa",
            "doc-aaaaaaaaaaaaaaaa:p:00002:bbbbbbbbbbbbbbbbbbbb",
        ],
        "answer_evidence": [
            AnswerEvidenceItem(
                block_id="doc-aaaaaaaaaaaaaaaa:p:00001:aaaaaaaaaaaaaaaaaaaa",
                excerpt="excerpt one",
            ),
            AnswerEvidenceItem(
                block_id="doc-aaaaaaaaaaaaaaaa:p:00002:bbbbbbbbbbbbbbbbbbbb",
                excerpt="excerpt two",
            ),
        ],
        "factual_facets": ["facet_a"],
        "legal_issue_facets": ["issue_a"],
        "procedural_facets": ["proc_a"],
        "similarity_rationale": "test rationale",
        "hard_negative_rationales": [
            HardNegativeRationale(
                document_id="doc-bbbbbbbbbbbbbbbb",
                looks_similar_because="looks similar",
                materially_incorrect_because="wrong",
            )
        ],
        "provenance": CaseSimilarityProvenance(
            builder="test",
            review_number=1,
        ),
        "hard_negative_evaluable": True,
        "hard_negative_blocker": None,
    }
    base.update(overrides)
    return CaseSimilarityGoldenItem(**base)


def test_hard_negative_fields_default_evaluable() -> None:
    item = _minimal_item()
    assert item.hard_negative_evaluable is True
    assert item.hard_negative_blocker is None


def test_pilot_007_is_explicitly_blocked() -> None:
    items = load_case_similarity_golden_jsonl(PILOT_PATH)
    item = next(row for row in items if row.benchmark_id == "nalus-cs-pilot-007")
    assert item.hard_negative_evaluable is False
    assert item.hard_negative_blocker == HARD_NEGATIVE_BLOCKER_INSUFFICIENT_SAME_DOMAIN_CORPUS


def test_validator_rejects_evaluable_with_blocker() -> None:
    with pytest.raises(ValidationError):
        _minimal_item(
            hard_negative_evaluable=True,
            hard_negative_blocker=HARD_NEGATIVE_BLOCKER_INSUFFICIENT_SAME_DOMAIN_CORPUS,
        )


def test_validator_rejects_blocked_without_blocker() -> None:
    with pytest.raises(ValidationError):
        _minimal_item(hard_negative_evaluable=False, hard_negative_blocker=None)


def test_blocked_row_included_in_hit_and_mrr_excluded_from_hn_denominator() -> None:
    blocked = evaluate_ranked_documents(
        query_id="nalus-cs-pilot-007",
        query="q",
        query_style="multi_issue_client_narrative",
        difficulty="hard",
        expected_primary_document_id="doc-primary",
        accepted_alternative_document_ids=[],
        hard_negative_document_ids=["doc-hn"],
        hard_negative_evaluable=False,
        hard_negative_blocker=HARD_NEGATIVE_BLOCKER_INSUFFICIENT_SAME_DOMAIN_CORPUS,
        ranked_document_ids=["doc-primary", "doc-hn"],
    )
    evaluable = evaluate_ranked_documents(
        query_id="nalus-cs-pilot-001",
        query="q",
        query_style="client_narrative",
        difficulty="medium",
        expected_primary_document_id="doc-primary",
        accepted_alternative_document_ids=[],
        hard_negative_document_ids=["doc-hn"],
        hard_negative_evaluable=True,
        hard_negative_blocker=None,
        ranked_document_ids=["doc-hn", "doc-primary"],
    )
    assert blocked.hard_negative_before_positive is None
    assert blocked.hit_at_1 is True
    assert blocked.reciprocal_rank == 1.0
    metrics = aggregate_case_similarity_metrics([blocked, evaluable])
    assert metrics.hard_negative_evaluable_query_count == 1
    assert metrics.hard_negative_blocked_query_count == 1
    assert metrics.hard_negative_outrank_count == 1
    assert metrics.hard_negative_outrank_rate == 1.0
    assert metrics.hard_negative_outrank_query_ids == ["nalus-cs-pilot-001"]
    assert metrics.hit_at_1 == 0.5
    assert metrics.mrr == pytest.approx(0.75)


def test_primary_at_rank_1() -> None:
    row = evaluate_ranked_documents(
        query_id="q1",
        query="q",
        query_style="client_narrative",
        difficulty="easy",
        expected_primary_document_id="doc-p",
        accepted_alternative_document_ids=[],
        hard_negative_document_ids=["doc-hn"],
        hard_negative_evaluable=True,
        hard_negative_blocker=None,
        ranked_document_ids=["doc-p", "doc-x"],
    )
    assert row.primary_rank == 1
    assert row.hit_at_1 and row.hit_at_10
    assert row.reciprocal_rank == 1.0


def test_primary_at_rank_5() -> None:
    ranked = [f"doc-{i}" for i in range(1, 5)] + ["doc-p"]
    row = evaluate_ranked_documents(
        query_id="q1",
        query="q",
        query_style="client_narrative",
        difficulty="medium",
        expected_primary_document_id="doc-p",
        accepted_alternative_document_ids=[],
        hard_negative_document_ids=["doc-hn"],
        hard_negative_evaluable=True,
        hard_negative_blocker=None,
        ranked_document_ids=ranked,
    )
    assert row.primary_rank == 5
    assert row.hit_at_1 is False
    assert row.hit_at_5 is True
    assert row.reciprocal_rank == pytest.approx(0.2)


def test_accepted_alternative_wins_when_primary_absent() -> None:
    row = evaluate_ranked_documents(
        query_id="q1",
        query="q",
        query_style="client_narrative",
        difficulty="medium",
        expected_primary_document_id="doc-p",
        accepted_alternative_document_ids=["doc-alt"],
        hard_negative_document_ids=["doc-hn"],
        hard_negative_evaluable=True,
        hard_negative_blocker=None,
        ranked_document_ids=["doc-alt", "doc-x"],
    )
    assert row.primary_rank is None
    assert row.best_positive_document_id == "doc-alt"
    assert row.best_positive_rank == 1
    assert row.hit_at_1 is True


def test_no_positive_in_top_10() -> None:
    row = evaluate_ranked_documents(
        query_id="q1",
        query="q",
        query_style="client_narrative",
        difficulty="hard",
        expected_primary_document_id="doc-p",
        accepted_alternative_document_ids=[],
        hard_negative_document_ids=["doc-hn"],
        hard_negative_evaluable=True,
        hard_negative_blocker=None,
        ranked_document_ids=[f"doc-{i}" for i in range(10)],
    )
    assert row.best_positive_rank is None
    assert row.hit_at_10 is False
    assert row.failure_type == FAILURE_POSITIVE_NOT_RETRIEVED


def test_hard_negative_above_positive_evaluable() -> None:
    row = evaluate_ranked_documents(
        query_id="q1",
        query="q",
        query_style="client_narrative",
        difficulty="hard",
        expected_primary_document_id="doc-p",
        accepted_alternative_document_ids=[],
        hard_negative_document_ids=["doc-hn"],
        hard_negative_evaluable=True,
        hard_negative_blocker=None,
        ranked_document_ids=["doc-hn", "doc-p"],
    )
    assert row.hard_negative_before_positive is True
    assert row.hit_at_1 is False
    assert row.hit_at_3 is True
    assert row.failure_type == FAILURE_HARD_NEGATIVE_ABOVE_POSITIVE


def test_duplicate_chunks_deduped_stable_order() -> None:
    assert dedupe_document_ids(["a", "a", "b", "a", "c", "b"]) == ["a", "b", "c"]
    row = evaluate_ranked_documents(
        query_id="q1",
        query="q",
        query_style="client_narrative",
        difficulty="easy",
        expected_primary_document_id="doc-p",
        accepted_alternative_document_ids=[],
        hard_negative_document_ids=["doc-hn"],
        hard_negative_evaluable=True,
        hard_negative_blocker=None,
        ranked_document_ids=["doc-p", "doc-p", "doc-x", "doc-p"],
    )
    assert row.retrieved_document_ids == ["doc-p", "doc-x"]
    assert row.primary_rank == 1


def test_missing_primary_in_corpus_failure() -> None:
    row = evaluate_ranked_documents(
        query_id="q1",
        query="q",
        query_style="client_narrative",
        difficulty="hard",
        expected_primary_document_id="doc-p",
        accepted_alternative_document_ids=[],
        hard_negative_document_ids=["doc-hn"],
        hard_negative_evaluable=True,
        hard_negative_blocker=None,
        ranked_document_ids=[],
        corpus_compatible=False,
        failure_type=FAILURE_EXPECTED_DOCUMENT_MISSING,
        error="missing",
    )
    assert row.corpus_compatible is False
    assert row.failure_type == FAILURE_EXPECTED_DOCUMENT_MISSING
    metrics = aggregate_case_similarity_metrics([row])
    assert metrics.corpus_index_failures == 1
    assert metrics.evaluable_positive_retrieval_queries == 0


def test_correct_mrr_and_grouped_metrics() -> None:
    rows = [
        evaluate_ranked_documents(
            query_id="a",
            query="q",
            query_style="client_narrative",
            difficulty="easy",
            expected_primary_document_id="doc-p",
            accepted_alternative_document_ids=[],
            hard_negative_document_ids=["doc-hn"],
            hard_negative_evaluable=True,
            hard_negative_blocker=None,
            ranked_document_ids=["doc-p"],
        ),
        evaluate_ranked_documents(
            query_id="b",
            query="q",
            query_style="noisy_client_narrative",
            difficulty="hard",
            expected_primary_document_id="doc-p",
            accepted_alternative_document_ids=[],
            hard_negative_document_ids=["doc-hn"],
            hard_negative_evaluable=True,
            hard_negative_blocker=None,
            ranked_document_ids=["doc-x", "doc-p"],
        ),
    ]
    metrics = aggregate_case_similarity_metrics(rows)
    assert metrics.mrr == pytest.approx((1.0 + 0.5) / 2)
    assert metrics.by_difficulty["easy"]["hit_at_1"] == 1.0
    assert metrics.by_query_style["noisy_client_narrative"]["hit_at_1"] == 0.0


def test_007_does_not_alter_hard_negative_outrank_metric() -> None:
    blocked = evaluate_ranked_documents(
        query_id="nalus-cs-pilot-007",
        query="q",
        query_style="multi_issue_client_narrative",
        difficulty="hard",
        expected_primary_document_id="doc-p",
        accepted_alternative_document_ids=[],
        hard_negative_document_ids=["doc-hn"],
        hard_negative_evaluable=False,
        hard_negative_blocker=HARD_NEGATIVE_BLOCKER_INSUFFICIENT_SAME_DOMAIN_CORPUS,
        ranked_document_ids=["doc-hn", "doc-p"],
    )
    alone = aggregate_case_similarity_metrics(
        [
            evaluate_ranked_documents(
                query_id="nalus-cs-pilot-001",
                query="q",
                query_style="client_narrative",
                difficulty="medium",
                expected_primary_document_id="doc-p",
                accepted_alternative_document_ids=[],
                hard_negative_document_ids=["doc-hn"],
                hard_negative_evaluable=True,
                hard_negative_blocker=None,
                ranked_document_ids=["doc-p", "doc-hn"],
            )
        ]
    )
    with_blocked = aggregate_case_similarity_metrics(
        [
            evaluate_ranked_documents(
                query_id="nalus-cs-pilot-001",
                query="q",
                query_style="client_narrative",
                difficulty="medium",
                expected_primary_document_id="doc-p",
                accepted_alternative_document_ids=[],
                hard_negative_document_ids=["doc-hn"],
                hard_negative_evaluable=True,
                hard_negative_blocker=None,
                ranked_document_ids=["doc-p", "doc-hn"],
            ),
            blocked,
        ]
    )
    assert alone.hard_negative_outrank_rate == 0.0
    assert with_blocked.hard_negative_outrank_rate == 0.0
    assert with_blocked.hard_negative_blocked_query_count == 1
