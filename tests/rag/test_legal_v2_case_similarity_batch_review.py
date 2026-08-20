"""Tests for Golden v3 batch relevance review tooling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.rag.legal_v2.benchmark.case_similarity_batch_review import (
    apply_confirmed_to_queue,
    assert_freeze_allowed,
    build_proposal_rows,
    normalize_confirmed_rows,
    propose_grade_from_content,
    qrels_to_jsonl_rows,
    reviewed_qrel_entries,
    select_batch_query_ids,
    split_review_complete,
)
from app.rag.legal_v2.benchmark.case_similarity_graded_eval import (
    infer_relevance_from_retrieval_rank,
)


def _queue_row(
    *,
    query_id: str,
    document_id: str,
    split: str = "dev",
    query_text: str = "Musí Ústavní soud poučit o advokátovi při vadném podání?",
    summary: str = "Ústavní soud odmítl vadné podání bez advokáta.",
    review_status: str = "pending",
    relevance_grade: int | None = None,
) -> dict:
    return {
        "query_id": query_id,
        "split": split,
        "query_text": query_text,
        "document_id": document_id,
        "ecli": document_id,
        "candidate_summary": summary,
        "central_legal_issue": summary,
        "reasoning_excerpt": summary,
        "dense_rank": 1,
        "bm25_rank": 2,
        "hybrid_rank": 1,
        "relevance_grade": relevance_grade,
        "relevance_label": None,
        "reviewer_notes": "",
        "review_status": review_status,
    }


def test_select_batch_pending_dev() -> None:
    rows = [
        _queue_row(query_id="q-a", document_id="d1"),
        _queue_row(query_id="q-b", document_id="d1"),
        _queue_row(query_id="q-c", document_id="d1"),
        _queue_row(query_id="q-d", document_id="d1"),
        _queue_row(query_id="q-e", document_id="d1"),
        _queue_row(query_id="q-f", document_id="d1"),
    ]
    batch = select_batch_query_ids(rows, split="dev", batch_size=5, batch_index=1)
    assert batch.query_ids == ["q-a", "q-b", "q-c", "q-d", "q-e"]


def test_propose_ignores_ranks_and_uses_content() -> None:
    high = propose_grade_from_content(
        query_text="Musí Ústavní soud poučit o povinném zastoupení advokátem při vadném podání?",
        candidate_summary="Ústavní soud odmítl vadné podání bez advokáta a bez dalšího poučení.",
        reasoning_excerpt="Formální vady a chybějící zastoupení advokátem.",
        central_legal_issue="vadné podání advokát",
        is_legacy_primary=False,
    )
    low = propose_grade_from_content(
        query_text="Musí Ústavní soud poučit o povinném zastoupení advokátem při vadném podání?",
        candidate_summary="Spor o celní dluh a vymáhání cla.",
        reasoning_excerpt="Celní orgán zahájil vymáhací řízení.",
        central_legal_issue="clo",
        is_legacy_primary=False,
    )
    assert high[0] in {2, 3}
    assert low[0] in {0, 1, None}


def test_build_proposal_does_not_copy_rank_into_grade() -> None:
    rows = [
        _queue_row(
            query_id="q1",
            document_id="ECLI:CZ:US:2024:1.US.1.1",
            summary="Nepříbuzný restituční spor o knihovní vložku z roku 1948.",
        )
    ]
    proposals = build_proposal_rows(
        rows,
        query_ids=["q1"],
        legacy_by_query={"q1": "ECLI:CZ:US:2024:1.US.9.9.1"},
    )
    assert proposals[0]["dense_rank"] == 1
    assert proposals[0]["proposed_grade"] != proposals[0]["dense_rank"]


def test_apply_only_updates_confirmed_rows() -> None:
    queue = [
        _queue_row(query_id="q1", document_id="d1"),
        _queue_row(query_id="q1", document_id="d2"),
    ]
    confirmed = [
        {
            "query_id": "q1",
            "document_id": "d1",
            "proposed_grade": 3,
            "proposed_reason": "direct match",
            "needs_human_check": False,
            "final_grade": 3,
            "final_reason": "direct match",
        }
    ]
    updated, count = apply_confirmed_to_queue(queue, confirmed)
    assert count == 1
    by_doc = {row["document_id"]: row for row in updated}
    assert by_doc["d1"]["review_status"] == "reviewed"
    assert by_doc["d1"]["relevance_grade"] == 3
    assert by_doc["d2"]["review_status"] == "pending"
    assert by_doc["d2"]["relevance_grade"] is None


def test_needs_human_check_cannot_enter_qrels_without_final_grade() -> None:
    with pytest.raises(ValueError, match="NEEDS_HUMAN_CHECK"):
        normalize_confirmed_rows(
            [
                {
                    "query_id": "q1",
                    "document_id": "d1",
                    "proposed_grade": None,
                    "needs_human_check": True,
                    "final_grade": None,
                }
            ]
        )


def test_unjudged_not_in_reviewed_qrels() -> None:
    queue = [
        _queue_row(query_id="q1", document_id="d1", review_status="reviewed", relevance_grade=0),
        _queue_row(query_id="q1", document_id="d2", review_status="pending", relevance_grade=None),
        _queue_row(query_id="q1", document_id="d3", review_status="reviewed", relevance_grade=3),
    ]
    entries = reviewed_qrel_entries(queue, split="dev")
    ids = {entry.document_id for entry in entries}
    assert ids == {"d1", "d3"}
    grades = {entry.document_id: entry.grade for entry in entries}
    assert grades["d1"] == 0
    assert grades["d3"] == 3


def test_exclude_from_qrels_skips_merged_duplicate_alias() -> None:
    queue = [
        _queue_row(
            query_id="nalus-cs-v2-015",
            document_id="ECLI:CZ:US:1999:4.US.23.99.1",
            review_status="human_reviewed",
            relevance_grade=3,
        ),
        {
            **_queue_row(
                query_id="nalus-cs-v2-015",
                document_id="ECLI:CZ:US:1999:4.US.23.99",
                review_status="human_reviewed",
                relevance_grade=3,
            ),
            "exclude_from_qrels": True,
            "is_duplicate": True,
            "dedup_status": "merged_into_canonical",
            "duplicate_of": "ECLI:CZ:US:1999:4.US.23.99.1",
        },
    ]
    entries = reviewed_qrel_entries(queue, split="dev")
    assert [entry.document_id for entry in entries] == ["ECLI:CZ:US:1999:4.US.23.99.1"]
    assert split_review_complete(queue, "dev") is True


def test_split_dev_export_and_freeze_gate(tmp_path: Path) -> None:
    queue = [
        _queue_row(query_id="q1", document_id="d1", split="dev", review_status="reviewed", relevance_grade=2),
        _queue_row(query_id="q2", document_id="d1", split="test", review_status="pending"),
    ]
    assert split_review_complete(queue, "dev") is True
    assert split_review_complete(queue, "test") is False
    with pytest.raises(RuntimeError, match="TEST"):
        assert_freeze_allowed(queue)
    rows = qrels_to_jsonl_rows(reviewed_qrel_entries(queue, split="dev"))
    assert len(rows) == 1
    assert rows[0]["grade"] == 2


def test_no_rank_to_grade_path() -> None:
    with pytest.raises(RuntimeError, match="forbidden"):
        infer_relevance_from_retrieval_rank(rank=1)
