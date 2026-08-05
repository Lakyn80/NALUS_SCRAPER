from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.legal_v2.benchmark.corpus import load_development_corpus
from app.rag.legal_v2.benchmark.retrieval_golden import (
    DEFAULT_PILOT_DATASET,
    RetrievalGoldenItem,
    evidence_excerpt_in_block,
    load_retrieval_golden_jsonl,
    normalize_query_text,
    validate_retrieval_golden_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PILOT_PATH = PROJECT_ROOT / "benchmarks" / "legal_v2" / "retrieval_golden_v1_pilot.jsonl"


@pytest.fixture(scope="module")
def pilot_items() -> list[RetrievalGoldenItem]:
    assert PILOT_PATH.exists(), f"missing pilot dataset: {PILOT_PATH}"
    return load_retrieval_golden_jsonl(PILOT_PATH)


@pytest.fixture(scope="module")
def development_corpus():
    return load_development_corpus()


def test_default_pilot_path_points_at_tracked_dataset() -> None:
    assert DEFAULT_PILOT_DATASET.resolve() == PILOT_PATH.resolve()


def test_pilot_counts_and_unique_ids(pilot_items: list[RetrievalGoldenItem]) -> None:
    assert len(pilot_items) == 30
    positives = [item for item in pilot_items if not item.is_negative]
    negatives = [item for item in pilot_items if item.is_negative]
    assert len(positives) == 29
    assert len(negatives) == 1
    assert len({item.query_id for item in pilot_items}) == 30
    assert len({normalize_query_text(item.query) for item in pilot_items}) == 30
    assert all(item.split == "development" for item in pilot_items)


def test_pilot_validates_against_development_corpus(
    pilot_items: list[RetrievalGoldenItem],
    development_corpus,
) -> None:
    report = validate_retrieval_golden_dataset(
        pilot_items,
        blocks_by_id=development_corpus.blocks_by_id,
        dataset_path=str(PILOT_PATH),
    )
    assert report.ok, report.model_dump()
    assert report.failure_count == 0


def test_positive_evidence_excerpts_are_verbatim_substrings(
    pilot_items: list[RetrievalGoldenItem],
    development_corpus,
) -> None:
    for item in pilot_items:
        if item.is_negative:
            continue
        assert item.primary_expected_block_id
        block = development_corpus.blocks_by_id[item.primary_expected_block_id]
        assert evidence_excerpt_in_block(item.evidence_excerpt or "", block.raw_text)
        assert item.source_document_id == block.document_id
        assert item.expected_block_ids
        assert item.primary_expected_block_id in item.expected_block_ids


def test_negative_entry_shape(pilot_items: list[RetrievalGoldenItem]) -> None:
    negatives = [item for item in pilot_items if item.is_negative]
    assert len(negatives) == 1
    item = negatives[0]
    assert item.query_id == "nalus-rg-pilot-030"
    assert item.expected_document_ids == []
    assert item.expected_block_ids == []
    assert item.primary_expected_block_id is None
    assert item.evidence_excerpt is None
    assert item.negative_rationale
    assert len(item.inspected_negative_candidates) >= 3


def test_hard_negatives_exist_and_do_not_overlap_expected(
    pilot_items: list[RetrievalGoldenItem],
    development_corpus,
) -> None:
    for item in pilot_items:
        if item.is_negative:
            continue
        for block_id in item.hard_negative_block_ids:
            assert block_id in development_corpus.blocks_by_id
            assert block_id not in item.expected_block_ids
        for block_id in item.accepted_alternative_block_ids:
            assert block_id in development_corpus.blocks_by_id


def test_only_development_documents_used(
    pilot_items: list[RetrievalGoldenItem],
    development_corpus,
) -> None:
    allowed = {ref.document_id for ref in development_corpus.documents}
    for item in pilot_items:
        if item.is_negative:
            continue
        assert item.source_document_id in allowed
        for document_id in item.expected_document_ids:
            assert document_id in allowed


def test_pilot_008_uses_operative_disposition_alternative(
    pilot_items: list[RetrievalGoldenItem],
    development_corpus,
) -> None:
    item = next(entry for entry in pilot_items if entry.query_id == "nalus-rg-pilot-008")
    caption_id = "doc-a5292901931de05a:p:00004:ca14de3c7403e6c824c8"
    assert caption_id not in item.accepted_alternative_block_ids
    assert caption_id not in item.hard_negative_block_ids
    assert item.primary_expected_block_id not in item.hard_negative_block_ids
    assert len(item.accepted_alternative_block_ids) == 1
    alt_id = item.accepted_alternative_block_ids[0]
    alt = development_corpus.blocks_by_id[alt_id]
    assert "Ústavní stížnost se odmítá." in alt.raw_text
    assert alt_id not in item.hard_negative_block_ids
    assert item.hard_negative_block_ids == [
        "doc-a5292901931de05a:p:00010:c24aba59016b3431f019"
    ]


def test_pilot_010_and_024_have_no_accepted_alternatives(
    pilot_items: list[RetrievalGoldenItem],
) -> None:
    for query_id in ("nalus-rg-pilot-010", "nalus-rg-pilot-024"):
        item = next(entry for entry in pilot_items if entry.query_id == query_id)
        assert item.accepted_alternative_block_ids == []


def test_deterministic_reload_matches(pilot_items: list[RetrievalGoldenItem]) -> None:
    again = load_retrieval_golden_jsonl(PILOT_PATH)
    assert [item.model_dump() for item in again] == [item.model_dump() for item in pilot_items]
