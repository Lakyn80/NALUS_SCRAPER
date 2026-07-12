from __future__ import annotations

import pytest

from app.rag.retrieval.document_retrieval import (
    DocumentRetrievalConfig,
    build_document_level_results,
    canonical_document_id,
    document_retrieval_config_from_env,
)
from app.rag.retrieval.errors import RetrievalConfigurationError
from app.rag.retrieval.models import RetrievedChunk


def _chunk(
    chunk_id: str,
    *,
    score: float,
    text: str | None = None,
    document_id: str | None = "ECLI:CZ:NS:2026:1.TEST.1",
    metadata: dict | None = None,
) -> RetrievedChunk:
    payload = dict(metadata or {})
    if document_id is not None:
        payload.setdefault("document_id", document_id)
    return RetrievedChunk(
        id=chunk_id,
        text=text if text is not None else f"text {chunk_id}",
        score=score,
        source="hybrid",
        metadata=payload,
    )


def _config(**overrides) -> DocumentRetrievalConfig:
    data = {
        "enabled": True,
        "max_candidate_chunks": 200,
        "max_returned_documents": 50,
        "max_supporting_chunks_per_document": 3,
        "document_relevance_threshold": 0.0,
    }
    data.update(overrides)
    return DocumentRetrievalConfig(**data)


def test_groups_chunks_by_canonical_document_id_and_deduplicates_documents() -> None:
    result = build_document_level_results(
        candidate_chunks=[
            _chunk("a-1", score=0.9, document_id="DOC-A"),
            _chunk("a-2", score=0.7, document_id="DOC-A"),
            _chunk("b-1", score=0.8, document_id="DOC-B"),
        ],
        config=_config(),
    )

    assert [document.document_id for document in result.documents] == ["DOC-A", "DOC-B"]
    assert result.documents[0].candidate_chunk_count == 2
    assert result.diagnostics.candidate_chunks_retrieved == 3
    assert result.diagnostics.unique_documents_produced == 2
    assert result.diagnostics.duplicate_document_hits_removed == 1


def test_document_score_uses_best_and_average_top_supporting_chunks() -> None:
    result = build_document_level_results(
        candidate_chunks=[
            _chunk("a-1", score=1.0, document_id="DOC-A"),
            _chunk("a-2", score=0.5, document_id="DOC-A"),
        ],
        config=_config(max_supporting_chunks_per_document=2),
    )

    assert result.documents[0].score == pytest.approx(0.925)


def test_dynamic_threshold_returns_empty_without_hidden_fallback() -> None:
    result = build_document_level_results(
        candidate_chunks=[_chunk("a-1", score=0.6, document_id="DOC-A")],
        config=_config(document_relevance_threshold=0.8),
    )

    assert result.documents == []
    assert result.diagnostics.documents_filtered == 1
    assert result.diagnostics.final_document_count == 0


def test_limits_candidate_chunks_returned_documents_and_supporting_chunks() -> None:
    result = build_document_level_results(
        candidate_chunks=[
            _chunk("a-1", score=0.9, document_id="DOC-A"),
            _chunk("a-2", score=0.8, document_id="DOC-A"),
            _chunk("b-1", score=0.7, document_id="DOC-B"),
            _chunk("c-1", score=0.6, document_id="DOC-C"),
        ],
        config=_config(
            max_candidate_chunks=3,
            max_returned_documents=1,
            max_supporting_chunks_per_document=1,
        ),
    )

    assert result.diagnostics.candidate_chunks_retrieved == 3
    assert len(result.documents) == 1
    assert result.documents[0].document_id == "DOC-A"
    assert len(result.documents[0].best_passages) == 1


def test_best_passages_remove_identical_text_and_order_by_relevance() -> None:
    result = build_document_level_results(
        candidate_chunks=[
            _chunk("a-1", score=0.7, text="same text", document_id="DOC-A"),
            _chunk("a-2", score=0.9, text="same   text", document_id="DOC-A"),
            _chunk("a-3", score=0.8, text="different text", document_id="DOC-A"),
        ],
        config=_config(max_supporting_chunks_per_document=3),
    )

    assert [passage.chunk_id for passage in result.documents[0].best_passages] == [
        "a-2",
        "a-3",
    ]


def test_missing_document_id_chunks_are_skipped_with_diagnostics() -> None:
    result = build_document_level_results(
        candidate_chunks=[
            _chunk("missing", score=0.9, document_id=None),
            _chunk("valid", score=0.8, document_id="DOC-A"),
        ],
        config=_config(),
    )

    assert [document.document_id for document in result.documents] == ["DOC-A"]
    assert result.diagnostics.chunks_missing_document_id == 1


def test_duplicate_chunk_id_keeps_highest_scored_chunk() -> None:
    result = build_document_level_results(
        candidate_chunks=[
            _chunk("a-1", score=0.3, document_id="DOC-A"),
            _chunk("a-1", score=0.9, document_id="DOC-A"),
        ],
        config=_config(),
    )

    assert result.documents[0].best_chunk_score == 0.9
    assert result.diagnostics.duplicate_chunks_removed == 1


def test_canonical_document_id_prefers_source_document_id() -> None:
    chunk = _chunk(
        "a-1",
        score=0.9,
        metadata={
            "source_document_id": "SOURCE-DOC",
            "document_id": "DOC",
            "ecli": "ECLI",
        },
    )

    assert canonical_document_id(chunk) == "SOURCE-DOC"


def test_config_from_env_validates_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NALUS_DOCUMENT_RETRIEVAL_ENABLED", "1")
    monkeypatch.setenv("NALUS_DOCUMENT_MAX_CANDIDATE_CHUNKS", "10")
    monkeypatch.setenv("NALUS_DOCUMENT_MAX_RETURNED_DOCUMENTS", "5")
    monkeypatch.setenv("NALUS_DOCUMENT_MAX_SUPPORTING_CHUNKS_PER_DOCUMENT", "2")
    monkeypatch.setenv("NALUS_DOCUMENT_RELEVANCE_THRESHOLD", "0.25")
    monkeypatch.setenv("NALUS_DOCUMENT_LATENCY_BUDGET_MS", "1000")

    config = document_retrieval_config_from_env()

    assert config.enabled is True
    assert config.max_candidate_chunks == 10
    assert config.max_returned_documents == 5
    assert config.max_supporting_chunks_per_document == 2
    assert config.document_relevance_threshold == 0.25
    assert config.latency_budget_ms == 1000


def test_invalid_config_rejects_unsafe_values() -> None:
    with pytest.raises(RetrievalConfigurationError):
        _config(max_candidate_chunks=0).validate()
