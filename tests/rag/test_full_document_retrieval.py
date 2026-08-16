from __future__ import annotations

import pytest

from app.rag.retrieval.full_document import (
    build_full_document_from_payloads,
    resolve_full_document_collection_name,
    validate_document_id,
)


def test_builds_full_document_from_contiguous_indexed_chunks() -> None:
    result = build_full_document_from_payloads(
        document_id="ECLI:CZ:US:2026:3.US.446.26.1",
        collection_name="test-collection",
        payloads=[
            {
                "original_id": "chunk-1",
                "document_id": "ECLI:CZ:US:2026:3.US.446.26.1",
                "text": "Druhá část.",
                "chunk_index": 1,
            },
            {
                "original_id": "chunk-0",
                "source_document_id": "ECLI:CZ:US:2026:3.US.446.26.1",
                "text": "První část.",
                "chunk_index": 0,
                "source": "nalus",
            },
        ],
    )

    assert result.full_text == "První část.\n\nDruhá část."
    assert result.full_text_availability_status == "available"
    assert result.metadata["ecli"] == "ECLI:CZ:US:2026:3.US.446.26.1"
    assert result.metadata["court_name"] == "Ústavní soud"
    assert result.diagnostics.missing_chunk_indexes == []
    assert result.diagnostics.duplicate_chunk_indexes == []


def test_partial_status_when_chunk_index_is_missing() -> None:
    result = build_full_document_from_payloads(
        document_id="DOC-A",
        collection_name="test-collection",
        payloads=[
            {"original_id": "chunk-a", "document_id": "DOC-A", "text": "Text bez indexu."}
        ],
    )

    assert result.full_text == "Text bez indexu."
    assert result.full_text_availability_status == "partial"
    assert result.diagnostics.all_chunks_have_index is False


def test_partial_status_when_indexes_have_gaps_and_duplicates() -> None:
    result = build_full_document_from_payloads(
        document_id="DOC-A",
        collection_name="test-collection",
        payloads=[
            {
                "original_id": "chunk-0",
                "document_id": "DOC-A",
                "text": "První.",
                "chunk_index": 0,
            },
            {
                "original_id": "chunk-2a",
                "document_id": "DOC-A",
                "text": "Třetí A.",
                "chunk_index": 2,
            },
            {
                "original_id": "chunk-2b",
                "document_id": "DOC-A",
                "text": "Třetí B.",
                "chunk_index": 2,
            },
        ],
    )

    assert result.full_text_availability_status == "partial"
    assert result.diagnostics.missing_chunk_indexes == [1]
    assert result.diagnostics.duplicate_chunk_indexes == [2]


def test_returns_not_found_when_no_matching_payload_exists() -> None:
    result = build_full_document_from_payloads(
        document_id="DOC-A",
        collection_name="test-collection",
        payloads=[{"document_id": "DOC-B", "text": "Jiný dokument.", "chunk_index": 0}],
    )

    assert result.full_text_availability_status == "not_found"
    assert result.full_text == ""
    assert result.chunks == []


@pytest.mark.parametrize("document_id", ["", "   ", "../secret", "DOC\\A", "A\x00B"])
def test_validate_document_id_rejects_unsafe_values(document_id: str) -> None:
    with pytest.raises(ValueError):
        validate_document_id(document_id)


def test_resolve_full_document_collection_prefers_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NALUS_FULL_DOCUMENT_QDRANT_COLLECTION", "full-doc-override")
    monkeypatch.setenv("NALUS_LEGAL_V2_CASE_SIMILARITY_ENABLED", "1")
    monkeypatch.setenv("NALUS_LEGAL_V2_QDRANT_COLLECTION", "legal-v2-pilot")
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "legacy")
    assert resolve_full_document_collection_name() == "full-doc-override"


def test_resolve_full_document_collection_follows_stage1_search_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NALUS_FULL_DOCUMENT_QDRANT_COLLECTION", raising=False)
    monkeypatch.setenv("NALUS_LEGAL_V2_CASE_SIMILARITY_ENABLED", "1")
    monkeypatch.setenv("NALUS_LEGAL_V2_QDRANT_COLLECTION", "nalus_legal_paragraph_chunks_v2_pilot_600")
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "nalus_bge_m3_chunks_v1")
    assert (
        resolve_full_document_collection_name()
        == "nalus_legal_paragraph_chunks_v2_pilot_600"
    )


def test_resolve_full_document_collection_falls_back_to_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NALUS_FULL_DOCUMENT_QDRANT_COLLECTION", raising=False)
    monkeypatch.delenv("NALUS_LEGAL_V2_CASE_SIMILARITY_ENABLED", raising=False)
    monkeypatch.delenv("NALUS_LEGAL_V2_SEARCH_ENABLED", raising=False)
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "nalus_bge_m3_chunks_v1")
    assert resolve_full_document_collection_name() == "nalus_bge_m3_chunks_v1"


def test_build_full_document_matches_canonical_document_id() -> None:
    ecli = "ECLI:CZ:US:2022:2.US.62.22.1"
    result = build_full_document_from_payloads(
        document_id=ecli,
        collection_name="pilot",
        payloads=[
            {
                "original_id": "chunk-0",
                "canonical_document_id": ecli,
                "text": "Text z legal_v2 chunku.",
                "chunk_index": 0,
            }
        ],
    )
    assert result.full_text_availability_status == "available"
    assert result.full_text == "Text z legal_v2 chunku."


def test_document_id_filter_omits_unindexed_secondary_keys() -> None:
    from app.rag.retrieval.full_document import (
        _DOCUMENT_ID_FILTER_KEYS,
        _DOCUMENT_ID_KEYS,
        _document_id_filter,
    )

    assert "case_number" not in _DOCUMENT_ID_FILTER_KEYS
    assert "reference" not in _DOCUMENT_ID_FILTER_KEYS
    assert "case_number" in _DOCUMENT_ID_KEYS
    assert "reference" in _DOCUMENT_ID_KEYS

    scroll_filter = _document_id_filter("ECLI:CZ:US:2009:2.US.1255.07.1")
    filter_keys = [condition.key for condition in scroll_filter.should]
    assert filter_keys == list(_DOCUMENT_ID_FILTER_KEYS)
