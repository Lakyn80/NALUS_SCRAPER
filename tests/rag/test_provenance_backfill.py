from __future__ import annotations

import pytest

from app.rag.retrieval.errors import RetrievalConfigurationError
from app.rag.retrieval.production_profile import BGE_M3_DENSE_BM25_RRF
from app.rag.retrieval.provenance import (
    derive_document_id,
    ensure_embedding_provenance,
    validate_embedding_provenance,
)


def test_derive_document_id_from_ecli() -> None:
    payload = {"ecli": "ECLI:CZ:US:2026:1.US.927.25.1", "source_document_id": "other"}
    assert derive_document_id(payload) == "ECLI:CZ:US:2026:1.US.927.25.1"


def test_ensure_embedding_provenance_backfills_missing_fields() -> None:
    payload = {
        "text": "právo na spravedlivý proces",
        "source": "usoud / nalus",
        "ecli": "ECLI:CZ:US:2026:1.US.927.25.1",
        "chunk_index": 0,
        "chunk_id": 1,
    }
    enriched = ensure_embedding_provenance(
        payload,
        profile=BGE_M3_DENSE_BM25_RRF,
        qdrant_collection="nalus_us_bge_m3_rag_combined_20260709",
        bm25_index_id=BGE_M3_DENSE_BM25_RRF.name,
        ingest_run_id="test-backfill",
    )
    validate_embedding_provenance(
        enriched,
        profile=BGE_M3_DENSE_BM25_RRF,
        qdrant_collection="nalus_us_bge_m3_rag_combined_20260709",
        bm25_index_id=BGE_M3_DENSE_BM25_RRF.name,
    )
    assert enriched["document_id"] == "ECLI:CZ:US:2026:1.US.927.25.1"
    assert enriched["embedding_model"] == "BAAI/bge-m3"
    assert enriched["embedding_dimension"] == 1024
    assert enriched["content_checksum"]


def test_ensure_embedding_provenance_requires_text() -> None:
    with pytest.raises(RetrievalConfigurationError, match="content_checksum"):
        ensure_embedding_provenance(
            {"ecli": "ECLI:CZ:US:2026:1"},
            profile=BGE_M3_DENSE_BM25_RRF,
            qdrant_collection="nalus_us_bge_m3_rag_combined_20260709",
            bm25_index_id=BGE_M3_DENSE_BM25_RRF.name,
            ingest_run_id="test-backfill",
        )
