from __future__ import annotations

import pytest

from app.rag.retrieval.errors import RetrievalConfigurationError
from app.rag.retrieval.production_profile import BGE_M3_DENSE_BM25_RRF
from app.rag.retrieval.provenance import (
    derive_chunk_index,
    derive_document_id,
    derive_source,
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
            {"ecli": "ECLI:CZ:US:2026:1", "chunk_index": 0},
            profile=BGE_M3_DENSE_BM25_RRF,
            qdrant_collection="nalus_us_bge_m3_rag_combined_20260709",
            bm25_index_id=BGE_M3_DENSE_BM25_RRF.name,
            ingest_run_id="test-backfill",
        )


def test_derive_document_id_from_nsoud_chunk_metadata() -> None:
    payload = {
        "chunk_id": 923,
        "chunk_metadata": {
            "source_document_id": "ECLI:CZ:NS:2025:3.TDO.1120.2024.1",
            "chunk_id": "ECLI:CZ:NS:2025:3.TDO.1120.2024.1__chunk_0003",
            "case_number": "3 Tdo 1120/2024",
        },
        "text": "náhrada nemajetkové újmy",
    }
    assert derive_document_id(payload) == "ECLI:CZ:NS:2025:3.TDO.1120.2024.1"
    assert derive_chunk_index(payload) == 3
    assert derive_source(payload) == "nsoud"


def test_ensure_embedding_provenance_backfills_nsoud_chunk_metadata() -> None:
    payload = {
        "chunk_id": 923,
        "chunk_metadata": {
            "source_document_id": "ECLI:CZ:NS:2025:3.TDO.1120.2024.1",
            "chunk_id": "ECLI:CZ:NS:2025:3.TDO.1120.2024.1__chunk_0003",
            "case_number": "3 Tdo 1120/2024",
            "section_type": "reasoning",
            "legal_area": "criminal",
        },
        "model_code": "bge_m3",
        "source_id": 1,
        "text": "náhrada nemajetkové újmy",
    }
    collection = "nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1"
    enriched = ensure_embedding_provenance(
        payload,
        profile=BGE_M3_DENSE_BM25_RRF,
        qdrant_collection=collection,
        bm25_index_id=BGE_M3_DENSE_BM25_RRF.name,
        ingest_run_id="nsoud-bge-m3-provenance-backfill-v1",
    )
    validate_embedding_provenance(
        enriched,
        profile=BGE_M3_DENSE_BM25_RRF,
        qdrant_collection=collection,
        bm25_index_id=BGE_M3_DENSE_BM25_RRF.name,
    )
    assert enriched["document_id"] == "ECLI:CZ:NS:2025:3.TDO.1120.2024.1"
    assert enriched["chunk_index"] == 3
    assert enriched["source"] == "nsoud"
