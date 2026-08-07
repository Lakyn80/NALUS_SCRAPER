"""API integration tests for long-input preprocessing on Stage 1 search."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.rag.legal_v2.query_input.service import reset_query_input_service_for_tests
from app.rag.legal_v2.retrieve.case_similarity_search import (
    CaseSimilarityStage1Runtime,
    reset_case_similarity_stage1_runtime_for_tests,
)


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch):
    reset_case_similarity_stage1_runtime_for_tests()
    reset_query_input_service_for_tests()
    monkeypatch.setenv("NALUS_LEGAL_V2_CASE_SIMILARITY_ENABLED", "1")
    monkeypatch.delenv("NALUS_LEGAL_V2_STAGE1_WARMUP_ON_START", raising=False)
    monkeypatch.delenv("NALUS_LEGAL_V2_LONG_INPUT_ENABLED", raising=False)
    monkeypatch.delenv("NALUS_LEGAL_V2_LONG_INPUT_HARD_LIMIT", raising=False)
    yield
    reset_case_similarity_stage1_runtime_for_tests()
    reset_query_input_service_for_tests()


def _install_fake_runtime(monkeypatch: pytest.MonkeyPatch, observed: dict[str, Any]) -> None:
    class _FakeRetriever:
        def retrieve(self, query_spec):
            observed["spec"] = query_spec
            from app.rag.legal_v2.evidence.selection import CandidateEvidenceDocument
            from app.rag.legal_v2.models import LegalParagraph, MetadataProvenance, SectionType
            from app.rag.legal_v2.retrieve.retriever import LegalV2RetrievalResult

            paragraph = LegalParagraph(
                document_id="ECLI:CZ:US:2025:1.US.3575.25.1",
                paragraph_id="p1",
                paragraph_index=0,
                original_text="bez advokáta",
                normalized_text="bez advokáta",
                section_type=SectionType.FACTS,
                start_offset=0,
                end_offset=12,
                source_order=0,
                heading_context=[],
                is_boilerplate=False,
                is_citation_block=False,
                language="cs",
                metadata_provenance=MetadataProvenance(
                    source="test",
                    extraction_method="unit",
                ),
            )
            document = CandidateEvidenceDocument(
                document_id="ECLI:CZ:US:2025:1.US.3575.25.1",
                metadata={
                    "ecli": "ECLI:CZ:US:2025:1.US.3575.25.1",
                    "canonical_document_id": "ECLI:CZ:US:2025:1.US.3575.25.1",
                    "court_name": "Ústavní soud",
                    "source_document_id": "doc-16b9100a8b9122dd",
                },
                paragraphs=[paragraph],
                score=0.1,
            )
            return LegalV2RetrievalResult(
                documents=[document],
                dense_results=[],
                bm25_results=[],
                fused_results=[],
                diagnostics={
                    "dense_latency_ms": 1.0,
                    "bm25_latency_ms": 1.0,
                    "rrf_latency_ms": 0.1,
                    "total_retrieval_latency_ms": 2.0,
                    "dense_candidate_chunks": 1,
                    "bm25_candidate_chunks": 1,
                    "fused_candidate_chunks": 1,
                    "candidate_documents": 1,
                },
            )

    runtime = CaseSimilarityStage1Runtime(
        retriever=_FakeRetriever(),  # type: ignore[arg-type]
        config=MagicMock(
            qdrant_collection="nalus_legal_paragraph_chunks_v2_pilot_600",
            bm25_index_id="nalus_legal_paragraph_bm25_v2_pilot_600",
        ),
        ready=True,
    )
    monkeypatch.setattr(
        "app.rag.legal_v2.retrieve.case_similarity_search.get_case_similarity_stage1_runtime",
        lambda: runtime,
    )


def test_flag_off_short_query_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, Any] = {}
    _install_fake_runtime(monkeypatch, observed)
    monkeypatch.setenv("NALUS_LEGAL_V2_LONG_INPUT_ENABLED", "0")
    reset_query_input_service_for_tests()
    client = TestClient(app)
    query = "Hledám rozhodnutí o úpravě styku rodiče s nezletilým dítětem."
    resp = client.post(
        "/api/rag/legal-v2/case-similarity/search",
        json={"query": query, "limit": 5},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["diagnostics"]["input_processing"]["was_condensed"] is False
    assert observed["spec"].original_query == query


def test_flag_on_long_query_uses_brief(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, Any] = {}
    _install_fake_runtime(monkeypatch, observed)
    monkeypatch.setenv("NALUS_LEGAL_V2_LONG_INPUT_ENABLED", "1")
    reset_query_input_service_for_tests()
    client = TestClient(app)
    raw = """
Ústavní stížností se stěžovatel domáhá zrušení rozhodnutí.
Nehledám meritorní spor o péči, ale odmítnutí ústavní stížnosti pro formální vady.
Nebyl zastoupen advokátem a neodstranil vady podání.
Odůvodnění je nedostatečné a stížnost byla odmítnuta.
""".strip()
    raw = (raw + "\n\n") * 3
    resp = client.post(
        "/api/rag/legal-v2/case-similarity/search",
        json={"query": raw, "limit": 5},
    )
    assert resp.status_code == 200
    payload = resp.json()
    processing = payload["diagnostics"]["input_processing"]
    assert processing["was_condensed"] is True
    assert processing["classification"] == "long_legal_input"
    assert observed["spec"].original_query != raw
    assert len(observed["spec"].original_query) < len(raw)


def test_flag_on_oversized_returns_422(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, Any] = {}
    _install_fake_runtime(monkeypatch, observed)
    monkeypatch.setenv("NALUS_LEGAL_V2_LONG_INPUT_ENABLED", "1")
    monkeypatch.setenv("NALUS_LEGAL_V2_LONG_INPUT_HARD_LIMIT", "500")
    reset_query_input_service_for_tests()
    client = TestClient(app)
    resp = client.post(
        "/api/rag/legal-v2/case-similarity/search",
        json={"query": "a" * 501, "limit": 3},
    )
    assert resp.status_code == 422
