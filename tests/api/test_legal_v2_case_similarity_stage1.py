"""API tests for Legal v2 Stage 1 case-similarity search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.rag.legal_v2.retrieve.case_similarity_search import (
    CaseSimilarityStage1Runtime,
    Stage1DocumentResult,
    Stage1Passage,
    Stage1SearchResult,
    reset_case_similarity_stage1_runtime_for_tests,
)
from app.rag.retrieval.errors import RetrievalConfigurationError


@pytest.fixture(autouse=True)
def _reset_runtime(monkeypatch: pytest.MonkeyPatch):
    reset_case_similarity_stage1_runtime_for_tests()
    monkeypatch.setenv("NALUS_LEGAL_V2_CASE_SIMILARITY_ENABLED", "1")
    monkeypatch.delenv("NALUS_LEGAL_V2_DEBUG", raising=False)
    yield
    reset_case_similarity_stage1_runtime_for_tests()


def _sample_result(*, include_debug: bool = False) -> Stage1SearchResult:
    diagnostics: dict[str, Any] = {
        "query_length": 12,
        "generated_query_count": 3,
        "result_count": 1,
        "collection": "nalus_legal_paragraph_chunks_v2_pilot_600",
        "bm25_index_id": "nalus_legal_paragraph_bm25_v2_pilot_600",
        "total_latency_ms": 12.5,
        "retrieval_status": "ok",
    }
    if include_debug:
        diagnostics["debug"] = {
            "retrieval_queries": ["q"],
            "negative_constraints": [],
        }
    return Stage1SearchResult(
        query="ústavní stížnost formální vady",
        result_count=1,
        retrieval_stage="hybrid_rrf_stage_1",
        results=[
            Stage1DocumentResult(
                rank=1,
                document_id="ECLI:CZ:US:2025:1.US.3575.25.1",
                canonical_document_id="ECLI:CZ:US:2025:1.US.3575.25.1",
                ecli="ECLI:CZ:US:2025:1.US.3575.25.1",
                court="Ústavní soud",
                case_number="I. ÚS 3575/25",
                decision_date="2025-11-01",
                document_type="usnesení",
                score=0.09,
                relevant_passages=[
                    Stage1Passage(
                        text="Stěžovatel není zastoupen advokátem.",
                        chunk_id="chunk-1",
                        section="facts",
                    )
                ],
                source_document_id="doc-16b9100a8b9122dd",
            )
        ],
        diagnostics=diagnostics,
    )


def test_stage1_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NALUS_LEGAL_V2_CASE_SIMILARITY_ENABLED", "0")
    monkeypatch.setenv("NALUS_LEGAL_V2_SEARCH_ENABLED", "0")
    client = TestClient(app)
    resp = client.post(
        "/api/rag/legal-v2/case-similarity/search",
        json={"query": "ústavní stížnost"},
    )
    assert resp.status_code == 404


def test_stage1_empty_query_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(app)
    resp = client.post(
        "/api/rag/legal-v2/case-similarity/search",
        json={"query": "   "},
    )
    assert resp.status_code == 422


def test_stage1_limit_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NALUS_LEGAL_V2_MAX_RESULT_LIMIT", "10")

    def _boom(**kwargs):
        raise AssertionError("search should not run")

    monkeypatch.setattr(
        "app.rag.legal_v2.retrieve.case_similarity_search.search_case_similarity_stage1",
        _boom,
    )
    # Patch the import path used inside the endpoint.
    monkeypatch.setattr(
        "app.api.rag_router.search_case_similarity_stage1",
        _boom,
        raising=False,
    )
    client = TestClient(app)
    resp = client.post(
        "/api/rag/legal-v2/case-similarity/search",
        json={"query": "náklady řízení", "limit": 99},
    )
    assert resp.status_code == 422


def test_stage1_success_uses_authoritative_search(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, Any] = {}

    def _fake_search(**kwargs):
        called.update(kwargs)
        return _sample_result()

    monkeypatch.setattr(
        "app.rag.legal_v2.retrieve.case_similarity_search.search_case_similarity_stage1",
        _fake_search,
    )
    client = TestClient(app)
    resp = client.post(
        "/api/rag/legal-v2/case-similarity/search",
        json={"query": "Nehledám meritorní spor o péči, ale formální vady.", "limit": 5},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["retrieval_stage"] == "hybrid_rrf_stage_1"
    assert payload["result_count"] == 1
    doc = payload["results"][0]
    assert doc["document_id"] == doc["canonical_document_id"] == doc["ecli"]
    assert not str(doc["document_id"]).startswith("doc-")
    assert doc["source_document_id"].startswith("doc-")
    assert doc["relevant_passages"]
    assert called["query"].startswith("Nehledám")
    assert called["limit"] == 5
    assert called["include_debug"] is False
    assert "debug" not in payload["diagnostics"]


def test_stage1_missing_dependencies_return_503(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(**kwargs):
        raise RetrievalConfigurationError("BM25 sidecar missing")

    monkeypatch.setattr(
        "app.rag.legal_v2.retrieve.case_similarity_search.search_case_similarity_stage1",
        _fail,
    )
    client = TestClient(app)
    resp = client.post(
        "/api/rag/legal-v2/case-similarity/search",
        json={"query": "ústavní stížnost"},
    )
    assert resp.status_code == 503
    assert resp.json()["detail"]


def test_stage1_unexpected_error_is_not_empty_200(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(**kwargs):
        raise RuntimeError("dense failed")

    monkeypatch.setattr(
        "app.rag.legal_v2.retrieve.case_similarity_search.search_case_similarity_stage1",
        _fail,
    )
    client = TestClient(app)
    resp = client.post(
        "/api/rag/legal-v2/case-similarity/search",
        json={"query": "ústavní stížnost"},
    )
    assert resp.status_code == 503


def test_stage1_ready_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.rag.legal_v2.retrieve.case_similarity_search.probe_case_similarity_stage1_readiness",
        lambda: {
            "ready": True,
            "status": "ready",
            "collection": "nalus_legal_paragraph_chunks_v2_pilot_600",
            "bm25_index_id": "nalus_legal_paragraph_bm25_v2_pilot_600",
            "bm25_sidecar_exists": True,
            "retrieval_stage": "hybrid_rrf_stage_1",
        },
    )
    client = TestClient(app)
    resp = client.get("/api/rag/legal-v2/case-similarity/ready")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ready"] is True
    assert payload["enabled"] is True


def test_stage1_openapi_contains_models() -> None:
    client = TestClient(app)
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    assert "/api/rag/legal-v2/case-similarity/search" in paths
    assert "/api/rag/legal-v2/case-similarity/ready" in paths


def test_search_case_similarity_stage1_builds_queryspec(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.rag.legal_v2.query_spec import build_query_spec_v2
    from app.rag.legal_v2.retrieve import case_similarity_search as module

    observed: dict[str, Any] = {}

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
                    "total_retrieval_latency_ms": 2.5,
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
    result = module.search_case_similarity_stage1(
        query="Nehledám meritorní spor o péči, ale odmítnutí ústavní stížnosti pro vady. Bez advokáta.",
        limit=5,
        runtime=runtime,
        query_spec_builder=build_query_spec_v2,
    )
    assert result.results[0].ecli.startswith("ECLI:")
    assert "child_custody_merits" in {
        item["name"]
        for item in (
            observed["spec"].structured_query.get("negated_requested_concepts") or []
        )
    }
