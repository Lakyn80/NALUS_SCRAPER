"""
Integration tests for POST /api/rag/query endpoint.

Uses FastAPI TestClient with dependency_overrides so no real LLM or
retrieval services are called.

Run:
    pytest tests/api/test_rag_api.py -v
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.rag_router as rtr
from app.api.query_cache import CachedQueryResponse
from app.api.rag_router import (
    get_answer_service,
    get_full_document_store,
    get_orchestrator,
    get_pipeline,
    router,
)
from app.rag.legal_v2.pipeline import LegalV2VerifiedDocument
from app.rag.orchestrator.orchestrator_service import OrchestratorResult
from app.rag.retrieval.full_document import (
    FullDocumentChunk,
    FullDocumentDiagnostics,
    FullDocumentLookupError,
    FullDocumentResult,
)
from app.rag.retrieval.models import RetrievedChunk


# ---------------------------------------------------------------------------
# Test app factory
# ---------------------------------------------------------------------------


def _make_app(
    orchestrator_override=None,
    *,
    pipeline_override=None,
    answer_service_override=None,
    full_document_store_override=None,
    legal_v2_runtime_provider_override=None,
) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    if orchestrator_override is not None:
        app.dependency_overrides[get_orchestrator] = lambda: orchestrator_override
    if pipeline_override is not None:
        app.dependency_overrides[get_pipeline] = lambda: pipeline_override
    if answer_service_override is not None:
        app.dependency_overrides[get_answer_service] = lambda: answer_service_override
    if full_document_store_override is not None:
        app.dependency_overrides[get_full_document_store] = lambda: full_document_store_override
    if legal_v2_runtime_provider_override is not None:
        app.dependency_overrides[rtr.get_legal_v2_runtime_provider] = (
            lambda: legal_v2_runtime_provider_override
        )
    return app


# ---------------------------------------------------------------------------
# Fake orchestrators
# ---------------------------------------------------------------------------


class _FakeOrchestrator:
    """Returns a configurable fixed result."""

    def __init__(
        self,
        answer: str = "Syntetická odpověď",
        sources: list[str] | None = None,
        plan_steps: list[str] | None = None,
        retrieve_results: list[RetrievedChunk] | None = None,
    ) -> None:
        self._answer = answer
        self._sources = sources if sources is not None else ["1", "2"]
        self._plan_steps = plan_steps if plan_steps is not None else ["krok 1", "krok 2"]
        self._retrieve_results = retrieve_results if retrieve_results is not None else []
        self.calls: list[str] = []
        self.retrieve_calls: list[tuple[str, int]] = []

    def run(self, query: str) -> OrchestratorResult:
        self.calls.append(query)
        return OrchestratorResult(
            answer=self._answer,
            sources=self._sources,
            plan_steps=self._plan_steps,
        )

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        self.retrieve_calls.append((query, top_k))
        return self._retrieve_results[:top_k]


class _ExplodingOrchestrator:
    """Always raises — simulates unexpected orchestrator failure."""

    def run(self, query: str) -> OrchestratorResult:
        raise RuntimeError("orchestrator exploded")

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        del query, top_k
        raise RuntimeError("orchestrator exploded")


class _FakePipeline:
    def __init__(self, results: list[RetrievedChunk] | None = None) -> None:
        self._results = results if results is not None else []
        self.calls: list[tuple[str, int]] = []

    def run(self, query: str, top_k: int = 5):
        self.calls.append((query, top_k))
        return SimpleNamespace(results=self._results[:top_k])


class _FakeAnswerService:
    def __init__(
        self,
        *,
        summary: str = "souhrn",
        top_cases: list[str] | None = None,
        excerpts: list[str] | None = None,
    ) -> None:
        self._summary = summary
        self._top_cases = top_cases if top_cases is not None else ["III.ÚS 255/26"]
        self._excerpts = excerpts if excerpts is not None else ["relevantní excerpt"]
        self.calls: list[tuple[str, list[RetrievedChunk]]] = []

    def generate(self, query: str, chunks: list[RetrievedChunk]):
        self.calls.append((query, chunks))
        return SimpleNamespace(
            query=query,
            summary=self._summary,
            top_cases=self._top_cases,
            excerpts=self._excerpts,
        )


class _FakeFullDocumentStore:
    def __init__(
        self,
        *,
        result: FullDocumentResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.calls: list[str] = []

    def get(self, document_id: str) -> FullDocumentResult | None:
        self.calls.append(document_id)
        if self._error is not None:
            raise self._error
        return self._result


class _MemoryCache:
    def __init__(self) -> None:
        self.store: dict[str, CachedQueryResponse] = {}
        self.reads = 0
        self.writes = 0

    def get(self, key: str) -> CachedQueryResponse | None:
        self.reads += 1
        return self.store.get(key)

    def set(
        self,
        key: str,
        value: CachedQueryResponse,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        del ttl_seconds
        self.writes += 1
        self.store[key] = value

    def close(self) -> None:
        return None


def _chunk(
    chunk_id: str = "III.ÚS_255_26_0",
    *,
    score: float = 0.91,
    source: str = "dense",
    text: str = "Rozhodnutí Ústavního soudu.",
    metadata: dict | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        text=text,
        score=score,
        source=source,
        metadata=metadata or {},
    )


def _full_document_result(document_id: str = "ECLI:CZ:US:2026:3.US.446.26.1") -> FullDocumentResult:
    chunks = [
        FullDocumentChunk(
            chunk_id="chunk-0",
            chunk_index=0,
            text="První část rozsudku.",
            metadata={"document_id": document_id, "chunk_index": 0},
        ),
        FullDocumentChunk(
            chunk_id="chunk-1",
            chunk_index=1,
            text="Druhá část rozsudku.",
            metadata={"document_id": document_id, "chunk_index": 1},
        ),
    ]
    return FullDocumentResult(
        document_id=document_id,
        metadata={
            "document_id": document_id,
            "ecli": document_id,
            "case_reference": "III. ÚS 446/26",
            "court_name": "Ústavní soud",
            "decision_date": "2026-04-01",
        },
        full_text="\n\n".join(chunk.text for chunk in chunks),
        chunks=chunks,
        source_url=None,
        provenance_status="overeno",
        full_text_availability_status="available",
        diagnostics=FullDocumentDiagnostics(
            collection_name="test-collection",
            chunk_count=2,
            missing_chunk_indexes=[],
            duplicate_chunk_indexes=[],
            all_chunks_have_index=True,
            reconstruction_method="qdrant_payload_chunk_index",
            max_chunks=2000,
        ),
    )


def _constraint_full_document(document_id: str, text: str) -> FullDocumentResult:
    chunks = [
        FullDocumentChunk(
            chunk_id=f"{document_id}-chunk-0",
            chunk_index=0,
            text=text,
            metadata={"document_id": document_id, "chunk_index": 0},
        )
    ]
    return FullDocumentResult(
        document_id=document_id,
        metadata={
            "document_id": document_id,
            "court_name": "Ústavní soud",
        },
        full_text=text,
        chunks=chunks,
        source_url=None,
        provenance_status="overeno",
        full_text_availability_status="available",
        diagnostics=FullDocumentDiagnostics(
            collection_name="test-collection",
            chunk_count=1,
            missing_chunk_indexes=[],
            duplicate_chunk_indexes=[],
            all_chunks_have_index=True,
            reconstruction_method="test",
            max_chunks=2000,
        ),
    )


def _legal_v2_config(returned_verified_documents: int = 3) -> rtr.LegalV2RetrieverConfig:
    return rtr.LegalV2RetrieverConfig(
        qdrant_collection="nalus_legal_paragraph_chunks_v2_test",
        bm25_sidecar_path="storage/rag/bm25/nalus_legal_paragraph_bm25_v2_test.sqlite",
        bm25_index_id="nalus_legal_paragraph_bm25_v2_test",
        model_path="/models/BAAI/bge-m3",
        dense_candidate_chunks=5,
        bm25_candidate_chunks=6,
        fused_candidate_chunks=7,
        candidate_documents=4,
        returned_verified_documents=returned_verified_documents,
        evidence_windows_per_constraint=2,
    )


def _legal_v2_document(
    document_id: str = "ECLI:CZ:US:2026:3.US.446.26.1",
    *,
    status: str = "verified_match",
    relevance_classification: str = "materially_relevant",
) -> LegalV2VerifiedDocument:
    return LegalV2VerifiedDocument(
        document_id=document_id,
        score=0.91,
        status=status,
        relevance_classification=relevance_classification,
        metadata={
            "document_id": document_id,
            "ecli": document_id,
            "court_name": "Ústavní soud",
            "source": "nalus",
            "case_reference": "III. ÚS 446/26",
            "paragraph_texts": {"p1": "this must not leak"},
        },
        evidence=[
            {
                "constraint_id": "constraint_child_abduction",
                "paragraph_ids": [f"{document_id}__p1"],
                "section_types": ["reasoning"],
                "quote": "Soud zjistil relevantní skutkový stav.",
                "source_of_claim": "court_finding",
            }
        ],
        constraint_results=[
            {
                "constraint_id": "constraint_child_abduction",
                "status": "proven",
                "reason": "bounded verifier reason",
                "evidence_paragraph_ids": [f"{document_id}__p1"],
            }
        ],
        dense_rank=1,
        bm25_rank=2,
        rrf_score=0.42,
        verification_reason="ověřeno z odstavce soudu",
        verifier_diagnostics={"constraint_result_count": 1, "raw_provider_response": "must not leak"},
    )


def _legal_v2_result(
    *,
    status: str = "verified_match",
    interpretation_status: str = "ok",
    verified_documents: list[LegalV2VerifiedDocument] | None = None,
    rejected_documents: list[LegalV2VerifiedDocument] | None = None,
    diagnostics: dict | None = None,
    provider: dict | None = None,
) -> rtr.RuntimeLegalV2SearchResult:
    return rtr.RuntimeLegalV2SearchResult(
        status=status,
        interpretation_status=interpretation_status,
        query_spec_summary={
            "intent": "legal_research",
            "requires_verification": True,
            "retrieval_queries": ["mezinárodní únos dítěte"],
        },
        verified_documents=verified_documents if verified_documents is not None else [_legal_v2_document()],
        rejected_documents=rejected_documents if rejected_documents is not None else [],
        rejection_counts={document.status: 1 for document in rejected_documents or []},
        latency_ms_by_stage={"total": 12.5},
        provider=provider or {"query_interpreter": "fake", "verifier": "fake"},
        index={
            "collection": "nalus_legal_paragraph_chunks_v2_test",
            "bm25_index_id": "nalus_legal_paragraph_bm25_v2_test",
            "bm25_sidecar_path": "C:/secret/local/path.sqlite",
        },
        diagnostics=diagnostics or {"candidate_documents": 1},
    )


class _FakeLegalV2RuntimeProvider:
    def __init__(
        self,
        *,
        result: rtr.RuntimeLegalV2SearchResult | None = None,
        error: Exception | None = None,
        config: rtr.LegalV2RetrieverConfig | None = None,
    ) -> None:
        self.result = result if result is not None else _legal_v2_result()
        self.error = error
        self.config = config if config is not None else _legal_v2_config()
        self.runtime_calls = 0
        self.search_calls: list[dict] = []

    def __call__(self) -> rtr.LegalV2Runtime:
        self.runtime_calls += 1
        if self.error is not None:
            raise self.error
        return rtr.LegalV2Runtime(
            retriever=object(),
            query_provider=object(),
            verifier=object(),
            config=self.config,
            search=self._search,
        )

    def _search(self, **kwargs):
        self.search_calls.append(kwargs)
        return self.result


def _search_v2_route_count(app: FastAPI) -> int:
    count = 0
    for route in app.routes:
        nested_router = getattr(route, "original_router", None)
        candidates = nested_router.routes if nested_router is not None else [route]
        count += sum(
            1
            for candidate in candidates
            if getattr(candidate, "path", None) == "/api/rag/search-v2"
            and "POST" in getattr(candidate, "methods", set())
        )
    return count


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    fake = _FakeOrchestrator()
    return TestClient(_make_app(fake))


@pytest.fixture()
def fake_orchestrator() -> _FakeOrchestrator:
    return _FakeOrchestrator()


@pytest.fixture()
def client_with_fake(fake_orchestrator: _FakeOrchestrator) -> TestClient:
    return TestClient(_make_app(fake_orchestrator))


@pytest.fixture(autouse=True)
def _reset_router_cache_state() -> None:
    original_cache = rtr._query_cache
    original_backend = rtr._query_cache_backend
    original_error = rtr._query_cache_error
    original_corpus_version = rtr._corpus_version
    rtr._query_cache = None
    rtr._query_cache_backend = "none"
    rtr._query_cache_error = None
    rtr._corpus_version = "test-corpus"
    rtr.reset_legal_v2_runtime_for_tests()
    yield
    rtr.reset_legal_v2_runtime_for_tests()
    rtr._query_cache = original_cache
    rtr._query_cache_backend = original_backend
    rtr._query_cache_error = original_error
    rtr._corpus_version = original_corpus_version


# ---------------------------------------------------------------------------
# Search endpoint compatibility
# ---------------------------------------------------------------------------


class TestSearchEndpointCompatibility:
    def test_search_returns_existing_shape(self) -> None:
        pipeline = _FakePipeline(results=[_chunk()])
        answer_service = _FakeAnswerService(
            summary="stručný souhrn",
            top_cases=["III.ÚS 255/26"],
            excerpts=["relevantní excerpt"],
        )
        client = TestClient(
            _make_app(
                _FakeOrchestrator(),
                pipeline_override=pipeline,
                answer_service_override=answer_service,
            )
        )

        resp = client.post("/api/rag/search", json={"query": "dotaz", "top_k": 3})

        assert resp.status_code == 200
        assert resp.json() == {
            "query": "dotaz",
            "summary": "stručný souhrn",
            "top_cases": ["III.ÚS 255/26"],
            "excerpts": ["relevantní excerpt"],
        }
        assert pipeline.calls == [("dotaz", 3)]
        assert len(answer_service.calls) == 1


# ---------------------------------------------------------------------------
# Raw retrieval endpoint
# ---------------------------------------------------------------------------


class TestRawRetrieveEndpoint:
    def test_returns_raw_chunks_with_metadata(self) -> None:
        fake = _FakeOrchestrator(
            retrieve_results=[
                _chunk(
                    metadata={
                        "source": "nalus",
                        "case_reference": "III.ÚS 255/26",
                        "decision_date": "2026-01-15",
                        "court_name": "Ústavní soud",
                        "document_id": 136186,
                        "chunk_index": 3,
                    }
                )
            ]
        )
        client = TestClient(_make_app(fake))

        resp = client.post("/api/rag/retrieve", json={"query": "únos dítěte", "top_k": 5})

        assert resp.status_code == 200
        assert resp.json() == {
            "results": [
                {
                    "chunk_id": "III.ÚS_255_26_0",
                    "text": "Rozhodnutí Ústavního soudu.",
                    "score": 0.91,
                    "source": "nalus",
                    "reference": "III.ÚS 255/26",
                    "case_reference": "III.ÚS 255/26",
                    "court_name": "Ústavní soud",
                    "date": "2026-01-15",
                    "document_id": 136186,
                    "chunk_index": 3,
                    "metadata": {
                        "source": "nalus",
                        "case_reference": "III.ÚS 255/26",
                        "decision_date": "2026-01-15",
                        "court_name": "Ústavní soud",
                        "document_id": 136186,
                        "chunk_index": 3,
                    },
                }
            ]
        }
        assert fake.retrieve_calls == [("únos dítěte", 5)]

    def test_constitutional_filter_matches_nalus_source(self) -> None:
        fake = _FakeOrchestrator(
            retrieve_results=[
                _chunk(
                    chunk_id="constitutional-hit",
                    metadata={"source": "nalus", "case_reference": "III.ÚS 255/26"},
                ),
                _chunk(
                    chunk_id="supreme-hit",
                    metadata={"source": "supreme", "case_reference": "30 Cdo 1/2026"},
                ),
            ]
        )
        client = TestClient(_make_app(fake))

        resp = client.post(
            "/api/rag/retrieve",
            json={"query": "dotaz", "top_k": 10, "sources": ["constitutional"]},
        )

        assert resp.status_code == 200
        assert [item["chunk_id"] for item in resp.json()["results"]] == ["constitutional-hit"]

    def test_constitutional_filter_matches_usoud_source_and_ecli_identity(self) -> None:
        fake = _FakeOrchestrator(
            retrieve_results=[
                _chunk(
                    chunk_id="usoud-source-hit",
                    metadata={
                        "source": "usoud / nalus",
                        "court": "Ústavní soud",
                        "document_id": "ECLI:CZ:US:2023:4.US.652.22.2",
                    },
                ),
                _chunk(
                    chunk_id="ecli-only-hit",
                    metadata={"document_id": "ECLI:CZ:US:2023:3.US.3469.22.1"},
                ),
                _chunk(
                    chunk_id="supreme-hit",
                    metadata={"source": "supreme", "document_id": "ECLI:CZ:NS:2024:1.TDO.1.1"},
                ),
            ]
        )
        client = TestClient(_make_app(fake))

        resp = client.post(
            "/api/rag/retrieve",
            json={"query": "dotaz", "top_k": 10, "sources": ["constitutional"]},
        )

        assert resp.status_code == 200
        payload = resp.json()
        assert [item["chunk_id"] for item in payload["results"]] == [
            "usoud-source-hit",
            "ecli-only-hit",
        ]
        assert payload["results"][0]["court_name"] == "Ústavní soud"
        assert payload["results"][1]["court_name"] == "Ústavní soud"

    def test_supreme_filter_matches_nsoud_source_and_ecli_identity(self) -> None:
        fake = _FakeOrchestrator(
            retrieve_results=[
                _chunk(
                    chunk_id="constitutional-hit",
                    metadata={"document_id": "ECLI:CZ:US:2023:3.US.3469.22.1"},
                ),
                _chunk(
                    chunk_id="nsoud-source-hit",
                    metadata={"source": "nsoud", "document_id": "ECLI:CZ:NS:2024:1.TDO.1.1"},
                ),
                _chunk(
                    chunk_id="nsoud-ecli-hit",
                    metadata={"document_id": "ECLI:CZ:NS:2025:5.TDO.1086.2024.1"},
                ),
            ]
        )
        client = TestClient(_make_app(fake))

        resp = client.post(
            "/api/rag/retrieve",
            json={"query": "dotaz", "top_k": 10, "sources": ["supreme"]},
        )

        assert resp.status_code == 200
        payload = resp.json()
        assert [item["chunk_id"] for item in payload["results"]] == [
            "nsoud-source-hit",
            "nsoud-ecli-hit",
        ]
        assert payload["results"][0]["court_name"] == "Nejvyšší soud"
        assert payload["results"][1]["court_name"] == "Nejvyšší soud"

    def test_retrieve_failure_returns_empty_results(self) -> None:
        client = TestClient(_make_app(_ExplodingOrchestrator()))

        resp = client.post("/api/rag/retrieve", json={"query": "dotaz"})

        assert resp.status_code == 200
        assert resp.json() == {"results": []}

    def test_document_retrieve_disabled_by_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("NALUS_DOCUMENT_RETRIEVAL_ENABLED", raising=False)
        client = TestClient(_make_app(_FakeOrchestrator()))

        resp = client.post("/api/rag/retrieve-documents", json={"query": "dotaz"})

        assert resp.status_code == 404
        assert "Document-level retrieval is disabled" in resp.json()["detail"]

    def test_document_retrieve_returns_unique_documents_with_diagnostics(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("NALUS_DOCUMENT_RETRIEVAL_ENABLED", "1")
        monkeypatch.setenv("NALUS_DOCUMENT_MAX_CANDIDATE_CHUNKS", "4")
        monkeypatch.setenv("NALUS_DOCUMENT_MAX_RETURNED_DOCUMENTS", "10")
        monkeypatch.setenv("NALUS_DOCUMENT_MAX_SUPPORTING_CHUNKS_PER_DOCUMENT", "2")
        monkeypatch.setenv("NALUS_DOCUMENT_RELEVANCE_THRESHOLD", "0.0")
        fake = _FakeOrchestrator(
            retrieve_results=[
                _chunk(
                    chunk_id="doc-a-1",
                    score=0.9,
                    text="A first passage",
                    metadata={
                        "source": "nalus",
                        "document_id": "DOC-A",
                        "case_reference": "III.ÚS 1/26",
                        "chunk_index": 1,
                    },
                ),
                _chunk(
                    chunk_id="doc-a-2",
                    score=0.7,
                    text="A second passage",
                    metadata={"source": "nalus", "document_id": "DOC-A", "chunk_index": 2},
                ),
                _chunk(
                    chunk_id="doc-b-1",
                    score=0.8,
                    text="B passage",
                    metadata={"source": "nalus", "document_id": "DOC-B", "chunk_index": 1},
                ),
            ]
        )
        client = TestClient(_make_app(fake))

        resp = client.post("/api/rag/retrieve-documents", json={"query": "dotaz"})

        assert resp.status_code == 200
        payload = resp.json()
        assert [item["document_id"] for item in payload["documents"]] == ["DOC-A", "DOC-B"]
        assert payload["documents"][0]["best_passages"][0]["chunk_id"] == "doc-a-1"
        assert payload["diagnostics"]["candidate_chunks_retrieved"] == 3
        assert payload["diagnostics"]["unique_documents_produced"] == 2
        assert payload["diagnostics"]["duplicate_document_hits_removed"] == 1
        assert fake.retrieve_calls == [("dotaz", 4)]

    def test_document_retrieve_threshold_empty_result_has_no_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("NALUS_DOCUMENT_RETRIEVAL_ENABLED", "1")
        monkeypatch.setenv("NALUS_DOCUMENT_RELEVANCE_THRESHOLD", "0.95")
        fake = _FakeOrchestrator(
            retrieve_results=[
                _chunk(score=0.6, metadata={"source": "nalus", "document_id": "DOC-A"})
            ]
        )
        client = TestClient(_make_app(fake))

        resp = client.post("/api/rag/retrieve-documents", json={"query": "dotaz"})

        assert resp.status_code == 200
        assert resp.json()["documents"] == []
        assert resp.json()["diagnostics"]["documents_filtered"] == 1

    def test_existing_retrieve_response_shape_remains_backward_compatible(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("NALUS_DOCUMENT_RETRIEVAL_ENABLED", "1")
        fake = _FakeOrchestrator(
            retrieve_results=[
                _chunk(metadata={"source": "nalus", "document_id": "DOC-A"})
            ]
        )
        client = TestClient(_make_app(fake))

        resp = client.post("/api/rag/retrieve", json={"query": "dotaz", "top_k": 1})

        assert resp.status_code == 200
        assert set(resp.json().keys()) == {"results"}

    def test_verified_retrieve_disabled_by_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("NALUS_CONSTRAINT_RETRIEVAL_ENABLED", raising=False)
        client = TestClient(_make_app(_FakeOrchestrator()))

        resp = client.post("/api/rag/retrieve-verified", json={"query": "dotaz"})

        assert resp.status_code == 404
        assert "Constraint-aware document verification is disabled" in resp.json()["detail"]

    def test_verified_retrieve_returns_only_verified_documents(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("NALUS_CONSTRAINT_RETRIEVAL_ENABLED", "1")
        monkeypatch.setenv("NALUS_CONSTRAINT_MAX_CANDIDATE_CHUNKS", "10")
        fake = _FakeOrchestrator(
            retrieve_results=[
                _chunk(
                    chunk_id="doc-ok-0",
                    score=0.9,
                    metadata={"document_id": "DOC-OK", "court_name": "Ústavní soud"},
                ),
                _chunk(
                    chunk_id="doc-bad-0",
                    score=0.95,
                    metadata={"document_id": "DOC-BAD", "court_name": "Ústavní soud"},
                ),
            ]
        )
        store = _FakeFullDocumentStore()

        def get(document_id: str) -> FullDocumentResult | None:
            store.calls.append(document_id)
            if document_id == "DOC-OK":
                return _constraint_full_document(
                    document_id,
                    "Stěžovatel je státní občan Ruské federace a podal žádost o udělení státního občanství České republiky.",
                )
            return _constraint_full_document(
                document_id,
                "Stěžovatel je občan Ukrajiny a žádal o udělení státního občanství České republiky.",
            )

        store.get = get  # type: ignore[method-assign]
        client = TestClient(_make_app(fake, full_document_store_override=store))

        resp = client.post(
            "/api/rag/retrieve-verified",
            json={"query": "udělení českého občanství ruskému občanu", "debug": True},
        )

        assert resp.status_code == 200
        payload = resp.json()
        assert [item["document_id"] for item in payload["documents"]] == ["DOC-OK"]
        assert payload["diagnostics"]["verified_document_count"] == 1
        assert payload["diagnostics"]["excluded_hard_mismatch_count"] == 1
        assert payload["rejected_documents"][0]["document_id"] == "DOC-BAD"
        assert fake.retrieve_calls == [("udělení českého občanství ruskému občanu", 10)]

    def test_verified_retrieve_empty_result_has_no_unrelated_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("NALUS_CONSTRAINT_RETRIEVAL_ENABLED", "1")
        fake = _FakeOrchestrator(
            retrieve_results=[
                _chunk(metadata={"document_id": "DOC-IRRELEVANT", "court_name": "Ústavní soud"})
            ]
        )
        store = _FakeFullDocumentStore(
            result=_constraint_full_document(
                "DOC-IRRELEVANT",
                "Rozhodnutí o místním referendu a územním plánování.",
            )
        )
        client = TestClient(_make_app(fake, full_document_store_override=store))

        resp = client.post(
            "/api/rag/retrieve-verified",
            json={"query": "udělení českého občanství ruskému občanu"},
        )

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["documents"] == []
        assert payload["rejected_documents"] == []
        assert payload["diagnostics"]["final_document_count"] == 0
        assert payload["diagnostics"]["excluded_not_proven_count"] == 1

    def test_verified_retrieve_provider_failure_returns_503(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("NALUS_CONSTRAINT_RETRIEVAL_ENABLED", "1")
        client = TestClient(_make_app(_ExplodingOrchestrator()))

        resp = client.post(
            "/api/rag/retrieve-verified",
            json={"query": "udělení českého občanství ruskému občanu"},
        )

        assert resp.status_code == 503
        assert resp.json()["detail"] == "Constraint-aware retrieval is temporarily unavailable."


class TestLegalV2SearchEndpoint:
    def test_registered_once_in_main_app(self) -> None:
        from app.api_app import app as main_app

        assert _search_v2_route_count(main_app) == 1

    def test_disabled_returns_controlled_response_without_runtime(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("NALUS_LEGAL_V2_SEARCH_ENABLED", raising=False)
        provider = _FakeLegalV2RuntimeProvider()
        client = TestClient(
            _make_app(
                _FakeOrchestrator(),
                legal_v2_runtime_provider_override=provider,
            )
        )

        resp = client.post("/api/rag/search-v2", json={"query": "únos dítěte"})

        assert resp.status_code == 404
        assert "Legal Retrieval v2 search is disabled" in resp.json()["detail"]
        assert provider.runtime_calls == 0

    def test_enabled_success_uses_fake_runtime_and_preserves_contract(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("NALUS_LEGAL_V2_SEARCH_ENABLED", "1")
        provider = _FakeLegalV2RuntimeProvider()
        client = TestClient(
            _make_app(
                _FakeOrchestrator(),
                legal_v2_runtime_provider_override=provider,
            )
        )

        resp = client.post(
            "/api/rag/search-v2",
            json={
                "query": "mezinárodní únos dítěte matkou do Ruska",
                "sources": ["constitutional"],
                "max_results": 2,
                "debug": True,
            },
        )

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "verified_match"
        assert payload["interpretation_status"] == "ok"
        assert payload["verified_documents"][0]["document_id"] == "ECLI:CZ:US:2026:3.US.446.26.1"
        assert payload["verified_documents"][0]["metadata"]["court_name"] == "Ústavní soud"
        assert payload["verified_documents"][0]["relevance_classification"] == "materially_relevant"
        assert "paragraph_texts" not in payload["verified_documents"][0]["metadata"]
        assert payload["verified_documents"][0]["verification_reason"] == "ověřeno z odstavce soudu"
        assert payload["verified_documents"][0]["verifier_diagnostics"]["raw_provider_response"] == "[redacted]"
        assert payload["verified_documents"][0]["evidence"][0]["paragraph_ids"] == [
            "ECLI:CZ:US:2026:3.US.446.26.1__p1"
        ]
        assert payload["index"]["bm25_sidecar_path"] == "[redacted]"
        assert provider.runtime_calls == 1
        assert provider.search_calls[0]["source_filter"] == {"constitutional"}

    def test_enabled_zero_results_do_not_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("NALUS_LEGAL_V2_SEARCH_ENABLED", "1")
        provider = _FakeLegalV2RuntimeProvider(
            result=_legal_v2_result(
                status="no_verified_results",
                verified_documents=[],
                diagnostics={"candidate_documents": 0, "zero_result": True},
            )
        )
        client = TestClient(
            _make_app(
                _FakeOrchestrator(retrieve_results=[_chunk(chunk_id="legacy-hit")]),
                legal_v2_runtime_provider_override=provider,
            )
        )

        resp = client.post("/api/rag/search-v2", json={"query": "nenalezitelný právní dotaz"})

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "no_verified_results"
        assert payload["verified_documents"] == []
        assert payload["diagnostics"]["zero_result"] is True
        assert provider.search_calls

    def test_missing_runtime_configuration_returns_503_without_secret_leak(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv("NALUS_LEGAL_V2_SEARCH_ENABLED", "1")
        secret = "sk-test-secret-value"
        raw_query = "RAW_QUERY_SHOULD_NOT_APPEAR"
        provider = _FakeLegalV2RuntimeProvider(
            error=rtr.RetrievalConfigurationError(
                f"missing config {secret} {raw_query}"
            )
        )
        client = TestClient(
            _make_app(
                _FakeOrchestrator(),
                legal_v2_runtime_provider_override=provider,
            )
        )

        with caplog.at_level(logging.WARNING, logger="app.api.rag_router"):
            resp = client.post("/api/rag/search-v2", json={"query": raw_query})

        body = str(resp.json())
        logs = "\n".join(record.getMessage() for record in caplog.records)
        assert resp.status_code == 503
        assert resp.json()["detail"] == "Legal Retrieval v2 search is temporarily unavailable."
        assert secret not in body
        assert secret not in logs
        assert raw_query not in body
        assert raw_query not in logs

    @pytest.mark.parametrize(
        ("error_text", "expected_status"),
        [
            ("missing qdrant collection nalus_legal_paragraph_chunks_v2_test", 503),
            ("missing bm25 sidecar nalus_legal_paragraph_bm25_v2_test", 503),
        ],
    )
    def test_missing_isolated_index_dependencies_return_503(
        self,
        monkeypatch: pytest.MonkeyPatch,
        error_text: str,
        expected_status: int,
    ) -> None:
        monkeypatch.setenv("NALUS_LEGAL_V2_SEARCH_ENABLED", "1")
        provider = _FakeLegalV2RuntimeProvider(error=rtr.RetrievalConfigurationError(error_text))
        client = TestClient(
            _make_app(
                _FakeOrchestrator(),
                legal_v2_runtime_provider_override=provider,
            )
        )

        resp = client.post("/api/rag/search-v2", json={"query": "únos dítěte"})

        assert resp.status_code == expected_status
        assert "temporarily unavailable" in resp.json()["detail"]

    def test_queryspec_provider_error_uses_existing_fail_closed_result_and_redacts_raw_payload(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv("NALUS_LEGAL_V2_SEARCH_ENABLED", "1")
        raw_query = "DISTINCTIVE_RAW_QUERY"
        provider = _FakeLegalV2RuntimeProvider(
            result=_legal_v2_result(
                status="query_interpretation_error",
                interpretation_status="failed",
                verified_documents=[],
                provider={
                    "query_interpreter": "fake",
                    "reason": "query_interpreter_provider_error",
                    "error": "RAW_PROVIDER_BODY sk-test-secret",
                },
                diagnostics={"raw_provider_response": "RAW_PROVIDER_BODY sk-test-secret"},
            )
        )
        client = TestClient(
            _make_app(
                _FakeOrchestrator(),
                legal_v2_runtime_provider_override=provider,
            )
        )

        with caplog.at_level(logging.WARNING, logger="app.api.rag_router"):
            resp = client.post("/api/rag/search-v2", json={"query": raw_query})

        body = str(resp.json())
        logs = "\n".join(record.getMessage() for record in caplog.records)
        assert resp.status_code == 200
        assert resp.json()["status"] == "query_interpretation_error"
        assert "RAW_PROVIDER_BODY" not in body
        assert "sk-test-secret" not in body
        assert raw_query not in logs

    def test_verifier_failure_returns_no_verified_candidate(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("NALUS_LEGAL_V2_SEARCH_ENABLED", "1")
        rejected = _legal_v2_document(document_id="DOC-REJECTED", status="verifier_error")
        provider = _FakeLegalV2RuntimeProvider(
            result=_legal_v2_result(
                status="no_verified_results",
                verified_documents=[],
                rejected_documents=[rejected],
            )
        )
        client = TestClient(
            _make_app(
                _FakeOrchestrator(),
                legal_v2_runtime_provider_override=provider,
            )
        )

        resp = client.post("/api/rag/search-v2", json={"query": "únos dítěte", "debug": True})

        assert resp.status_code == 200
        assert resp.json()["verified_documents"] == []
        assert resp.json()["rejected_documents"][0]["document_id"] == "DOC-REJECTED"

    @pytest.mark.parametrize("error", [TimeoutError("RAW_PROVIDER_BODY"), ValueError("invalid RAW_PROVIDER_BODY")])
    def test_provider_timeout_or_invalid_output_returns_503_without_raw_body(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        error: Exception,
    ) -> None:
        monkeypatch.setenv("NALUS_LEGAL_V2_SEARCH_ENABLED", "1")
        provider = _FakeLegalV2RuntimeProvider()

        def failing_search(**kwargs):
            provider.search_calls.append(kwargs)
            raise error

        provider.result = _legal_v2_result()
        client = TestClient(
            _make_app(
                _FakeOrchestrator(),
                legal_v2_runtime_provider_override=lambda: rtr.LegalV2Runtime(
                    retriever=object(),
                    query_provider=object(),
                    verifier=object(),
                    config=_legal_v2_config(),
                    search=failing_search,
                ),
            )
        )

        with caplog.at_level(logging.WARNING, logger="app.api.rag_router"):
            resp = client.post("/api/rag/search-v2", json={"query": "citlivý dotaz"})

        combined = str(resp.json()) + "\n".join(record.getMessage() for record in caplog.records)
        assert resp.status_code == 503
        assert "RAW_PROVIDER_BODY" not in combined
        assert "citlivý dotaz" not in combined

    @pytest.mark.parametrize(
        "payload",
        [
            {"query": ""},
            {"query": "   \t"},
            {"query": "a" * (rtr.LEGAL_V2_MAX_QUERY_LENGTH + 1)},
            {"query": "dotaz", "max_results": 0},
            {"query": "dotaz", "max_results": rtr.LEGAL_V2_MAX_REQUESTED_RESULTS + 1},
        ],
    )
    def test_request_validation_rejects_invalid_payloads(
        self,
        monkeypatch: pytest.MonkeyPatch,
        payload: dict,
    ) -> None:
        monkeypatch.setenv("NALUS_LEGAL_V2_SEARCH_ENABLED", "1")
        provider = _FakeLegalV2RuntimeProvider()
        client = TestClient(
            _make_app(
                _FakeOrchestrator(),
                legal_v2_runtime_provider_override=provider,
            )
        )

        resp = client.post("/api/rag/search-v2", json=payload)

        assert resp.status_code == 422
        assert provider.runtime_calls == 0

    def test_czech_unicode_and_extra_fields_follow_pydantic_ignore_policy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("NALUS_LEGAL_V2_SEARCH_ENABLED", "1")
        provider = _FakeLegalV2RuntimeProvider()
        client = TestClient(
            _make_app(
                _FakeOrchestrator(),
                legal_v2_runtime_provider_override=provider,
            )
        )
        query = "Ústavní soud řešil únos dítěte podle Haagské úmluvy"

        resp = client.post(
            "/api/rag/search-v2",
            json={"query": query, "unsupported": "ignored"},
        )

        assert resp.status_code == 200
        assert provider.search_calls[0]["query"] == query

    def test_requested_final_limit_is_bounded_by_runtime_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("NALUS_LEGAL_V2_SEARCH_ENABLED", "1")
        provider = _FakeLegalV2RuntimeProvider(config=_legal_v2_config(returned_verified_documents=3))
        client = TestClient(
            _make_app(
                _FakeOrchestrator(),
                legal_v2_runtime_provider_override=provider,
            )
        )

        resp = client.post("/api/rag/search-v2", json={"query": "únos dítěte", "max_results": 20})

        assert resp.status_code == 200
        bounded_config = provider.search_calls[0]["config"]
        assert bounded_config.returned_verified_documents == 3
        assert bounded_config.dense_candidate_chunks == 5
        assert bounded_config.bm25_candidate_chunks == 6
        assert bounded_config.fused_candidate_chunks == 7
        assert bounded_config.candidate_documents == 4

    def test_runtime_cache_reuse_and_reset_without_real_services(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = {"qdrant": 0, "build": 0}

        class FakeQdrantClient:
            def __init__(self, **kwargs) -> None:
                del kwargs
                calls["qdrant"] += 1

        monkeypatch.setenv("LLM_API_KEY", "valid-test-key")
        monkeypatch.setattr(rtr, "import_module", lambda name: SimpleNamespace(QdrantClient=FakeQdrantClient))
        monkeypatch.setattr(rtr, "BgeM3Embedder", lambda config: object())
        monkeypatch.setattr(rtr, "DeepSeekQuerySpecProvider", lambda *args, **kwargs: object())
        monkeypatch.setattr(rtr, "DeepSeekSemanticVerifierProvider", lambda *args, **kwargs: object())

        def fake_build_live_legal_v2_retriever(client, embedder, config):
            del client, embedder, config
            calls["build"] += 1
            return object()

        monkeypatch.setattr(rtr, "build_live_legal_v2_retriever", fake_build_live_legal_v2_retriever)

        first = rtr.get_legal_v2_runtime()
        second = rtr.get_legal_v2_runtime()
        rtr.reset_legal_v2_runtime_for_tests()
        third = rtr.get_legal_v2_runtime()

        assert first is second
        assert third is not first
        assert calls == {"qdrant": 2, "build": 2}

    def test_evidence_provenance_belongs_to_returned_document(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("NALUS_LEGAL_V2_SEARCH_ENABLED", "1")
        document_id = "ECLI:CZ:US:2026:3.US.446.26.1"
        provider = _FakeLegalV2RuntimeProvider(
            result=_legal_v2_result(verified_documents=[_legal_v2_document(document_id=document_id)])
        )
        client = TestClient(
            _make_app(
                _FakeOrchestrator(),
                legal_v2_runtime_provider_override=provider,
            )
        )

        resp = client.post("/api/rag/search-v2", json={"query": "únos dítěte"})

        document = resp.json()["verified_documents"][0]
        assert all(
            paragraph_id.startswith(document["document_id"])
            for evidence in document["evidence"]
            for paragraph_id in evidence["paragraph_ids"]
        )


class TestFullDocumentEndpoint:
    def test_returns_reconstructed_full_document(self) -> None:
        result = _full_document_result()
        store = _FakeFullDocumentStore(result=result)
        client = TestClient(
            _make_app(_FakeOrchestrator(), full_document_store_override=store)
        )

        resp = client.get(f"/api/rag/documents/{result.document_id}")

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["document_id"] == result.document_id
        assert payload["metadata"]["ecli"] == result.document_id
        assert payload["full_text"] == "První část rozsudku.\n\nDruhá část rozsudku."
        assert [chunk["chunk_index"] for chunk in payload["chunks"]] == [0, 1]
        assert payload["full_text_availability_status"] == "available"
        assert payload["diagnostics"]["chunk_count"] == 2
        assert store.calls == [result.document_id]

    def test_invalid_document_id_returns_400(self) -> None:
        store = _FakeFullDocumentStore(result=_full_document_result())
        client = TestClient(
            _make_app(_FakeOrchestrator(), full_document_store_override=store)
        )

        resp = client.get(f"/api/rag/documents/{'A' * 257}")

        assert resp.status_code == 400
        assert store.calls == []

    def test_missing_document_returns_404(self) -> None:
        store = _FakeFullDocumentStore(result=None)
        client = TestClient(
            _make_app(_FakeOrchestrator(), full_document_store_override=store)
        )

        resp = client.get("/api/rag/documents/DOC-404")

        assert resp.status_code == 404
        assert store.calls == ["DOC-404"]

    def test_lookup_failure_returns_503(self) -> None:
        store = _FakeFullDocumentStore(
            error=FullDocumentLookupError("qdrant unavailable")
        )
        client = TestClient(
            _make_app(_FakeOrchestrator(), full_document_store_override=store)
        )

        resp = client.get("/api/rag/documents/DOC-1")

        assert resp.status_code == 503
        assert resp.json()["detail"] == "Full document lookup is temporarily unavailable."


# ---------------------------------------------------------------------------
# Success — response shape
# ---------------------------------------------------------------------------


class TestSuccessResponseShape:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.post("/api/rag/query", json={"query": "únos dítěte"})
        assert resp.status_code == 200

    def test_response_has_answer_field(self, client: TestClient) -> None:
        resp = client.post("/api/rag/query", json={"query": "dotaz"})
        assert "answer" in resp.json()

    def test_response_has_sources_field(self, client: TestClient) -> None:
        resp = client.post("/api/rag/query", json={"query": "dotaz"})
        assert "sources" in resp.json()

    def test_response_has_plan_steps_field(self, client: TestClient) -> None:
        resp = client.post("/api/rag/query", json={"query": "dotaz"})
        assert "plan_steps" in resp.json()

    def test_answer_is_string(self, client: TestClient) -> None:
        resp = client.post("/api/rag/query", json={"query": "dotaz"})
        assert isinstance(resp.json()["answer"], str)

    def test_sources_is_list(self, client: TestClient) -> None:
        resp = client.post("/api/rag/query", json={"query": "dotaz"})
        assert isinstance(resp.json()["sources"], list)

    def test_plan_steps_is_list(self, client: TestClient) -> None:
        resp = client.post("/api/rag/query", json={"query": "dotaz"})
        assert isinstance(resp.json()["plan_steps"], list)


# ---------------------------------------------------------------------------
# Success — content passthrough
# ---------------------------------------------------------------------------


class TestSuccessContent:
    def test_answer_from_orchestrator(self, fake_orchestrator: _FakeOrchestrator) -> None:
        fake_orchestrator._answer = "Konkrétní právní odpověď"
        client = TestClient(_make_app(fake_orchestrator))
        resp = client.post("/api/rag/query", json={"query": "dotaz"})
        assert resp.json()["answer"] == "Konkrétní právní odpověď"

    def test_sources_from_orchestrator(self, fake_orchestrator: _FakeOrchestrator) -> None:
        fake_orchestrator._sources = ["ABC", "DEF"]
        client = TestClient(_make_app(fake_orchestrator))
        resp = client.post("/api/rag/query", json={"query": "dotaz"})
        assert resp.json()["sources"] == ["ABC", "DEF"]

    def test_plan_steps_from_orchestrator(self, fake_orchestrator: _FakeOrchestrator) -> None:
        fake_orchestrator._plan_steps = ["krok A", "krok B", "krok C"]
        client = TestClient(_make_app(fake_orchestrator))
        resp = client.post("/api/rag/query", json={"query": "dotaz"})
        assert resp.json()["plan_steps"] == ["krok A", "krok B", "krok C"]

    def test_query_passed_to_orchestrator(self, fake_orchestrator: _FakeOrchestrator) -> None:
        client = TestClient(_make_app(fake_orchestrator))
        client.post("/api/rag/query", json={"query": "haagská úmluva"})
        assert "haagská úmluva" in fake_orchestrator.calls


# ---------------------------------------------------------------------------
# Empty query
# ---------------------------------------------------------------------------


class TestEmptyQuery:
    def test_empty_string_returns_200(self, client: TestClient) -> None:
        resp = client.post("/api/rag/query", json={"query": ""})
        assert resp.status_code == 200

    def test_empty_string_has_valid_shape(self, client: TestClient) -> None:
        resp = client.post("/api/rag/query", json={"query": ""})
        body = resp.json()
        assert "answer" in body
        assert "sources" in body
        assert "plan_steps" in body


# ---------------------------------------------------------------------------
# Long query
# ---------------------------------------------------------------------------


class TestLongQuery:
    def test_long_query_returns_200(self, client: TestClient) -> None:
        long_query = "únos dítěte " * 200  # ~2 400 chars
        resp = client.post("/api/rag/query", json={"query": long_query})
        assert resp.status_code == 200

    def test_long_query_has_valid_shape(self, client: TestClient) -> None:
        long_query = "právní dotaz " * 200
        resp = client.post("/api/rag/query", json={"query": long_query})
        body = resp.json()
        assert isinstance(body["answer"], str)
        assert isinstance(body["sources"], list)


# ---------------------------------------------------------------------------
# Orchestrator failure fallback
# ---------------------------------------------------------------------------


class TestOrchestratorFailure:
    @pytest.fixture()
    def failing_client(self) -> TestClient:
        return TestClient(_make_app(_ExplodingOrchestrator()))

    def test_failure_returns_200(self, failing_client: TestClient) -> None:
        resp = failing_client.post("/api/rag/query", json={"query": "dotaz"})
        assert resp.status_code == 200

    def test_failure_returns_empty_answer(self, failing_client: TestClient) -> None:
        resp = failing_client.post("/api/rag/query", json={"query": "dotaz"})
        assert resp.json()["answer"] == ""

    def test_failure_returns_empty_sources(self, failing_client: TestClient) -> None:
        resp = failing_client.post("/api/rag/query", json={"query": "dotaz"})
        assert resp.json()["sources"] == []

    def test_failure_returns_empty_plan_steps(self, failing_client: TestClient) -> None:
        resp = failing_client.post("/api/rag/query", json={"query": "dotaz"})
        assert resp.json()["plan_steps"] == []

    def test_failure_response_has_all_fields(self, failing_client: TestClient) -> None:
        resp = failing_client.post("/api/rag/query", json={"query": "dotaz"})
        body = resp.json()
        assert set(body.keys()) >= {"answer", "sources", "plan_steps"}


# ---------------------------------------------------------------------------
# Missing request body / bad input
# ---------------------------------------------------------------------------


class TestBadInput:
    def test_missing_body_returns_422(self, client: TestClient) -> None:
        resp = client.post("/api/rag/query")
        assert resp.status_code == 422

    def test_missing_query_field_returns_422(self, client: TestClient) -> None:
        resp = client.post("/api/rag/query", json={"not_query": "x"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestLogging:
    def test_received_logged(self, fake_orchestrator, caplog) -> None:
        client = TestClient(_make_app(fake_orchestrator))
        with caplog.at_level(logging.INFO, logger="app.api.rag_router"):
            client.post("/api/rag/query", json={"query": "dotaz"})
        msgs = [r.getMessage() for r in caplog.records]
        assert any("[api]" in m and "query received" in m for m in msgs)

    def test_completed_logged(self, fake_orchestrator, caplog) -> None:
        client = TestClient(_make_app(fake_orchestrator))
        with caplog.at_level(logging.INFO, logger="app.api.rag_router"):
            client.post("/api/rag/query", json={"query": "dotaz"})
        msgs = [r.getMessage() for r in caplog.records]
        assert any("[api]" in m and "query completed" in m for m in msgs)

    def test_warning_logged_on_failure(self, caplog) -> None:
        client = TestClient(_make_app(_ExplodingOrchestrator()))
        with caplog.at_level(logging.WARNING, logger="app.api.rag_router"):
            client.post("/api/rag/query", json={"query": "dotaz"})
        msgs = [r.getMessage() for r in caplog.records]
        assert any("[api]" in m and "orchestrator" in m for m in msgs)


class TestQueryCache:
    def test_repeated_query_uses_cache(self, fake_orchestrator: _FakeOrchestrator) -> None:
        cache = _MemoryCache()
        rtr._query_cache = cache
        rtr._query_cache_backend = "memory"
        client = TestClient(_make_app(fake_orchestrator))

        first = client.post("/api/rag/query", json={"query": "dotaz"})
        second = client.post("/api/rag/query", json={"query": "dotaz"})

        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json() == second.json()
        assert fake_orchestrator.calls == ["dotaz"]
        assert cache.reads == 2
        assert cache.writes == 1
