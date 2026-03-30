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
    QueryResponse,
    get_answer_service,
    get_orchestrator,
    get_pipeline,
    router,
)
from app.rag.orchestrator.orchestrator_service import OrchestratorResult, OrchestratorService
from app.rag.retrieval.models import RetrievedChunk


# ---------------------------------------------------------------------------
# Test app factory
# ---------------------------------------------------------------------------


def _make_app(
    orchestrator_override=None,
    *,
    pipeline_override=None,
    answer_service_override=None,
) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    if orchestrator_override is not None:
        app.dependency_overrides[get_orchestrator] = lambda: orchestrator_override
    if pipeline_override is not None:
        app.dependency_overrides[get_pipeline] = lambda: pipeline_override
    if answer_service_override is not None:
        app.dependency_overrides[get_answer_service] = lambda: answer_service_override
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
    yield
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

    def test_retrieve_failure_returns_empty_results(self) -> None:
        client = TestClient(_make_app(_ExplodingOrchestrator()))

        resp = client.post("/api/rag/retrieve", json={"query": "dotaz"})

        assert resp.status_code == 200
        assert resp.json() == {"results": []}


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
