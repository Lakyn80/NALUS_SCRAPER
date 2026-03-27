"""
Integration tests for POST /api/rag/query endpoint.

Uses FastAPI TestClient with dependency_overrides so no real LLM or
retrieval services are called.

Run:
    pytest tests/api/test_rag_api.py -v
"""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.rag_router as rtr
from app.api.query_cache import CachedQueryResponse
from app.api.rag_router import QueryResponse, get_orchestrator, router
from app.rag.orchestrator.orchestrator_service import OrchestratorResult, OrchestratorService


# ---------------------------------------------------------------------------
# Test app factory
# ---------------------------------------------------------------------------


def _make_app(orchestrator_override=None) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    if orchestrator_override is not None:
        app.dependency_overrides[get_orchestrator] = lambda: orchestrator_override
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
    ) -> None:
        self._answer = answer
        self._sources = sources if sources is not None else ["1", "2"]
        self._plan_steps = plan_steps if plan_steps is not None else ["krok 1", "krok 2"]
        self.calls: list[str] = []

    def run(self, query: str) -> OrchestratorResult:
        self.calls.append(query)
        return OrchestratorResult(
            answer=self._answer,
            sources=self._sources,
            plan_steps=self._plan_steps,
        )


class _ExplodingOrchestrator:
    """Always raises — simulates unexpected orchestrator failure."""

    def run(self, query: str) -> OrchestratorResult:
        raise RuntimeError("orchestrator exploded")


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
