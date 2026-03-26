from __future__ import annotations

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

import app.api.main as main
import app.api.rag_router as rtr
from app.rag.orchestrator.orchestrator_service import OrchestratorResult


class _FakeOrchestrator:
    def run(self, query: str) -> OrchestratorResult:
        return OrchestratorResult(answer=f"ready:{query}", sources=[], plan_steps=[])


@pytest.fixture(autouse=True)
def _reset_live_orchestrator(monkeypatch) -> None:
    monkeypatch.delenv("RAG_STRICT_REAL_MODE", raising=False)
    monkeypatch.setattr(main, "_STARTUP_RETRY_DELAY_SECONDS", 0.01)
    main._deferred_ingest_task = None
    rtr._live_orchestrator = None
    rtr._live_orchestrator_status = "pending"
    rtr._live_orchestrator_error = None
    rtr._background_ingest_status = "idle"
    rtr._background_ingest_error = None
    yield
    main._deferred_ingest_task = None
    rtr._live_orchestrator = None
    rtr._live_orchestrator_status = "pending"
    rtr._live_orchestrator_error = None
    rtr._background_ingest_status = "idle"
    rtr._background_ingest_error = None


def test_docs_are_available_while_orchestrator_initializes(monkeypatch) -> None:
    async def fake_initialize() -> None:
        rtr._live_orchestrator_status = "initializing"
        await asyncio.sleep(0.2)
        rtr._live_orchestrator = _FakeOrchestrator()
        rtr._live_orchestrator_status = "ready"
        rtr._live_orchestrator_error = None

    monkeypatch.setenv("RAG_STRICT_REAL_MODE", "1")
    monkeypatch.setattr(main, "_initialize_orchestrator", fake_initialize)

    with TestClient(main.app) as client:
        docs = client.get("/docs")
        health = client.get("/health")

    assert docs.status_code == 200
    assert health.status_code == 200
    assert health.json()["orchestrator_ready"] is False
    assert health.json()["strict_real_mode"] is True
    assert health.json()["orchestrator_status"] in {"pending", "initializing"}
    assert health.json()["background_ingest_status"] == "idle"


def test_strict_mode_query_returns_503_while_initializing(monkeypatch) -> None:
    async def fake_initialize() -> None:
        rtr._live_orchestrator_status = "initializing"
        await asyncio.sleep(0.2)
        rtr._live_orchestrator = _FakeOrchestrator()
        rtr._live_orchestrator_status = "ready"
        rtr._live_orchestrator_error = None

    monkeypatch.setenv("RAG_STRICT_REAL_MODE", "1")
    monkeypatch.setattr(main, "_initialize_orchestrator", fake_initialize)

    with TestClient(main.app) as client:
        response = client.post("/api/rag/query", json={"query": "dotaz"})

    assert response.status_code == 503
    assert "still initializing" in response.json()["detail"]


def test_background_retry_recovers_after_startup_failure(monkeypatch) -> None:
    attempts = {"count": 0}

    async def fake_initialize() -> None:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("qdrant unavailable")

        rtr._live_orchestrator_status = "initializing"
        await asyncio.sleep(0.01)
        rtr._live_orchestrator = _FakeOrchestrator()
        rtr._live_orchestrator_status = "ready"
        rtr._live_orchestrator_error = None

    monkeypatch.setenv("RAG_STRICT_REAL_MODE", "1")
    monkeypatch.setattr(main, "_initialize_orchestrator", fake_initialize)
    monkeypatch.setattr(main, "_STARTUP_RETRY_DELAY_SECONDS", 0.05)

    with TestClient(main.app) as client:
        time.sleep(0.02)
        health_after_failure = client.get("/health")

        time.sleep(0.08)
        query_after_retry = client.post("/api/rag/query", json={"query": "dotaz"})

    assert health_after_failure.status_code == 200
    assert health_after_failure.json()["orchestrator_status"] == "error"
    assert health_after_failure.json()["orchestrator_error"] == "qdrant unavailable"
    assert query_after_retry.status_code == 200
    assert query_after_retry.json()["answer"] == "ready:dotaz"
    assert attempts["count"] >= 2
