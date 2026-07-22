from __future__ import annotations

import asyncio
import logging

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.middleware import install_observability_middleware
from app.core.context import (
    CORRELATION_ID_HEADER,
    REQUEST_ID_HEADER,
    clear_context,
    get_context,
    is_valid_context_id,
)


@pytest.fixture(autouse=True)
def _clear_context() -> None:
    clear_context()
    yield
    clear_context()


def _make_app() -> FastAPI:
    app = FastAPI()
    install_observability_middleware(app)

    @app.get("/context")
    async def context() -> dict[str, str | None]:
        return get_context()

    @app.get("/sleep")
    async def sleep() -> dict[str, str | None]:
        await asyncio.sleep(0.05)
        return get_context()

    @app.get("/fail")
    async def fail() -> None:
        raise RuntimeError("boom")

    return app


def test_valid_inbound_correlation_id_is_accepted() -> None:
    app = _make_app()
    with TestClient(app) as client:
        response = client.get("/context", headers={CORRELATION_ID_HEADER: "corr-12345678"})

    assert response.status_code == 200
    assert response.headers[CORRELATION_ID_HEADER] == "corr-12345678"
    assert response.json()["correlation_id"] == "corr-12345678"


def test_missing_correlation_id_is_generated() -> None:
    app = _make_app()
    with TestClient(app) as client:
        response = client.get("/context")

    correlation_id = response.headers[CORRELATION_ID_HEADER]
    assert is_valid_context_id(correlation_id)
    assert response.json()["correlation_id"] == correlation_id


def test_invalid_correlation_id_is_rejected_and_replaced(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = _make_app()
    with TestClient(app) as client:
        with caplog.at_level(logging.WARNING, logger="app.api.middleware"):
            response = client.get(
                "/context",
                headers={CORRELATION_ID_HEADER: "bad value with spaces"},
            )

    replacement = response.headers[CORRELATION_ID_HEADER]
    assert replacement != "bad value with spaces"
    assert is_valid_context_id(replacement)
    assert any(
        record.event_name == "security.invalid_correlation_id"
        for record in caplog.records
    )


def test_request_id_is_unique_per_request() -> None:
    app = _make_app()
    with TestClient(app) as client:
        first = client.get("/context")
        second = client.get("/context")

    assert first.headers[REQUEST_ID_HEADER] != second.headers[REQUEST_ID_HEADER]
    assert is_valid_context_id(first.headers[REQUEST_ID_HEADER])
    assert is_valid_context_id(second.headers[REQUEST_ID_HEADER])


def test_response_headers_are_present() -> None:
    app = _make_app()
    with TestClient(app) as client:
        response = client.get("/context")

    assert CORRELATION_ID_HEADER in response.headers
    assert REQUEST_ID_HEADER in response.headers


def test_context_is_cleared_after_successful_request() -> None:
    app = _make_app()
    with TestClient(app) as client:
        client.get("/context", headers={CORRELATION_ID_HEADER: "corr-12345678"})

    assert all(value is None for value in get_context().values())


def test_context_is_cleared_after_failed_request() -> None:
    app = _make_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/fail", headers={CORRELATION_ID_HEADER: "corr-12345678"})

    assert response.status_code == 500
    assert response.headers[CORRELATION_ID_HEADER] == "corr-12345678"
    assert all(value is None for value in get_context().values())


def test_no_context_leakage_between_sequential_requests() -> None:
    app = _make_app()
    with TestClient(app) as client:
        first = client.get("/context", headers={CORRELATION_ID_HEADER: "corr-12345678"})
        second = client.get("/context")

    assert first.json()["correlation_id"] == "corr-12345678"
    assert second.json()["correlation_id"] != "corr-12345678"
    assert second.headers[CORRELATION_ID_HEADER] == second.json()["correlation_id"]


@pytest.mark.asyncio
async def test_no_context_leakage_between_concurrent_requests() -> None:
    app = _make_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first, second = await asyncio.gather(
            client.get("/sleep", headers={CORRELATION_ID_HEADER: "corr-11111111"}),
            client.get("/sleep", headers={CORRELATION_ID_HEADER: "corr-22222222"}),
        )

    assert first.json()["correlation_id"] == "corr-11111111"
    assert second.json()["correlation_id"] == "corr-22222222"
    assert first.json()["request_id"] != second.json()["request_id"]


def test_request_start_and_completion_events_are_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = _make_app()
    with TestClient(app) as client:
        with caplog.at_level(logging.INFO, logger="app.api.middleware"):
            client.get("/context")

    middleware_records = [
        record for record in caplog.records if record.name == "app.api.middleware"
    ]
    event_names = [record.event_name for record in middleware_records]
    assert "http.request.started" in event_names
    assert "http.request.completed" in event_names
    assert all(record.correlation_id for record in middleware_records)
    assert all(record.request_id for record in middleware_records)


def test_request_failure_event_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    app = _make_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        with caplog.at_level(logging.WARNING, logger="app.api.middleware"):
            client.get("/fail")

    assert any(record.event_name == "http.request.failed" for record in caplog.records)


def test_middleware_does_not_log_authorization_cookies_or_query_secrets(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = _make_app()
    with TestClient(app) as client:
        with caplog.at_level(logging.INFO, logger="app.api.middleware"):
            client.get(
                "/context?api_key=query-secret",
                headers={
                    "authorization": "Bearer header-secret",
                    "cookie": "session=cookie-secret",
                },
            )

    rendered = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name == "app.api.middleware"
    )
    assert "header-secret" not in rendered
    assert "cookie-secret" not in rendered
    assert "query-secret" not in rendered
