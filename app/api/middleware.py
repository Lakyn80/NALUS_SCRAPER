"""FastAPI/ASGI observability middleware for Phase A."""

from __future__ import annotations

import os
import time

from fastapi import FastAPI
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.context import (
    CORRELATION_ID_HEADER,
    REQUEST_ID_HEADER,
    bind_context,
    clear_context,
    generate_correlation_id,
    generate_request_id,
    get_context,
    normalize_inbound_correlation_id,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


class ObservabilityMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        correlation_header: str = CORRELATION_ID_HEADER,
        request_header: str = REQUEST_ID_HEADER,
    ) -> None:
        self.app = app
        self.correlation_header = correlation_header
        self.request_header = request_header

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        inbound_correlation = headers.get(self.correlation_header)
        correlation_id = normalize_inbound_correlation_id(inbound_correlation)
        invalid_correlation = inbound_correlation is not None and correlation_id is None
        if correlation_id is None:
            correlation_id = generate_correlation_id()
        request_id = generate_request_id()

        binding = bind_context(correlation_id=correlation_id, request_id=request_id)
        method = str(scope.get("method") or "")
        route = _safe_http_route(scope)
        started = time.monotonic()
        status_code = 500
        response_started = False

        if invalid_correlation:
            logger.warning(
                "Invalid inbound correlation ID was replaced.",
                extra={
                    "event_name": "security.invalid_correlation_id",
                    "http_method": method,
                    "http_route": route,
                    "safe_error_code": "invalid_correlation_id",
                },
            )

        logger.info(
            "HTTP request started.",
            extra={
                "event_name": "http.request.started",
                "http_method": method,
                "http_route": route,
            },
        )

        async def send_with_context(message: Message) -> None:
            nonlocal response_started, status_code, route
            if message["type"] == "http.response.start":
                response_started = True
                status_code = int(message["status"])
                route = _safe_http_route(scope)
                response_headers = MutableHeaders(scope=message)
                response_headers[self.correlation_header] = correlation_id
                response_headers[self.request_header] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_context)
        except Exception as exc:
            duration_ms = int((time.monotonic() - started) * 1000)
            logger.warning(
                "HTTP request failed.",
                extra={
                    "event_name": "http.request.failed",
                    "http_method": method,
                    "http_route": route,
                    "http_status": status_code,
                    "duration_ms": duration_ms,
                    "safe_error_code": type(exc).__name__,
                },
            )
            if not response_started:
                await send(
                    {
                        "type": "http.response.start",
                        "status": 500,
                        "headers": [
                            (
                                self.correlation_header.lower().encode("latin-1"),
                                correlation_id.encode("latin-1"),
                            ),
                            (
                                self.request_header.lower().encode("latin-1"),
                                request_id.encode("latin-1"),
                            ),
                            (b"content-type", b"text/plain; charset=utf-8"),
                        ],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b"Internal Server Error",
                    }
                )
                return
            raise
        else:
            duration_ms = int((time.monotonic() - started) * 1000)
            logger.info(
                "HTTP request completed.",
                extra={
                    "event_name": "http.request.completed",
                    "http_method": method,
                    "http_route": route,
                    "http_status": status_code,
                    "duration_ms": duration_ms,
                },
            )
        finally:
            del binding
            try:
                clear_context()
            except Exception:  # noqa: BLE001
                logger.error(
                    "Observability context cleanup failed.",
                    extra={"event_name": "observability.context_cleanup_failed"},
                )
                return
            if any(value is not None for value in get_context().values()):
                logger.error(
                    "Observability context cleanup failed.",
                    extra={"event_name": "observability.context_cleanup_failed"},
                )


def install_observability_middleware(app: FastAPI) -> None:
    if getattr(app.state, "observability_middleware_installed", False):
        return
    app.add_middleware(
        ObservabilityMiddleware,
        correlation_header=os.getenv("NALUS_CORRELATION_ID_HEADER", CORRELATION_ID_HEADER),
        request_header=os.getenv("NALUS_REQUEST_ID_HEADER", REQUEST_ID_HEADER),
    )
    app.state.observability_middleware_installed = True


def _safe_http_route(scope: Scope) -> str:
    route = scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str) and path:
        return path
    return str(scope.get("path") or "")
