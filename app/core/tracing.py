"""
Lightweight structured tracing for the NALUS RAG pipeline.

Each trace event is a named, dict-like payload logged at DEBUG level through a
standard Python logger.  The interface is intentionally minimal so it can be
replaced later with LangSmith / OpenTelemetry without touching call sites.

Usage:
    from app.core.logging import get_logger
    from app.core.tracing import trace_event

    logger = get_logger(__name__)

    trace_event(logger, "chunking.start", doc_id="abc-123", text_length=4800)
    trace_event(logger, "chunking.done",  doc_id="abc-123", num_chunks=12)
    trace_event(logger, "retrieval.result", query="únos dítěte", top_k=5, scores=[0.91, 0.88])

Output (one line per event):
    2024-01-01T12:00:00 | DEBUG    | app.rag.chunking | TRACE chunking.start | doc_id=abc-123 text_length=4800
"""

import json
import logging
from typing import Any


def trace_event(logger: logging.Logger, name: str, **payload: Any) -> None:
    """Log a structured trace event.

    Args:
        logger:  The caller's module logger (use get_logger(__name__)).
        name:    Dot-separated event name, e.g. "chunking.done".
        **payload: Arbitrary key-value pairs describing the event.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return

    payload_str = _format_payload(payload)
    logger.debug("TRACE %s | %s", name, payload_str)


def _format_payload(payload: dict[str, Any]) -> str:
    """Serialize payload to a compact, human-readable string.

    Values that are not JSON-serializable fall back to repr().
    """
    parts: list[str] = []
    for key, value in payload.items():
        try:
            serialized = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            serialized = str(value)
        parts.append(f"{key}={serialized}")
    return " ".join(parts)
