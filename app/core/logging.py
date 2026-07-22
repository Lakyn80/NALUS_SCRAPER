"""Centralized logging configuration for the NALUS RAG pipeline."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional

from app.core.context import get_context
from app.core.redaction import redact_sensitive

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
_JSON_FORMAT_VALUES = {"1", "true", "yes", "on", "json"}

_RESERVED_LOG_RECORD_ATTRS = set(logging.makeLogRecord({}).__dict__) | {
    "message",
    "asctime",
}

_STRUCTURED_ATTRS = (
    "event_name",
    "http_method",
    "http_route",
    "http_status",
    "duration_ms",
    "safe_error_code",
    "build_version",
    "operation_type",
    "workflow_status",
    "adapter_or_provider",
    "idempotency_key_fingerprint",
    "external_reference",
    "retry_attempt",
    "reconciliation_status",
)

# Module-level flag so we only configure handlers once per process.
_configured = False
_record_factory_installed = False
_previous_record_factory: Any = None


class ContextEnrichingFilter(logging.Filter):
    """Ensure records created before factory installation still carry context."""

    def filter(self, record: logging.LogRecord) -> bool:
        _enrich_record(record)
        return True


class JsonLogFormatter(logging.Formatter):
    """Production-compatible JSON formatter with redacted structured extras."""

    def format(self, record: logging.LogRecord) -> str:
        _enrich_record(record)
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "severity": record.levelname,
            "service": os.getenv("SERVICE_NAME", "nalus-scraper"),
            "environment": os.getenv(
                "APP_ENV",
                os.getenv("ENVIRONMENT", "local"),
            ),
            "logger": record.name,
            "event_name": getattr(record, "event_name", "log"),
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", None),
            "request_id": getattr(record, "request_id", None),
            "operation_id": getattr(record, "operation_id", None),
            "workflow_id": getattr(record, "workflow_id", None),
            "job_id": getattr(record, "job_id", None),
            "task_id": getattr(record, "task_id", None),
        }
        for attr in _STRUCTURED_ATTRS:
            if attr in payload:
                continue
            value = getattr(record, attr, None)
            if value is not None:
                payload[attr] = value
        extras = _record_extras(record)
        if extras:
            payload["extra"] = redact_sensitive(extras)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(redact_sensitive(payload), ensure_ascii=False, default=str)


def _configure_root_logger() -> None:
    global _configured
    _install_record_factory()
    if _configured:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    formatter: logging.Formatter
    if _json_logging_enabled():
        formatter = JsonLogFormatter()
    else:
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(level)

    # Use exact type checks — FileHandler inherits StreamHandler, so isinstance()
    # would incorrectly treat a FileHandler as a StreamHandler guard match.
    if not any(type(h) is logging.StreamHandler for h in root.handlers):
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(ContextEnrichingFilter())
        root.addHandler(stream_handler)

    log_file: Optional[str] = os.getenv("LOG_FILE")
    if log_file and not any(type(h) is logging.FileHandler for h in root.handlers):
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(ContextEnrichingFilter())
        root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with the project-wide configuration applied."""
    _configure_root_logger()
    return logging.getLogger(name)


def _install_record_factory() -> None:
    global _record_factory_installed, _previous_record_factory
    if _record_factory_installed:
        return
    _previous_record_factory = logging.getLogRecordFactory()

    def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = _previous_record_factory(*args, **kwargs)
        _enrich_record(record)
        return record

    logging.setLogRecordFactory(record_factory)
    _record_factory_installed = True


def _enrich_record(record: logging.LogRecord) -> None:
    for key, value in get_context().items():
        if not hasattr(record, key):
            setattr(record, key, value)


def _record_extras(record: logging.LogRecord) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    for key, value in record.__dict__.items():
        if key in _RESERVED_LOG_RECORD_ATTRS:
            continue
        if key in _STRUCTURED_ATTRS:
            continue
        if key in {
            "correlation_id",
            "request_id",
            "operation_id",
            "workflow_id",
            "job_id",
            "task_id",
        }:
            continue
        extras[key] = value
    return extras


def _json_logging_enabled() -> bool:
    raw = os.getenv("LOG_FORMAT", os.getenv("LOG_JSON", "")).strip().lower()
    return raw in _JSON_FORMAT_VALUES
