from __future__ import annotations

import json
import logging

import pytest

from app.core.context import bound_context, clear_context
from app.core.logging import JsonLogFormatter, get_logger
from app.core.redaction import REDACTED
from app.core.tracing import trace_event


@pytest.fixture(autouse=True)
def _clear_context() -> None:
    clear_context()
    yield
    clear_context()


def test_log_records_are_enriched_from_context(caplog: pytest.LogCaptureFixture) -> None:
    logger = get_logger("tests.phase_a.context")

    with bound_context(correlation_id="corr-12345678", request_id="req-12345678"):
        with caplog.at_level(logging.INFO, logger="tests.phase_a.context"):
            logger.info("hello", extra={"event_name": "test.event"})

    record = caplog.records[0]
    assert record.correlation_id == "corr-12345678"
    assert record.request_id == "req-12345678"
    assert record.event_name == "test.event"


def test_json_logging_output_is_valid_and_contains_context() -> None:
    formatter = JsonLogFormatter()
    record = logging.LogRecord(
        "tests.phase_a.json",
        logging.INFO,
        __file__,
        10,
        "message %s",
        ("ok",),
        None,
    )
    record.event_name = "test.json"
    record.correlation_id = "corr-12345678"
    record.request_id = "req-12345678"
    record.authorization = "Bearer token-value"

    payload = json.loads(formatter.format(record))

    assert payload["event_name"] == "test.json"
    assert payload["message"] == "message ok"
    assert payload["correlation_id"] == "corr-12345678"
    assert payload["request_id"] == "req-12345678"
    assert payload["extra"]["authorization"] == REDACTED
    assert "token-value" not in json.dumps(payload)


def test_full_idempotency_key_is_not_logged() -> None:
    formatter = JsonLogFormatter()
    record = logging.LogRecord(
        "tests.phase_a.idempotency",
        logging.INFO,
        __file__,
        10,
        "idempotency",
        (),
        None,
    )
    record.idempotency_key = "full-idempotency-key-value"

    rendered = formatter.format(record)

    assert "full-idempotency-key-value" not in rendered
    assert REDACTED in rendered


def test_existing_logging_calls_remain_compatible(caplog: pytest.LogCaptureFixture) -> None:
    logger = get_logger("tests.phase_a.compat")

    with caplog.at_level(logging.INFO, logger="tests.phase_a.compat"):
        logger.info("plain %s", "message")

    assert caplog.records[0].getMessage() == "plain message"


def test_get_logger_does_not_create_duplicate_handlers() -> None:
    root = logging.getLogger()
    before = len(root.handlers)

    for index in range(10):
        get_logger(f"tests.phase_a.dup.{index}")

    assert len(root.handlers) == before


def test_trace_event_redacts_sensitive_payload(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("tests.phase_a.trace")
    logger.setLevel(logging.DEBUG)

    with caplog.at_level(logging.DEBUG, logger="tests.phase_a.trace"):
        trace_event(
            logger,
            "trace.redaction",
            api_key="key-value",
            idempotency_key="full-idempotency-key-value",
        )

    message = caplog.records[0].getMessage()
    assert "key-value" not in message
    assert "full-idempotency-key-value" not in message
    assert REDACTED in message
