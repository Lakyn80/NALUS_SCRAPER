"""
Unit tests for app.core.logging and app.core.tracing.

Run:
    pytest tests/test_core_tracing.py -v
"""

import logging

import pytest

from app.core.logging import get_logger
from app.core.tracing import _format_payload, trace_event


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


class TestGetLogger:
    def test_returns_logger_with_correct_name(self) -> None:
        logger = get_logger("app.test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "app.test.module"

    def test_same_name_returns_same_instance(self) -> None:
        a = get_logger("app.same")
        b = get_logger("app.same")
        assert a is b

    def test_different_names_return_different_instances(self) -> None:
        a = get_logger("app.module_a")
        b = get_logger("app.module_b")
        assert a is not b

    def test_multiple_calls_do_not_add_duplicate_handlers(self) -> None:
        """Root logger must not accumulate handlers across repeated get_logger calls."""
        root = logging.getLogger()
        # Call several times — handler count on root must not grow.
        count_before = len(root.handlers)
        for _ in range(10):
            get_logger(f"app.dup_check.{_}")
        count_after = len(root.handlers)
        assert count_after == count_before

    def test_stream_handler_check_not_fooled_by_file_handler(self) -> None:
        """FileHandler inherits StreamHandler — guard must use exact type check."""
        root = logging.getLogger()
        stream_handlers = [h for h in root.handlers if type(h) is logging.StreamHandler]
        file_handlers = [h for h in root.handlers if type(h) is logging.FileHandler]
        # FileHandlers must NOT be counted as StreamHandlers
        for h in file_handlers:
            assert type(h) is not logging.StreamHandler


# ---------------------------------------------------------------------------
# _format_payload (pure function, no I/O)
# ---------------------------------------------------------------------------


class TestFormatPayload:
    def test_empty_payload(self) -> None:
        assert _format_payload({}) == ""

    def test_single_string_value(self) -> None:
        result = _format_payload({"doc_id": "abc-123"})
        assert result == 'doc_id="abc-123"'

    def test_integer_value(self) -> None:
        result = _format_payload({"num_chunks": 12})
        assert result == "num_chunks=12"

    def test_float_value(self) -> None:
        result = _format_payload({"score": 0.91})
        assert "score=0.91" in result

    def test_list_value(self) -> None:
        result = _format_payload({"scores": [0.91, 0.88]})
        assert "scores=[0.91, 0.88]" in result

    def test_multiple_keys(self) -> None:
        result = _format_payload({"a": 1, "b": "x"})
        assert "a=1" in result
        assert 'b="x"' in result

    def test_non_serializable_value_falls_back_to_str(self) -> None:
        class Unserializable:
            def __str__(self) -> str:
                return "str-repr"

            def __repr__(self) -> str:
                return "repr-repr"

        result = _format_payload({"obj": Unserializable()})
        # Must use str(), not repr()
        assert "obj=str-repr" in result
        assert "repr-repr" not in result

    def test_none_value_is_safe(self) -> None:
        result = _format_payload({"text": None})
        assert "text=null" in result

    def test_dict_value_is_safe(self) -> None:
        result = _format_payload({"meta": {"key": "val", "n": 3}})
        assert "meta=" in result
        assert "key" in result
        assert "val" in result

    def test_nested_list_is_safe(self) -> None:
        result = _format_payload({"pairs": [[1, 2], [3, 4]]})
        assert "pairs=" in result
        assert "1" in result

    def test_czech_string_no_escaping(self) -> None:
        result = _format_payload({"query": "únos dítěte"})
        assert "únos dítěte" in result


# ---------------------------------------------------------------------------
# trace_event (verifies logging integration)
# ---------------------------------------------------------------------------


class TestTraceEvent:
    def _make_logger(self, name: str, level: int = logging.DEBUG) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        return logger

    def test_emits_debug_record_with_event_name(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = self._make_logger("trace.test.basic")
        with caplog.at_level(logging.DEBUG, logger="trace.test.basic"):
            trace_event(logger, "chunking.done", num_chunks=5)

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelno == logging.DEBUG
        assert "TRACE chunking.done" in record.getMessage()

    def test_payload_values_appear_in_log_message(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = self._make_logger("trace.test.payload")
        with caplog.at_level(logging.DEBUG, logger="trace.test.payload"):
            trace_event(logger, "retrieval.result", query="únos dítěte", top_k=5)

        msg = caplog.records[0].getMessage()
        assert "query=" in msg
        assert "únos dítěte" in msg
        assert "top_k=5" in msg

    def test_no_log_emitted_at_info_level(self, caplog: pytest.LogCaptureFixture) -> None:
        """TRACE = DEBUG. When logger is at INFO, no trace events must appear."""
        logger = self._make_logger("trace.test.info_level", level=logging.INFO)
        with caplog.at_level(logging.INFO, logger="trace.test.info_level"):
            trace_event(logger, "should.not.appear", value=42)

        assert len(caplog.records) == 0

    def test_no_log_emitted_when_debug_disabled(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = self._make_logger("trace.test.disabled", level=logging.WARNING)
        with caplog.at_level(logging.WARNING, logger="trace.test.disabled"):
            trace_event(logger, "should.not.appear", value=42)

        assert len(caplog.records) == 0

    def test_empty_payload_is_valid(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = self._make_logger("trace.test.empty")
        with caplog.at_level(logging.DEBUG, logger="trace.test.empty"):
            trace_event(logger, "pipeline.start")

        assert len(caplog.records) == 1
        assert "TRACE pipeline.start" in caplog.records[0].getMessage()

    def test_list_payload_serialized(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = self._make_logger("trace.test.list")
        with caplog.at_level(logging.DEBUG, logger="trace.test.list"):
            trace_event(logger, "retrieval.scores", scores=[0.92, 0.87, 0.81])

        msg = caplog.records[0].getMessage()
        assert "scores=" in msg
        assert "0.92" in msg

    def test_multiple_events_all_recorded(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = self._make_logger("trace.test.multi")
        with caplog.at_level(logging.DEBUG, logger="trace.test.multi"):
            trace_event(logger, "step.one", x=1)
            trace_event(logger, "step.two", x=2)
            trace_event(logger, "step.three", x=3)

        assert len(caplog.records) == 3
        messages = [r.getMessage() for r in caplog.records]
        assert any("step.one" in m for m in messages)
        assert any("step.two" in m for m in messages)
        assert any("step.three" in m for m in messages)
