"""
Unit tests for app.rag.answer.hybrid_service.HybridAnswerService.

Run:
    pytest tests/rag/test_hybrid_answer_service.py -v
"""

import logging

import pytest

from app.rag.answer.answer_service import AnswerResult, AnswerService
from app.rag.answer.hybrid_service import HybridAnswerService
from app.rag.llm.base import BaseLLM
from app.rag.llm.mock_llm import MockLLM
from app.rag.llm.models import LLMInput, LLMOutput
from app.rag.llm.service import LLMService
from app.rag.retrieval.models import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(id: str, text: str = "Ústavní soud rozhodl.", score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(id=id, text=text, score=score, source="dense")


def _make_service(spy_answer: bool = False, spy_llm: bool = False) -> tuple[
    HybridAnswerService, list[str]
]:
    """Build HybridAnswerService with optional call-tracking spies."""
    calls: list[str] = []

    class SpyAnswerService(AnswerService):
        def generate(self, query, chunks):
            calls.append("answer")
            return super().generate(query, chunks)

    class SpyLLM(BaseLLM):
        def generate(self, data: LLMInput) -> LLMOutput:
            calls.append("llm")
            return LLMOutput(
                answer=f"LLM: {data.query}",
                reasoning="spy",
                sources=[c.id for c in data.chunks[:3]],
                confidence=0.7,
            )

    answer_svc = SpyAnswerService() if spy_answer else AnswerService()
    llm_svc = LLMService(SpyLLM() if spy_llm else MockLLM())
    return HybridAnswerService(answer_svc, llm_svc), calls


# ---------------------------------------------------------------------------
# Routing by chunk count
# ---------------------------------------------------------------------------


class TestHybridAnswerServiceRouting:
    def test_zero_chunks_routes_to_answer(self) -> None:
        svc, calls = _make_service(spy_answer=True, spy_llm=True)
        svc.generate("q", [])
        assert calls == ["answer"]

    def test_one_chunk_routes_to_answer(self) -> None:
        svc, calls = _make_service(spy_answer=True, spy_llm=True)
        svc.generate("q", [_chunk("a")])
        assert calls == ["answer"]

    def test_two_chunks_routes_to_llm(self) -> None:
        svc, calls = _make_service(spy_answer=True, spy_llm=True)
        svc.generate("q", [_chunk("a"), _chunk("b")])
        assert calls == ["llm"]

    def test_many_chunks_routes_to_llm(self) -> None:
        svc, calls = _make_service(spy_answer=True, spy_llm=True)
        svc.generate("q", [_chunk(str(i)) for i in range(10)])
        assert calls == ["llm"]

    def test_exactly_at_threshold_routes_to_llm(self) -> None:
        """Threshold is 2 — two chunks must take LLM path."""
        svc, calls = _make_service(spy_answer=True, spy_llm=True)
        svc.generate("q", [_chunk("x"), _chunk("y")])
        assert "llm" in calls
        assert "answer" not in calls


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


class TestHybridAnswerServiceReturnTypes:
    def test_answer_route_returns_answer_result(self) -> None:
        svc, _ = _make_service()
        result = svc.generate("q", [])
        assert isinstance(result, AnswerResult)

    def test_llm_route_returns_llm_output(self) -> None:
        svc, _ = _make_service()
        result = svc.generate("q", [_chunk("a"), _chunk("b")])
        assert isinstance(result, LLMOutput)

    def test_answer_result_preserves_query(self) -> None:
        svc, _ = _make_service()
        result = svc.generate("únos dítěte", [])
        assert isinstance(result, AnswerResult)
        assert result.query == "únos dítěte"

    def test_llm_output_preserves_query_in_answer(self) -> None:
        svc, _ = _make_service()
        result = svc.generate("únos dítěte", [_chunk("a"), _chunk("b")])
        assert isinstance(result, LLMOutput)
        assert "únos dítěte" in result.answer


# ---------------------------------------------------------------------------
# Service isolation — only one service called per generate()
# ---------------------------------------------------------------------------


class TestHybridAnswerServiceIsolation:
    def test_answer_route_does_not_call_llm(self) -> None:
        svc, calls = _make_service(spy_answer=True, spy_llm=True)
        svc.generate("q", [_chunk("a")])
        assert "llm" not in calls

    def test_llm_route_does_not_call_answer_service(self) -> None:
        svc, calls = _make_service(spy_answer=True, spy_llm=True)
        svc.generate("q", [_chunk("a"), _chunk("b")])
        assert "answer" not in calls


# ---------------------------------------------------------------------------
# Trace events
# ---------------------------------------------------------------------------


class TestHybridAnswerServiceTrace:
    def test_start_and_done_emitted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        svc, _ = _make_service()
        with caplog.at_level(logging.DEBUG, logger="app.rag.answer.hybrid_service"):
            svc.generate("q", [_chunk("a")])

        msgs = [r.getMessage() for r in caplog.records]
        assert any("TRACE hybrid.start" in m for m in msgs)
        assert any("TRACE hybrid.done" in m for m in msgs)

    def test_route_event_emitted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        svc, _ = _make_service()
        with caplog.at_level(logging.DEBUG, logger="app.rag.answer.hybrid_service"):
            svc.generate("q", [_chunk("a")])

        msgs = [r.getMessage() for r in caplog.records]
        assert any("TRACE hybrid.route" in m for m in msgs)

    def test_answer_route_logged_as_answer(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        svc, _ = _make_service()
        with caplog.at_level(logging.DEBUG, logger="app.rag.answer.hybrid_service"):
            svc.generate("q", [_chunk("only")])

        route_msg = next(
            r.getMessage() for r in caplog.records
            if "hybrid.route" in r.getMessage()
        )
        assert "answer" in route_msg

    def test_llm_route_logged_as_llm(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        svc, _ = _make_service()
        with caplog.at_level(logging.DEBUG, logger="app.rag.answer.hybrid_service"):
            svc.generate("q", [_chunk("a"), _chunk("b")])

        route_msg = next(
            r.getMessage() for r in caplog.records
            if "hybrid.route" in r.getMessage()
        )
        assert "llm" in route_msg

    def test_start_includes_query_and_num_chunks(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        svc, _ = _make_service()
        with caplog.at_level(logging.DEBUG, logger="app.rag.answer.hybrid_service"):
            svc.generate("únos dítěte", [_chunk("a"), _chunk("b"), _chunk("c")])

        start = next(
            r.getMessage() for r in caplog.records
            if "hybrid.start" in r.getMessage()
        )
        assert "únos dítěte" in start
        assert "num_chunks=3" in start
