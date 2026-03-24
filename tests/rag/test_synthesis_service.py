"""
Unit tests for app.rag.synthesis.synthesis_service.

Run:
    pytest tests/rag/test_synthesis_service.py -v
"""

import logging

import pytest

from app.rag.execution.execution_service import ExecutionResult
from app.rag.retrieval.models import RetrievedChunk
from app.rag.synthesis.synthesis_service import (
    MockSynthesisLLM,
    SynthesisOutput,
    SynthesisService,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(id: str, score: float = 0.8, text: str = "text") -> RetrievedChunk:
    return RetrievedChunk(id=id, text=text, score=score, source="keyword")


def _result(*step_chunks: list[RetrievedChunk]) -> ExecutionResult:
    steps = [
        {"query": f"q{i}", "reason": f"r{i}", "results": chunks}
        for i, chunks in enumerate(step_chunks)
    ]
    return ExecutionResult(steps_results=steps)


def _make_service(llm=None) -> SynthesisService:
    return SynthesisService(llm=llm or MockSynthesisLLM())


# ---------------------------------------------------------------------------
# SynthesisOutput
# ---------------------------------------------------------------------------


class TestSynthesisOutput:
    def test_stores_answer(self) -> None:
        out = SynthesisOutput(answer="odpověď", sources=["1"])
        assert out.answer == "odpověď"

    def test_stores_sources(self) -> None:
        out = SynthesisOutput(answer="a", sources=["X", "Y"])
        assert out.sources == ["X", "Y"]

    def test_empty_sources(self) -> None:
        out = SynthesisOutput(answer="a", sources=[])
        assert out.sources == []


# ---------------------------------------------------------------------------
# SynthesisService — return type and basic structure
# ---------------------------------------------------------------------------


class TestSynthesisServiceReturnType:
    def test_returns_synthesis_output(self) -> None:
        service = _make_service()
        result = service.synthesize("dotaz", _result([_chunk("1")]))
        assert isinstance(result, SynthesisOutput)

    def test_answer_is_string(self) -> None:
        service = _make_service()
        result = service.synthesize("dotaz", _result([_chunk("1")]))
        assert isinstance(result.answer, str)

    def test_sources_is_list(self) -> None:
        service = _make_service()
        result = service.synthesize("dotaz", _result([_chunk("1")]))
        assert isinstance(result.sources, list)


# ---------------------------------------------------------------------------
# SynthesisService — answer content
# ---------------------------------------------------------------------------


class TestSynthesisServiceAnswer:
    def test_answer_not_empty_when_chunks_present(self) -> None:
        service = _make_service()
        result = service.synthesize("únos dítěte", _result([_chunk("A"), _chunk("B")]))
        assert len(result.answer) > 0

    def test_answer_with_empty_execution_result(self) -> None:
        service = _make_service()
        result = service.synthesize("dotaz", ExecutionResult(steps_results=[]))
        assert isinstance(result.answer, str)

    def test_mock_llm_returns_czech_content(self) -> None:
        service = _make_service()
        result = service.synthesize("dotaz", _result([_chunk("1")]))
        # MockSynthesisLLM always returns structured Czech answer
        assert "Shrnutí" in result.answer or len(result.answer) > 0


# ---------------------------------------------------------------------------
# SynthesisService — sources
# ---------------------------------------------------------------------------


class TestSynthesisServiceSources:
    def test_sources_contain_chunk_ids(self) -> None:
        service = _make_service()
        chunks = [_chunk("ABC"), _chunk("DEF")]
        result = service.synthesize("dotaz", _result(chunks))
        assert "ABC" in result.sources
        assert "DEF" in result.sources

    def test_sources_deduplicated(self) -> None:
        chunk_low = _chunk("X", score=0.5)
        chunk_high = _chunk("X", score=0.9)
        result = service = _make_service()
        out = service.synthesize("dotaz", _result([chunk_low], [chunk_high]))
        assert out.sources.count("X") == 1

    def test_sources_from_multiple_steps(self) -> None:
        service = _make_service()
        step1 = [_chunk("A"), _chunk("B")]
        step2 = [_chunk("C")]
        result = service.synthesize("dotaz", _result(step1, step2))
        assert set(result.sources) == {"A", "B", "C"}

    def test_sources_empty_when_no_chunks(self) -> None:
        service = _make_service()
        result = service.synthesize("dotaz", ExecutionResult(steps_results=[]))
        assert result.sources == []

    def test_sources_order_deterministic(self) -> None:
        service = _make_service()
        # all_chunks returns sorted by score desc
        chunks = [_chunk("low", 0.3), _chunk("high", 0.9), _chunk("mid", 0.6)]
        result = service.synthesize("dotaz", _result(chunks))
        # high > mid > low
        assert result.sources == ["high", "mid", "low"]


# ---------------------------------------------------------------------------
# SynthesisService — LLM interaction
# ---------------------------------------------------------------------------


class TestSynthesisServiceLLM:
    def test_llm_called_with_query_in_prompt(self) -> None:
        received: list[str] = []

        class CaptureLLM(MockSynthesisLLM):
            def generate_text(self, prompt: str) -> str:
                received.append(prompt)
                return super().generate_text(prompt)

        service = _make_service(CaptureLLM())
        service.synthesize("haagská úmluva", _result([_chunk("1")]))
        assert received, "LLM was not called"
        assert "haagská úmluva" in received[0]

    def test_llm_called_with_chunk_text_in_prompt(self) -> None:
        received: list[str] = []

        class CaptureLLM(MockSynthesisLLM):
            def generate_text(self, prompt: str) -> str:
                received.append(prompt)
                return super().generate_text(prompt)

        service = _make_service(CaptureLLM())
        service.synthesize("dotaz", _result([_chunk("Z", text="specifický text rozhodnutí")]))
        assert "specifický text rozhodnutí" in received[0]

    def test_llm_exception_returns_empty_answer(self) -> None:
        class BrokenLLM(MockSynthesisLLM):
            def generate_text(self, prompt: str) -> str:
                raise RuntimeError("LLM unavailable")

        service = _make_service(BrokenLLM())
        result = service.synthesize("dotaz", _result([_chunk("1")]))
        assert result.answer == ""

    def test_llm_exception_still_returns_sources(self) -> None:
        class BrokenLLM(MockSynthesisLLM):
            def generate_text(self, prompt: str) -> str:
                raise RuntimeError("LLM unavailable")

        service = _make_service(BrokenLLM())
        result = service.synthesize("dotaz", _result([_chunk("A"), _chunk("B")]))
        assert set(result.sources) == {"A", "B"}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestSynthesisServiceLogging:
    def test_synthesis_info_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = _make_service()
        with caplog.at_level(logging.INFO, logger="app.rag.synthesis.synthesis_service"):
            service.synthesize("dotaz", _result([_chunk("1")]))
        msgs = [r.getMessage() for r in caplog.records]
        assert any("[synthesis]" in m for m in msgs)

    def test_log_contains_source_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = _make_service()
        with caplog.at_level(logging.INFO, logger="app.rag.synthesis.synthesis_service"):
            service.synthesize("dotaz", _result([_chunk("A"), _chunk("B")]))
        log = next(r.getMessage() for r in caplog.records if "[synthesis]" in r.getMessage())
        assert "sources=2" in log

    def test_trace_events_emitted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = _make_service()
        with caplog.at_level(logging.DEBUG, logger="app.rag.synthesis.synthesis_service"):
            service.synthesize("dotaz", _result([_chunk("1")]))
        msgs = [r.getMessage() for r in caplog.records]
        assert any("synthesis.start" in m for m in msgs)
        assert any("synthesis.done" in m for m in msgs)

    def test_empty_result_still_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = _make_service()
        with caplog.at_level(logging.INFO, logger="app.rag.synthesis.synthesis_service"):
            service.synthesize("dotaz", ExecutionResult(steps_results=[]))
        msgs = [r.getMessage() for r in caplog.records]
        assert any("[synthesis]" in m for m in msgs)
