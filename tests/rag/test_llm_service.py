"""
Unit tests for app.rag.llm.service.LLMService.

Run:
    pytest tests/rag/test_llm_service.py -v
"""

import logging

import pytest

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


class _SpyLLM(BaseLLM):
    """Records every call to generate() for assertion."""

    def __init__(self) -> None:
        self.calls: list[LLMInput] = []

    def generate(self, data: LLMInput) -> LLMOutput:
        self.calls.append(data)
        return LLMOutput(
            answer=f"Spy odpověď na: {data.query}",
            reasoning="Spy reasoning",
            sources=[c.id for c in data.chunks[:3]],
            confidence=0.7,
        )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestLLMServiceReturnType:
    def test_returns_llm_output(self) -> None:
        service = LLMService(MockLLM())
        result = service.generate("únos dítěte", [_chunk("a")])
        assert isinstance(result, LLMOutput)

    def test_answer_is_str(self) -> None:
        result = LLMService(MockLLM()).generate("q", [_chunk("a")])
        assert isinstance(result.answer, str)

    def test_sources_is_list(self) -> None:
        result = LLMService(MockLLM()).generate("q", [_chunk("a")])
        assert isinstance(result.sources, list)

    def test_confidence_is_float(self) -> None:
        result = LLMService(MockLLM()).generate("q", [_chunk("a")])
        assert isinstance(result.confidence, float)


# ---------------------------------------------------------------------------
# Injection — service uses the injected llm
# ---------------------------------------------------------------------------


class TestLLMServiceInjection:
    def test_injected_llm_is_called(self) -> None:
        spy = _SpyLLM()
        LLMService(spy).generate("q", [_chunk("a")])
        assert len(spy.calls) == 1

    def test_injected_llm_called_once_per_generate(self) -> None:
        spy = _SpyLLM()
        service = LLMService(spy)
        service.generate("q1", [_chunk("a")])
        service.generate("q2", [_chunk("b")])
        assert len(spy.calls) == 2

    def test_mock_llm_works_via_service(self) -> None:
        result = LLMService(MockLLM()).generate("test", [_chunk("x")])
        assert result.answer == "Mock odpověď na: test"


# ---------------------------------------------------------------------------
# Input forwarding — query and chunks reach LLMInput correctly
# ---------------------------------------------------------------------------


class TestLLMServiceInputForwarding:
    def test_query_forwarded_to_llm(self) -> None:
        spy = _SpyLLM()
        LLMService(spy).generate("únos dítěte Rusko", [_chunk("a")])
        assert spy.calls[0].query == "únos dítěte Rusko"

    def test_chunks_forwarded_to_llm(self) -> None:
        spy = _SpyLLM()
        chunks = [_chunk("A"), _chunk("B")]
        LLMService(spy).generate("q", chunks)
        assert spy.calls[0].chunks == chunks

    def test_llm_receives_llm_input_instance(self) -> None:
        spy = _SpyLLM()
        LLMService(spy).generate("q", [_chunk("a")])
        assert isinstance(spy.calls[0], LLMInput)


# ---------------------------------------------------------------------------
# Empty chunks
# ---------------------------------------------------------------------------


class TestLLMServiceEmptyChunks:
    def test_empty_chunks_returns_llm_output(self) -> None:
        result = LLMService(MockLLM()).generate("q", [])
        assert isinstance(result, LLMOutput)

    def test_empty_chunks_gives_empty_sources(self) -> None:
        result = LLMService(MockLLM()).generate("q", [])
        assert result.sources == []

    def test_empty_chunks_llm_called_with_empty_list(self) -> None:
        spy = _SpyLLM()
        LLMService(spy).generate("q", [])
        assert spy.calls[0].chunks == []


# ---------------------------------------------------------------------------
# Trace events
# ---------------------------------------------------------------------------


class TestLLMServiceTrace:
    def test_start_and_done_emitted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="app.rag.llm.service"):
            LLMService(MockLLM()).generate("únos dítěte", [_chunk("a")])

        msgs = [r.getMessage() for r in caplog.records]
        assert any("TRACE llm_service.start" in m for m in msgs)
        assert any("TRACE llm_service.done" in m for m in msgs)

    def test_start_includes_query_and_num_chunks(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        chunks = [_chunk(str(i)) for i in range(4)]
        with caplog.at_level(logging.DEBUG, logger="app.rag.llm.service"):
            LLMService(MockLLM()).generate("únos dítěte Rusko", chunks)

        start = next(
            r.getMessage() for r in caplog.records
            if "llm_service.start" in r.getMessage()
        )
        assert "únos dítěte Rusko" in start
        assert "num_chunks=4" in start

    def test_done_includes_num_sources_and_confidence(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        chunks = [_chunk(str(i)) for i in range(3)]
        with caplog.at_level(logging.DEBUG, logger="app.rag.llm.service"):
            LLMService(MockLLM()).generate("q", chunks)

        done = next(
            r.getMessage() for r in caplog.records
            if "llm_service.done" in r.getMessage()
        )
        assert "num_sources=" in done
        assert "confidence=" in done

    def test_empty_input_still_emits_done(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="app.rag.llm.service"):
            LLMService(MockLLM()).generate("q", [])

        msgs = [r.getMessage() for r in caplog.records]
        assert any("llm_service.done" in m for m in msgs)
