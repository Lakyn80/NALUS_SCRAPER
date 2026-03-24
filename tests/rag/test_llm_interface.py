"""
Unit tests for app.rag.llm — BaseLLM interface + MockLLM.

Run:
    pytest tests/rag/test_llm_interface.py -v
"""

import pytest

from app.rag.llm.base import BaseLLM
from app.rag.llm.mock_llm import MockLLM
from app.rag.llm.models import LLMInput, LLMOutput
from app.rag.retrieval.models import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(id: str, text: str = "Ústavní soud rozhodl.", score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(id=id, text=text, score=score, source="dense")


def _input(*chunk_ids: str, query: str = "únos dítěte") -> LLMInput:
    return LLMInput(query=query, chunks=[_chunk(cid) for cid in chunk_ids])


# ---------------------------------------------------------------------------
# BaseLLM — abstract contract
# ---------------------------------------------------------------------------


class TestBaseLLMAbstract:
    def test_cannot_instantiate_base_llm(self) -> None:
        with pytest.raises(TypeError):
            BaseLLM()  # type: ignore[abstract]

    def test_subclass_without_generate_cannot_be_instantiated(self) -> None:
        class Incomplete(BaseLLM):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_subclass_with_generate_can_be_instantiated(self) -> None:
        class Minimal(BaseLLM):
            def generate(self, data: LLMInput) -> LLMOutput:
                return LLMOutput(answer="", reasoning="", sources=[], confidence=0.0)

        assert isinstance(Minimal(), BaseLLM)


# ---------------------------------------------------------------------------
# MockLLM — return type
# ---------------------------------------------------------------------------


class TestMockLLMReturnType:
    def test_returns_llm_output_instance(self) -> None:
        result = MockLLM().generate(_input("a"))
        assert isinstance(result, LLMOutput)

    def test_answer_is_str(self) -> None:
        result = MockLLM().generate(_input("a"))
        assert isinstance(result.answer, str)

    def test_reasoning_is_str(self) -> None:
        result = MockLLM().generate(_input("a"))
        assert isinstance(result.reasoning, str)

    def test_sources_is_list(self) -> None:
        result = MockLLM().generate(_input("a"))
        assert isinstance(result.sources, list)

    def test_confidence_is_float(self) -> None:
        result = MockLLM().generate(_input("a"))
        assert isinstance(result.confidence, float)


# ---------------------------------------------------------------------------
# MockLLM — answer content
# ---------------------------------------------------------------------------


class TestMockLLMAnswer:
    def test_answer_contains_query(self) -> None:
        result = MockLLM().generate(_input("a", query="únos dítěte Rusko"))
        assert "únos dítěte Rusko" in result.answer

    def test_reasoning_not_empty(self) -> None:
        result = MockLLM().generate(_input("a"))
        assert result.reasoning.strip() != ""

    def test_confidence_is_0_5(self) -> None:
        result = MockLLM().generate(_input("a"))
        assert result.confidence == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# MockLLM — sources
# ---------------------------------------------------------------------------


class TestMockLLMSources:
    def test_sources_match_first_three_chunk_ids(self) -> None:
        result = MockLLM().generate(_input("A", "B", "C", "D", "E"))
        assert result.sources == ["A", "B", "C"]

    def test_sources_at_most_three(self) -> None:
        result = MockLLM().generate(_input(*[str(i) for i in range(10)]))
        assert len(result.sources) <= 3

    def test_sources_exactly_three_when_enough_chunks(self) -> None:
        result = MockLLM().generate(_input("X", "Y", "Z"))
        assert len(result.sources) == 3

    def test_sources_fewer_than_three_when_not_enough_chunks(self) -> None:
        result = MockLLM().generate(_input("only"))
        assert result.sources == ["only"]

    def test_empty_chunks_gives_empty_sources(self) -> None:
        data = LLMInput(query="test", chunks=[])
        result = MockLLM().generate(data)
        assert result.sources == []


# ---------------------------------------------------------------------------
# MockLLM — determinism
# ---------------------------------------------------------------------------


class TestMockLLMDeterminism:
    def test_same_input_produces_same_output(self) -> None:
        llm = MockLLM()
        data = _input("A", "B", query="haagská úmluva")
        assert llm.generate(data) == llm.generate(data)

    def test_different_query_gives_different_answer(self) -> None:
        llm = MockLLM()
        r1 = llm.generate(_input("a", query="únos dítěte"))
        r2 = llm.generate(_input("a", query="rodinné právo"))
        assert r1.answer != r2.answer


# ---------------------------------------------------------------------------
# LLMInput / LLMOutput — dataclass integrity
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_llm_input_stores_query_and_chunks(self) -> None:
        chunks = [_chunk("a")]
        data = LLMInput(query="test", chunks=chunks)
        assert data.query == "test"
        assert data.chunks is chunks

    def test_llm_output_fields(self) -> None:
        out = LLMOutput(answer="a", reasoning="r", sources=["x"], confidence=0.9)
        assert out.answer == "a"
        assert out.reasoning == "r"
        assert out.sources == ["x"]
        assert out.confidence == pytest.approx(0.9)
