"""
Unit tests for app.rag.answer.answer_service.

Run:
    pytest tests/rag/test_answer_service.py -v
"""

import logging

import pytest

from app.rag.answer.answer_service import (
    AnswerResult,
    AnswerService,
    _extract_cases,
    _extract_excerpts,
)
from app.rag.retrieval.models import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(
    id: str,
    score: float = 0.80,
    text: str = "Ústavní soud rozhodl.",
    source: str = "dense",
) -> RetrievedChunk:
    return RetrievedChunk(id=id, text=text, score=score, source=source)


def _long_text(n: int = 500) -> str:
    base = "Ústavní soud rozhodl takto: "
    return (base * (n // len(base) + 1))[:n]


# ---------------------------------------------------------------------------
# _extract_cases (pure helper)
# ---------------------------------------------------------------------------


class TestExtractCases:
    def test_returns_chunk_ids(self) -> None:
        chunks = [_chunk("III.US_255_26_0"), _chunk("II.US_100_24_1")]
        assert _extract_cases(chunks) == ["III.US_255_26_0", "II.US_100_24_1"]

    def test_deduplicates_ids(self) -> None:
        chunks = [_chunk("A"), _chunk("A"), _chunk("B")]
        result = _extract_cases(chunks)
        assert result.count("A") == 1

    def test_max_five_cases(self) -> None:
        chunks = [_chunk(str(i)) for i in range(10)]
        assert len(_extract_cases(chunks)) <= 5

    def test_preserves_order(self) -> None:
        ids = ["C", "A", "B"]
        result = _extract_cases([_chunk(i) for i in ids])
        assert result == ids

    def test_empty_input_returns_empty(self) -> None:
        assert _extract_cases([]) == []

    def test_exactly_five_cases(self) -> None:
        chunks = [_chunk(str(i)) for i in range(5)]
        assert len(_extract_cases(chunks)) == 5


# ---------------------------------------------------------------------------
# _extract_excerpts (pure helper)
# ---------------------------------------------------------------------------


class TestExtractExcerpts:
    def test_returns_up_to_three_excerpts(self) -> None:
        chunks = [_chunk(str(i)) for i in range(5)]
        assert len(_extract_excerpts(chunks)) <= 3

    def test_exactly_three_when_enough_chunks(self) -> None:
        chunks = [_chunk(str(i)) for i in range(3)]
        assert len(_extract_excerpts(chunks)) == 3

    def test_fewer_than_three_when_not_enough_chunks(self) -> None:
        chunks = [_chunk("a"), _chunk("b")]
        assert len(_extract_excerpts(chunks)) == 2

    def test_excerpts_truncated_to_300_chars(self) -> None:
        chunk = _chunk("a", text=_long_text(500))
        excerpts = _extract_excerpts([chunk])
        assert len(excerpts[0]) <= 300

    def test_short_text_not_padded(self) -> None:
        chunk = _chunk("a", text="Krátký text.")
        assert _extract_excerpts([chunk])[0] == "Krátký text."

    def test_empty_text_chunk_skipped(self) -> None:
        chunks = [_chunk("a", text=""), _chunk("b", text="Relevantní text.")]
        excerpts = _extract_excerpts(chunks)
        assert "Relevantní text." in excerpts

    def test_empty_input_returns_empty(self) -> None:
        assert _extract_excerpts([]) == []


# ---------------------------------------------------------------------------
# AnswerService.generate — return type
# ---------------------------------------------------------------------------


class TestAnswerServiceReturnType:
    def test_returns_answer_result_instance(self) -> None:
        result = AnswerService().generate("test", [_chunk("a")])
        assert isinstance(result, AnswerResult)

    def test_query_preserved(self) -> None:
        result = AnswerService().generate("únos dítěte", [_chunk("a")])
        assert result.query == "únos dítěte"

    def test_top_cases_is_list(self) -> None:
        result = AnswerService().generate("q", [_chunk("a")])
        assert isinstance(result.top_cases, list)

    def test_excerpts_is_list(self) -> None:
        result = AnswerService().generate("q", [_chunk("a")])
        assert isinstance(result.excerpts, list)


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestAnswerServiceEmpty:
    def test_empty_chunks_returns_no_results_summary(self) -> None:
        result = AnswerService().generate("únos dítěte", [])
        assert "Nenalezeny" in result.summary

    def test_empty_chunks_gives_empty_top_cases(self) -> None:
        result = AnswerService().generate("q", [])
        assert result.top_cases == []

    def test_empty_chunks_gives_empty_excerpts(self) -> None:
        result = AnswerService().generate("q", [])
        assert result.excerpts == []


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestAnswerServiceSummary:
    def test_summary_contains_query(self) -> None:
        query = "žena cizinka unesla dítě do Ruska"
        result = AnswerService().generate(query, [_chunk("a")])
        assert query in result.summary

    def test_summary_not_empty_with_results(self) -> None:
        result = AnswerService().generate("test", [_chunk("a")])
        assert result.summary.strip() != ""

    def test_summary_references_ustavni_soud(self) -> None:
        result = AnswerService().generate("test", [_chunk("a")])
        assert "Ústavního soudu" in result.summary


# ---------------------------------------------------------------------------
# top_cases
# ---------------------------------------------------------------------------


class TestAnswerServiceTopCases:
    def test_top_cases_uses_chunk_ids(self) -> None:
        chunks = [_chunk("III.US_255_26_0"), _chunk("II.US_100_24_1")]
        result = AnswerService().generate("q", chunks)
        assert "III.US_255_26_0" in result.top_cases
        assert "II.US_100_24_1" in result.top_cases

    def test_top_cases_at_most_five(self) -> None:
        chunks = [_chunk(str(i)) for i in range(10)]
        result = AnswerService().generate("q", chunks)
        assert len(result.top_cases) <= 5

    def test_top_cases_deduplicated(self) -> None:
        chunks = [_chunk("same"), _chunk("same"), _chunk("other")]
        result = AnswerService().generate("q", chunks)
        assert result.top_cases.count("same") == 1

    def test_top_cases_preserves_score_order(self) -> None:
        chunks = [_chunk("first"), _chunk("second"), _chunk("third")]
        result = AnswerService().generate("q", chunks)
        assert result.top_cases[0] == "first"


# ---------------------------------------------------------------------------
# excerpts
# ---------------------------------------------------------------------------


class TestAnswerServiceExcerpts:
    def test_excerpts_at_most_three(self) -> None:
        chunks = [_chunk(str(i)) for i in range(6)]
        result = AnswerService().generate("q", chunks)
        assert len(result.excerpts) <= 3

    def test_excerpts_at_most_300_chars_each(self) -> None:
        chunks = [_chunk("a", text=_long_text(600))]
        result = AnswerService().generate("q", chunks)
        assert len(result.excerpts[0]) <= 300

    def test_excerpts_contain_chunk_text_prefix(self) -> None:
        text = "Ústavní soud rozhodl ve věci mezinárodního únosu dítěte."
        result = AnswerService().generate("q", [_chunk("a", text=text)])
        assert result.excerpts[0] == text


# ---------------------------------------------------------------------------
# Trace events
# ---------------------------------------------------------------------------


class TestAnswerServiceTrace:
    def test_start_and_done_emitted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="app.rag.answer.answer_service"):
            AnswerService().generate("únos dítěte", [_chunk("a")])

        msgs = [r.getMessage() for r in caplog.records]
        assert any("TRACE answer.start" in m for m in msgs)
        assert any("TRACE answer.done" in m for m in msgs)

    def test_start_includes_query_and_num_chunks(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        chunks = [_chunk(str(i)) for i in range(3)]
        with caplog.at_level(logging.DEBUG, logger="app.rag.answer.answer_service"):
            AnswerService().generate("únos dítěte", chunks)

        start = next(
            r.getMessage() for r in caplog.records
            if "answer.start" in r.getMessage()
        )
        assert "únos dítěte" in start
        assert "num_chunks=3" in start

    def test_done_includes_num_cases_and_excerpts(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        chunks = [_chunk(str(i)) for i in range(4)]
        with caplog.at_level(logging.DEBUG, logger="app.rag.answer.answer_service"):
            AnswerService().generate("q", chunks)

        done = next(
            r.getMessage() for r in caplog.records
            if "answer.done" in r.getMessage()
        )
        assert "num_cases=" in done
        assert "num_excerpts=" in done

    def test_empty_input_still_emits_done(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="app.rag.answer.answer_service"):
            AnswerService().generate("q", [])

        msgs = [r.getMessage() for r in caplog.records]
        assert any("answer.done" in m for m in msgs)
