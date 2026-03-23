"""
Unit tests for app.rag.chunking.chunker.

Run:
    pytest tests/rag/test_chunker.py -v
"""

import logging

import pytest

from app.models.search_result import NalusResult
from app.rag.chunking.chunker import (
    TextChunk,
    _clean_text,
    _normalize_case_reference,
    chunk_document,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    full_text: str | None = None,
    case_reference: str | None = "III.ÚS 255/26",
    ecli: str | None = "ECLI:CZ:US:2026:3.US.255.26.1",
    decision_date: str | None = "2026-01-15",
    judge_rapporteur: str | None = "Jan Novák",
    text_url: str | None = "https://nalus.usoud.cz/text/255",
) -> NalusResult:
    return NalusResult(
        result_id=1,
        case_reference=case_reference,
        ecli=ecli,
        judge_rapporteur=judge_rapporteur,
        petitioner=None,
        popular_name=None,
        decision_date=decision_date,
        announcement_date=None,
        filing_date=None,
        publication_date=None,
        text_url=text_url,
        full_text=full_text,
    )


def _long_text(n: int = 5000) -> str:
    """Return a deterministic text of exactly n characters."""
    base = "Ústavní soud rozhodl takto: "
    return (base * (n // len(base) + 1))[:n]


# ---------------------------------------------------------------------------
# _normalize_case_reference
# ---------------------------------------------------------------------------


class TestNormalizeCaseReference:
    def test_spaces_replaced_with_underscores(self) -> None:
        assert _normalize_case_reference("III.ÚS 255/26") == "III.ÚS_255_26"

    def test_slashes_replaced_with_underscores(self) -> None:
        assert _normalize_case_reference("I/II/26") == "I_II_26"

    def test_duplicate_underscores_collapsed(self) -> None:
        assert _normalize_case_reference("A  B//C") == "A_B_C"

    def test_none_returns_unknown(self) -> None:
        assert _normalize_case_reference(None) == "unknown"

    def test_empty_string_returns_unknown(self) -> None:
        assert _normalize_case_reference("") == "unknown"

    def test_no_spaces_or_slashes_unchanged(self) -> None:
        assert _normalize_case_reference("ABC123") == "ABC123"


# ---------------------------------------------------------------------------
# _clean_text
# ---------------------------------------------------------------------------


class TestCleanText:
    def test_strips_leading_trailing_whitespace(self) -> None:
        assert _clean_text("  hello  ") == "hello"

    def test_collapses_multiple_spaces(self) -> None:
        assert _clean_text("a   b   c") == "a b c"

    def test_collapses_tabs(self) -> None:
        assert _clean_text("a\t\tb") == "a b"

    def test_preserves_newlines(self) -> None:
        result = _clean_text("line1\nline2\n\nline3")
        assert "\n" in result
        assert "line1" in result
        assert "line3" in result

    def test_empty_string(self) -> None:
        assert _clean_text("") == ""

    def test_only_whitespace(self) -> None:
        assert _clean_text("   \t  ") == ""


# ---------------------------------------------------------------------------
# chunk_document — no text cases
# ---------------------------------------------------------------------------


class TestChunkDocumentNoText:
    def test_none_full_text_returns_empty_list(self) -> None:
        result = _make_result(full_text=None)
        assert chunk_document(result) == []

    def test_empty_full_text_returns_empty_list(self) -> None:
        result = _make_result(full_text="")
        assert chunk_document(result) == []

    def test_whitespace_only_full_text_returns_empty_list(self) -> None:
        result = _make_result(full_text="   \t\n  ")
        assert chunk_document(result) == []

    def test_invalid_params_raises(self) -> None:
        result = _make_result(full_text="some text")
        with pytest.raises(ValueError, match="chunk_size"):
            chunk_document(result, chunk_size=200, overlap=200)

    def test_overlap_exceeding_chunk_size_raises(self) -> None:
        result = _make_result(full_text="some text")
        with pytest.raises(ValueError):
            chunk_document(result, chunk_size=100, overlap=150)


# ---------------------------------------------------------------------------
# chunk_document — basic chunking
# ---------------------------------------------------------------------------


class TestChunkDocumentBasic:
    def test_short_text_produces_single_chunk(self) -> None:
        result = _make_result(full_text="Krátký text.")
        chunks = chunk_document(result, chunk_size=1500, overlap=200)
        assert len(chunks) == 1
        assert chunks[0].text == "Krátký text."

    def test_text_exactly_chunk_size_produces_one_chunk(self) -> None:
        text = _long_text(1500)
        result = _make_result(full_text=text)
        chunks = chunk_document(result, chunk_size=1500, overlap=200)
        # First chunk covers everything; loop breaks when end == len(text)
        assert chunks[0].text == text.strip()

    def test_long_text_produces_multiple_chunks(self) -> None:
        result = _make_result(full_text=_long_text(5000))
        chunks = chunk_document(result, chunk_size=1500, overlap=200)
        assert len(chunks) > 1

    def test_all_chunks_are_text_chunks(self) -> None:
        result = _make_result(full_text=_long_text(5000))
        for chunk in chunk_document(result):
            assert isinstance(chunk, TextChunk)

    def test_chunk_text_not_empty(self) -> None:
        result = _make_result(full_text=_long_text(5000))
        for chunk in chunk_document(result):
            assert chunk.text.strip() != ""

    def test_chunk_size_respected(self) -> None:
        chunk_size = 500
        result = _make_result(full_text=_long_text(3000))
        chunks = chunk_document(result, chunk_size=chunk_size, overlap=50)
        for chunk in chunks:
            assert len(chunk.text) <= chunk_size

    def test_last_chunk_contains_end_of_text(self) -> None:
        text = _long_text(3000)
        result = _make_result(full_text=text)
        chunks = chunk_document(result, chunk_size=1000, overlap=100)
        last_chunk = chunks[-1]
        clean = text.strip()
        assert clean.endswith(last_chunk.text)


# ---------------------------------------------------------------------------
# chunk_document — overlap
# ---------------------------------------------------------------------------


class TestChunkDocumentOverlap:
    def test_consecutive_chunks_share_overlap_content(self) -> None:
        chunk_size = 100
        overlap = 20
        text = _long_text(300)
        result = _make_result(full_text=text)
        chunks = chunk_document(result, chunk_size=chunk_size, overlap=overlap)

        assert len(chunks) >= 2
        # End of chunk[0] must match start of chunk[1] for exactly `overlap` chars
        end_of_first = chunks[0].text[-overlap:]
        start_of_second = chunks[1].text[:overlap]
        assert end_of_first == start_of_second

    def test_zero_overlap_chunks_do_not_share_content(self) -> None:
        chunk_size = 100
        text = "A" * 300
        result = _make_result(full_text=text)
        chunks = chunk_document(result, chunk_size=chunk_size, overlap=0)

        # With zero overlap step == chunk_size, so chunks are disjoint
        assert len(chunks) == 3
        # No character from chunk[0] tail appears as chunk[1] head overlap
        assert chunks[0].text[-1:] + chunks[1].text[:0] == chunks[0].text[-1:]


# ---------------------------------------------------------------------------
# chunk_document — chunk_index and ordering
# ---------------------------------------------------------------------------


class TestChunkDocumentIndex:
    def test_chunk_index_is_sequential_from_zero(self) -> None:
        result = _make_result(full_text=_long_text(5000))
        chunks = chunk_document(result)
        for expected, chunk in enumerate(chunks):
            assert chunk.chunk_index == expected

    def test_single_chunk_has_index_zero(self) -> None:
        result = _make_result(full_text="Krátký text.")
        chunks = chunk_document(result)
        assert chunks[0].chunk_index == 0

    def test_chunk_order_matches_document_order(self) -> None:
        text = _long_text(5000)
        result = _make_result(full_text=text)
        chunks = chunk_document(result, chunk_size=1000, overlap=100)
        # Reconstructing approximate positions: start of each chunk must be later
        positions = [text.find(c.text[:50]) for c in chunks]
        assert positions == sorted(positions)


# ---------------------------------------------------------------------------
# chunk_document — ID generation
# ---------------------------------------------------------------------------


class TestChunkDocumentIds:
    def test_id_contains_normalized_case_reference(self) -> None:
        result = _make_result(full_text="Text.", case_reference="III.ÚS 255/26")
        chunks = chunk_document(result)
        assert chunks[0].id.startswith("III.ÚS_255_26")

    def test_id_ends_with_chunk_index(self) -> None:
        result = _make_result(full_text=_long_text(5000))
        chunks = chunk_document(result, chunk_size=1000, overlap=100)
        for chunk in chunks:
            assert chunk.id.endswith(f"_{chunk.chunk_index}")

    def test_id_unique_per_chunk(self) -> None:
        result = _make_result(full_text=_long_text(5000))
        chunks = chunk_document(result)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_id_with_none_case_reference_uses_unknown(self) -> None:
        result = _make_result(full_text="Text.", case_reference=None)
        chunks = chunk_document(result)
        assert chunks[0].id.startswith("unknown")


# ---------------------------------------------------------------------------
# chunk_document — metadata propagation
# ---------------------------------------------------------------------------


class TestChunkDocumentMetadata:
    def test_metadata_propagated_to_all_chunks(self) -> None:
        result = _make_result(full_text=_long_text(5000))
        chunks = chunk_document(result, chunk_size=1000, overlap=100)
        for chunk in chunks:
            assert chunk.case_reference == result.case_reference
            assert chunk.ecli == result.ecli
            assert chunk.decision_date == result.decision_date
            assert chunk.judge == result.judge_rapporteur
            assert chunk.text_url == result.text_url

    def test_none_metadata_fields_propagated_as_none(self) -> None:
        result = _make_result(
            full_text="Text pro chunking.",
            ecli=None,
            decision_date=None,
            judge_rapporteur=None,
            text_url=None,
        )
        chunks = chunk_document(result)
        assert chunks[0].ecli is None
        assert chunks[0].decision_date is None
        assert chunks[0].judge is None
        assert chunks[0].text_url is None


# ---------------------------------------------------------------------------
# chunk_document — trace events
# ---------------------------------------------------------------------------


class TestChunkDocumentTrace:
    def test_trace_start_and_done_emitted_for_valid_text(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        result = _make_result(full_text=_long_text(3000))
        with caplog.at_level(logging.DEBUG, logger="app.rag.chunking.chunker"):
            chunk_document(result)

        messages = [r.getMessage() for r in caplog.records]
        assert any("TRACE chunking.start" in m for m in messages)
        assert any("TRACE chunking.done" in m for m in messages)

    def test_trace_start_includes_text_length(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        result = _make_result(full_text=_long_text(3000))
        with caplog.at_level(logging.DEBUG, logger="app.rag.chunking.chunker"):
            chunk_document(result)

        start_msgs = [
            r.getMessage()
            for r in caplog.records
            if "chunking.start" in r.getMessage()
        ]
        assert len(start_msgs) == 1
        assert "text_length=" in start_msgs[0]

    def test_trace_done_includes_num_chunks_and_avg(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        result = _make_result(full_text=_long_text(5000))
        with caplog.at_level(logging.DEBUG, logger="app.rag.chunking.chunker"):
            chunk_document(result, chunk_size=1000, overlap=100)

        done_msgs = [
            r.getMessage()
            for r in caplog.records
            if "chunking.done" in r.getMessage()
        ]
        assert len(done_msgs) == 1
        assert "num_chunks=" in done_msgs[0]
        assert "avg_chunk_len=" in done_msgs[0]

    def test_trace_skip_emitted_when_no_text(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        result = _make_result(full_text=None)
        with caplog.at_level(logging.DEBUG, logger="app.rag.chunking.chunker"):
            chunk_document(result)

        skip_msgs = [
            r.getMessage()
            for r in caplog.records
            if "chunking.skip" in r.getMessage()
        ]
        assert len(skip_msgs) == 1
        assert "no_full_text" in skip_msgs[0]

    def test_no_trace_start_when_text_missing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        result = _make_result(full_text=None)
        with caplog.at_level(logging.DEBUG, logger="app.rag.chunking.chunker"):
            chunk_document(result)

        assert not any("chunking.start" in r.getMessage() for r in caplog.records)
