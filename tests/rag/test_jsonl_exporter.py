"""
Unit tests for app.rag.exporter.jsonl_exporter.

Run:
    pytest tests/rag/test_jsonl_exporter.py -v
"""

import json
import logging
import os

import pytest

from app.rag.chunking.chunker import TextChunk
from app.rag.exporter.jsonl_exporter import export_chunks_to_jsonl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    index: int = 0,
    case_reference: str | None = "III.ÚS 255/26",
    ecli: str | None = "ECLI:CZ:US:2026:3.US.255.26.1",
    decision_date: str | None = "2026-01-15",
    judge: str | None = "Jan Novák",
    text_url: str | None = "https://nalus.usoud.cz/text/255",
    text: str = "Ústavní soud rozhodl takto.",
) -> TextChunk:
    return TextChunk(
        id=f"III.ÚS_255_26_{index}",
        text=text,
        case_reference=case_reference,
        ecli=ecli,
        decision_date=decision_date,
        judge=judge,
        text_url=text_url,
        chunk_index=index,
    )


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# Line count and file creation
# ---------------------------------------------------------------------------


class TestExportLineCount:
    def test_one_chunk_produces_one_line(self, tmp_path) -> None:
        out = str(tmp_path / "out.jsonl")
        export_chunks_to_jsonl([_make_chunk()], out)
        records = _read_jsonl(out)
        assert len(records) == 1

    def test_multiple_chunks_produce_correct_line_count(self, tmp_path) -> None:
        out = str(tmp_path / "out.jsonl")
        chunks = [_make_chunk(i) for i in range(5)]
        export_chunks_to_jsonl(chunks, out)
        records = _read_jsonl(out)
        assert len(records) == 5

    def test_empty_chunks_creates_empty_file(self, tmp_path) -> None:
        out = str(tmp_path / "out.jsonl")
        export_chunks_to_jsonl([], out)
        assert os.path.exists(out)
        assert os.path.getsize(out) == 0

    def test_overwrites_existing_file(self, tmp_path) -> None:
        out = str(tmp_path / "out.jsonl")
        # Write 3 chunks first
        export_chunks_to_jsonl([_make_chunk(i) for i in range(3)], out)
        # Overwrite with 1 chunk
        export_chunks_to_jsonl([_make_chunk(0)], out)
        records = _read_jsonl(out)
        assert len(records) == 1


# ---------------------------------------------------------------------------
# Valid JSON per line
# ---------------------------------------------------------------------------


class TestExportValidJson:
    def test_each_line_is_valid_json(self, tmp_path) -> None:
        out = str(tmp_path / "out.jsonl")
        export_chunks_to_jsonl([_make_chunk(i) for i in range(3)], out)
        with open(out, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    parsed = json.loads(line)  # must not raise
                    assert isinstance(parsed, dict)

    def test_each_line_ends_with_newline(self, tmp_path) -> None:
        out = str(tmp_path / "out.jsonl")
        export_chunks_to_jsonl([_make_chunk(i) for i in range(3)], out)
        with open(out, encoding="utf-8") as fh:
            content = fh.read()
        lines = content.split("\n")
        # Last element after split is empty string (trailing newline)
        assert lines[-1] == ""


# ---------------------------------------------------------------------------
# Record structure
# ---------------------------------------------------------------------------


class TestExportRecordStructure:
    def test_top_level_keys_present(self, tmp_path) -> None:
        out = str(tmp_path / "out.jsonl")
        export_chunks_to_jsonl([_make_chunk()], out)
        record = _read_jsonl(out)[0]
        assert set(record.keys()) == {"id", "text", "metadata"}

    def test_id_matches_chunk_id(self, tmp_path) -> None:
        out = str(tmp_path / "out.jsonl")
        chunk = _make_chunk(0)
        export_chunks_to_jsonl([chunk], out)
        record = _read_jsonl(out)[0]
        assert record["id"] == chunk.id

    def test_text_matches_chunk_text(self, tmp_path) -> None:
        out = str(tmp_path / "out.jsonl")
        chunk = _make_chunk(text="Rozhodnutí soudu.")
        export_chunks_to_jsonl([chunk], out)
        record = _read_jsonl(out)[0]
        assert record["text"] == "Rozhodnutí soudu."

    def test_metadata_keys_present(self, tmp_path) -> None:
        out = str(tmp_path / "out.jsonl")
        export_chunks_to_jsonl([_make_chunk()], out)
        meta = _read_jsonl(out)[0]["metadata"]
        expected_keys = {
            "case_reference", "ecli", "decision_date",
            "judge", "text_url", "chunk_index", "source",
        }
        assert set(meta.keys()) == expected_keys

    def test_source_is_nalus(self, tmp_path) -> None:
        out = str(tmp_path / "out.jsonl")
        export_chunks_to_jsonl([_make_chunk()], out)
        meta = _read_jsonl(out)[0]["metadata"]
        assert meta["source"] == "nalus"

    def test_chunk_index_in_metadata(self, tmp_path) -> None:
        out = str(tmp_path / "out.jsonl")
        chunks = [_make_chunk(i) for i in range(3)]
        export_chunks_to_jsonl(chunks, out)
        records = _read_jsonl(out)
        for i, record in enumerate(records):
            assert record["metadata"]["chunk_index"] == i

    def test_metadata_values_match_chunk_fields(self, tmp_path) -> None:
        out = str(tmp_path / "out.jsonl")
        chunk = _make_chunk()
        export_chunks_to_jsonl([chunk], out)
        meta = _read_jsonl(out)[0]["metadata"]
        assert meta["case_reference"] == chunk.case_reference
        assert meta["ecli"] == chunk.ecli
        assert meta["decision_date"] == chunk.decision_date
        assert meta["judge"] == chunk.judge
        assert meta["text_url"] == chunk.text_url


# ---------------------------------------------------------------------------
# None values
# ---------------------------------------------------------------------------


class TestExportNoneValues:
    def test_none_fields_serialised_as_null(self, tmp_path) -> None:
        out = str(tmp_path / "out.jsonl")
        chunk = _make_chunk(
            ecli=None,
            decision_date=None,
            judge=None,
            text_url=None,
        )
        export_chunks_to_jsonl([chunk], out)
        meta = _read_jsonl(out)[0]["metadata"]
        assert meta["ecli"] is None
        assert meta["decision_date"] is None
        assert meta["judge"] is None
        assert meta["text_url"] is None

    def test_none_case_reference_serialised_as_null(self, tmp_path) -> None:
        out = str(tmp_path / "out.jsonl")
        chunk = _make_chunk(case_reference=None)
        export_chunks_to_jsonl([chunk], out)
        meta = _read_jsonl(out)[0]["metadata"]
        assert meta["case_reference"] is None


# ---------------------------------------------------------------------------
# UTF-8 / Czech characters
# ---------------------------------------------------------------------------


class TestExportEncoding:
    def test_czech_text_preserved(self, tmp_path) -> None:
        out = str(tmp_path / "out.jsonl")
        czech = "Žena cizinka unesla dítě do Ruska. Ústavní soud rozhodl."
        chunk = _make_chunk(text=czech)
        export_chunks_to_jsonl([chunk], out)
        record = _read_jsonl(out)[0]
        assert record["text"] == czech

    def test_czech_metadata_preserved(self, tmp_path) -> None:
        out = str(tmp_path / "out.jsonl")
        chunk = _make_chunk(judge="Kateřina Šimáčková")
        export_chunks_to_jsonl([chunk], out)
        meta = _read_jsonl(out)[0]["metadata"]
        assert meta["judge"] == "Kateřina Šimáčková"

    def test_no_ascii_escaping_in_file(self, tmp_path) -> None:
        out = str(tmp_path / "out.jsonl")
        chunk = _make_chunk(text="případ č. 255")
        export_chunks_to_jsonl([chunk], out)
        raw = open(out, encoding="utf-8").read()
        # ensure_ascii=False → Czech chars written directly, not as \uXXXX
        assert "\\u" not in raw
        assert "případ" in raw


# ---------------------------------------------------------------------------
# Trace events
# ---------------------------------------------------------------------------


class TestExportTrace:
    def test_trace_start_emitted(
        self, tmp_path, caplog: pytest.LogCaptureFixture
    ) -> None:
        out = str(tmp_path / "out.jsonl")
        with caplog.at_level(logging.DEBUG, logger="app.rag.exporter.jsonl_exporter"):
            export_chunks_to_jsonl([_make_chunk()], out)

        messages = [r.getMessage() for r in caplog.records]
        assert any("TRACE export.start" in m for m in messages)

    def test_trace_done_emitted(
        self, tmp_path, caplog: pytest.LogCaptureFixture
    ) -> None:
        out = str(tmp_path / "out.jsonl")
        with caplog.at_level(logging.DEBUG, logger="app.rag.exporter.jsonl_exporter"):
            export_chunks_to_jsonl([_make_chunk()], out)

        messages = [r.getMessage() for r in caplog.records]
        assert any("TRACE export.done" in m for m in messages)

    def test_trace_start_includes_num_chunks(
        self, tmp_path, caplog: pytest.LogCaptureFixture
    ) -> None:
        out = str(tmp_path / "out.jsonl")
        with caplog.at_level(logging.DEBUG, logger="app.rag.exporter.jsonl_exporter"):
            export_chunks_to_jsonl([_make_chunk(i) for i in range(4)], out)

        start_msgs = [
            r.getMessage() for r in caplog.records if "export.start" in r.getMessage()
        ]
        assert len(start_msgs) == 1
        assert "num_chunks=4" in start_msgs[0]

    def test_trace_done_includes_file_size(
        self, tmp_path, caplog: pytest.LogCaptureFixture
    ) -> None:
        out = str(tmp_path / "out.jsonl")
        with caplog.at_level(logging.DEBUG, logger="app.rag.exporter.jsonl_exporter"):
            export_chunks_to_jsonl([_make_chunk()], out)

        done_msgs = [
            r.getMessage() for r in caplog.records if "export.done" in r.getMessage()
        ]
        assert len(done_msgs) == 1
        assert "file_size_bytes=" in done_msgs[0]
        assert "num_written=1" in done_msgs[0]
