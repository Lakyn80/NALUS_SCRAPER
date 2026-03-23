"""
Unit tests for app.rag.ingest.qdrant_ingest.

All tests use MagicMock as the Qdrant client — no real server needed.

Run:
    pytest tests/rag/test_qdrant_ingest.py -v
"""

import logging
from unittest.mock import MagicMock, call

import pytest

from app.rag.chunking.chunker import TextChunk
from app.rag.ingest.qdrant_ingest import (
    IngestPoint,
    QdrantIngestor,
    _batched,
    _make_payload,
    _make_point,
)


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


def _mock_ingestor(batch_size: int = 100) -> tuple[QdrantIngestor, MagicMock]:
    client = MagicMock()
    return QdrantIngestor(client, "test-collection", batch_size=batch_size), client


# ---------------------------------------------------------------------------
# _batched helper
# ---------------------------------------------------------------------------


class TestBatched:
    def test_single_batch_when_items_fit(self) -> None:
        result = _batched([1, 2, 3], size=10)
        assert result == [[1, 2, 3]]

    def test_multiple_batches(self) -> None:
        result = _batched(list(range(5)), size=2)
        assert result == [[0, 1], [2, 3], [4]]

    def test_exact_multiple(self) -> None:
        result = _batched(list(range(4)), size=2)
        assert result == [[0, 1], [2, 3]]

    def test_empty_list_returns_empty(self) -> None:
        assert _batched([], size=10) == []

    def test_batch_size_one(self) -> None:
        result = _batched([1, 2, 3], size=1)
        assert result == [[1], [2], [3]]


# ---------------------------------------------------------------------------
# _make_payload
# ---------------------------------------------------------------------------


class TestMakePayload:
    def test_required_keys_present(self) -> None:
        chunk = _make_chunk()
        payload = _make_payload(chunk)
        expected = {
            "text", "case_reference", "ecli", "decision_date",
            "judge", "text_url", "chunk_index", "source",
        }
        assert set(payload.keys()) == expected

    def test_source_is_nalus(self) -> None:
        assert _make_payload(_make_chunk())["source"] == "nalus"

    def test_text_matches_chunk(self) -> None:
        chunk = _make_chunk(text="Rozhodnutí soudu.")
        assert _make_payload(chunk)["text"] == "Rozhodnutí soudu."

    def test_chunk_index_matches(self) -> None:
        chunk = _make_chunk(index=3)
        assert _make_payload(chunk)["chunk_index"] == 3

    def test_none_fields_preserved_as_none(self) -> None:
        chunk = _make_chunk(ecli=None, decision_date=None, judge=None, text_url=None)
        payload = _make_payload(chunk)
        assert payload["ecli"] is None
        assert payload["decision_date"] is None
        assert payload["judge"] is None
        assert payload["text_url"] is None

    def test_metadata_values_match_chunk(self) -> None:
        chunk = _make_chunk()
        payload = _make_payload(chunk)
        assert payload["case_reference"] == chunk.case_reference
        assert payload["ecli"] == chunk.ecli
        assert payload["decision_date"] == chunk.decision_date
        assert payload["judge"] == chunk.judge
        assert payload["text_url"] == chunk.text_url


# ---------------------------------------------------------------------------
# _make_point
# ---------------------------------------------------------------------------


class TestMakePoint:
    def test_returns_ingest_point(self) -> None:
        assert isinstance(_make_point(_make_chunk()), IngestPoint)

    def test_id_matches_chunk_id(self) -> None:
        chunk = _make_chunk(index=2)
        assert _make_point(chunk).id == chunk.id

    def test_vector_is_list_of_floats(self) -> None:
        point = _make_point(_make_chunk())
        assert isinstance(point.vector, list)
        assert all(isinstance(v, float) for v in point.vector)

    def test_vector_not_empty(self) -> None:
        assert len(_make_point(_make_chunk()).vector) > 0

    def test_payload_has_text(self) -> None:
        chunk = _make_chunk(text="Test text.")
        assert _make_point(chunk).payload["text"] == "Test text."


# ---------------------------------------------------------------------------
# QdrantIngestor.ingest_chunks — client interaction
# ---------------------------------------------------------------------------


class TestQdrantIngestorClientCalls:
    def test_empty_chunks_does_not_call_upsert(self) -> None:
        ingestor, client = _mock_ingestor()
        ingestor.ingest_chunks([])
        client.upsert.assert_not_called()

    def test_non_empty_calls_upsert_at_least_once(self) -> None:
        ingestor, client = _mock_ingestor()
        ingestor.ingest_chunks([_make_chunk()])
        assert client.upsert.called

    def test_upsert_called_with_correct_collection_name(self) -> None:
        ingestor, client = _mock_ingestor()
        ingestor.ingest_chunks([_make_chunk()])
        _, kwargs = client.upsert.call_args
        assert kwargs["collection_name"] == "test-collection"

    def test_upsert_points_are_ingest_point_instances(self) -> None:
        ingestor, client = _mock_ingestor()
        ingestor.ingest_chunks([_make_chunk(), _make_chunk(index=1)])
        _, kwargs = client.upsert.call_args
        for point in kwargs["points"]:
            assert isinstance(point, IngestPoint)

    def test_all_chunks_passed_as_points(self) -> None:
        ingestor, client = _mock_ingestor()
        chunks = [_make_chunk(i) for i in range(5)]
        ingestor.ingest_chunks(chunks)
        all_points = [
            p for c in client.upsert.call_args_list for p in c[1]["points"]
        ]
        assert len(all_points) == 5

    def test_point_ids_match_chunk_ids(self) -> None:
        ingestor, client = _mock_ingestor()
        chunks = [_make_chunk(i) for i in range(3)]
        ingestor.ingest_chunks(chunks)
        all_points = [
            p for c in client.upsert.call_args_list for p in c[1]["points"]
        ]
        point_ids = {p.id for p in all_points}
        chunk_ids = {c.id for c in chunks}
        assert point_ids == chunk_ids


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


class TestQdrantIngestorBatching:
    def test_single_batch_when_chunks_fit(self) -> None:
        ingestor, client = _mock_ingestor(batch_size=10)
        ingestor.ingest_chunks([_make_chunk(i) for i in range(5)])
        assert client.upsert.call_count == 1

    def test_multiple_batches_when_chunks_exceed_batch_size(self) -> None:
        ingestor, client = _mock_ingestor(batch_size=3)
        ingestor.ingest_chunks([_make_chunk(i) for i in range(7)])
        assert client.upsert.call_count == 3  # batches: [3, 3, 1]

    def test_each_batch_does_not_exceed_batch_size(self) -> None:
        batch_size = 4
        ingestor, client = _mock_ingestor(batch_size=batch_size)
        ingestor.ingest_chunks([_make_chunk(i) for i in range(10)])
        for c in client.upsert.call_args_list:
            assert len(c[1]["points"]) <= batch_size

    def test_total_points_equals_chunk_count(self) -> None:
        ingestor, client = _mock_ingestor(batch_size=3)
        chunks = [_make_chunk(i) for i in range(8)]
        ingestor.ingest_chunks(chunks)
        total = sum(len(c[1]["points"]) for c in client.upsert.call_args_list)
        assert total == 8


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------


class TestQdrantIngestorMetadata:
    def _get_all_points(self, client: MagicMock) -> list[IngestPoint]:
        return [p for c in client.upsert.call_args_list for p in c[1]["points"]]

    def test_case_reference_in_payload(self) -> None:
        ingestor, client = _mock_ingestor()
        ingestor.ingest_chunks([_make_chunk(case_reference="I.ÚS 1/24")])
        point = self._get_all_points(client)[0]
        assert point.payload["case_reference"] == "I.ÚS 1/24"

    def test_ecli_in_payload(self) -> None:
        ingestor, client = _mock_ingestor()
        ecli = "ECLI:CZ:US:2026:3.US.255.26.1"
        ingestor.ingest_chunks([_make_chunk(ecli=ecli)])
        point = self._get_all_points(client)[0]
        assert point.payload["ecli"] == ecli

    def test_none_metadata_preserved(self) -> None:
        ingestor, client = _mock_ingestor()
        ingestor.ingest_chunks([_make_chunk(ecli=None, judge=None)])
        point = self._get_all_points(client)[0]
        assert point.payload["ecli"] is None
        assert point.payload["judge"] is None

    def test_source_always_nalus(self) -> None:
        ingestor, client = _mock_ingestor()
        ingestor.ingest_chunks([_make_chunk(i) for i in range(3)])
        for point in self._get_all_points(client):
            assert point.payload["source"] == "nalus"


# ---------------------------------------------------------------------------
# Trace events
# ---------------------------------------------------------------------------


class TestQdrantIngestorTrace:
    def test_trace_start_and_done_emitted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        ingestor, _ = _mock_ingestor()
        with caplog.at_level(logging.DEBUG, logger="app.rag.ingest.qdrant_ingest"):
            ingestor.ingest_chunks([_make_chunk()])

        messages = [r.getMessage() for r in caplog.records]
        assert any("TRACE ingest.start" in m for m in messages)
        assert any("TRACE ingest.done" in m for m in messages)

    def test_trace_batch_emitted_for_each_batch(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        ingestor, _ = _mock_ingestor(batch_size=2)
        with caplog.at_level(logging.DEBUG, logger="app.rag.ingest.qdrant_ingest"):
            ingestor.ingest_chunks([_make_chunk(i) for i in range(5)])

        batch_msgs = [
            r.getMessage()
            for r in caplog.records
            if "ingest.batch" in r.getMessage()
        ]
        assert len(batch_msgs) == 3  # batches: [2, 2, 1]

    def test_trace_start_includes_num_chunks_and_collection(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        ingestor, _ = _mock_ingestor()
        with caplog.at_level(logging.DEBUG, logger="app.rag.ingest.qdrant_ingest"):
            ingestor.ingest_chunks([_make_chunk(i) for i in range(4)])

        start = next(
            r.getMessage()
            for r in caplog.records
            if "ingest.start" in r.getMessage()
        )
        assert "num_chunks=4" in start
        assert "collection=" in start

    def test_no_trace_emitted_for_empty_input(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        ingestor, _ = _mock_ingestor()
        with caplog.at_level(logging.DEBUG, logger="app.rag.ingest.qdrant_ingest"):
            ingestor.ingest_chunks([])

        assert len(caplog.records) == 0
