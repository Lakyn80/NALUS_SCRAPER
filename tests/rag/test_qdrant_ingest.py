"""
Unit tests for app.rag.ingest.qdrant_ingest.

All tests use local doubles or MagicMock — no real Qdrant server needed.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.rag.chunking.chunker import TextChunk
from app.rag.ingest.qdrant_ingest import (
    POINT_ID_SCHEME,
    IngestPoint,
    IngestStats,
    QdrantIngestor,
    _batched,
    _make_payload,
    _make_point,
    payload_checksum,
    point_id_for_chunk,
    point_id_from_original_id,
)
from app.rag.retrieval.embedder import BaseEmbedder


def _make_chunk(
    index: int = 0,
    *,
    chunk_id: str | None = None,
    case_reference: str | None = "III.ÚS 255/26",
    ecli: str | None = "ECLI:CZ:US:2026:3.US.255.26.1",
    decision_date: str | None = "2026-01-15",
    judge: str | None = "Jan Novák",
    text_url: str | None = "https://nalus.usoud.cz/text/255",
    text: str = "Ústavní soud rozhodl takto.",
) -> TextChunk:
    return TextChunk(
        id=chunk_id or f"III.ÚS_255_26_{index}",
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
    client.retrieve.return_value = []
    return QdrantIngestor(client, "test-collection", batch_size=batch_size), client


class _FixedEmbedder(BaseEmbedder):
    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def embed_query(self, query: str) -> list[float]:
        return list(self._vector)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vector) for _ in texts]


class _StatefulClient:
    def __init__(self) -> None:
        self.points: dict[str, IngestPoint] = {}
        self.upsert_calls: list[list[IngestPoint]] = []
        self.retrieve_calls: list[list[str]] = []

    def retrieve(
        self,
        *,
        collection_name: str,
        ids: list[str],
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> list[SimpleNamespace]:
        del collection_name, with_payload, with_vectors
        self.retrieve_calls.append(list(ids))
        return [
            SimpleNamespace(id=point_id, payload=self.points[point_id].payload)
            for point_id in ids
            if point_id in self.points
        ]

    def upsert(self, *, collection_name: str, points: list[IngestPoint]) -> None:
        del collection_name
        self.upsert_calls.append(list(points))
        for point in points:
            self.points[point.id] = point


class TestBatched:
    def test_single_batch_when_items_fit(self) -> None:
        assert _batched([1, 2, 3], size=10) == [[1, 2, 3]]

    def test_multiple_batches(self) -> None:
        assert _batched(list(range(5)), size=2) == [[0, 1], [2, 3], [4]]

    def test_empty_list_returns_empty(self) -> None:
        assert _batched([], size=10) == []


class TestDeterministicIds:
    def test_point_id_is_stable_for_same_original_id(self) -> None:
        assert point_id_from_original_id("III.ÚS_255_26_0") == point_id_from_original_id(
            "III.ÚS_255_26_0"
        )

    def test_different_original_ids_produce_different_point_ids(self) -> None:
        assert point_id_from_original_id("A") != point_id_from_original_id("B")

    def test_make_point_uses_deterministic_qdrant_id(self) -> None:
        chunk = _make_chunk(index=2)
        point = _make_point(chunk)
        assert point.id == point_id_for_chunk(chunk)


class TestMakePayload:
    def test_required_keys_present(self) -> None:
        payload = _make_payload(_make_chunk())
        assert set(payload.keys()) == {
            "original_id",
            "text",
            "case_reference",
            "ecli",
            "decision_date",
            "judge",
            "text_url",
            "chunk_index",
            "source",
            "point_id_scheme",
            "content_checksum",
        }

    def test_original_id_matches_chunk_id(self) -> None:
        chunk = _make_chunk(index=3)
        assert _make_payload(chunk)["original_id"] == chunk.id

    def test_point_id_scheme_is_versioned(self) -> None:
        assert _make_payload(_make_chunk())["point_id_scheme"] == POINT_ID_SCHEME

    def test_checksum_matches_payload(self) -> None:
        payload = _make_payload(_make_chunk())
        assert payload["content_checksum"] == payload_checksum(payload)

    def test_metadata_values_match_chunk(self) -> None:
        chunk = _make_chunk()
        payload = _make_payload(chunk)
        assert payload["case_reference"] == chunk.case_reference
        assert payload["ecli"] == chunk.ecli
        assert payload["decision_date"] == chunk.decision_date
        assert payload["judge"] == chunk.judge
        assert payload["text_url"] == chunk.text_url


class TestMakePoint:
    def test_returns_ingest_point(self) -> None:
        assert isinstance(_make_point(_make_chunk()), IngestPoint)

    def test_vector_is_list_of_floats(self) -> None:
        point = _make_point(_make_chunk())
        assert all(isinstance(v, float) for v in point.vector)

    def test_payload_contains_original_id(self) -> None:
        chunk = _make_chunk(text="Test text.")
        assert _make_point(chunk).payload["original_id"] == chunk.id


class TestQdrantIngestorClientCalls:
    def test_empty_chunks_returns_zero_stats_and_no_upsert(self) -> None:
        ingestor, client = _mock_ingestor()
        stats = ingestor.ingest_chunks([])
        assert stats == IngestStats(0, 0, 0, 0)
        client.upsert.assert_not_called()

    def test_non_empty_calls_upsert(self) -> None:
        ingestor, client = _mock_ingestor()
        ingestor.ingest_chunks([_make_chunk()])
        assert client.upsert.called

    def test_upsert_called_with_correct_collection_name(self) -> None:
        ingestor, client = _mock_ingestor()
        ingestor.ingest_chunks([_make_chunk()])
        assert client.upsert.call_args[1]["collection_name"] == "test-collection"

    def test_upsert_points_are_ingest_point_instances(self) -> None:
        ingestor, client = _mock_ingestor()
        ingestor.ingest_chunks([_make_chunk(), _make_chunk(index=1)])
        for point in client.upsert.call_args[1]["points"]:
            assert isinstance(point, IngestPoint)


class TestQdrantIngestorBatching:
    def test_single_batch_when_chunks_fit(self) -> None:
        ingestor, client = _mock_ingestor(batch_size=10)
        ingestor.ingest_chunks([_make_chunk(i) for i in range(5)])
        assert client.upsert.call_count == 1

    def test_multiple_batches_when_chunks_exceed_batch_size(self) -> None:
        ingestor, client = _mock_ingestor(batch_size=3)
        ingestor.ingest_chunks([_make_chunk(i) for i in range(7)])
        assert client.upsert.call_count == 3

    def test_each_batch_does_not_exceed_batch_size(self) -> None:
        ingestor, client = _mock_ingestor(batch_size=4)
        ingestor.ingest_chunks([_make_chunk(i) for i in range(10)])
        for call in client.upsert.call_args_list:
            assert len(call[1]["points"]) <= 4


class TestIdempotentIngestion:
    def test_reingesting_same_chunks_creates_zero_duplicates(self) -> None:
        client = _StatefulClient()
        ingestor = QdrantIngestor(client, "test-collection", batch_size=2)
        chunks = [_make_chunk(i) for i in range(3)]

        first = ingestor.ingest_chunks(chunks)
        second = ingestor.ingest_chunks(chunks)

        assert first.inserted_points == 3
        assert first.updated_points == 0
        assert first.skipped_points == 0
        assert second.inserted_points == 0
        assert second.updated_points == 0
        assert second.skipped_points == 3
        assert len(client.points) == 3
        assert len(client.upsert_calls) == 2

    def test_mixed_old_and_new_chunks_only_inserts_new(self) -> None:
        client = _StatefulClient()
        ingestor = QdrantIngestor(client, "test-collection", batch_size=10)
        existing = [_make_chunk(0), _make_chunk(1)]
        mixed = [_make_chunk(0), _make_chunk(1), _make_chunk(2)]

        ingestor.ingest_chunks(existing)
        stats = ingestor.ingest_chunks(mixed)

        assert stats.inserted_points == 1
        assert stats.updated_points == 0
        assert stats.skipped_points == 2
        assert len(client.points) == 3

    def test_changed_text_updates_same_logical_point_in_place(self) -> None:
        client = _StatefulClient()
        ingestor = QdrantIngestor(client, "test-collection", batch_size=10)
        original = _make_chunk(0, text="Původní text")
        changed = _make_chunk(0, text="Nový text")

        first = ingestor.ingest_chunks([original])
        second = ingestor.ingest_chunks([changed])
        stored = client.points[point_id_for_chunk(changed)]

        assert first.inserted_points == 1
        assert second.inserted_points == 0
        assert second.updated_points == 1
        assert second.skipped_points == 0
        assert len(client.points) == 1
        assert stored.payload["text"] == "Nový text"
        assert stored.payload["original_id"] == original.id


class TestQdrantIngestorMetadata:
    def test_custom_embedder_vector_used(self) -> None:
        client = MagicMock()
        client.retrieve.return_value = []
        embedder = _FixedEmbedder([0.9, 0.8, 0.7])
        ingestor = QdrantIngestor(
            client,
            "test-collection",
            batch_size=10,
            embedder=embedder,
        )

        ingestor.ingest_chunks([_make_chunk()])

        point = client.upsert.call_args[1]["points"][0]
        assert point.vector == [0.9, 0.8, 0.7]


class TestQdrantIngestorTrace:
    def test_trace_start_and_done_emitted(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        ingestor, _ = _mock_ingestor()
        with caplog.at_level(logging.DEBUG, logger="app.rag.ingest.qdrant_ingest"):
            ingestor.ingest_chunks([_make_chunk()])

        messages = [record.getMessage() for record in caplog.records]
        assert any("TRACE ingest.start" in message for message in messages)
        assert any("TRACE ingest.done" in message for message in messages)

    def test_no_trace_emitted_for_empty_input(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        ingestor, _ = _mock_ingestor()
        with caplog.at_level(logging.DEBUG, logger="app.rag.ingest.qdrant_ingest"):
            ingestor.ingest_chunks([])

        assert caplog.records == []
