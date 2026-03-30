"""
Qdrant ingest layer.

Stores TextChunks as vector points in a Qdrant collection.
The client is injected — no connection is created here, making the
class trivially testable with mocks.

Ingestion is intentionally idempotent:
  - Qdrant point IDs are deterministic UUIDs derived from the logical chunk ID
  - identical chunk payloads are skipped without re-upserting
  - changed payloads update the same logical point in place

Usage:
    from qdrant_client import QdrantClient
    from app.rag.ingest.qdrant_ingest import QdrantIngestor

    client = QdrantClient(url="http://localhost:6333")
    ingestor = QdrantIngestor(client, collection_name="nalus")
    stats = ingestor.ingest_chunks(chunks)
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.rag.chunking.chunker import TextChunk
from app.rag.retrieval.embedder import BaseEmbedder, MockEmbedder

logger = get_logger(__name__)

# Placeholder vector used until a real embedding model is wired up (KROK 7).
_MOCK_VECTOR_DIM = 10
_MOCK_VECTOR: list[float] = [round(i * 0.1, 1) for i in range(_MOCK_VECTOR_DIM)]
_POINT_ID_SCOPE = "nalus-rag-chunk"
POINT_ID_SCHEME = "uuid5_original_id_v1"
_CHECKSUM_KEYS = (
    "original_id",
    "text",
    "case_reference",
    "ecli",
    "decision_date",
    "judge",
    "text_url",
    "chunk_index",
    "document_id",
    "source",
    "point_id_scheme",
)


class _UpsertableClient(Protocol):
    def upsert(self, collection_name: str, points: list[Any]) -> Any: ...


@dataclass(frozen=True)
class IngestPoint:
    """Internal representation of a single Qdrant point before upsert."""

    id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass(frozen=True)
class IngestStats:
    total_chunks: int
    inserted_points: int
    updated_points: int
    skipped_points: int


@dataclass(frozen=True)
class _PreparedPoint:
    point_id: str
    payload: dict[str, Any]
    text: str


class QdrantIngestor:
    """Ingests TextChunks into a Qdrant collection in configurable batches."""

    def __init__(
        self,
        client: _UpsertableClient,
        collection_name: str,
        batch_size: int = 100,
        embedder: BaseEmbedder | None = None,
    ) -> None:
        self._client = client
        self._collection = collection_name
        self._batch_size = batch_size
        self._embedder = embedder or MockEmbedder(dim=_MOCK_VECTOR_DIM)

    def ingest_chunks(self, chunks: list[TextChunk]) -> IngestStats:
        """Upsert all chunks into the collection.

        Identical existing points are skipped. Existing logical points with changed
        content are updated in place under the same deterministic point ID.
        """
        if not chunks:
            return IngestStats(
                total_chunks=0,
                inserted_points=0,
                updated_points=0,
                skipped_points=0,
            )

        trace_event(
            logger,
            "ingest.start",
            num_chunks=len(chunks),
            collection=self._collection,
        )

        inserted_points = 0
        updated_points = 0
        skipped_points = 0

        for batch_chunks in _batched(chunks, self._batch_size):
            prepared = [_prepare_point(chunk) for chunk in batch_chunks]
            existing = self._retrieve_existing_points([point.point_id for point in prepared])

            to_upsert: list[_PreparedPoint] = []
            for point in prepared:
                existing_point = existing.get(point.point_id)
                if existing_point is not None and _existing_matches_payload(
                    existing_point,
                    point.payload,
                ):
                    skipped_points += 1
                    continue

                if existing_point is None:
                    inserted_points += 1
                else:
                    updated_points += 1
                to_upsert.append(point)

            if not to_upsert:
                continue

            vectors = self._embedder.embed_documents([point.text for point in to_upsert])
            if len(vectors) != len(to_upsert):
                raise ValueError("Embedder returned vector count different from chunk count.")

            batch = [
                IngestPoint(
                    id=point.point_id,
                    vector=vector,
                    payload=point.payload,
                )
                for point, vector in zip(to_upsert, vectors, strict=True)
            ]

            trace_event(
                logger,
                "ingest.batch",
                batch_size=len(batch),
                collection=self._collection,
            )
            self._client.upsert(
                collection_name=self._collection,
                points=batch,
            )

        stats = IngestStats(
            total_chunks=len(chunks),
            inserted_points=inserted_points,
            updated_points=updated_points,
            skipped_points=skipped_points,
        )
        trace_event(
            logger,
            "ingest.done",
            num_inserted=stats.inserted_points,
            num_updated=stats.updated_points,
            num_skipped=stats.skipped_points,
            collection=self._collection,
        )
        return stats

    def _retrieve_existing_points(self, point_ids: list[str]) -> dict[str, Any]:
        retrieve = getattr(self._client, "retrieve", None)
        if retrieve is None or not point_ids:
            return {}

        raw_points = retrieve(
            collection_name=self._collection,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )
        if raw_points is None:
            return {}

        try:
            points = list(raw_points)
        except TypeError:
            return {}

        return {str(point.id): point for point in points}


def point_id_from_original_id(original_id: str) -> str:
    """Return a deterministic UUID string for a logical chunk identity."""

    normalized = original_id.strip()
    if not normalized:
        raise ValueError("original_id must not be empty.")
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{_POINT_ID_SCOPE}:{normalized}"))


def point_id_for_chunk(chunk: TextChunk) -> str:
    return point_id_from_original_id(chunk.id)


def payload_checksum(payload: dict[str, Any]) -> str:
    stable_payload = {key: payload.get(key) for key in _CHECKSUM_KEYS}
    encoded = json.dumps(
        stable_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _prepare_point(chunk: TextChunk) -> _PreparedPoint:
    payload = _make_payload(chunk)
    return _PreparedPoint(
        point_id=point_id_for_chunk(chunk),
        payload=payload,
        text=chunk.text,
    )


def _make_point(chunk: TextChunk, vector: list[float] | None = None) -> IngestPoint:
    payload = _make_payload(chunk)
    return IngestPoint(
        id=point_id_for_chunk(chunk),
        vector=vector if vector is not None else _make_vector(chunk),
        payload=payload,
    )


def _make_vector(chunk: TextChunk) -> list[float]:
    """Return a placeholder vector. Replace with real embeddings in KROK 7."""

    return list(_MOCK_VECTOR)


def _make_payload(chunk: TextChunk) -> dict[str, Any]:
    payload = {
        "original_id": chunk.id,
        "text": chunk.text,
        "case_reference": chunk.case_reference,
        "ecli": chunk.ecli,
        "decision_date": chunk.decision_date,
        "judge": chunk.judge,
        "text_url": chunk.text_url,
        "chunk_index": chunk.chunk_index,
        "document_id": chunk.document_id,
        "source": "nalus",
        "point_id_scheme": POINT_ID_SCHEME,
    }
    payload["content_checksum"] = payload_checksum(payload)
    return payload


def _existing_matches_payload(existing_point: Any, expected_payload: dict[str, Any]) -> bool:
    existing_payload = getattr(existing_point, "payload", None) or {}
    existing_original_id = str(
        existing_payload.get("original_id") or getattr(existing_point, "id", "")
    )
    if existing_original_id != expected_payload["original_id"]:
        return False

    existing_checksum = existing_payload.get("content_checksum")
    if not existing_checksum:
        existing_payload = {**existing_payload, "original_id": existing_original_id}
        existing_checksum = payload_checksum(existing_payload)

    return existing_checksum == expected_payload["content_checksum"]


def _batched(items: list[Any], size: int) -> list[list[Any]]:
    """Split a list into consecutive sublists of at most `size` elements."""

    return [items[i : i + size] for i in range(0, len(items), size)]
