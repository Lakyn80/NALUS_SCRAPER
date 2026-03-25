"""
Qdrant ingest layer.

Stores TextChunks as vector points in a Qdrant collection.
The client is injected — no connection is created here, making the
class trivially testable with a mock.

Vector generation is a placeholder (fixed mock vector).
Replace _make_vector() with a real embedding call in KROK 7.

Usage:
    from qdrant_client import QdrantClient
    from app.rag.ingest.qdrant_ingest import QdrantIngestor

    client = QdrantClient(url="http://localhost:6333")
    ingestor = QdrantIngestor(client, collection_name="nalus")
    ingestor.ingest_chunks(chunks)
"""

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


# ---------------------------------------------------------------------------
# Minimal structural typing for the Qdrant client.
# Avoids importing qdrant_client at the module level, so the code is
# testable without the package installed.
# ---------------------------------------------------------------------------


class _UpsertableClient(Protocol):
    def upsert(self, collection_name: str, points: list[Any]) -> Any: ...


@dataclass
class IngestPoint:
    """Internal representation of a single Qdrant point before upsert."""
    id: str
    vector: list[float]
    payload: dict[str, Any]


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

    def ingest_chunks(self, chunks: list[TextChunk]) -> None:
        """Upsert all chunks into the collection.

        No-op when chunks is empty.
        """
        if not chunks:
            return

        trace_event(
            logger,
            "ingest.start",
            num_chunks=len(chunks),
            collection=self._collection,
        )

        total_inserted = 0

        for batch_chunks in _batched(chunks, self._batch_size):
            vectors = self._embedder.embed_documents([chunk.text for chunk in batch_chunks])
            if len(vectors) != len(batch_chunks):
                raise ValueError("Embedder returned vector count different from chunk count.")
            batch = [
                _make_point(chunk, vector)
                for chunk, vector in zip(batch_chunks, vectors, strict=True)
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
            total_inserted += len(batch)

        trace_event(
            logger,
            "ingest.done",
            num_inserted=total_inserted,
            collection=self._collection,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_point(chunk: TextChunk, vector: list[float] | None = None) -> IngestPoint:
    return IngestPoint(
        id=chunk.id,
        vector=vector if vector is not None else _make_vector(chunk),
        payload=_make_payload(chunk),
    )


def _make_vector(chunk: TextChunk) -> list[float]:
    """Return a placeholder vector.  Replace with real embeddings in KROK 7."""
    return list(_MOCK_VECTOR)


def _make_payload(chunk: TextChunk) -> dict[str, Any]:
    return {
        "text": chunk.text,
        "case_reference": chunk.case_reference,
        "ecli": chunk.ecli,
        "decision_date": chunk.decision_date,
        "judge": chunk.judge,
        "text_url": chunk.text_url,
        "chunk_index": chunk.chunk_index,
        "source": "nalus",
    }


def _batched(items: list[Any], size: int) -> list[list[Any]]:
    """Split a list into consecutive sublists of at most `size` elements."""
    return [items[i : i + size] for i in range(0, len(items), size)]
