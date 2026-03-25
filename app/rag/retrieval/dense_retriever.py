"""
Dense retriever backed by Qdrant vector search.

The Qdrant client and embedder are injected — no connections or models
are created here, making the class trivially testable with mocks.

Usage:
    from qdrant_client import QdrantClient
    from app.rag.retrieval.dense_retriever import DenseRetriever
    from app.rag.retrieval.embedder import MockEmbedder   # or real embedder

    retriever = DenseRetriever(
        client=QdrantClient(url="http://localhost:6333"),
        collection_name="nalus",
        embedder=MockEmbedder(),
    )
    results = retriever.retrieve("únos dítěte", top_k=5)
"""

from typing import Any, Protocol

from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.rag.retrieval.base import BaseRetriever
from app.rag.retrieval.embedder import BaseEmbedder
from app.rag.retrieval.models import RetrievedChunk

logger = get_logger(__name__)


class _SearchableClient(Protocol):
    """Structural interface for the Qdrant client search method.

    Avoids importing qdrant_client at module level so the module is
    importable (and testable) without the package installed.
    """

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
    ) -> list[Any]: ...


class DenseRetriever(BaseRetriever):
    """Retrieves chunks from Qdrant using dense vector similarity."""

    def __init__(
        self,
        client: _SearchableClient,
        collection_name: str,
        embedder: BaseEmbedder,
    ) -> None:
        self._client = client
        self._collection = collection_name
        self._embedder = embedder

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        trace_event(
            logger,
            "dense.start",
            query=query,
            top_k=top_k,
            collection=self._collection,
        )

        vector = self._embedder.embed_query(query)

        raw = self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=top_k,
        )

        results = [_to_chunk(point) for point in raw]

        trace_event(logger, "dense.done", num_results=len(results))
        return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_chunk(point: Any) -> RetrievedChunk:
    """Map a Qdrant ScoredPoint (or any duck-typed equivalent) to RetrievedChunk."""
    payload: dict[str, Any] = point.payload or {}
    return RetrievedChunk(
        id=str(payload.get("original_id") or point.id),
        text=payload.get("text", ""),
        score=float(point.score),
        source="dense",
    )
