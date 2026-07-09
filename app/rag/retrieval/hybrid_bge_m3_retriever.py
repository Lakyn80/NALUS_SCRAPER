from __future__ import annotations

from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.rag.query.query_processor import process_query
from app.rag.retrieval.bm25_sidecar import Bm25Sidecar
from app.rag.retrieval.errors import RetrievalDependencyError
from app.rag.retrieval.models import RetrievedChunk
from app.rag.retrieval.production_profile import ProductionRetrievalConfig
from app.rag.retrieval.qdrant_dense_store import QdrantDenseStore
from app.rag.retrieval.rrf import rrf_fuse

logger = get_logger(__name__)


class HybridBgeM3Retriever:
    """Production BGE-M3 dense + BM25 + RRF retrieval service."""

    def __init__(
        self,
        *,
        dense_store: QdrantDenseStore,
        bm25_sidecar: Bm25Sidecar,
        config: ProductionRetrievalConfig,
    ) -> None:
        self._dense_store = dense_store
        self._bm25 = bm25_sidecar
        self._config = config

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        processed = process_query(query)
        if not processed.normalized_query:
            return []

        trace_event(
            logger,
            "production_retrieval.start",
            profile=self._config.profile.name,
            mode=self._config.profile.retrieval_mode,
            top_k=top_k,
        )
        dense_results = self._dense_store.search(processed.normalized_query, top_k=top_k)
        bm25_results = self._bm25.search(processed.normalized_query, top_k=top_k)

        if not dense_results:
            raise RetrievalDependencyError("Production dense retrieval returned no results.")
        if not bm25_results:
            raise RetrievalDependencyError("Production BM25 sidecar returned no results.")

        fused = rrf_fuse(
            [dense_results, bm25_results],
            top_k=top_k,
            rrf_k=self._config.profile.rrf_k,
        )
        trace_event(
            logger,
            "production_retrieval.done",
            dense=len(dense_results),
            bm25=len(bm25_results),
            fused=len(fused),
        )
        return fused

    def search_dense(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Compatibility entry point used by /api/rag/retrieve; returns the production hybrid ranking."""
        return self.search(query, top_k=top_k)
