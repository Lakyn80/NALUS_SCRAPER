from __future__ import annotations

from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.rag.query.query_processor import process_query
from app.rag.retrieval.bm25_sidecar import Bm25Sidecar
from app.rag.retrieval.errors import RetrievalDependencyError
from app.rag.retrieval.lexical_support import filter_supported_chunks
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
        if not processed.normalized_query or top_k <= 0:
            return []
        candidate_top_k = self._candidate_top_k(top_k)

        trace_event(
            logger,
            "production_retrieval.start",
            profile=self._config.profile.name,
            mode=self._config.profile.retrieval_mode,
            top_k=top_k,
            candidate_top_k=candidate_top_k,
        )
        dense_results = self._dense_store.search(processed.normalized_query, top_k=candidate_top_k)
        bm25_results = self._bm25.search(processed.normalized_query, top_k=candidate_top_k)

        if not dense_results:
            raise RetrievalDependencyError("Production dense retrieval returned no results.")
        if not bm25_results:
            raise RetrievalDependencyError("Production BM25 sidecar returned no results.")

        fused_candidates = rrf_fuse(
            [dense_results, bm25_results],
            top_k=candidate_top_k,
            rrf_k=self._config.profile.rrf_k,
        )
        fused = (
            filter_supported_chunks(processed.normalized_query, fused_candidates)
            if self._config.lexical_filter_enabled
            else fused_candidates
        )
        deduped = self._dedupe_by_document(fused)
        trace_event(
            logger,
            "production_retrieval.done",
            dense=len(dense_results),
            bm25=len(bm25_results),
            candidates=len(fused_candidates),
            fused=len(fused),
            deduped=len(deduped),
        )
        return deduped[:top_k]

    def search_dense(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Compatibility entry point used by /api/rag/retrieve; returns the production hybrid ranking."""
        return self.search(query, top_k=top_k)

    def _candidate_top_k(self, top_k: int) -> int:
        expanded = max(top_k * self._config.candidate_multiplier, self._config.min_candidate_count)
        return min(expanded, self._config.max_candidate_count)

    def _dedupe_by_document(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        best_by_document: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            key = self._document_key(chunk)
            current = best_by_document.get(key)
            if current is None or self._chunk_quality(chunk) > self._chunk_quality(current):
                best_by_document[key] = chunk

        return sorted(best_by_document.values(), key=lambda chunk: (-chunk.score, chunk.id))

    def _document_key(self, chunk: RetrievedChunk) -> str:
        metadata = chunk.metadata or {}
        for key in ("source_document_id", "ecli", "document_id", "case_reference"):
            value = metadata.get(key)
            if value is not None and str(value).strip():
                return f"{key}:{value}"
        return f"chunk:{chunk.id}"

    def _chunk_quality(self, chunk: RetrievedChunk) -> tuple[float, float, float]:
        metadata = chunk.metadata or {}
        lexical = metadata.get("lexical_support") or {}
        matched_terms = lexical.get("matched_terms") or []
        score_components = metadata.get("score_components") or {}
        bm25_score = float(score_components.get("bm25") or metadata.get("bm25_score") or 0.0)
        return (float(len(matched_terms)), bm25_score, float(chunk.score))
