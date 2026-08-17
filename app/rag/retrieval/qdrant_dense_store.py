from __future__ import annotations

import time
from typing import Any

from app.core.logging import get_logger
from app.rag.retrieval.errors import RetrievalConfigurationError
from app.rag.retrieval.models import RetrievedChunk
from app.rag.retrieval.production_profile import ProductionRetrievalConfig
from app.rag.retrieval.provenance import validate_embedding_provenance
from app.rag.retrieval.qdrant_quantization import qdrant_quantization_policy_from_env

logger = get_logger(__name__)


class QdrantDenseStore:
    def __init__(
        self,
        *,
        client: Any,
        embedder: Any,
        config: ProductionRetrievalConfig,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._config = config

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        started = time.perf_counter()

        embed_started = time.perf_counter()
        vector = self._embedder.embed_query(query)
        embedding_latency_ms = _elapsed_ms(embed_started)
        if len(vector) != self._config.profile.embedding_dimension:
            raise RetrievalConfigurationError(
                "BGE-M3 query vector dimension mismatch: "
                f"{len(vector)} != {self._config.profile.embedding_dimension}"
            )

        quantization = qdrant_quantization_policy_from_env()
        qdrant_started = time.perf_counter()
        result = self._client.query_points(
            collection_name=self._config.qdrant_collection,
            query=vector,
            limit=top_k,
            with_payload=True,
            search_params=quantization.to_search_params(),
        )
        qdrant_latency_ms = _elapsed_ms(qdrant_started)

        conversion_started = time.perf_counter()
        chunks = [self._to_chunk(point) for point in result.points]
        conversion_latency_ms = _elapsed_ms(conversion_started)
        total_latency_ms = _elapsed_ms(started)

        quantization_diag = quantization.diagnostics()
        logger.info(
            "[dense_store] search completed embedding_latency_ms=%.3f "
            "qdrant_latency_ms=%.3f conversion_latency_ms=%.3f "
            "total_latency_ms=%.3f top_k=%s query_length=%s "
            "quantization_enabled=%s quantization_ignore=%s",
            embedding_latency_ms,
            qdrant_latency_ms,
            conversion_latency_ms,
            total_latency_ms,
            top_k,
            len(query),
            quantization_diag["quantization_enabled"],
            quantization_diag["quantization_ignore"],
            extra={
                "event_name": "dense_store.search.completed",
                "embedding_latency_ms": round(embedding_latency_ms, 3),
                "qdrant_latency_ms": round(qdrant_latency_ms, 3),
                "conversion_latency_ms": round(conversion_latency_ms, 3),
                "dense_conversion_latency_ms": round(conversion_latency_ms, 3),
                "total_latency_ms": round(total_latency_ms, 3),
                "dense_store_total_latency_ms": round(total_latency_ms, 3),
                "top_k": int(top_k),
                "query_length": len(query),
                **quantization_diag,
            },
        )
        return chunks

    def _to_chunk(self, point: Any) -> RetrievedChunk:
        payload = dict(point.payload or {})
        validate_embedding_provenance(
            payload,
            profile=self._config.profile,
            qdrant_collection=self._config.qdrant_collection,
            bm25_index_id=self._config.bm25_index_id,
        )
        metadata = dict(payload)
        metadata["qdrant_score"] = float(point.score)
        return RetrievedChunk(
            id=str(
                payload.get("original_id")
                or payload.get("chunk_id")
                or payload.get("id")
                or point.id
            ),
            text=str(payload.get("text") or payload.get("chunk_text") or ""),
            score=float(point.score),
            source="dense",
            metadata=metadata,
        )


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
