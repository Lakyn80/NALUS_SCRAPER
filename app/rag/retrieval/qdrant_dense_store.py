from __future__ import annotations

from typing import Any

from app.rag.retrieval.errors import RetrievalConfigurationError
from app.rag.retrieval.models import RetrievedChunk
from app.rag.retrieval.production_profile import ProductionRetrievalConfig
from app.rag.retrieval.provenance import validate_embedding_provenance


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
        vector = self._embedder.embed_query(query)
        if len(vector) != self._config.profile.embedding_dimension:
            raise RetrievalConfigurationError(
                "BGE-M3 query vector dimension mismatch: "
                f"{len(vector)} != {self._config.profile.embedding_dimension}"
            )

        result = self._client.query_points(
            collection_name=self._config.qdrant_collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        return [self._to_chunk(point) for point in result.points]

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
