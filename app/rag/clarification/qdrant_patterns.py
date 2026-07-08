from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from app.core.logging import get_logger
from app.rag.clarification.models import RuleAssessment
from app.rag.clarification.semantic import (
    DEFAULT_AMBIGUITY_PATTERNS,
    AmbiguityPattern,
    ClarificationPatternStore,
    SignatureEmbedder,
    _ambiguity_types_compatible,
)
from app.rag.retrieval.embedder import BaseEmbedder

logger = get_logger(__name__)

DEFAULT_QDRANT_COLLECTION = "legal_query_clarification_patterns"


class BaseClarificationPatternIndex(Protocol):
    def find_similar_pattern(self, query: str, *, assessment: RuleAssessment) -> AmbiguityPattern | None: ...


@dataclass(frozen=True)
class QdrantPatternHit:
    query_signature: str
    score: float


class InMemoryClarificationPatternIndex:
    """Default semantic index for clarification patterns (not court documents)."""

    def __init__(
        self,
        *,
        embedder: BaseEmbedder | None = None,
        patterns: tuple[AmbiguityPattern, ...] = DEFAULT_AMBIGUITY_PATTERNS,
        similarity_threshold: float = 0.55,
    ) -> None:
        self._store = ClarificationPatternStore(
            embedder=embedder,
            patterns=patterns,
            similarity_threshold=similarity_threshold,
        )

    def find_similar_pattern(self, query: str, *, assessment: RuleAssessment) -> AmbiguityPattern | None:
        return self._store.find_similar_pattern(query, assessment=assessment)


class QdrantClarificationPatternIndex:
    """Optional Qdrant collection for clarification-pattern similarity only."""

    def __init__(
        self,
        *,
        client: Any,
        collection_name: str = DEFAULT_QDRANT_COLLECTION,
        embedder: BaseEmbedder | None = None,
        similarity_threshold: float = 0.55,
        fallback: BaseClarificationPatternIndex | None = None,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._embedder = embedder or SignatureEmbedder()
        self._similarity_threshold = similarity_threshold
        self._fallback = fallback or InMemoryClarificationPatternIndex(embedder=self._embedder)
        self._pattern_lookup = {pattern.query_signature: pattern for pattern in DEFAULT_AMBIGUITY_PATTERNS}
        self._ensure_collection()

    def find_similar_pattern(self, query: str, *, assessment: RuleAssessment) -> AmbiguityPattern | None:
        try:
            vector = self._embedder.embed_query(query)
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=vector,
                limit=3,
                with_payload=True,
                with_vectors=False,
            )
            for point in response.points:
                payload = dict(point.payload or {})
                signature = str(payload.get("query_signature", ""))
                pattern = self._pattern_lookup.get(signature)
                if pattern is None:
                    continue
                if not _ambiguity_types_compatible(assessment.ambiguity_types, pattern.ambiguity_types):
                    continue
                if float(point.score) < self._similarity_threshold:
                    continue
                return pattern
        except Exception as exc:  # noqa: BLE001
            logger.warning("[clarification_patterns] qdrant lookup failed (%s); using fallback", exc)
        return self._fallback.find_similar_pattern(query, assessment=assessment)

    def _ensure_collection(self) -> None:
        if self._client.collection_exists(self._collection_name):
            return
        vector_size = len(self._embedder.embed_query("seed"))
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config={"size": vector_size, "distance": "Cosine"},
        )
        for pattern in DEFAULT_AMBIGUITY_PATTERNS:
            vector = self._embedder.embed_query(pattern.example_query)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, pattern.query_signature))
            self._client.upsert(
                collection_name=self._collection_name,
                points=[
                    {
                        "id": point_id,
                        "vector": vector,
                        "payload": {
                            "query_signature": pattern.query_signature,
                            "ambiguity_types": list(pattern.ambiguity_types),
                            "missing_slots": list(pattern.missing_slots),
                        },
                    }
                ],
            )


def build_clarification_pattern_index(
    *,
    embedder: BaseEmbedder | None = None,
) -> BaseClarificationPatternIndex:
    backend = os.getenv("LEGAL_QUERY_CLARIFICATION_PATTERN_BACKEND", "memory").strip().lower() or "memory"
    resolved_embedder = embedder or SignatureEmbedder()
    fallback = InMemoryClarificationPatternIndex(embedder=resolved_embedder)
    if backend != "qdrant":
        return fallback

    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333").strip()
    collection_name = os.getenv(
        "LEGAL_QUERY_CLARIFICATION_PATTERN_COLLECTION",
        DEFAULT_QDRANT_COLLECTION,
    ).strip()
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=qdrant_url, timeout=10, check_compatibility=False)
        return QdrantClarificationPatternIndex(
            client=client,
            collection_name=collection_name,
            embedder=resolved_embedder,
            fallback=fallback,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[clarification_patterns] qdrant unavailable (%s); using in-memory index", exc)
        return fallback
