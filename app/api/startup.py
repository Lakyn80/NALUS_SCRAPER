"""
Startup logic — ingestuje corpus do Qdrant a sestaví živý OrchestratorService.

Volá se z lifespan eventu v app/api/main.py.
Pokud Qdrant není dostupný, vrátí stub orchestrátor (žádná výjimka nepropadne).
"""

from __future__ import annotations

import os
from typing import Any

from app.core.logging import get_logger
from app.data.corpus import CORPUS
from app.rag.execution.execution_service import ExecutionService
from app.rag.orchestrator.orchestrator_service import OrchestratorService
from app.rag.planner.planner_service import MockPlannerLLM, PlannerService
from app.rag.retrieval.dense_retriever import DenseRetriever
from app.rag.retrieval.embedder import MockEmbedder
from app.rag.retrieval.keyword_retriever import KeywordRetriever
from app.rag.retrieval.retrieval_service import RetrievalService
from app.rag.llm.providers.deepseek import DeepSeekTextLLM
from app.rag.synthesis.synthesis_service import MockSynthesisLLM, SynthesisService

logger = get_logger(__name__)

COLLECTION = "nalus"
VECTOR_DIM = 10


# ---------------------------------------------------------------------------
# Qdrant search adapter
# ---------------------------------------------------------------------------
# New qdrant_client (>=1.9) removed .search() in favour of .query_points().
# DenseRetriever expects the old .search() protocol.
# This adapter bridges the two without touching DenseRetriever.


class _QdrantSearchAdapter:
    """Wraps a real QdrantClient, exposing the .search() signature DenseRetriever expects."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
    ) -> list[Any]:
        result = self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
        )
        return result.points


# ---------------------------------------------------------------------------
# Qdrant upsert adapter
# ---------------------------------------------------------------------------
# QdrantIngestor.upsert() receives IngestPoint dataclasses.
# Real client expects PointStruct (pydantic model).


class _QdrantUpsertAdapter:
    def __init__(self, client: Any, id_map: dict[str, int]) -> None:
        self._client = client
        self._id_map = id_map

    def upsert(self, collection_name: str, points: list[Any]) -> None:
        from qdrant_client.models import PointStruct

        converted = [
            PointStruct(
                id=self._id_map[p.id],
                vector=p.vector,
                payload={**p.payload, "original_id": p.id},
            )
            for p in points
        ]
        self._client.upsert(collection_name=collection_name, points=converted)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_live_orchestrator(qdrant_url: str | None = None) -> OrchestratorService:
    """
    Connect to Qdrant, ingest corpus if needed, return a wired OrchestratorService.

    Falls back to stub orchestrator (empty retrievers) on any failure so the
    API always starts successfully.
    """
    url = qdrant_url or os.getenv("QDRANT_URL", "http://qdrant:6333")
    try:
        return _build(url)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[startup] Qdrant unavailable (%s) — starting with stub orchestrator", exc
        )
        return _stub_orchestrator()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build(qdrant_url: str) -> OrchestratorService:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    logger.info("[startup] connecting to Qdrant at %s", qdrant_url)
    client = QdrantClient(url=qdrant_url, timeout=10)

    # Create collection if it doesn't exist yet
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION not in existing:
        logger.info("[startup] creating collection '%s' (dim=%d)", COLLECTION, VECTOR_DIM)
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        _ingest(client)
    else:
        count = client.count(collection_name=COLLECTION).count
        logger.info("[startup] collection '%s' already exists (%d points)", COLLECTION, count)
        if count == 0:
            _ingest(client)

    # Wire up retrievers
    search_adapter = _QdrantSearchAdapter(client)
    embedder = MockEmbedder(dim=VECTOR_DIM)

    dense = DenseRetriever(
        client=search_adapter,
        collection_name=COLLECTION,
        embedder=embedder,
    )
    keyword = KeywordRetriever(corpus=list(CORPUS))
    retrieval = RetrievalService(dense=dense, keyword=keyword)

    synthesis_llm = _build_synthesis_llm()
    logger.info("[startup] OrchestratorService built with real retrievers (%d docs)", len(CORPUS))

    return OrchestratorService(
        planner=PlannerService(llm=MockPlannerLLM()),
        execution=ExecutionService(retrieval_service=retrieval),
        synthesis=SynthesisService(llm=synthesis_llm),
    )


def _ingest(client: Any) -> None:
    """Upsert all corpus entries into Qdrant."""
    from app.rag.chunking.chunker import TextChunk
    from app.rag.ingest.qdrant_ingest import QdrantIngestor

    logger.info("[startup] ingesting %d documents into Qdrant", len(CORPUS))

    id_map = {str_id: idx for idx, (str_id, _) in enumerate(CORPUS, start=1)}
    upsert_adapter = _QdrantUpsertAdapter(client, id_map)
    ingestor = QdrantIngestor(upsert_adapter, collection_name=COLLECTION)

    chunks = [
        TextChunk(
            id=str_id,
            text=text,
            case_reference=str_id,
            ecli="",
            decision_date="",
            judge="",
            text_url="",
            chunk_index=0,
        )
        for str_id, text in CORPUS
    ]
    ingestor.ingest_chunks(chunks)
    logger.info("[startup] ingest complete")


def _build_synthesis_llm():
    """Return DeepSeekTextLLM when configured, otherwise MockSynthesisLLM."""
    provider = os.getenv("LLM_PROVIDER", "").lower()
    api_key = os.getenv("LLM_API_KEY", "")

    if provider == "deepseek" and api_key:
        logger.info("[startup] synthesis LLM: DeepSeek (deepseek-chat)")
        return DeepSeekTextLLM(api_key=api_key)

    if provider == "deepseek" and not api_key:
        logger.warning("[startup] LLM_PROVIDER=deepseek but LLM_API_KEY is not set — falling back to Mock")

    logger.info("[startup] synthesis LLM: MockSynthesisLLM (set LLM_PROVIDER=deepseek + LLM_API_KEY to enable real LLM)")
    return MockSynthesisLLM()


def _stub_orchestrator() -> OrchestratorService:
    """Minimal working orchestrator with empty retrievers (no Qdrant needed)."""
    retrieval = RetrievalService(
        dense=DenseRetriever(
            client=_EmptyClient(),
            collection_name=COLLECTION,
            embedder=MockEmbedder(dim=VECTOR_DIM),
        ),
        keyword=KeywordRetriever(corpus=list(CORPUS)),  # keyword still works offline
    )
    return OrchestratorService(
        planner=PlannerService(llm=MockPlannerLLM()),
        execution=ExecutionService(retrieval_service=retrieval),
        synthesis=SynthesisService(llm=MockSynthesisLLM()),
    )


class _EmptyClient:
    """No-op Qdrant client used when Qdrant is unreachable."""

    def search(self, collection_name: str, query_vector: list[float], limit: int) -> list:
        return []
