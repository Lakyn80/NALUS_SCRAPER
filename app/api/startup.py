"""
Startup logic — ingestuje runtime corpus do Qdrant a sestaví živý OrchestratorService.

Volá se z lifespan eventu v app/api/main.py.
Pokud Qdrant není dostupný, vrátí stub orchestrátor (žádná výjimka nepropadne).
"""

from __future__ import annotations

import os
from typing import Any

from app.core.logging import get_logger
from app.data.runtime_corpus import (
    RuntimeCorpus,
    build_seed_runtime_corpus,
    load_runtime_corpus,
)
from app.rag.execution.execution_service import ExecutionService
from app.rag.llm.provider_factory import get_text_llm
from app.rag.orchestrator.orchestrator_service import OrchestratorService
from app.rag.planner.planner_service import MockPlannerLLM, PlannerService
from app.rag.retrieval.dense_retriever import DenseRetriever
from app.rag.retrieval.embedder import MockEmbedder, SentenceTransformersEmbedder
from app.rag.retrieval.keyword_retriever import KeywordRetriever
from app.rag.retrieval.retrieval_service import RetrievalService
from app.rag.rewrite.query_rewrite_service import MockTextLLM, QueryRewriteService
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
    Connect to Qdrant, ingest runtime corpus if needed, return a wired OrchestratorService.

    Falls back to stub orchestrator (empty retrievers) on any failure so the
    API always starts successfully.
    """
    url = qdrant_url or os.getenv("QDRANT_URL", "http://qdrant:6333")
    runtime_corpus = _load_runtime_corpus_from_env()
    strict_real_mode = _read_bool_env("RAG_STRICT_REAL_MODE", default=False)
    try:
        return _build(url, runtime_corpus)
    except Exception as exc:  # noqa: BLE001
        if strict_real_mode:
            logger.exception("[startup] strict real mode enabled; refusing mock fallback")
            raise
        logger.warning(
            "[startup] Qdrant unavailable (%s) — starting with stub orchestrator", exc
        )
        return _stub_orchestrator(runtime_corpus)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build(qdrant_url: str, runtime_corpus: RuntimeCorpus) -> OrchestratorService:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    logger.info("[startup] connecting to Qdrant at %s", qdrant_url)
    client = QdrantClient(url=qdrant_url, timeout=10)
    embedder = _build_embedder()
    vector_dim = _embedder_dim(embedder)

    _ensure_collection_populated(
        client=client,
        runtime_corpus=runtime_corpus,
        vector_dim=vector_dim,
        distance=Distance.COSINE,
        vector_params_factory=VectorParams,
        embedder=embedder,
    )

    # Wire up retrievers
    dense = DenseRetriever(
        client=_QdrantSearchAdapter(client),
        collection_name=COLLECTION,
        embedder=embedder,
    )
    keyword = KeywordRetriever(corpus=list(runtime_corpus.keyword_corpus))
    retrieval = RetrievalService(dense=dense, keyword=keyword)

    text_llm = _build_text_llm()
    synthesis_llm = text_llm if not isinstance(text_llm, MockTextLLM) else MockSynthesisLLM()
    logger.info(
        "[startup] OrchestratorService built with real retrievers (%d docs, %d chunks, source=%s)",
        runtime_corpus.document_count,
        len(runtime_corpus.chunks),
        runtime_corpus.source_label,
    )

    return OrchestratorService(
        planner=PlannerService(llm=text_llm),
        execution=ExecutionService(retrieval_service=retrieval),
        synthesis=SynthesisService(llm=synthesis_llm),
        rewrite=QueryRewriteService(llm=text_llm),
    )


def _ensure_collection_populated(
    client: Any,
    runtime_corpus: RuntimeCorpus,
    vector_dim: int,
    distance: Any,
    vector_params_factory: Any,
    embedder: Any,
) -> None:
    existing = {c.name for c in client.get_collections().collections}
    expected_points = len(runtime_corpus.chunks)

    if COLLECTION not in existing:
        logger.info("[startup] creating collection '%s' (dim=%d)", COLLECTION, vector_dim)
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=vector_params_factory(size=vector_dim, distance=distance),
        )
        _ingest(client, runtime_corpus, embedder)
        return

    count = client.count(collection_name=COLLECTION).count
    current_vector_dim = _current_vector_dim(client)
    logger.info("[startup] collection '%s' already exists (%d points)", COLLECTION, count)
    if count == expected_points and current_vector_dim == vector_dim:
        return

    logger.info(
        "[startup] recreating collection '%s' because collection shape differs"
        " (existing_points=%d expected_points=%d existing_dim=%s expected_dim=%d)",
        COLLECTION,
        count,
        expected_points,
        current_vector_dim,
        vector_dim,
    )
    client.delete_collection(collection_name=COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=vector_params_factory(size=vector_dim, distance=distance),
    )
    _ingest(client, runtime_corpus, embedder)


def _ingest(client: Any, runtime_corpus: RuntimeCorpus, embedder: Any) -> None:
    """Upsert all runtime corpus chunks into Qdrant."""
    from app.rag.ingest.qdrant_ingest import QdrantIngestor

    logger.info(
        "[startup] ingesting %d documents / %d chunks into Qdrant from %s",
        runtime_corpus.document_count,
        len(runtime_corpus.chunks),
        runtime_corpus.source_label,
    )

    id_map = {chunk.id: idx for idx, chunk in enumerate(runtime_corpus.chunks, start=1)}
    upsert_adapter = _QdrantUpsertAdapter(client, id_map)
    ingestor = QdrantIngestor(
        upsert_adapter,
        collection_name=COLLECTION,
        embedder=embedder,
    )

    ingestor.ingest_chunks(runtime_corpus.chunks)
    logger.info("[startup] ingest complete")


def _build_text_llm():
    """Return a real BaseTextLLM when configured, otherwise a mock fallback."""
    provider = os.getenv("LLM_PROVIDER", "").lower()
    api_key = os.getenv("LLM_API_KEY", "")
    strict_real_mode = _read_bool_env("RAG_STRICT_REAL_MODE", default=False)

    if provider and api_key:
        logger.info("[startup] text LLM provider=%s", provider)
        return get_text_llm(provider, api_key)

    if strict_real_mode:
        raise RuntimeError(
            "RAG_STRICT_REAL_MODE=1 requires LLM_PROVIDER and LLM_API_KEY."
        )

    if provider and not api_key:
        logger.warning("[startup] LLM_PROVIDER=%s but LLM_API_KEY is not set — falling back to mock text LLM", provider)

    logger.info("[startup] text LLM: MockTextLLM / MockSynthesisLLM fallback")
    return MockTextLLM()


def _stub_orchestrator(runtime_corpus: RuntimeCorpus) -> OrchestratorService:
    """Minimal working orchestrator with keyword-only fallback (no Qdrant needed)."""
    retrieval = RetrievalService(
        dense=DenseRetriever(
            client=_EmptyClient(),
            collection_name=COLLECTION,
            embedder=MockEmbedder(dim=VECTOR_DIM),
        ),
        keyword=KeywordRetriever(corpus=list(runtime_corpus.keyword_corpus)),
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


def _load_runtime_corpus_from_env() -> RuntimeCorpus:
    results_path = os.getenv("NALUS_RESULTS_PATH")
    chunk_size = int(os.getenv("NALUS_CHUNK_SIZE", "1500"))
    overlap = int(os.getenv("NALUS_CHUNK_OVERLAP", "200"))

    try:
        runtime_corpus = load_runtime_corpus(
            results_path=results_path,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[startup] failed to load runtime corpus (%s) — falling back to seed corpus",
            exc,
        )
        runtime_corpus = build_seed_runtime_corpus()

    logger.info(
        "[startup] runtime corpus loaded source=%s docs=%d chunks=%d",
        runtime_corpus.source_label,
        runtime_corpus.document_count,
        len(runtime_corpus.chunks),
    )
    return runtime_corpus


def _read_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(f"{name} must be a boolean value.")


def _build_embedder() -> SentenceTransformersEmbedder:
    model_name = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    normalize = _read_bool_env("EMBEDDING_NORMALIZE", default=True)
    logger.info("[startup] loading embedder model=%s", model_name)
    return SentenceTransformersEmbedder(
        model_name=model_name,
        batch_size=batch_size,
        normalize_embeddings=normalize,
    )


def _embedder_dim(embedder: Any) -> int:
    dim = getattr(embedder, "dim", None)
    if dim:
        return int(dim)
    return len(embedder.embed_query("dimension probe"))


def _current_vector_dim(client: Any) -> int | None:
    try:
        info = client.get_collection(COLLECTION)
    except Exception:  # noqa: BLE001
        return None

    vectors = info.config.params.vectors
    size = getattr(vectors, "size", None)
    if size is not None:
        return int(size)
    return None
