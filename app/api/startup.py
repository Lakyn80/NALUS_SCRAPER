"""
Startup logic — sestaví živý OrchestratorService nad Qdrant kolekcí.

Runtime corpus se vždy načítá z lokálních batch JSON souborů pro keyword
retrieval. Synchronizace do Qdrantu je oddělena:
  - pokud je kolekce prázdná nebo už používá stabilní point ID schéma,
    background sync je bezpečný a idempotentní
  - pokud kolekce stále používá legacy count-based point IDs, sync se blokuje
    a je nutné spustit jednorázovou repair/migration cestu
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.data.runtime_corpus import (
    RuntimeCorpus,
    build_runtime_corpus,
    build_seed_runtime_corpus,
    load_results_from_json,
)
from app.rag.execution.execution_service import ExecutionService
from app.rag.ingest.qdrant_ingest import (
    POINT_ID_SCHEME,
    QdrantIngestor,
    point_id_from_original_id,
)
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

DEFAULT_COLLECTION = "nalus"
VECTOR_DIM = 10


@dataclass(frozen=True)
class LiveOrchestratorBuild:
    orchestrator: OrchestratorService
    deferred_ingest: Callable[[], None] | None = None
    ingest_status: str = "idle"
    ingest_message: str | None = None


class _QdrantSearchAdapter:
    """Wrap a real Qdrant client with the legacy .search() signature."""

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


class _QdrantCollectionAdapter:
    """Expose the subset of Qdrant operations the ingestor needs."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def retrieve(
        self,
        *,
        collection_name: str,
        ids: list[str],
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> list[Any]:
        return self._client.retrieve(
            collection_name=collection_name,
            ids=ids,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )

    def upsert(self, *, collection_name: str, points: list[Any]) -> None:
        from qdrant_client.models import PointStruct

        converted = [
            PointStruct(
                id=point.id,
                vector=point.vector,
                payload=point.payload,
            )
            for point in points
        ]
        self._client.upsert(collection_name=collection_name, points=converted)


class _LockedEmbedder:
    """Serialize embedder access so query embedding and background sync do not race."""

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder
        self._lock = threading.Lock()

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            return self._embedder.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            return self._embedder.embed_documents(texts)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._embedder, name)


def build_live_orchestrator(qdrant_url: str | None = None) -> LiveOrchestratorBuild:
    """
    Connect to Qdrant and build a live OrchestratorService.

    In non-strict mode, any failure falls back to the stub orchestrator.
    """

    url = qdrant_url or os.getenv("QDRANT_URL", "http://qdrant:6333")
    strict_real_mode = _read_bool_env("RAG_STRICT_REAL_MODE", default=False)
    try:
        return _build(url)
    except Exception as exc:  # noqa: BLE001
        if strict_real_mode:
            logger.exception("[startup] strict real mode enabled; refusing mock fallback")
            raise
        logger.warning(
            "[startup] Qdrant unavailable (%s) — starting with stub orchestrator",
            exc,
        )
        return LiveOrchestratorBuild(
            orchestrator=_stub_orchestrator(build_seed_runtime_corpus())
        )


def _build(qdrant_url: str) -> LiveOrchestratorBuild:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    collection_name = _collection_name()
    logger.info("[startup] connecting to Qdrant at %s", qdrant_url)
    client = QdrantClient(url=qdrant_url, timeout=10)
    embedder = _LockedEmbedder(_build_embedder())
    vector_dim = _embedder_dim(embedder)

    _ensure_collection(client, collection_name, vector_dim, Distance.COSINE, VectorParams)

    batches_dir = _resolve_batches_dir()
    runtime_corpus = _load_runtime_corpus_from_batches(batches_dir)

    dense = DenseRetriever(
        client=_QdrantSearchAdapter(client),
        collection_name=collection_name,
        embedder=embedder,
    )
    keyword = KeywordRetriever(corpus=list(runtime_corpus.keyword_corpus))
    retrieval = RetrievalService(dense=dense, keyword=keyword)

    text_llm = _build_text_llm()
    synthesis_llm = text_llm if not isinstance(text_llm, MockTextLLM) else MockSynthesisLLM()
    logger.info(
        "[startup] OrchestratorService ready (%d docs, %d chunks)",
        runtime_corpus.document_count,
        len(runtime_corpus.chunks),
    )

    orchestrator = OrchestratorService(
        planner=PlannerService(llm=text_llm),
        execution=ExecutionService(retrieval_service=retrieval),
        synthesis=SynthesisService(llm=synthesis_llm),
        rewrite=QueryRewriteService(llm=text_llm),
    )

    if not runtime_corpus.chunks:
        return LiveOrchestratorBuild(orchestrator=orchestrator)

    if _collection_supports_stable_sync(client, collection_name):
        return LiveOrchestratorBuild(
            orchestrator=orchestrator,
            deferred_ingest=_make_deferred_sync(
                qdrant_url,
                collection_name,
                embedder,
                runtime_corpus.chunks,
            ),
            ingest_status="pending",
            ingest_message=(
                f"{len(runtime_corpus.chunks)} runtime chunks scheduled for idempotent Qdrant sync."
            ),
        )

    message = (
        "Automatic sync blocked because the collection still uses legacy point IDs. "
        "Run scripts/repair_qdrant_collection.py and switch QDRANT_COLLECTION_NAME "
        "to the repaired collection before enabling append sync."
    )
    logger.warning("[startup] %s", message)
    return LiveOrchestratorBuild(
        orchestrator=orchestrator,
        ingest_status="blocked",
        ingest_message=message,
    )


def _ensure_collection(
    client: Any,
    collection_name: str,
    vector_dim: int,
    distance: Any,
    vector_params_factory: Any,
) -> None:
    if not _collection_target_exists(client, collection_name):
        logger.info("[startup] creating collection '%s' (dim=%d)", collection_name, vector_dim)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vector_params_factory(size=vector_dim, distance=distance),
        )
        return

    count = client.count(collection_name=collection_name).count
    logger.info("[startup] collection '%s' exists (%d points)", collection_name, count)


def _load_runtime_corpus_from_batches(batches_dir: Path) -> RuntimeCorpus:
    all_results = []
    for json_file in sorted(batches_dir.glob("*.json")):
        if json_file.name == "manifest.json":
            continue
        all_results.extend(load_results_from_json(json_file))

    if not all_results:
        logger.warning("[startup] no batch files found in %s, using seed corpus", batches_dir)
        return build_seed_runtime_corpus()

    return build_runtime_corpus(all_results, source_label=str(batches_dir))


def _collection_target_exists(client: Any, collection_name: str) -> bool:
    collection_names = {collection.name for collection in client.get_collections().collections}
    if collection_name in collection_names:
        return True

    get_aliases = getattr(client, "get_aliases", None)
    if get_aliases is None:
        return False

    aliases = get_aliases().aliases
    return any(alias.alias_name == collection_name for alias in aliases)


def _collection_supports_stable_sync(
    client: Any,
    collection_name: str,
    sample_size: int = 128,
) -> bool:
    count = client.count(collection_name=collection_name).count
    if count == 0:
        return True

    remaining = sample_size
    offset = None
    while remaining > 0:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=min(remaining, 64),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        for point in points:
            payload = point.payload or {}
            original_id = payload.get("original_id")
            if not original_id:
                return False
            if payload.get("point_id_scheme") != POINT_ID_SCHEME:
                return False
            if str(point.id) != point_id_from_original_id(str(original_id)):
                return False

        remaining -= len(points)
        if offset is None:
            break

    return True


def _make_deferred_sync(
    qdrant_url: str,
    collection_name: str,
    embedder: Any,
    chunks: list[Any],
) -> Callable[[], None]:
    def _run() -> None:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=qdrant_url, timeout=10)
        adapter = _QdrantCollectionAdapter(client)
        ingestor = QdrantIngestor(
            adapter,
            collection_name=collection_name,
            batch_size=_sync_batch_size(),
            embedder=embedder,
        )
        stats = ingestor.ingest_chunks(chunks)
        logger.info(
            "[startup] background Qdrant sync finished inserted=%d updated=%d skipped=%d",
            stats.inserted_points,
            stats.updated_points,
            stats.skipped_points,
        )

    return _run


def _collection_name() -> str:
    return os.getenv("QDRANT_COLLECTION_NAME", DEFAULT_COLLECTION)


def _sync_batch_size() -> int:
    return int(os.getenv("QDRANT_SYNC_BATCH_SIZE", "100"))


def _resolve_batches_dir() -> Path:
    batches_dir = os.getenv("NALUS_BATCHES_DIR")
    if batches_dir:
        return Path(batches_dir)
    results_path = os.getenv("NALUS_RESULTS_PATH")
    if results_path:
        return Path(results_path).parent
    return Path(__file__).resolve().parents[2] / "batches"


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
        logger.warning(
            "[startup] LLM_PROVIDER=%s but LLM_API_KEY is not set — falling back to mock text LLM",
            provider,
        )

    logger.info("[startup] text LLM: MockTextLLM / MockSynthesisLLM fallback")
    return MockTextLLM()


def _stub_orchestrator(runtime_corpus: RuntimeCorpus) -> OrchestratorService:
    """Minimal working orchestrator with keyword-only fallback (no Qdrant needed)."""

    retrieval = RetrievalService(
        dense=DenseRetriever(
            client=_EmptyClient(),
            collection_name=_collection_name(),
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
