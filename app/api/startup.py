"""
Startup logic — sestaví živý OrchestratorService nad produkčním BGE-M3
dense+BM25+RRF retrieval stackem.

Production startup is intentionally read-only for retrieval data:
  - Qdrant collection must already exist and have 1024-dimensional vectors
  - BM25 sidecar must already exist
  - BGE-M3 model is lazy-loaded only on first query, not during API startup
"""

from __future__ import annotations

import os
from hashlib import sha256
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
from app.rag.clarification.orchestrator import ClarifyingOrchestratorService
from app.rag.execution.execution_service import ExecutionService
from app.rag.ingest.qdrant_ingest import POINT_ID_SCHEME, point_id_from_original_id
from app.rag.llm.provider_factory import get_text_llm
from app.rag.orchestrator.orchestrator_service import OrchestratorService
from app.rag.planner.planner_service import MockPlannerLLM, PlannerService
from app.rag.retrieval.cached_bge_m3_embedder import build_cached_bge_m3_embedder
from app.rag.retrieval.embedding_cache import EmbeddingCacheBuild
from app.rag.retrieval.bm25_sidecar import Bm25Sidecar
from app.rag.retrieval.hybrid_bge_m3_retriever import HybridBgeM3Retriever
from app.rag.retrieval.production_profile import (
    DEFAULT_QDRANT_COLLECTION,
    ProductionRetrievalConfig,
    production_retrieval_config_from_env,
)
from app.rag.retrieval.qdrant_dense_store import QdrantDenseStore
from app.rag.rewrite.query_rewrite_service import MockTextLLM, QueryRewriteService
from app.rag.synthesis.synthesis_service import MockSynthesisLLM, SynthesisService

logger = get_logger(__name__)

DEFAULT_COLLECTION = DEFAULT_QDRANT_COLLECTION


@dataclass(frozen=True)
class LiveOrchestratorBuild:
    orchestrator: Any
    corpus_version: str
    deferred_ingest: Callable[[], None] | None = None
    ingest_status: str = "idle"
    ingest_message: str | None = None
    embedding_cache_backend: str = "none"
    embedding_cache_enabled: bool = False
    embedding_cache_error: str | None = None


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
            orchestrator=_stub_orchestrator(build_seed_runtime_corpus()),
            corpus_version="seed-corpus",
        )


def _build(qdrant_url: str) -> LiveOrchestratorBuild:
    from qdrant_client import QdrantClient

    config = production_retrieval_config_from_env()
    collection_name = config.qdrant_collection
    logger.info("[startup] connecting to Qdrant at %s", qdrant_url)
    client = QdrantClient(url=qdrant_url, timeout=10)
    _assert_collection_ready(client, collection_name, config.profile.embedding_dimension)
    _assert_bge_m3_model_ready(config)
    _assert_bm25_sidecar_ready(config)

    batches_dir = _resolve_batches_dir()
    corpus_version = _runtime_corpus_version(batches_dir)
    retrieval, embedding_cache_build = _build_production_retrieval(client, config)

    text_llm = _build_text_llm()
    synthesis_llm = text_llm if not isinstance(text_llm, MockTextLLM) else MockSynthesisLLM()
    logger.info(
        "[startup] OrchestratorService ready profile=%s collection=%s bm25=%s",
        config.profile.name,
        collection_name,
        config.bm25_sidecar_path,
    )

    orchestrator = OrchestratorService(
        planner=PlannerService(llm=text_llm),
        execution=ExecutionService(retrieval_service=retrieval),
        synthesis=SynthesisService(llm=synthesis_llm),
        rewrite=QueryRewriteService(llm=text_llm),
    )

    return LiveOrchestratorBuild(
        orchestrator=ClarifyingOrchestratorService(orchestrator),
        corpus_version=corpus_version,
        ingest_status="external",
        ingest_message=(
            "Production retrieval uses prebuilt BGE-M3 Qdrant collection and BM25 sidecar; "
            "API startup performs no ingest or Qdrant writes."
        ),
        embedding_cache_backend=embedding_cache_build.backend,
        embedding_cache_enabled=embedding_cache_build.enabled,
        embedding_cache_error=embedding_cache_build.error,
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


def _assert_collection_ready(client: Any, collection_name: str, expected_dim: int) -> None:
    if not _collection_target_exists(client, collection_name):
        raise RuntimeError(
            f"Production Qdrant collection/alias '{collection_name}' does not exist. "
            "Build and smoke-test the BGE-M3 collection before starting production retrieval."
        )

    info = client.get_collection(collection_name=collection_name)
    vectors = info.config.params.vectors
    actual_dim = int(getattr(vectors, "size", 0) or 0)
    if actual_dim != expected_dim:
        raise RuntimeError(
            f"Production Qdrant collection '{collection_name}' has vector dimension "
            f"{actual_dim}; expected {expected_dim} for BGE-M3."
        )

    count = client.count(collection_name=collection_name).count
    logger.info("[startup] production collection '%s' ready (%d points)", collection_name, count)


def _assert_bge_m3_model_ready(config: ProductionRetrievalConfig) -> None:
    model_path = Path(config.model_path)
    if not model_path.exists():
        raise RuntimeError(
            f"BGE-M3 model path is missing: {model_path}. "
            "Set EMBEDDING_MODEL_NAME to an offline local snapshot (no HuggingFace download). "
            "In Docker, use the huggingface_cache volume snapshot under "
            "/root/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/<hash>."
        )
    if not (model_path / "config.json").is_file():
        raise RuntimeError(
            f"BGE-M3 model path exists but is not a valid model directory (missing config.json): {model_path}"
        )
    logger.info("[startup] BGE-M3 model path ready at %s", model_path)


def _assert_bm25_sidecar_ready(config: ProductionRetrievalConfig) -> None:
    Bm25Sidecar(
        config.bm25_sidecar_path,
        k1=config.profile.bm25_k1,
        b=config.profile.bm25_b,
        index_id=config.bm25_index_id,
    ).assert_ready()


def _build_production_retrieval(
    client: Any,
    config: ProductionRetrievalConfig,
) -> tuple[HybridBgeM3Retriever, EmbeddingCacheBuild]:
    embedder, cache_build = build_cached_bge_m3_embedder(config)
    bm25 = Bm25Sidecar(
        config.bm25_sidecar_path,
        k1=config.profile.bm25_k1,
        b=config.profile.bm25_b,
        index_id=config.bm25_index_id,
    )
    dense = QdrantDenseStore(client=client, embedder=embedder, config=config)
    return HybridBgeM3Retriever(dense_store=dense, bm25_sidecar=bm25, config=config), cache_build


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


def _runtime_corpus_version(batches_dir: Path) -> str:
    hasher = sha256()
    files = sorted(path for path in batches_dir.glob("*.json") if path.name != "manifest.json")
    if not files:
        return "seed-corpus"

    for path in files:
        stat = path.stat()
        hasher.update(path.name.encode("utf-8"))
        hasher.update(str(stat.st_size).encode("ascii"))
        hasher.update(str(stat.st_mtime_ns).encode("ascii"))

    return hasher.hexdigest()[:16]


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


def _collection_name() -> str:
    return os.getenv("QDRANT_COLLECTION_NAME", DEFAULT_COLLECTION)


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


def _stub_orchestrator(runtime_corpus: RuntimeCorpus) -> Any:
    """Minimal non-production orchestrator used only when strict mode is disabled."""

    retrieval = _EmptyRetrievalService()
    return ClarifyingOrchestratorService(
        OrchestratorService(
        planner=PlannerService(llm=MockPlannerLLM()),
        execution=ExecutionService(retrieval_service=retrieval),
        synthesis=SynthesisService(llm=MockSynthesisLLM()),
        )
    )


class _EmptyRetrievalService:
    """No-op retrieval used for local non-strict startup only."""

    def search(self, query: str, top_k: int = 5) -> list:
        del query, top_k
        return []

    def search_dense(self, query: str, top_k: int = 5) -> list:
        del query, top_k
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
