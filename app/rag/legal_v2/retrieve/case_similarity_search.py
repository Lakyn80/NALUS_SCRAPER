"""Stage 1 case-similarity search: deterministic QuerySpec + hybrid retrieval (no LLM).

Public search API is async-first. Blocking hybrid/CE work runs via
``asyncio.to_thread``; ColBERT uses the existing async backend.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.rag.legal_v2.identity import is_valid_ecli, normalize_ecli
from app.rag.legal_v2.indexing import LEGAL_V2_PROFILE
from app.rag.legal_v2.query_spec import QuerySpecV2, build_query_spec_v2
from app.rag.legal_v2.retrieve.retrieval_profiles import (
    RetrievalStage,
    build_retrieval_stage,
    ce_index_binding,
    colbert_master_allow_enabled,
    fast_index_binding,
    resolve_retrieval_profile,
)
from app.rag.legal_v2.retrieve.retriever import (
    LegalV2HybridRetriever,
    LegalV2RetrievalResult,
    LegalV2RetrieverConfig,
    build_live_legal_v2_retriever,
    legal_v2_retriever_config_from_env,
)
from app.rag.retrieval.bge_m3_embedder import BgeM3Embedder
from app.rag.retrieval.production_profile import ProductionRetrievalConfig
from app.rag.retrieval.errors import RetrievalConfigurationError

logger = get_logger(__name__)

STAGE_1_RETRIEVAL = RetrievalStage.HYBRID_RRF_STAGE_1.value
DEFAULT_RESULT_LIMIT = 10
MAX_RESULT_LIMIT = 50
MAX_QUERY_LENGTH = 8000
MIN_QUERY_LENGTH = 1
_TRUTHY = {"1", "true", "yes", "on"}
_WARMUP_QUERY = "warmup"


def case_similarity_stage1_enabled() -> bool:
    raw = os.getenv("NALUS_LEGAL_V2_CASE_SIMILARITY_ENABLED", "").strip().lower()
    if raw in _TRUTHY:
        return True
    # Allow Stage 1 when the broader Legal v2 search flag is on.
    return os.getenv("NALUS_LEGAL_V2_SEARCH_ENABLED", "").strip().lower() in _TRUTHY


def stage1_debug_allowed() -> bool:
    return os.getenv("NALUS_LEGAL_V2_DEBUG", "").strip().lower() in _TRUTHY


def stage1_warmup_on_start_enabled() -> bool:
    """When enabled, API preloads BGE-M3 + BM25 after boot so first search is warm."""
    return os.getenv("NALUS_LEGAL_V2_STAGE1_WARMUP_ON_START", "").strip().lower() in _TRUTHY


def default_result_limit() -> int:
    return _bounded_int(
        os.getenv("NALUS_LEGAL_V2_DEFAULT_RESULT_LIMIT"),
        default=DEFAULT_RESULT_LIMIT,
        minimum=1,
        maximum=MAX_RESULT_LIMIT,
    )


def max_result_limit() -> int:
    return _bounded_int(
        os.getenv("NALUS_LEGAL_V2_MAX_RESULT_LIMIT"),
        default=MAX_RESULT_LIMIT,
        minimum=1,
        maximum=MAX_RESULT_LIMIT,
    )


@dataclass(frozen=True)
class Stage1Passage:
    text: str
    chunk_id: str
    section: str | None = None
    page: int | None = None
    score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_rank: int | None = None
    retrieval_channels: tuple[str, ...] = ()
    chunk_position: int | None = None


@dataclass(frozen=True)
class Stage1DocumentResult:
    rank: int
    document_id: str
    canonical_document_id: str
    ecli: str
    court: str | None
    case_number: str | None
    decision_date: str | None
    document_type: str | None
    score: float
    relevant_passages: list[Stage1Passage]
    source_document_id: str | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    stage1_rank: int | None = None
    stage1_score: float | None = None
    ce_rank: int | None = None
    ce_score: float | None = None
    chunk_evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class Stage1SearchResult:
    query: str
    result_count: int
    retrieval_stage: str
    results: list[Stage1DocumentResult]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseSimilarityStage1Runtime:
    retriever: LegalV2HybridRetriever
    config: LegalV2RetrieverConfig
    embedder: Any | None = None
    ready: bool = False
    ready_error: str | None = None
    model_loaded: bool = False
    bm25_loaded: bool = False
    warmup_status: str = "cold"  # cold | warming | warm | failed | skipped
    warmup_error_type: str | None = None
    warmup_latency_ms: float | None = None
    # Dual Slice-4 bindings: FAST=A; BALANCED/PRECISE share B contextual.
    ce_retriever: LegalV2HybridRetriever | None = None
    ce_config: LegalV2RetrieverConfig | None = None
    # Lazy ColBERT handle for BALANCED (populated on first balanced search).
    colbert_retriever: Any | None = None
    colbert_config: Any | None = None
    colbert_ready: bool = False
    colbert_error_type: str | None = None

    def _uses_b_indexes(self, profile_id: str) -> bool:
        return profile_id in {"precise", "balanced", "ce7"}

    def retriever_for_profile(self, profile_id: str) -> LegalV2HybridRetriever:
        if self._uses_b_indexes(profile_id):
            if self.ce_retriever is None:
                raise RetrievalConfigurationError(
                    "B contextual retriever is not configured "
                    "(expected Slice 4 B indexes for BALANCED/PRECISE)."
                )
            return self.ce_retriever
        return self.retriever

    def config_for_profile(self, profile_id: str) -> LegalV2RetrieverConfig:
        if self._uses_b_indexes(profile_id):
            if self.ce_config is None:
                raise RetrievalConfigurationError(
                    "B contextual retriever config is not configured "
                    "(expected Slice 4 B indexes for BALANCED/PRECISE)."
                )
            return self.ce_config
        return self.config


_runtime: CaseSimilarityStage1Runtime | None = None
_runtime_lock = Lock()
_colbert_init_lock = asyncio.Lock()


def reset_case_similarity_stage1_runtime_for_tests() -> None:
    global _runtime
    with _runtime_lock:
        _runtime = None


def _retriever_config_for_index(
    *,
    base: LegalV2RetrieverConfig,
    qdrant_collection: str,
    bm25_index_id: str,
    bm25_sidecar_path: Path,
) -> LegalV2RetrieverConfig:
    from dataclasses import replace

    return replace(
        base,
        qdrant_collection=qdrant_collection,
        bm25_index_id=bm25_index_id,
        bm25_sidecar_path=bm25_sidecar_path,
    )


def _require_collection(client: Any, collection: str) -> Any:
    try:
        info = client.get_collection(collection)
    except Exception as exc:  # noqa: BLE001
        raise RetrievalConfigurationError(
            f"Qdrant collection unavailable: {collection}"
        ) from exc
    if int(getattr(info, "points_count", 0) or 0) <= 0:
        raise RetrievalConfigurationError(f"Qdrant collection is empty: {collection}")
    return info


def _resolve_profile_bm25_path(binding_path: Path) -> Path:
    """Map relative Slice-4 BM25 paths onto the Docker /app/storage mount when present."""
    if binding_path.is_absolute():
        return binding_path
    env_path = os.getenv("NALUS_LEGAL_V2_BM25_SIDECAR_PATH", "").strip()
    if env_path.startswith("/app/storage"):
        return Path("/app") / binding_path.as_posix()
    sidecar_dir = os.getenv("NALUS_LEGAL_V2_BM25_SIDECAR_DIR", "").strip()
    if sidecar_dir:
        return Path(sidecar_dir) / binding_path.name
    return binding_path


def get_case_similarity_stage1_runtime() -> CaseSimilarityStage1Runtime:
    global _runtime
    if _runtime is not None:
        return _runtime
    with _runtime_lock:
        if _runtime is not None:
            return _runtime
        # Shared knobs (dense/BM25 candidate sizes, model path) from env;
        # profile-specific index bindings come from Slice-4 pin constants.
        base = legal_v2_retriever_config_from_env()
        fast_bind = fast_index_binding()
        ce_bind = ce_index_binding()
        fast_config = _retriever_config_for_index(
            base=base,
            qdrant_collection=fast_bind.qdrant_collection,
            bm25_index_id=fast_bind.bm25_index_id,
            bm25_sidecar_path=_resolve_profile_bm25_path(fast_bind.bm25_sidecar_path),
        )
        ce_config = _retriever_config_for_index(
            base=base,
            qdrant_collection=ce_bind.qdrant_collection,
            bm25_index_id=ce_bind.bm25_index_id,
            bm25_sidecar_path=_resolve_profile_bm25_path(ce_bind.bm25_sidecar_path),
        )
        fast_config.validate()
        ce_config.validate()
        if not fast_config.bm25_sidecar_path.exists():
            raise RetrievalConfigurationError(
                f"FAST BM25 sidecar missing: {fast_config.bm25_sidecar_path}"
            )
        if not ce_config.bm25_sidecar_path.exists():
            raise RetrievalConfigurationError(
                f"CE BM25 sidecar missing: {ce_config.bm25_sidecar_path}"
            )
        qdrant_module = __import__("qdrant_client", fromlist=["QdrantClient"])
        client = qdrant_module.QdrantClient(
            url=os.getenv("QDRANT_URL", "http://qdrant:6333"),
            timeout=10,
        )
        fast_info = _require_collection(client, fast_config.qdrant_collection)
        ce_info = _require_collection(client, ce_config.qdrant_collection)
        prod_config = ProductionRetrievalConfig(
            profile=LEGAL_V2_PROFILE,
            qdrant_collection=fast_config.qdrant_collection,
            bm25_sidecar_path=fast_config.bm25_sidecar_path,
            bm25_index_id=fast_config.bm25_index_id,
            model_path=fast_config.model_path,
            local_files_only=True,
            trust_remote_code=False,
            device=os.getenv("EMBEDDING_DEVICE", "cpu"),
            candidate_multiplier=1,
            min_candidate_count=1,
            max_candidate_count=max(
                fast_config.dense_candidate_chunks, fast_config.bm25_candidate_chunks
            ),
            lexical_filter_enabled=False,
        )
        embedder = BgeM3Embedder(prod_config)
        fast_retriever = build_live_legal_v2_retriever(client, embedder, fast_config)
        ce_retriever = build_live_legal_v2_retriever(client, embedder, ce_config)
        _runtime = CaseSimilarityStage1Runtime(
            retriever=fast_retriever,
            config=fast_config,
            embedder=embedder,
            ready=True,
            ready_error=None,
            model_loaded=bool(getattr(embedder, "loaded", False)),
            bm25_loaded=False,
            warmup_status="cold",
            ce_retriever=ce_retriever,
            ce_config=ce_config,
        )
        logger.info(
            "[legal_v2.stage1] runtime ready "
            "fast_collection=%s fast_bm25=%s fast_points=%s "
            "ce_collection=%s ce_bm25=%s ce_points=%s",
            fast_config.qdrant_collection,
            fast_config.bm25_index_id,
            getattr(fast_info, "points_count", None),
            ce_config.qdrant_collection,
            ce_config.bm25_index_id,
            getattr(ce_info, "points_count", None),
        )
        return _runtime


def _embedder_is_loaded(runtime: CaseSimilarityStage1Runtime) -> bool:
    if runtime.model_loaded:
        return True
    embedder = runtime.embedder
    return bool(embedder is not None and getattr(embedder, "loaded", False))


def _bm25_is_loaded(runtime: CaseSimilarityStage1Runtime) -> bool:
    if runtime.bm25_loaded:
        return True
    bm25 = getattr(runtime.retriever, "_bm25", None)
    fast_ok = bool(bm25 is not None and getattr(bm25, "_index", None) is not None)
    if runtime.ce_retriever is None:
        return fast_ok
    ce_bm25 = getattr(runtime.ce_retriever, "_bm25", None)
    ce_ok = bool(ce_bm25 is not None and getattr(ce_bm25, "_index", None) is not None)
    return fast_ok and ce_ok


def warmup_case_similarity_stage1_runtime() -> dict[str, Any]:
    """Load BGE-M3 + BM25 into process memory (blocking; call from a worker thread)."""
    if not case_similarity_stage1_enabled():
        return {
            "warmup_status": "skipped",
            "reason": "stage1_disabled",
            "model_loaded": False,
            "bm25_loaded": False,
        }

    runtime = get_case_similarity_stage1_runtime()
    if runtime.warmup_status == "warm" and _embedder_is_loaded(runtime) and _bm25_is_loaded(runtime):
        return {
            "warmup_status": "warm",
            "model_loaded": True,
            "bm25_loaded": True,
            "warmup_latency_ms": runtime.warmup_latency_ms,
            "collection": runtime.config.qdrant_collection,
            "fast_collection": runtime.config.qdrant_collection,
            "ce_collection": (
                runtime.ce_config.qdrant_collection if runtime.ce_config else None
            ),
        }

    runtime.warmup_status = "warming"
    runtime.warmup_error_type = None
    started = time.perf_counter()
    try:
        embedder = runtime.embedder
        if embedder is None:
            raise RetrievalConfigurationError("Stage 1 embedder is not configured.")
        embedder.load()
        # Force a real encode so first user query does not pay JIT / graph costs.
        vector = embedder.embed_query(_WARMUP_QUERY)
        if not vector:
            raise RetrievalConfigurationError("Stage 1 warmup embedder returned an empty vector.")
        runtime.model_loaded = True

        bm25 = getattr(runtime.retriever, "_bm25", None)
        if bm25 is None:
            raise RetrievalConfigurationError("Stage 1 BM25 sidecar is not configured.")
        bm25.search(_WARMUP_QUERY, top_k=1)
        if runtime.ce_retriever is not None:
            ce_bm25 = getattr(runtime.ce_retriever, "_bm25", None)
            if ce_bm25 is None:
                raise RetrievalConfigurationError(
                    "CE BM25 sidecar is not configured."
                )
            ce_bm25.search(_WARMUP_QUERY, top_k=1)
        runtime.bm25_loaded = True

        runtime.warmup_status = "warm"
        runtime.warmup_latency_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "[legal_v2.stage1] warmup complete fast_collection=%s ce_collection=%s "
            "model_loaded=1 bm25_loaded=1 warmup_latency_ms=%.1f",
            runtime.config.qdrant_collection,
            runtime.ce_config.qdrant_collection if runtime.ce_config else None,
            runtime.warmup_latency_ms,
        )
        return {
            "warmup_status": "warm",
            "model_loaded": True,
            "bm25_loaded": True,
            "warmup_latency_ms": runtime.warmup_latency_ms,
            "collection": runtime.config.qdrant_collection,
            "fast_collection": runtime.config.qdrant_collection,
            "ce_collection": (
                runtime.ce_config.qdrant_collection if runtime.ce_config else None
            ),
        }
    except Exception as exc:  # noqa: BLE001
        runtime.warmup_status = "failed"
        runtime.warmup_error_type = type(exc).__name__
        runtime.warmup_latency_ms = (time.perf_counter() - started) * 1000.0
        logger.warning(
            "[legal_v2.stage1] warmup failed error_type=%s warmup_latency_ms=%.1f",
            runtime.warmup_error_type,
            runtime.warmup_latency_ms,
        )
        raise


def _cross_encoder_readiness_payload() -> dict[str, Any]:
    from app.rag.legal_v2.rerank.service import get_cross_encoder_reranking_service

    try:
        return get_cross_encoder_reranking_service().readiness()
    except Exception as exc:  # noqa: BLE001
        return {
            "enabled": False,
            "status": "unavailable",
            "model": None,
            "device": None,
            "error_type": exc.__class__.__name__,
        }


def probe_case_similarity_stage1_readiness() -> dict[str, Any]:
    """Readiness probe. With warmup-on-start, ready means model+BM25 are loaded."""
    try:
        runtime = get_case_similarity_stage1_runtime()
    except Exception as exc:  # noqa: BLE001
        fast_bind = fast_index_binding()
        ce_bind = ce_index_binding()
        return {
            "ready": False,
            "status": "unavailable",
            "error_type": exc.__class__.__name__,
            "collection": fast_bind.qdrant_collection,
            "bm25_index_id": fast_bind.bm25_index_id,
            "fast_collection": fast_bind.qdrant_collection,
            "fast_bm25_index_id": fast_bind.bm25_index_id,
            "ce_collection": ce_bind.qdrant_collection,
            "ce_bm25_index_id": ce_bind.bm25_index_id,
            "model_loaded": False,
            "bm25_loaded": False,
            "warmup_status": "failed",
            "warmup_required": stage1_warmup_on_start_enabled(),
            "cross_encoder": _cross_encoder_readiness_payload(),
        }

    model_loaded = _embedder_is_loaded(runtime)
    bm25_loaded = _bm25_is_loaded(runtime)
    warmup_required = stage1_warmup_on_start_enabled()
    warm_enough = model_loaded and bm25_loaded

    if warmup_required and not warm_enough:
        if runtime.warmup_status == "failed":
            status = "unavailable"
        elif runtime.warmup_status == "warming":
            status = "warming"
        else:
            status = "cold"
        ready = False
    else:
        status = "ready"
        ready = True

    return {
        "ready": ready,
        "status": status,
        "collection": runtime.config.qdrant_collection,
        "bm25_index_id": runtime.config.bm25_index_id,
        "bm25_sidecar_exists": runtime.config.bm25_sidecar_path.exists(),
        "fast_collection": runtime.config.qdrant_collection,
        "fast_bm25_index_id": runtime.config.bm25_index_id,
        "ce_collection": (
            runtime.ce_config.qdrant_collection if runtime.ce_config else None
        ),
        "ce_bm25_index_id": (
            runtime.ce_config.bm25_index_id if runtime.ce_config else None
        ),
        "retrieval_stage": STAGE_1_RETRIEVAL,
        "model_loaded": model_loaded,
        "bm25_loaded": bm25_loaded,
        "warmup_status": runtime.warmup_status,
        "warmup_required": warmup_required,
        "warmup_latency_ms": runtime.warmup_latency_ms,
        "error_type": runtime.warmup_error_type if not ready else None,
        "cross_encoder": _cross_encoder_readiness_payload(),
    }


def _colbert_config_from_env() -> Any:
    from app.rag.legal_v2.retrieve.colbert import (
        DEFAULT_COLBERT_MODEL,
        DEFAULT_INDEX_NAME,
        ColbertConfig,
    )

    index_path = Path(
        os.getenv(
            "NALUS_LEGAL_V2_COLBERT_INDEX_PATH",
            "artifacts/legal_v2/chunking_ab_pilot_300_v1/colbert_v1/index",
        )
    )
    mapping_path = Path(
        os.getenv(
            "NALUS_LEGAL_V2_COLBERT_MAPPING_PATH",
            "artifacts/legal_v2/chunking_ab_pilot_300_v1/colbert_v1/"
            "colbert_chunk_mapping.jsonl",
        )
    )
    device = os.getenv("NALUS_LEGAL_V2_COLBERT_DEVICE", "cuda").strip() or "cuda"
    batch_size = int(os.getenv("NALUS_LEGAL_V2_COLBERT_BATCH_SIZE", "16") or "16")
    allow_download = (
        os.getenv("NALUS_LEGAL_V2_COLBERT_ALLOW_DOWNLOAD", "0").strip().lower() in _TRUTHY
    )
    cfg = ColbertConfig(
        model_name=os.getenv("NALUS_LEGAL_V2_COLBERT_MODEL", DEFAULT_COLBERT_MODEL),
        index_path=index_path,
        index_name=os.getenv("NALUS_LEGAL_V2_COLBERT_INDEX_NAME", DEFAULT_INDEX_NAME),
        device=device,
        top_k=int(os.getenv("NALUS_LEGAL_V2_COLBERT_CANDIDATE_CHUNKS", "80") or "80"),
        batch_size=batch_size,
        concurrency_limit=1,
        mapping_path=mapping_path,
        allow_download=allow_download,
    )
    cfg.validate()
    return cfg


async def _ensure_colbert_retriever(runtime: CaseSimilarityStage1Runtime) -> Any:
    """Lazily initialize ColBERT for BALANCED. Safe to call concurrently."""
    if runtime.colbert_retriever is not None and runtime.colbert_ready:
        return runtime.colbert_retriever
    async with _colbert_init_lock:
        if runtime.colbert_retriever is not None and runtime.colbert_ready:
            return runtime.colbert_retriever
        if not colbert_master_allow_enabled():
            raise RetrievalConfigurationError(
                "ColBERT is disabled (NALUS_LEGAL_V2_COLBERT_ENABLED!=1)."
            )
        try:
            from app.rag.legal_v2.retrieve.colbert import (
                ColbertRetriever,
                PyLateColbertBackend,
            )

            cfg = _colbert_config_from_env()
            if not Path(cfg.index_path).exists():
                raise RetrievalConfigurationError(
                    f"ColBERT index missing: {cfg.index_path}"
                )
            mapping = cfg.resolved_mapping_path()
            if not mapping.exists():
                raise RetrievalConfigurationError(
                    f"ColBERT mapping missing: {mapping}"
                )
            backend = PyLateColbertBackend(cfg)
            await backend.initialize()
            retriever = ColbertRetriever(cfg, backend=backend)
            runtime.colbert_config = cfg
            runtime.colbert_retriever = retriever
            runtime.colbert_ready = True
            runtime.colbert_error_type = None
            logger.info(
                "[legal_v2.stage1] ColBERT ready index=%s device=%s",
                cfg.index_name,
                cfg.device,
            )
            return retriever
        except Exception as exc:  # noqa: BLE001
            runtime.colbert_ready = False
            runtime.colbert_error_type = type(exc).__name__
            raise RetrievalConfigurationError(
                f"ColBERT initialization failed: {type(exc).__name__}"
            ) from exc


async def search_case_similarity_stage1(
    *,
    query: str,
    limit: int | None = None,
    include_debug: bool = False,
    runtime: CaseSimilarityStage1Runtime | None = None,
    query_spec_builder=build_query_spec_v2,
    retrieval_profile: str | None = None,
) -> Stage1SearchResult:
    from app.rag.legal_v2.query_input.errors import (
        InputTooLargeError,
        NoUsefulContentError,
        UnsupportedCondensationModeError,
    )
    from app.rag.legal_v2.query_input.service import get_query_input_service

    raw = str(query or "").strip()
    if not raw:
        raise ValueError("query must not be blank")

    try:
        prepared = get_query_input_service().prepare(raw)
    except InputTooLargeError as exc:
        raise ValueError(str(exc)) from exc
    except (NoUsefulContentError, UnsupportedCondensationModeError) as exc:
        raise ValueError(str(exc)) from exc

    cleaned = prepared.retrieval_query
    if not cleaned:
        raise ValueError("query must not be blank")
    if len(cleaned) > MAX_QUERY_LENGTH:
        raise ValueError(f"query exceeds maximum length of {MAX_QUERY_LENGTH}")

    max_limit = max_result_limit()
    resolved_limit = default_result_limit() if limit is None else int(limit)
    if resolved_limit < 1 or resolved_limit > max_limit:
        raise ValueError(f"limit must be between 1 and {max_limit}")

    active = runtime or get_case_similarity_stage1_runtime()
    started = time.perf_counter()
    query_started = time.perf_counter()
    query_spec = query_spec_builder(cleaned)
    query_ms = (time.perf_counter() - query_started) * 1000.0

    from app.rag.legal_v2.rerank.errors import (
        RerankerInferenceError,
        RerankerInvalidCandidateError,
        RerankerModelLoadError,
        RerankerUnavailableError,
    )
    from app.rag.legal_v2.rerank.service import get_cross_encoder_reranking_service
    from app.rag.legal_v2.retrieve.colbert_hybrid import retrieve_hybrid_plus_colbert

    profile = resolve_retrieval_profile(retrieval_profile)
    active_retriever = active.retriever_for_profile(profile.profile_id)
    active_config = active.config_for_profile(profile.profile_id)
    colbert_applied = False

    if profile.use_colbert:
        colbert_retriever = await _ensure_colbert_retriever(active)
        retrieval = await retrieve_hybrid_plus_colbert(
            hybrid_retriever=active_retriever,
            colbert_retriever=colbert_retriever,
            query_spec=query_spec,
            colbert_candidate_chunks=int(profile.colbert_candidate_chunks),
            fused_candidate_chunks=int(active_config.fused_candidate_chunks),
            candidate_documents=int(active_config.candidate_documents),
            rrf_k=int(LEGAL_V2_PROFILE.rrf_k),
        )
        colbert_applied = True
    else:
        retrieval = await asyncio.to_thread(active_retriever.retrieve, query_spec)

    ce_config = profile.cross_encoder_config
    rerank_diagnostics: dict[str, Any] | None = None

    if not profile.use_cross_encoder or ce_config is None:
        documents = [
            _to_stage1_document(document, rank=index)
            for index, document in enumerate(retrieval.documents[:resolved_limit], start=1)
        ]
        rerank_diagnostics = {
            "rerank_enabled": False,
            "rerank_applied": False,
            "experiment_mode": profile.profile_id,
            "retrieval_profile": profile.profile_id,
            "colbert_applied": colbert_applied,
        }
    else:
        shortlist_n = min(
            ce_config.candidate_documents,
            len(retrieval.documents),
        )
        shortlist = [
            _to_stage1_document(
                document,
                rank=index,
                evidence_limit=ce_config.evidence_pool_limit,
                prefer_chunk_evidence=True,
            )
            for index, document in enumerate(retrieval.documents[:shortlist_n], start=1)
        ]
        by_ecli = {item.ecli: item for item in shortlist}
        service = get_cross_encoder_reranking_service(ce_config)
        try:
            reranked = await asyncio.to_thread(
                lambda: service.rerank(cleaned, shortlist, require_success=True)
            )
            rerank_diagnostics = reranked.diagnostics.as_dict()
            rerank_diagnostics["retrieval_profile"] = profile.profile_id
            documents = []
            for item in reranked.documents[:resolved_limit]:
                source = by_ecli.get(item.ecli)
                if source is None:
                    continue
                public_passages = list(source.relevant_passages)[:5]
                documents.append(
                    Stage1DocumentResult(
                        rank=item.ce_rank if item.ce_rank <= resolved_limit else len(documents) + 1,
                        document_id=source.document_id,
                        canonical_document_id=source.canonical_document_id,
                        ecli=source.ecli,
                        court=source.court,
                        case_number=source.case_number,
                        decision_date=source.decision_date,
                        document_type=source.document_type,
                        score=source.score,
                        relevant_passages=public_passages,
                        source_document_id=source.source_document_id,
                        dense_rank=source.dense_rank,
                        bm25_rank=source.bm25_rank,
                        rrf_score=source.rrf_score,
                        metadata=dict(source.metadata),
                        stage1_rank=item.stage1_rank,
                        stage1_score=item.stage1_score,
                        ce_rank=item.ce_rank,
                        ce_score=item.ce_score,
                        chunk_evidence=list(source.chunk_evidence),
                    )
                )
            documents = [
                Stage1DocumentResult(
                    rank=index,
                    document_id=doc.document_id,
                    canonical_document_id=doc.canonical_document_id,
                    ecli=doc.ecli,
                    court=doc.court,
                    case_number=doc.case_number,
                    decision_date=doc.decision_date,
                    document_type=doc.document_type,
                    score=doc.score,
                    relevant_passages=list(doc.relevant_passages),
                    source_document_id=doc.source_document_id,
                    dense_rank=doc.dense_rank,
                    bm25_rank=doc.bm25_rank,
                    rrf_score=doc.rrf_score,
                    metadata=dict(doc.metadata),
                    stage1_rank=doc.stage1_rank,
                    stage1_score=doc.stage1_score,
                    ce_rank=doc.ce_rank,
                    ce_score=doc.ce_score,
                    chunk_evidence=list(doc.chunk_evidence),
                )
                for index, doc in enumerate(documents, start=1)
            ]
        except (
            RerankerUnavailableError,
            RerankerModelLoadError,
            RerankerInferenceError,
            RerankerInvalidCandidateError,
        ) as exc:
            raise ValueError(f"cross-encoder reranking failed: {exc}") from exc

    for document in documents:
        _assert_ecli_identity(document)

    total_ms = (time.perf_counter() - started) * 1000.0
    diagnostics = {
        "query_length": len(cleaned),
        "original_query_length": len(prepared.original_query),
        "generated_query_count": len(query_spec.retrieval_queries),
        "result_count": len(documents),
        "collection": active_config.qdrant_collection,
        "bm25_index_id": active_config.bm25_index_id,
        "queryspec_latency_ms": query_ms,
        "dense_latency_ms": retrieval.diagnostics.get("dense_latency_ms"),
        "bm25_latency_ms": retrieval.diagnostics.get("bm25_latency_ms"),
        "colbert_latency_ms": retrieval.diagnostics.get("colbert_latency_ms"),
        "rrf_latency_ms": retrieval.diagnostics.get("rrf_latency_ms"),
        "aggregation_latency_ms": max(
            0.0,
            float(retrieval.diagnostics.get("total_retrieval_latency_ms") or 0.0)
            - float(retrieval.diagnostics.get("dense_latency_ms") or 0.0)
            - float(retrieval.diagnostics.get("bm25_latency_ms") or 0.0)
            - float(retrieval.diagnostics.get("colbert_latency_ms") or 0.0)
            - float(retrieval.diagnostics.get("rrf_latency_ms") or 0.0),
        ),
        "total_latency_ms": total_ms,
        "dense_candidate_chunks": retrieval.diagnostics.get("dense_candidate_chunks"),
        "bm25_candidate_chunks": retrieval.diagnostics.get("bm25_candidate_chunks"),
        "colbert_candidate_chunks": retrieval.diagnostics.get("colbert_candidate_chunks"),
        "fused_candidate_chunks": retrieval.diagnostics.get("fused_candidate_chunks"),
        "aggregated_documents": retrieval.diagnostics.get("candidate_documents"),
        "retrieval_status": "ok",
        "input_processing": prepared.input_processing_diagnostics(
            include_brief_text=False
        ),
        "rerank": rerank_diagnostics
        or {
            "rerank_enabled": False,
            "rerank_applied": False,
            "experiment_mode": profile.profile_id,
            "retrieval_profile": profile.profile_id,
            "colbert_applied": colbert_applied,
        },
        "retrieval_profile": profile.profile_id,
        "profile_label": profile.label,
        "profile_index_notes": profile.notes,
        "colbert_applied": colbert_applied,
    }
    rerank_payload = diagnostics["rerank"]
    passages_per_document = rerank_payload.get("requested_passages_per_document")
    if not isinstance(passages_per_document, int):
        passages_per_document = (
            ce_config.passages_per_document
            if bool(rerank_payload.get("rerank_applied")) and ce_config is not None
            else None
        )
    retrieval_stage = build_retrieval_stage(
        rerank_applied=bool(rerank_payload.get("rerank_applied")),
        passages_per_document=passages_per_document,
        colbert_applied=colbert_applied,
    )
    if include_debug and stage1_debug_allowed():
        diagnostics["debug"] = _safe_debug_payload(query_spec, retrieval)
        diagnostics["debug"]["input_processing"] = prepared.input_processing_diagnostics(
            include_brief_text=True
        )

    trace_event(
        logger,
        "legal_v2.stage1.search.completed",
        result_count=len(documents),
        query_length=len(cleaned),
        collection=active_config.qdrant_collection,
    )
    logger.info(
        "[legal_v2.stage1] search done result_count=%s query_length=%s "
        "generated_query_count=%s collection=%s bm25_index_id=%s "
        "dense_latency_ms=%s bm25_latency_ms=%s colbert_latency_ms=%s "
        "total_latency_ms=%s was_condensed=%s classification=%s "
        "retrieval_profile=%s retrieval_stage=%s",
        len(documents),
        len(cleaned),
        len(query_spec.retrieval_queries),
        active_config.qdrant_collection,
        active_config.bm25_index_id,
        diagnostics.get("dense_latency_ms"),
        diagnostics.get("bm25_latency_ms"),
        diagnostics.get("colbert_latency_ms"),
        diagnostics.get("total_latency_ms"),
        prepared.was_condensed,
        prepared.classification.value,
        profile.profile_id,
        retrieval_stage,
    )
    return Stage1SearchResult(
        query=cleaned,
        result_count=len(documents),
        retrieval_stage=retrieval_stage,
        results=documents,
        diagnostics=diagnostics,
    )


def search_case_similarity_stage1_sync(
    *,
    query: str,
    limit: int | None = None,
    include_debug: bool = False,
    runtime: CaseSimilarityStage1Runtime | None = None,
    query_spec_builder=build_query_spec_v2,
    retrieval_profile: str | None = None,
) -> Stage1SearchResult:
    """Sync CLI/test boundary only. Prefer ``await search_case_similarity_stage1``."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            search_case_similarity_stage1(
                query=query,
                limit=limit,
                include_debug=include_debug,
                runtime=runtime,
                query_spec_builder=query_spec_builder,
                retrieval_profile=retrieval_profile,
            )
        )
    raise RuntimeError(
        "search_case_similarity_stage1_sync() cannot be called from a running "
        "event loop; await search_case_similarity_stage1() instead"
    )



def _to_stage1_document(
    document,
    *,
    rank: int,
    evidence_limit: int = 5,
    prefer_chunk_evidence: bool = False,
) -> Stage1DocumentResult:
    metadata = dict(document.metadata or {})
    ecli_raw = metadata.get("ecli") or document.document_id
    ecli = normalize_ecli(str(ecli_raw)) if is_valid_ecli(str(ecli_raw)) else str(ecli_raw)
    if not is_valid_ecli(ecli):
        raise RetrievalConfigurationError(
            f"aggregated document missing valid ECLI identity: {document.document_id!r}"
        )
    source_document_id = metadata.get("source_document_id")
    if isinstance(source_document_id, str) and source_document_id.startswith("doc-"):
        source_id = source_document_id
    else:
        source_id = None
        raw_doc = str(document.document_id or "")
        if raw_doc.startswith("doc-"):
            # Never promote doc-* to the public identity.
            source_id = raw_doc

    chunk_evidence_raw = list(getattr(document, "chunk_evidence", None) or [])
    chunk_evidence = [
        dict(item) for item in chunk_evidence_raw[: max(0, int(evidence_limit))] if isinstance(item, dict)
    ]

    passages: list[Stage1Passage] = []
    if prefer_chunk_evidence and chunk_evidence:
        for item in chunk_evidence:
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            channels = tuple(item.get("retrieval_channels") or ())
            passages.append(
                Stage1Passage(
                    text=text,
                    chunk_id=str(item.get("chunk_id") or ""),
                    section=item.get("section"),
                    page=item.get("page"),
                    score=item.get("rrf_score"),
                    dense_rank=item.get("dense_rank"),
                    bm25_rank=item.get("bm25_rank"),
                    rrf_rank=item.get("rrf_rank"),
                    retrieval_channels=channels,
                    chunk_position=item.get("chunk_position"),
                )
            )
    else:
        # FAST / historical path: first N paragraphs in document source order.
        for paragraph in list(document.paragraphs or [])[: max(0, int(evidence_limit))]:
            section = getattr(paragraph.section_type, "value", None) or str(
                getattr(paragraph, "section_type", "") or ""
            ) or None
            passages.append(
                Stage1Passage(
                    text=str(paragraph.normalized_text or paragraph.original_text or "").strip(),
                    chunk_id=str(paragraph.paragraph_id or ""),
                    section=section,
                    page=None,
                    score=None,
                )
            )
    passages = [item for item in passages if item.text]

    stage1_score = float(document.score or 0.0)
    return Stage1DocumentResult(
        rank=rank,
        document_id=ecli,
        canonical_document_id=ecli,
        ecli=ecli,
        court=_optional_str(metadata.get("court_name") or metadata.get("court")),
        case_number=_optional_str(
            metadata.get("case_reference")
            or metadata.get("case_number")
            or metadata.get("spisova_znacka")
        ),
        decision_date=_optional_str(metadata.get("decision_date") or metadata.get("date")),
        document_type=_optional_str(
            metadata.get("document_type") or metadata.get("decision_type") or metadata.get("title")
        ),
        score=stage1_score,
        relevant_passages=passages,
        source_document_id=source_id,
        dense_rank=document.dense_rank,
        bm25_rank=document.bm25_rank,
        rrf_score=document.rrf_score,
        metadata={
            key: value
            for key, value in metadata.items()
            if key
            in {
                "court_name",
                "court",
                "case_reference",
                "case_number",
                "decision_date",
                "document_type",
                "source",
                "source_document_id",
            }
        },
        stage1_rank=rank,
        stage1_score=stage1_score,
        ce_rank=None,
        ce_score=None,
        chunk_evidence=chunk_evidence,
    )


def _assert_ecli_identity(document: Stage1DocumentResult) -> None:
    if not (
        document.document_id == document.canonical_document_id == document.ecli
        and is_valid_ecli(document.ecli)
    ):
        raise RetrievalConfigurationError(
            "Stage 1 result violated ECLI identity invariant "
            f"document_id={document.document_id!r} "
            f"canonical_document_id={document.canonical_document_id!r} "
            f"ecli={document.ecli!r}"
        )
    if document.document_id.startswith("doc-"):
        raise RetrievalConfigurationError("Stage 1 must not expose doc-* as document_id")


def _safe_debug_payload(query_spec: QuerySpecV2, retrieval: LegalV2RetrievalResult) -> dict[str, Any]:
    return {
        "retrieval_queries": list(query_spec.retrieval_queries),
        "legal_concepts": list(query_spec.structured_query.get("legal_concepts") or []),
        "candidate_retrieval_concepts": list(
            query_spec.structured_query.get("candidate_retrieval_concepts") or []
        ),
        "negated_requested_concepts": list(
            query_spec.structured_query.get("negated_requested_concepts") or []
        ),
        "suppressed_expansions": list(
            query_spec.structured_query.get("suppressed_expansions") or []
        ),
        "procedural_posture": list(query_spec.procedural_posture),
        "decision_outcome": list(query_spec.decision_outcome),
        "negative_constraints": [
            {
                "attribute": item.attribute,
                "value": item.value,
            }
            for item in query_spec.negative_constraints
        ],
        "candidate_counts": {
            "dense": len(retrieval.dense_results),
            "bm25": len(retrieval.bm25_results),
            "fused": len(retrieval.fused_results),
            "aggregated": len(retrieval.documents),
        },
    }


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _bounded_int(
    raw: str | None,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(str(raw).strip()) if raw not in {None, ""} else default
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))
