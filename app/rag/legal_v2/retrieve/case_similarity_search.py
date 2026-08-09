"""Stage 1 case-similarity search: deterministic QuerySpec + hybrid retrieval (no LLM)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.rag.legal_v2.identity import is_valid_ecli, normalize_ecli
from app.rag.legal_v2.indexing import LEGAL_V2_PROFILE
from app.rag.legal_v2.query_spec import QuerySpecV2, build_query_spec_v2
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

STAGE_1_RETRIEVAL = "hybrid_rrf_stage_1"
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


_runtime: CaseSimilarityStage1Runtime | None = None
_runtime_lock = Lock()


def reset_case_similarity_stage1_runtime_for_tests() -> None:
    global _runtime
    with _runtime_lock:
        _runtime = None


def get_case_similarity_stage1_runtime() -> CaseSimilarityStage1Runtime:
    global _runtime
    if _runtime is not None:
        return _runtime
    with _runtime_lock:
        if _runtime is not None:
            return _runtime
        config = legal_v2_retriever_config_from_env()
        config.validate()
        if not config.bm25_sidecar_path.exists():
            raise RetrievalConfigurationError(
                f"BM25 sidecar missing: {config.bm25_sidecar_path}"
            )
        qdrant_module = __import__("qdrant_client", fromlist=["QdrantClient"])
        client = qdrant_module.QdrantClient(
            url=os.getenv("QDRANT_URL", "http://qdrant:6333"),
            timeout=10,
        )
        collection = config.qdrant_collection
        try:
            info = client.get_collection(collection)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalConfigurationError(
                f"Qdrant collection unavailable: {collection}"
            ) from exc
        if int(getattr(info, "points_count", 0) or 0) <= 0:
            raise RetrievalConfigurationError(
                f"Qdrant collection is empty: {collection}"
            )
        prod_config = ProductionRetrievalConfig(
            profile=LEGAL_V2_PROFILE,
            qdrant_collection=config.qdrant_collection,
            bm25_sidecar_path=config.bm25_sidecar_path,
            bm25_index_id=config.bm25_index_id,
            model_path=config.model_path,
            local_files_only=True,
            trust_remote_code=False,
            device=os.getenv("EMBEDDING_DEVICE", "cpu"),
            candidate_multiplier=1,
            min_candidate_count=1,
            max_candidate_count=max(
                config.dense_candidate_chunks, config.bm25_candidate_chunks
            ),
            lexical_filter_enabled=False,
        )
        embedder = BgeM3Embedder(prod_config)
        retriever = build_live_legal_v2_retriever(client, embedder, config)
        _runtime = CaseSimilarityStage1Runtime(
            retriever=retriever,
            config=config,
            embedder=embedder,
            ready=True,
            ready_error=None,
            model_loaded=bool(getattr(embedder, "loaded", False)),
            bm25_loaded=False,
            warmup_status="cold",
        )
        logger.info(
            "[legal_v2.stage1] runtime ready collection=%s bm25_index_id=%s points=%s",
            config.qdrant_collection,
            config.bm25_index_id,
            getattr(info, "points_count", None),
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
    return bool(bm25 is not None and getattr(bm25, "_index", None) is not None)


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
        runtime.bm25_loaded = True

        runtime.warmup_status = "warm"
        runtime.warmup_latency_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "[legal_v2.stage1] warmup complete collection=%s model_loaded=1 bm25_loaded=1 "
            "warmup_latency_ms=%.1f",
            runtime.config.qdrant_collection,
            runtime.warmup_latency_ms,
        )
        return {
            "warmup_status": "warm",
            "model_loaded": True,
            "bm25_loaded": True,
            "warmup_latency_ms": runtime.warmup_latency_ms,
            "collection": runtime.config.qdrant_collection,
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
        return {
            "ready": False,
            "status": "unavailable",
            "error_type": exc.__class__.__name__,
            "collection": os.getenv("NALUS_LEGAL_V2_QDRANT_COLLECTION"),
            "bm25_index_id": os.getenv("NALUS_LEGAL_V2_BM25_INDEX_ID"),
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
        "retrieval_stage": STAGE_1_RETRIEVAL,
        "model_loaded": model_loaded,
        "bm25_loaded": bm25_loaded,
        "warmup_status": runtime.warmup_status,
        "warmup_required": warmup_required,
        "warmup_latency_ms": runtime.warmup_latency_ms,
        "error_type": runtime.warmup_error_type if not ready else None,
        "cross_encoder": _cross_encoder_readiness_payload(),
    }


def search_case_similarity_stage1(
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
    retrieval = active.retriever.retrieve(query_spec)

    from app.rag.legal_v2.rerank.errors import (
        RerankerInferenceError,
        RerankerInvalidCandidateError,
        RerankerModelLoadError,
        RerankerUnavailableError,
    )
    from app.rag.legal_v2.rerank.service import get_cross_encoder_reranking_service
    from app.rag.legal_v2.retrieve.retrieval_profiles import resolve_retrieval_profile

    # Env alone no longer forces CE on every API call; request profile selects mode.
    # Master-allow still gates CE via resolve_retrieval_profile (ce7 requires env ON).
    profile = resolve_retrieval_profile(retrieval_profile)
    ce_config = profile.cross_encoder_config
    rerank_diagnostics: dict[str, Any] | None = None

    if not profile.use_cross_encoder or ce_config is None:
        # FAST path — preserve historical public passage truncation (first 5 paragraphs).
        documents = [
            _to_stage1_document(document, rank=index)
            for index, document in enumerate(retrieval.documents[:resolved_limit], start=1)
        ]
        rerank_diagnostics = {
            "rerank_enabled": False,
            "rerank_applied": False,
            "experiment_mode": "fast",
            "retrieval_profile": profile.profile_id,
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
            reranked = service.rerank(cleaned, shortlist, require_success=True)
            rerank_diagnostics = reranked.diagnostics.as_dict()
            rerank_diagnostics["retrieval_profile"] = profile.profile_id
            documents = []
            for item in reranked.documents[:resolved_limit]:
                source = by_ecli.get(item.ecli)
                if source is None:
                    continue
                # Public response keeps a short passage preview; CE used the fuller pool.
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
            # Re-number public ranks 1..N after truncation.
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
            # Experimental CE mode: fail clearly (no silent FAST claim).
            raise ValueError(f"cross-encoder reranking failed: {exc}") from exc

    for document in documents:
        _assert_ecli_identity(document)

    total_ms = (time.perf_counter() - started) * 1000.0
    diagnostics = {
        "query_length": len(cleaned),
        "original_query_length": len(prepared.original_query),
        "generated_query_count": len(query_spec.retrieval_queries),
        "result_count": len(documents),
        "collection": active.config.qdrant_collection,
        "bm25_index_id": active.config.bm25_index_id,
        "queryspec_latency_ms": query_ms,
        "dense_latency_ms": retrieval.diagnostics.get("dense_latency_ms"),
        "bm25_latency_ms": retrieval.diagnostics.get("bm25_latency_ms"),
        "rrf_latency_ms": retrieval.diagnostics.get("rrf_latency_ms"),
        "aggregation_latency_ms": max(
            0.0,
            float(retrieval.diagnostics.get("total_retrieval_latency_ms") or 0.0)
            - float(retrieval.diagnostics.get("dense_latency_ms") or 0.0)
            - float(retrieval.diagnostics.get("bm25_latency_ms") or 0.0)
            - float(retrieval.diagnostics.get("rrf_latency_ms") or 0.0),
        ),
        "total_latency_ms": total_ms,
        "dense_candidate_chunks": retrieval.diagnostics.get("dense_candidate_chunks"),
        "bm25_candidate_chunks": retrieval.diagnostics.get("bm25_candidate_chunks"),
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
            "experiment_mode": "fast",
            "retrieval_profile": profile.profile_id,
        },
        "retrieval_profile": profile.profile_id,
    }
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
        collection=active.config.qdrant_collection,
    )
    logger.info(
        "[legal_v2.stage1] search done result_count=%s query_length=%s "
        "generated_query_count=%s collection=%s bm25_index_id=%s "
        "dense_latency_ms=%s bm25_latency_ms=%s total_latency_ms=%s "
        "was_condensed=%s classification=%s",
        len(documents),
        len(cleaned),
        len(query_spec.retrieval_queries),
        active.config.qdrant_collection,
        active.config.bm25_index_id,
        diagnostics.get("dense_latency_ms"),
        diagnostics.get("bm25_latency_ms"),
        diagnostics.get("total_latency_ms"),
        prepared.was_condensed,
        prepared.classification.value,
    )
    return Stage1SearchResult(
        query=cleaned,
        result_count=len(documents),
        retrieval_stage=STAGE_1_RETRIEVAL,
        results=documents,
        diagnostics=diagnostics,
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
