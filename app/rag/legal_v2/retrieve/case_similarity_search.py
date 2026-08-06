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
MAX_RESULT_LIMIT = 20
MAX_QUERY_LENGTH = 8000
MIN_QUERY_LENGTH = 1


def case_similarity_stage1_enabled() -> bool:
    raw = os.getenv("NALUS_LEGAL_V2_CASE_SIMILARITY_ENABLED", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    # Allow Stage 1 when the broader Legal v2 search flag is on.
    return os.getenv("NALUS_LEGAL_V2_SEARCH_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def stage1_debug_allowed() -> bool:
    return os.getenv("NALUS_LEGAL_V2_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


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
        maximum=50,
    )


@dataclass(frozen=True)
class Stage1Passage:
    text: str
    chunk_id: str
    section: str | None = None
    page: int | None = None
    score: float | None = None


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
    ready: bool = False
    ready_error: str | None = None


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
            ready=True,
            ready_error=None,
        )
        logger.info(
            "[legal_v2.stage1] runtime ready collection=%s bm25_index_id=%s points=%s",
            config.qdrant_collection,
            config.bm25_index_id,
            getattr(info, "points_count", None),
        )
        return _runtime


def probe_case_similarity_stage1_readiness() -> dict[str, Any]:
    """Lightweight readiness probe without requiring a search."""
    try:
        runtime = get_case_similarity_stage1_runtime()
    except Exception as exc:  # noqa: BLE001
        return {
            "ready": False,
            "status": "unavailable",
            "error_type": exc.__class__.__name__,
            "collection": os.getenv("NALUS_LEGAL_V2_QDRANT_COLLECTION"),
            "bm25_index_id": os.getenv("NALUS_LEGAL_V2_BM25_INDEX_ID"),
        }
    return {
        "ready": True,
        "status": "ready",
        "collection": runtime.config.qdrant_collection,
        "bm25_index_id": runtime.config.bm25_index_id,
        "bm25_sidecar_exists": runtime.config.bm25_sidecar_path.exists(),
        "retrieval_stage": STAGE_1_RETRIEVAL,
    }


def search_case_similarity_stage1(
    *,
    query: str,
    limit: int | None = None,
    include_debug: bool = False,
    runtime: CaseSimilarityStage1Runtime | None = None,
    query_spec_builder=build_query_spec_v2,
) -> Stage1SearchResult:
    cleaned = " ".join(str(query or "").split())
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
    documents = [
        _to_stage1_document(document, rank=index)
        for index, document in enumerate(retrieval.documents[:resolved_limit], start=1)
    ]
    for document in documents:
        _assert_ecli_identity(document)

    total_ms = (time.perf_counter() - started) * 1000.0
    diagnostics = {
        "query_length": len(cleaned),
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
    }
    if include_debug and stage1_debug_allowed():
        diagnostics["debug"] = _safe_debug_payload(query_spec, retrieval)

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
        "dense_latency_ms=%s bm25_latency_ms=%s total_latency_ms=%s",
        len(documents),
        len(cleaned),
        len(query_spec.retrieval_queries),
        active.config.qdrant_collection,
        active.config.bm25_index_id,
        diagnostics.get("dense_latency_ms"),
        diagnostics.get("bm25_latency_ms"),
        diagnostics.get("total_latency_ms"),
    )
    return Stage1SearchResult(
        query=cleaned,
        result_count=len(documents),
        retrieval_stage=STAGE_1_RETRIEVAL,
        results=documents,
        diagnostics=diagnostics,
    )


def _to_stage1_document(document, *, rank: int) -> Stage1DocumentResult:
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

    passages: list[Stage1Passage] = []
    for paragraph in list(document.paragraphs or [])[:5]:
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
        score=float(document.score or 0.0),
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
