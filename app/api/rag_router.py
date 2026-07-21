"""
FastAPI router for RAG search and orchestrator pipeline.

Usage (mount in your FastAPI app):
    from app.api.rag_router import router as rag_router
    app.include_router(rag_router)

Endpoints:
    POST /api/rag/search  — legacy retrieval pipeline (RetrievalPipeline + AnswerService)
    POST /api/rag/query   — full orchestrated pipeline (OrchestratorService)

Override dependency providers via app.dependency_overrides for production
or test injection.
"""

import os
import time
from dataclasses import asdict
from typing import Any
from unittest.mock import MagicMock

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.query_cache import CachedQueryResponse, build_cache_key, query_cache_ttl_seconds
from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.observability.constraint_retrieval_metrics import (
    record_constraint_retrieval_error,
    record_constraint_retrieval_metrics,
)
from app.rag.answer.answer_service import AnswerService
from app.rag.clarification.orchestrator import ClarifyingOrchestratorService
from app.rag.execution.execution_service import ExecutionService
from app.rag.orchestration.pipeline import RetrievalPipeline
from app.rag.orchestrator.orchestrator_service import OrchestratorResult, OrchestratorService
from app.rag.planner.planner_service import MockPlannerLLM, PlannerService
from app.rag.retrieval.constraint_config import (
    ConstraintRetrievalConfig,
    constraint_retrieval_config_from_env,
)
from app.rag.retrieval.constraint_models import ConstraintRetrievalResult
from app.rag.retrieval.constraint_pipeline import retrieve_verified_documents
from app.rag.retrieval.document_retrieval import (
    DocumentRetrievalConfig,
    build_document_level_results,
    document_retrieval_config_from_env,
)
from app.rag.retrieval.errors import RetrievalConfigurationError
from app.rag.retrieval.full_document import (
    FullDocumentLookupError,
    FullDocumentResult,
    FullDocumentStore,
    QdrantFullDocumentStore,
    validate_document_id,
)
from app.rag.retrieval.production_profile import DEFAULT_QDRANT_COLLECTION
from app.rag.synthesis.synthesis_service import MockSynthesisLLM, SynthesisService

logger = get_logger(__name__)

router = APIRouter(prefix="/api/rag", tags=["rag"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResponse(BaseModel):
    query: str
    summary: str
    top_cases: list[str]
    excerpts: list[str]


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    plan_steps: list[str]


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 10
    sources: list[str] | None = None


class RetrievedResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    source: str | None = None
    reference: str | None = None
    case_reference: str | None = None
    court_name: str | None = None
    date: str | None = None
    document_id: int | str | None = None
    chunk_index: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieveResponse(BaseModel):
    results: list[RetrievedResult]


class DocumentRetrieveRequest(BaseModel):
    query: str
    sources: list[str] | None = None


class SupportingPassageResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    source: str | None = None
    chunk_index: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentRetrievedResult(BaseModel):
    document_id: str
    score: float
    best_passages: list[SupportingPassageResult]
    metadata: dict[str, Any] = Field(default_factory=dict)
    candidate_chunk_count: int
    best_chunk_score: float


class DocumentRetrievalDiagnosticsResult(BaseModel):
    candidate_chunks_retrieved: int
    unique_documents_produced: int
    duplicate_document_hits_removed: int
    duplicate_chunks_removed: int
    chunks_missing_document_id: int
    documents_filtered: int
    final_document_count: int
    scoring_strategy: str
    document_relevance_threshold: float
    max_candidate_chunks: int
    max_returned_documents: int
    max_supporting_chunks_per_document: int
    retrieval_latency_ms: float | None = None
    aggregation_latency_ms: float | None = None
    latency_budget_ms: int | None = None
    latency_budget_exceeded: bool = False


class DocumentRetrieveResponse(BaseModel):
    documents: list[DocumentRetrievedResult]
    diagnostics: DocumentRetrievalDiagnosticsResult


class FullDocumentChunkResult(BaseModel):
    chunk_id: str
    chunk_index: int | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class FullDocumentDiagnosticsResult(BaseModel):
    collection_name: str
    chunk_count: int
    missing_chunk_indexes: list[int]
    duplicate_chunk_indexes: list[int]
    all_chunks_have_index: bool
    reconstruction_method: str
    truncated: bool = False
    max_chunks: int


class FullDocumentResponse(BaseModel):
    document_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    full_text: str
    chunks: list[FullDocumentChunkResult]
    source_url: str | None = None
    provenance_status: str
    full_text_availability_status: str
    diagnostics: FullDocumentDiagnosticsResult


class VerifiedRetrieveRequest(BaseModel):
    query: str
    sources: list[str] | None = None
    debug: bool = False


class StructuredEntityResult(BaseModel):
    id: str
    entity_type: str
    role: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class StructuredRelationResult(BaseModel):
    subject: str
    predicate: str
    object: str
    requirement: str


class StructuredConstraintResult(BaseModel):
    id: str
    category: str
    value: str
    requirement: str
    relation: StructuredRelationResult | None = None
    description: str


class StructuredQueryResult(BaseModel):
    intent: str
    status: str
    constraints: list[StructuredConstraintResult]
    entities: list[StructuredEntityResult]
    relations: list[StructuredRelationResult]
    ambiguities: list[str]
    retrieval_expansions: list[str]
    interpreter: str


class ConstraintEvidenceResult(BaseModel):
    document_id: str
    chunk_id: str | None = None
    quote: str
    source_field: str | None = None


class ConstraintVerificationResultModel(BaseModel):
    constraint_id: str
    category: str
    status: str
    required_value: str
    detected_value: str | None = None
    evidence: list[ConstraintEvidenceResult]
    verification_method: str
    confidence: float
    reason: str


class VerifiedDocumentResult(BaseModel):
    document_id: str
    score: float
    decision_status: str
    constraint_results: list[ConstraintVerificationResultModel]
    supporting_passages: list[ConstraintEvidenceResult]
    metadata: dict[str, Any] = Field(default_factory=dict)
    candidate_chunk_count: int


class ConstraintRetrievalDiagnosticsResult(BaseModel):
    query_interpretation_status: str
    hard_constraint_count: int
    soft_constraint_count: int
    candidate_chunks_retrieved: int
    candidate_documents_produced: int
    documents_verified: int
    verified_document_count: int
    excluded_hard_mismatch_count: int
    excluded_not_proven_count: int
    verification_error_count: int
    final_document_count: int
    retrieval_latency_ms: float | None
    verification_latency_ms: float | None
    total_latency_ms: float | None
    latency_budget_ms: int | None
    latency_budget_exceeded: bool


class VerifiedRetrieveResponse(BaseModel):
    structured_query: StructuredQueryResult
    documents: list[VerifiedDocumentResult]
    rejected_documents: list[VerifiedDocumentResult] = Field(default_factory=list)
    diagnostics: ConstraintRetrievalDiagnosticsResult


# ---------------------------------------------------------------------------
# Live orchestrator — set by startup lifespan, None until Qdrant is ready
# ---------------------------------------------------------------------------

_live_orchestrator: OrchestratorService | None = None
_live_orchestrator_status: str = "pending"
_live_orchestrator_error: str | None = None
_background_ingest_status: str = "idle"
_background_ingest_error: str | None = None
_corpus_version: str = "unknown"
_query_cache = None
_query_cache_backend: str = "none"
_query_cache_error: str | None = None
_embedding_cache_enabled: bool = False
_embedding_cache_backend: str = "none"
_embedding_cache_error: str | None = None
_SOURCE_FILTER_ALIASES: dict[str, set[str]] = {
    "constitutional": {"constitutional", "nalus"},
    "supreme": {"supreme"},
    "administrative": {"administrative"},
}

# ---------------------------------------------------------------------------
# Dependency providers
# ---------------------------------------------------------------------------


def _collection_name() -> str:
    return os.getenv("QDRANT_COLLECTION_NAME", DEFAULT_QDRANT_COLLECTION)


def get_pipeline() -> RetrievalPipeline:
    """
    Legacy /search endpoint compatibility pipeline.

    Production retrieval is wired through startup and /query or /retrieve. This
    fallback intentionally returns no retrieval results instead of using the old
    substring KeywordRetriever.
    """
    return _EmptyPipeline()


def get_answer_service() -> AnswerService:
    return AnswerService()


def get_orchestrator() -> OrchestratorService:
    """
    Returns the live orchestrator (real Qdrant + corpus) if startup succeeded,
    otherwise falls back to a stub with empty retrieval.
    """
    if _live_orchestrator is not None:
        return _live_orchestrator

    if os.getenv("RAG_STRICT_REAL_MODE", "").strip().lower() in {"1", "true", "yes", "on"}:
        if _live_orchestrator_error:
            detail = (
                "Live orchestrator is unavailable in strict real mode. "
                f"Last startup error: {_live_orchestrator_error}"
            )
        else:
            detail = (
                "Live orchestrator is still initializing in strict real mode."
            )
        raise HTTPException(
            status_code=503,
            detail=detail,
        )

    # Fallback stub (used only when strict mode is disabled or in tests).
    return ClarifyingOrchestratorService(
        OrchestratorService(
            planner=PlannerService(llm=MockPlannerLLM()),
            execution=ExecutionService(retrieval_service=_EmptyRetrievalService()),
            synthesis=SynthesisService(llm=MockSynthesisLLM()),
        )
    )


def get_full_document_store() -> FullDocumentStore:
    return QdrantFullDocumentStore(
        qdrant_url=os.getenv("QDRANT_URL", "http://qdrant:6333"),
        collection_name=_collection_name(),
    )


class _EmptyPipeline:
    def run(self, query: str, top_k: int = 5):
        del top_k
        return MagicMock(query=query, results=[])


class _EmptyRetrievalService:
    def search(self, query: str, top_k: int = 5) -> list:
        del query, top_k
        return []

    def search_dense(self, query: str, top_k: int = 5) -> list:
        del query, top_k
        return []


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_filter_values(values: list[str] | None) -> set[str]:
    normalized: set[str] = set()
    for value in values or []:
        text = _normalize_text(value)
        if text:
            normalized.add(text.lower())
    return normalized


def _chunk_source_tags(chunk) -> set[str]:
    metadata = chunk.metadata or {}
    tags: set[str] = set()

    raw_source = _normalize_text(metadata.get("source"))
    if raw_source:
        normalized_source = raw_source.lower()
        tags.add(normalized_source)
        if (
            normalized_source == "nalus"
            or "nalus" in normalized_source
            or "usoud" in normalized_source
            or "ústav" in normalized_source
            or "ustav" in normalized_source
            or "constitutional" in normalized_source
        ):
            tags.add("constitutional")
        if "supreme" in normalized_source or "nsoud" in normalized_source:
            tags.add("supreme")

    court_name = _normalize_text(metadata.get("court_name") or metadata.get("court"))
    if court_name:
        normalized_court = court_name.lower()
        tags.add(normalized_court)
        if "ústavní" in normalized_court or "ustavni" in normalized_court:
            tags.add("constitutional")
        elif "nejvyšší správní" in normalized_court or "nejvyssi spravni" in normalized_court:
            tags.add("administrative")
        elif "nejvyšší" in normalized_court or "nejvyssi" in normalized_court:
            tags.add("supreme")

    document_identity = " ".join(
        value
        for value in (
            _normalize_text(metadata.get("ecli")),
            _normalize_text(metadata.get("document_id")),
            _normalize_text(metadata.get("source_document_id")),
            _normalize_text(metadata.get("case_reference")),
            _normalize_text(metadata.get("reference")),
        )
        if value
    ).lower()
    if "ecli:cz:us" in document_identity:
        tags.add("constitutional")
    if "ecli:cz:ns" in document_identity:
        tags.add("supreme")

    return tags


def _matches_source_filters(chunk, requested_sources: set[str]) -> bool:
    if not requested_sources:
        return True

    tags = _chunk_source_tags(chunk)
    if not tags:
        return False

    for requested in requested_sources:
        allowed = _SOURCE_FILTER_ALIASES.get(requested, {requested})
        if tags.intersection(allowed):
            return True
    return False


def _raw_retrieve_limit(top_k: int, requested_sources: set[str]) -> int:
    if not requested_sources:
        return top_k
    return max(top_k, top_k * 5)


def _to_retrieved_result(chunk) -> RetrievedResult:
    metadata = dict(chunk.metadata or {})
    reference = _normalize_text(
        metadata.get("case_reference")
        or metadata.get("reference")
        or metadata.get("spisova_znacka")
        or metadata.get("case_number")
        or metadata.get("sp_zn")
    )
    source = _normalize_text(metadata.get("source"))
    court_name = (
        _normalize_text(metadata.get("court_name") or metadata.get("court"))
        or _infer_court_name_from_metadata(metadata)
    )
    date = _normalize_text(metadata.get("decision_date") or metadata.get("date"))
    chunk_index = metadata.get("chunk_index")
    if chunk_index is not None:
        try:
            chunk_index = int(chunk_index)
        except (TypeError, ValueError):
            chunk_index = None

    return RetrievedResult(
        chunk_id=chunk.id,
        text=chunk.text,
        score=chunk.score,
        source=source,
        reference=reference,
        case_reference=reference,
        court_name=court_name,
        date=date,
        document_id=metadata.get("document_id"),
        chunk_index=chunk_index,
        metadata=metadata,
    )


def _infer_court_name_from_metadata(metadata: dict[str, Any]) -> str | None:
    identity = " ".join(
        value
        for value in (
            _normalize_text(metadata.get("source")),
            _normalize_text(metadata.get("ecli")),
            _normalize_text(metadata.get("document_id")),
            _normalize_text(metadata.get("source_document_id")),
        )
        if value
    ).lower()
    if (
        "ecli:cz:us" in identity
        or "nalus" in identity
        or "usoud" in identity
        or "ústav" in identity
        or "ustav" in identity
        or "constitutional" in identity
    ):
        return "Ústavní soud"
    if "ecli:cz:ns" in identity or "nsoud" in identity or "supreme" in identity:
        return "Nejvyšší soud"
    return None


def _document_retrieval_config() -> DocumentRetrievalConfig:
    try:
        return document_retrieval_config_from_env()
    except RetrievalConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _constraint_retrieval_config() -> ConstraintRetrievalConfig:
    try:
        return constraint_retrieval_config_from_env()
    except RetrievalConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _to_document_retrieve_response(result) -> DocumentRetrieveResponse:
    return DocumentRetrieveResponse(
        documents=[
            DocumentRetrievedResult(
                document_id=document.document_id,
                score=document.score,
                best_passages=[
                    SupportingPassageResult(**asdict(passage))
                    for passage in document.best_passages
                ],
                metadata=document.metadata,
                candidate_chunk_count=document.candidate_chunk_count,
                best_chunk_score=document.best_chunk_score,
            )
            for document in result.documents
        ],
        diagnostics=DocumentRetrievalDiagnosticsResult(**asdict(result.diagnostics)),
    )


def _to_verified_retrieve_response(
    result: ConstraintRetrievalResult,
    *,
    include_rejected_documents: bool,
) -> VerifiedRetrieveResponse:
    return VerifiedRetrieveResponse(
        structured_query=_structured_query_result(result.structured_query),
        documents=[_verified_document_result(document) for document in result.verified_documents],
        rejected_documents=(
            [_verified_document_result(document) for document in result.rejected_documents]
            if include_rejected_documents
            else []
        ),
        diagnostics=ConstraintRetrievalDiagnosticsResult(
            query_interpretation_status=result.diagnostics.query_interpretation_status.value,
            hard_constraint_count=result.diagnostics.hard_constraint_count,
            soft_constraint_count=result.diagnostics.soft_constraint_count,
            candidate_chunks_retrieved=result.diagnostics.candidate_chunks_retrieved,
            candidate_documents_produced=result.diagnostics.candidate_documents_produced,
            documents_verified=result.diagnostics.documents_verified,
            verified_document_count=result.diagnostics.verified_document_count,
            excluded_hard_mismatch_count=result.diagnostics.excluded_hard_mismatch_count,
            excluded_not_proven_count=result.diagnostics.excluded_not_proven_count,
            verification_error_count=result.diagnostics.verification_error_count,
            final_document_count=result.diagnostics.final_document_count,
            retrieval_latency_ms=result.diagnostics.retrieval_latency_ms,
            verification_latency_ms=result.diagnostics.verification_latency_ms,
            total_latency_ms=result.diagnostics.total_latency_ms,
            latency_budget_ms=result.diagnostics.latency_budget_ms,
            latency_budget_exceeded=result.diagnostics.latency_budget_exceeded,
        ),
    )


def _structured_query_result(structured_query) -> StructuredQueryResult:
    return StructuredQueryResult(
        intent=structured_query.intent,
        status=structured_query.status.value,
        constraints=[
            StructuredConstraintResult(
                id=constraint.id,
                category=constraint.category.value,
                value=constraint.value,
                requirement=constraint.requirement.value,
                relation=(
                    _structured_relation_result(constraint.relation)
                    if constraint.relation is not None
                    else None
                ),
                description=constraint.description,
            )
            for constraint in structured_query.constraints
        ],
        entities=[
            StructuredEntityResult(
                id=entity.id,
                entity_type=entity.entity_type,
                role=entity.role,
                attributes=entity.attributes,
            )
            for entity in structured_query.entities
        ],
        relations=[
            _structured_relation_result(relation)
            for relation in structured_query.relations
        ],
        ambiguities=structured_query.ambiguities,
        retrieval_expansions=structured_query.retrieval_expansions,
        interpreter=structured_query.interpreter,
    )


def _structured_relation_result(relation) -> StructuredRelationResult:
    return StructuredRelationResult(
        subject=relation.subject,
        predicate=relation.predicate.value,
        object=relation.object,
        requirement=relation.requirement.value,
    )


def _verified_document_result(document) -> VerifiedDocumentResult:
    return VerifiedDocumentResult(
        document_id=document.document_id,
        score=document.score,
        decision_status=document.decision_status.value,
        constraint_results=[
            ConstraintVerificationResultModel(
                constraint_id=result.constraint_id,
                category=result.category.value,
                status=result.status.value,
                required_value=result.required_value,
                detected_value=result.detected_value,
                evidence=[
                    ConstraintEvidenceResult(
                        document_id=evidence.document_id,
                        chunk_id=evidence.chunk_id,
                        quote=evidence.quote,
                        source_field=evidence.source_field,
                    )
                    for evidence in result.evidence
                ],
                verification_method=result.verification_method.value,
                confidence=result.confidence,
                reason=result.reason,
            )
            for result in document.constraint_results
        ],
        supporting_passages=[
            ConstraintEvidenceResult(
                document_id=evidence.document_id,
                chunk_id=evidence.chunk_id,
                quote=evidence.quote,
                source_field=evidence.source_field,
            )
            for evidence in document.supporting_passages
        ],
        metadata=document.metadata,
        candidate_chunk_count=document.candidate_chunk_count,
    )


def _to_full_document_response(result: FullDocumentResult) -> FullDocumentResponse:
    return FullDocumentResponse(
        document_id=result.document_id,
        metadata=result.metadata,
        full_text=result.full_text,
        chunks=[
            FullDocumentChunkResult(
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                metadata=chunk.metadata,
            )
            for chunk in result.chunks
        ],
        source_url=result.source_url,
        provenance_status=result.provenance_status,
        full_text_availability_status=result.full_text_availability_status,
        diagnostics=FullDocumentDiagnosticsResult(**asdict(result.diagnostics)),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    pipeline: RetrievalPipeline = Depends(get_pipeline),
    answer_service: AnswerService = Depends(get_answer_service),
) -> SearchResponse:
    trace_event(logger, "api.rag.start", query_length=len(request.query), top_k=request.top_k)

    result = pipeline.run(request.query, top_k=request.top_k)
    answer = answer_service.generate(request.query, result.results)

    trace_event(logger, "api.rag.done", num_results=len(result.results))

    return SearchResponse(
        query=answer.query,
        summary=answer.summary,
        top_cases=answer.top_cases,
        excerpts=answer.excerpts,
    )


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(
    req: RetrieveRequest,
    orchestrator: OrchestratorService = Depends(get_orchestrator),
) -> RetrieveResponse:
    logger.info("[api] retrieve received query_length=%d", len(req.query))

    requested_sources = _normalize_filter_values(req.sources)
    fetch_limit = _raw_retrieve_limit(req.top_k, requested_sources)

    trace_event(
        logger,
        "api.retrieve.start",
        query_length=len(req.query),
        top_k=req.top_k,
        fetch_limit=fetch_limit,
        sources=sorted(requested_sources),
    )

    try:
        chunks = orchestrator.retrieve(req.query, top_k=fetch_limit)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[api] retrieve raised unexpectedly (%s); returning empty response", exc)
        return RetrieveResponse(results=[])

    filtered = [
        chunk for chunk in chunks
        if _matches_source_filters(chunk, requested_sources)
    ]
    if req.top_k >= 0:
        filtered = filtered[:req.top_k]

    results = [_to_retrieved_result(chunk) for chunk in filtered]

    logger.info("[api] retrieve completed results=%d", len(results))
    trace_event(logger, "api.retrieve.done", results=len(results))
    return RetrieveResponse(results=results)


@router.post("/retrieve-documents", response_model=DocumentRetrieveResponse)
def retrieve_documents(
    req: DocumentRetrieveRequest,
    orchestrator: OrchestratorService = Depends(get_orchestrator),
) -> DocumentRetrieveResponse:
    config = _document_retrieval_config()
    if not config.enabled:
        raise HTTPException(
            status_code=404,
            detail=(
                "Document-level retrieval is disabled. Set "
                "NALUS_DOCUMENT_RETRIEVAL_ENABLED=1 to enable the additive endpoint."
            ),
        )

    requested_sources = _normalize_filter_values(req.sources)
    trace_event(
        logger,
        "api.retrieve_documents.start",
        query_length=len(req.query),
        max_candidate_chunks=config.max_candidate_chunks,
        sources=sorted(requested_sources),
    )
    retrieval_started = time.perf_counter()
    try:
        candidate_chunks = orchestrator.retrieve(req.query, top_k=config.max_candidate_chunks)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[api] document retrieve raised unexpectedly (%s); returning empty response", exc)
        candidate_chunks = []
    retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000

    filtered_candidates = [
        chunk for chunk in candidate_chunks
        if _matches_source_filters(chunk, requested_sources)
    ]
    document_result = build_document_level_results(
        candidate_chunks=filtered_candidates,
        config=config,
        retrieval_latency_ms=retrieval_latency_ms,
    )
    trace_event(
        logger,
        "api.retrieve_documents.done",
        candidate_chunks_retrieved=document_result.diagnostics.candidate_chunks_retrieved,
        final_document_count=document_result.diagnostics.final_document_count,
        retrieval_latency_ms=round(retrieval_latency_ms, 3),
        aggregation_latency_ms=(
            round(document_result.diagnostics.aggregation_latency_ms, 3)
            if document_result.diagnostics.aggregation_latency_ms is not None
            else None
        ),
    )
    return _to_document_retrieve_response(document_result)


@router.post("/retrieve-verified", response_model=VerifiedRetrieveResponse)
def retrieve_verified(
    req: VerifiedRetrieveRequest,
    orchestrator: OrchestratorService = Depends(get_orchestrator),
    store: FullDocumentStore = Depends(get_full_document_store),
) -> VerifiedRetrieveResponse:
    endpoint_label = "/api/rag/retrieve-verified"
    config = _constraint_retrieval_config()
    if not config.enabled:
        record_constraint_retrieval_error(endpoint=endpoint_label, status="disabled")
        raise HTTPException(
            status_code=404,
            detail=(
                "Constraint-aware document verification is disabled. Set "
                "NALUS_CONSTRAINT_RETRIEVAL_ENABLED=1 to enable the additive endpoint."
            ),
        )

    requested_sources = _normalize_filter_values(req.sources)
    trace_event(
        logger,
        "api.retrieve_verified.start",
        query_length=len(req.query),
        max_candidate_chunks=config.max_candidate_chunks,
        sources=sorted(requested_sources),
    )

    try:
        result = retrieve_verified_documents(
            query=req.query,
            retriever=lambda query, top_k: orchestrator.retrieve(query, top_k=top_k),
            full_document_store=store,
            config=config,
            candidate_filter=(
                (lambda chunk: _matches_source_filters(chunk, requested_sources))
                if requested_sources
                else None
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[api] verified retrieve failed (%s)", exc)
        record_constraint_retrieval_error(endpoint=endpoint_label, status="error")
        raise HTTPException(
            status_code=503,
            detail="Constraint-aware retrieval is temporarily unavailable.",
        ) from exc

    record_constraint_retrieval_metrics(
        result,
        endpoint=endpoint_label,
        status="success",
    )
    trace_event(
        logger,
        "api.retrieve_verified.done",
        final_document_count=result.diagnostics.final_document_count,
        verified_document_count=result.diagnostics.verified_document_count,
        excluded_hard_mismatch_count=result.diagnostics.excluded_hard_mismatch_count,
        excluded_not_proven_count=result.diagnostics.excluded_not_proven_count,
    )
    return _to_verified_retrieve_response(
        result,
        include_rejected_documents=req.debug or config.include_rejected_documents,
    )


@router.get("/documents/{document_id}", response_model=FullDocumentResponse)
def get_full_document(
    document_id: str,
    store: FullDocumentStore = Depends(get_full_document_store),
) -> FullDocumentResponse:
    try:
        normalized_id = validate_document_id(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    trace_event(
        logger,
        "api.full_document.start",
        document_id_length=len(normalized_id),
    )

    try:
        result = store.get(normalized_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FullDocumentLookupError as exc:
        logger.warning("[api] full document lookup failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Full document lookup is temporarily unavailable.",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.warning("[api] full document lookup raised unexpectedly: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Full document lookup is temporarily unavailable.",
        ) from exc

    if result is None:
        trace_event(logger, "api.full_document.not_found")
        raise HTTPException(status_code=404, detail="Document was not found.")

    trace_event(
        logger,
        "api.full_document.done",
        chunk_count=result.diagnostics.chunk_count,
        full_text_availability_status=result.full_text_availability_status,
    )
    return _to_full_document_response(result)


@router.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    orchestrator: OrchestratorService = Depends(get_orchestrator),
) -> QueryResponse:
    logger.info("[api] query received query_length=%d", len(req.query))
    trace_event(logger, "api.query.start", query_length=len(req.query))

    cache_key = None
    if _query_cache is not None:
        cache_key = build_cache_key(req.query, corpus_version=_corpus_version)
        try:
            cached = _query_cache.get(cache_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[api] query cache read failed (%s)", exc)
            cached = None
        if cached is not None:
            logger.info("[api] query cache hit backend=%s", _query_cache_backend)
            return QueryResponse(
                answer=cached.answer,
                sources=cached.sources,
                plan_steps=cached.plan_steps,
            )

    try:
        result: OrchestratorResult = orchestrator.run(req.query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[api] orchestrator raised unexpectedly (%s); returning empty response", exc)
        return QueryResponse(answer="", sources=[], plan_steps=[])

    logger.info(
        "[api] query completed answer_length=%d sources=%d plan_steps=%d",
        len(result.answer),
        len(result.sources),
        len(result.plan_steps),
    )
    trace_event(
        logger, "api.query.done",
        answer_length=len(result.answer),
        sources=len(result.sources),
    )

    if _query_cache is not None and cache_key is not None:
        try:
            _query_cache.set(
                cache_key,
                CachedQueryResponse(
                    answer=result.answer,
                    sources=result.sources,
                    plan_steps=result.plan_steps,
                ),
                ttl_seconds=query_cache_ttl_seconds(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[api] query cache write failed (%s)", exc)

    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        plan_steps=result.plan_steps,
    )
