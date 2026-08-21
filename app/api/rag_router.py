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
from dataclasses import asdict, dataclass
from importlib import import_module
from threading import Lock
from typing import Any, Callable, Protocol
from unittest.mock import MagicMock

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.api.query_cache import (
    BaseQueryCache,
    CachedQueryResponse,
    build_cache_key,
    query_cache_ttl_seconds,
)
from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.observability.constraint_retrieval_metrics import (
    record_constraint_retrieval_error,
    record_constraint_retrieval_metrics,
)
from app.observability.legal_v2_metrics import record_request
from app.rag.answer.answer_service import AnswerService
from app.rag.clarification.orchestrator import ClarifyingOrchestratorService
from app.rag.execution.execution_service import ExecutionService
from app.rag.legal_v2.interpreter import DeepSeekQuerySpecProvider, QuerySpecProvider
from app.rag.legal_v2.indexing import LEGAL_V2_PROFILE
from app.rag.legal_v2.pipeline import (
    LegalV2SearchResult as RuntimeLegalV2SearchResult,
    legal_v2_search_enabled,
    search_legal_v2,
)
from app.rag.legal_v2.retriever import (
    LegalV2HybridRetriever,
    LegalV2RetrieverConfig,
    build_live_legal_v2_retriever,
    legal_v2_retriever_config_from_env,
)
from app.rag.legal_v2.verifier import DeepSeekSemanticVerifierProvider, SemanticVerifierProvider
from app.rag.llm.providers.deepseek import DeepSeekThinkingMode
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
from app.rag.retrieval.bge_m3_embedder import BgeM3Embedder
from app.rag.retrieval.full_document import (
    FullDocumentLookupError,
    FullDocumentResult,
    FullDocumentStore,
    QdrantFullDocumentStore,
    resolve_full_document_collection_name,
    validate_document_id,
)
from app.rag.retrieval.production_profile import ProductionRetrievalConfig
from app.rag.synthesis.synthesis_service import MockSynthesisLLM, SynthesisService

logger = get_logger(__name__)

router = APIRouter(prefix="/api/rag", tags=["rag"])

LEGAL_V2_MAX_QUERY_LENGTH = 4000
LEGAL_V2_MAX_REQUESTED_RESULTS = 50
_SAFE_TEXT_LIMIT = 500
_SAFE_LIST_LIMIT = 50
_SENSITIVE_RESPONSE_KEYS = {
    "api_key",
    "authorization",
    "bm25_sidecar_path",
    "body",
    "error",
    "full_text",
    "headers",
    "path",
    "prompt",
    "provider_error",
    "raw",
    "raw_body",
    "raw_diagnostics",
    "raw_provider_response",
    "response_body",
    "secret",
    "token",
}
_SENSITIVE_RESPONSE_KEY_FRAGMENTS = (
    "api_key",
    "authorization",
    "raw_",
    "_raw",
    "secret",
    "token",
    "prompt",
    "full_text",
    "judgment_text",
)


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


class LegalV2SearchRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str = Field(min_length=1, max_length=LEGAL_V2_MAX_QUERY_LENGTH)
    sources: list[str] | None = None
    max_results: int = Field(default=10, ge=1, le=LEGAL_V2_MAX_REQUESTED_RESULTS)
    debug: bool = False

    @field_validator("query")
    @classmethod
    def _query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class LegalV2EvidenceResult(BaseModel):
    constraint_id: str
    paragraph_ids: list[str]
    section_types: list[str]
    quote: str
    source_of_claim: str


class LegalV2VerifiedDocumentResult(BaseModel):
    document_id: str
    score: float
    status: str
    relevance_classification: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)
    evidence: list[LegalV2EvidenceResult]
    constraint_results: list[dict[str, Any]]
    verification_reason: str = ""
    verifier_diagnostics: dict[str, Any] = Field(default_factory=dict)
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None


class LegalV2SearchResponse(BaseModel):
    status: str
    interpretation_status: str
    query_spec_summary: dict[str, Any] | None = None
    verified_documents: list[LegalV2VerifiedDocumentResult]
    related_documents: list[LegalV2VerifiedDocumentResult] = Field(default_factory=list)
    rejected_documents: list[LegalV2VerifiedDocumentResult] = Field(default_factory=list)
    rejection_counts: dict[str, int] = Field(default_factory=dict)
    latency_ms_by_stage: dict[str, float] = Field(default_factory=dict)
    provider: dict[str, Any] = Field(default_factory=dict)
    index: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class SearchPipelineLike(Protocol):
    def run(self, query: str, top_k: int = 5) -> Any:
        ...


class OrchestratorLike(Protocol):
    def retrieve(self, query: str, top_k: int = 10) -> list[Any]:
        ...

    def run(self, query: str) -> OrchestratorResult:
        ...


LegalV2SearchCallable = Callable[..., RuntimeLegalV2SearchResult]


@dataclass(frozen=True)
class LegalV2Runtime:
    retriever: LegalV2HybridRetriever
    query_provider: QuerySpecProvider
    verifier: SemanticVerifierProvider
    config: LegalV2RetrieverConfig
    thinking_verifier: SemanticVerifierProvider | None = None
    search: LegalV2SearchCallable = search_legal_v2


LegalV2RuntimeProvider = Callable[[], LegalV2Runtime]


# ---------------------------------------------------------------------------
# Live orchestrator — set by startup lifespan, None until Qdrant is ready
# ---------------------------------------------------------------------------

_live_orchestrator: OrchestratorService | None = None
_live_orchestrator_status: str = "pending"
_live_orchestrator_error: str | None = None
_background_ingest_status: str = "idle"
_background_ingest_error: str | None = None
_corpus_version: str = "unknown"
_query_cache: BaseQueryCache | None = None
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
_legal_v2_runtime: LegalV2Runtime | None = None
_legal_v2_runtime_lock = Lock()

# ---------------------------------------------------------------------------
# Dependency providers
# ---------------------------------------------------------------------------


def _collection_name() -> str:
    """Collection for full-document reconstruction and related Qdrant reads.

    When Stage 1 / Legal v2 search is enabled, this follows
    ``NALUS_LEGAL_V2_QDRANT_COLLECTION`` so FE \"Celý rozsudek\" hits the same
    indexed corpus as search results.
    """
    return resolve_full_document_collection_name()


def get_pipeline() -> SearchPipelineLike:
    """
    Legacy /search endpoint compatibility pipeline.

    Production retrieval is wired through startup and /query or /retrieve. This
    fallback intentionally returns no retrieval results instead of using the old
    substring KeywordRetriever.
    """
    return _EmptyPipeline()


def get_answer_service() -> AnswerService:
    return AnswerService()


def get_orchestrator() -> OrchestratorLike:
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


def get_legal_v2_runtime() -> LegalV2Runtime:
    global _legal_v2_runtime
    if _legal_v2_runtime is not None:
        return _legal_v2_runtime
    with _legal_v2_runtime_lock:
        if _legal_v2_runtime is not None:
            return _legal_v2_runtime
        config = legal_v2_retriever_config_from_env()
        config.validate()
        api_key = os.getenv("LLM_API_KEY", "").strip()
        if not api_key or api_key == "your-api-key-here":
            raise RetrievalConfigurationError("Legal v2 search requires configured DeepSeek credentials.")
        qdrant_module = import_module("qdrant_client")
        qdrant_client_type = getattr(qdrant_module, "QdrantClient")
        client = qdrant_client_type(url=os.getenv("QDRANT_URL", "http://qdrant:6333"), timeout=10)
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
            max_candidate_count=max(config.dense_candidate_chunks, config.bm25_candidate_chunks),
            lexical_filter_enabled=False,
        )
        embedder = BgeM3Embedder(prod_config)
        retriever = build_live_legal_v2_retriever(client, embedder, config)
        query_provider = DeepSeekQuerySpecProvider(
            api_key,
            thinking=DeepSeekThinkingMode.ENABLED,
            timeout_seconds=float(os.getenv("NALUS_LEGAL_V2_QUERYSPEC_TIMEOUT_SECONDS", "120")),
        )
        verifier = DeepSeekSemanticVerifierProvider(
            api_key,
            thinking=DeepSeekThinkingMode.DISABLED,
            timeout_seconds=float(os.getenv("NALUS_LEGAL_V2_VERIFIER_TIMEOUT_SECONDS", "30")),
            max_tokens=1024,
        )
        thinking_verifier = DeepSeekSemanticVerifierProvider(
            api_key,
            thinking=DeepSeekThinkingMode.ENABLED,
            timeout_seconds=float(
                os.getenv("NALUS_LEGAL_V2_VERIFIER_THINKING_TIMEOUT_SECONDS", "120")
            ),
        )
        _legal_v2_runtime = LegalV2Runtime(
            retriever=retriever,
            query_provider=query_provider,
            verifier=verifier,
            thinking_verifier=thinking_verifier,
            config=config,
        )
        return _legal_v2_runtime


def get_legal_v2_runtime_provider() -> LegalV2RuntimeProvider:
    return get_legal_v2_runtime


def reset_legal_v2_runtime_for_tests() -> None:
    global _legal_v2_runtime
    with _legal_v2_runtime_lock:
        _legal_v2_runtime = None


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


def _to_legal_v2_search_response(result: RuntimeLegalV2SearchResult) -> LegalV2SearchResponse:
    return LegalV2SearchResponse(
        status=result.status,
        interpretation_status=result.interpretation_status,
        query_spec_summary=_safe_payload(result.query_spec_summary),
        verified_documents=[_legal_v2_document(document) for document in result.verified_documents],
        related_documents=[
            _legal_v2_document(document)
            for document in getattr(result, "related_documents", []) or []
        ],
        rejected_documents=[_legal_v2_document(document) for document in result.rejected_documents],
        rejection_counts=_safe_payload(result.rejection_counts),
        latency_ms_by_stage=_safe_payload(result.latency_ms_by_stage),
        provider=_safe_payload(result.provider),
        index=_safe_payload(result.index),
        diagnostics=_safe_payload(result.diagnostics),
    )


def _legal_v2_document(document) -> LegalV2VerifiedDocumentResult:
    return LegalV2VerifiedDocumentResult(
        document_id=document.document_id,
        score=document.score,
        status=document.status,
        relevance_classification=_bounded_safe_text(
            getattr(document, "relevance_classification", "unknown") or "unknown",
            limit=100,
        ),
        metadata=_safe_metadata_payload(document.metadata),
        evidence=[
            LegalV2EvidenceResult(
                constraint_id=str(item.get("constraint_id") or ""),
                paragraph_ids=[
                    _bounded_safe_text(str(paragraph_id), limit=200)
                    for paragraph_id in item.get("paragraph_ids", [])
                ],
                section_types=[
                    _bounded_safe_text(str(section_type), limit=100)
                    for section_type in item.get("section_types", [])
                ],
                quote=_bounded_safe_text(str(item.get("quote") or "")),
                source_of_claim=_bounded_safe_text(str(item.get("source_of_claim") or ""), limit=100),
            )
            for item in document.evidence
        ],
        constraint_results=_safe_payload(document.constraint_results),
        verification_reason=_bounded_safe_text(getattr(document, "verification_reason", "") or ""),
        verifier_diagnostics=_safe_payload(getattr(document, "verifier_diagnostics", {}) or {}),
        dense_rank=document.dense_rank,
        bm25_rank=document.bm25_rank,
        rrf_score=document.rrf_score,
    )


def _bounded_safe_text(text: str, *, limit: int = _SAFE_TEXT_LIMIT) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def _safe_payload(value: Any, *, key: str = "") -> Any:
    if _redact_response_key(key):
        return "[redacted]"
    if isinstance(value, str):
        return _bounded_safe_text(value)
    if isinstance(value, dict):
        return {
            str(item_key): _safe_payload(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [_safe_payload(item) for item in value[:_SAFE_LIST_LIMIT]]
    if isinstance(value, tuple):
        return [_safe_payload(item) for item in value[:_SAFE_LIST_LIMIT]]
    if isinstance(value, bool | int | float) or value is None:
        return value
    return _bounded_safe_text(str(value))


def _safe_metadata_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    forbidden = {
        "chunk_text",
        "full_text",
        "paragraph_original_texts",
        "paragraph_texts",
        "text",
    }
    return {
        str(key): _safe_payload(value, key=str(key))
        for key, value in metadata.items()
        if str(key) not in forbidden
    }


def _redact_response_key(key: str) -> bool:
    normalized = key.strip().lower()
    if normalized in _SENSITIVE_RESPONSE_KEYS:
        return True
    if normalized.endswith("_path"):
        return True
    return any(fragment in normalized for fragment in _SENSITIVE_RESPONSE_KEY_FRAGMENTS)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    pipeline: SearchPipelineLike = Depends(get_pipeline),
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
    orchestrator: OrchestratorLike = Depends(get_orchestrator),
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
    orchestrator: OrchestratorLike = Depends(get_orchestrator),
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
    orchestrator: OrchestratorLike = Depends(get_orchestrator),
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


class CaseSimilarityStage1SearchRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Raw paste may be long when long-input preprocessing is enabled; Stage 1
    # still searches only the condensed retrieval_query (<= 8000).
    query: str = Field(min_length=1, max_length=100_000)
    limit: int | None = Field(default=None, ge=1, le=50)
    include_debug: bool = False
    # Request-level profile: fast (default) | balanced | precise | ce7(alias of precise).
    # BALANCED requires NALUS_LEGAL_V2_COLBERT_ENABLED=1.
    # PRECISE/ce7 require NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED=1.
    retrieval_profile: str | None = Field(default="fast", max_length=32)

    @field_validator("query")
    @classmethod
    def _query_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be blank")
        return cleaned


class CaseSimilarityStage1PassageResult(BaseModel):
    text: str
    chunk_id: str
    section: str | None = None
    page: int | None = None
    score: float | None = None


class CaseSimilarityStage1DocumentResult(BaseModel):
    rank: int
    document_id: str
    canonical_document_id: str
    ecli: str
    court: str | None = None
    case_number: str | None = None
    decision_date: str | None = None
    document_type: str | None = None
    score: float
    relevant_passages: list[CaseSimilarityStage1PassageResult] = Field(default_factory=list)
    source_document_id: str | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    stage1_rank: int | None = None
    stage1_score: float | None = None
    ce_rank: int | None = None
    ce_score: float | None = None


class CaseSimilarityStage1SearchResponse(BaseModel):
    query: str
    result_count: int
    retrieval_stage: str
    results: list[CaseSimilarityStage1DocumentResult]
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class CaseSimilarityStage1ReadyResponse(BaseModel):
    ready: bool
    status: str
    collection: str | None = None
    bm25_index_id: str | None = None
    bm25_sidecar_exists: bool | None = None
    retrieval_stage: str | None = None
    error_type: str | None = None
    enabled: bool = False
    model_loaded: bool | None = None
    bm25_loaded: bool | None = None
    warmup_status: str | None = None
    warmup_required: bool | None = None
    warmup_latency_ms: float | None = None
    cross_encoder: dict[str, Any] | None = None


@router.get(
    "/legal-v2/case-similarity/ready",
    response_model=CaseSimilarityStage1ReadyResponse,
)
def case_similarity_stage1_ready() -> CaseSimilarityStage1ReadyResponse:
    from app.rag.legal_v2.retrieve.case_similarity_search import (
        case_similarity_stage1_enabled,
        probe_case_similarity_stage1_readiness,
    )

    enabled = case_similarity_stage1_enabled()
    if not enabled:
        return CaseSimilarityStage1ReadyResponse(
            ready=False,
            status="disabled",
            enabled=False,
            retrieval_stage="hybrid_rrf_stage_1",
        )
    payload = probe_case_similarity_stage1_readiness()
    return CaseSimilarityStage1ReadyResponse(
        ready=bool(payload.get("ready")),
        status=str(payload.get("status") or "unavailable"),
        collection=payload.get("collection"),
        bm25_index_id=payload.get("bm25_index_id"),
        bm25_sidecar_exists=payload.get("bm25_sidecar_exists"),
        retrieval_stage=payload.get("retrieval_stage"),
        error_type=payload.get("error_type"),
        enabled=True,
        model_loaded=payload.get("model_loaded"),
        bm25_loaded=payload.get("bm25_loaded"),
        warmup_status=payload.get("warmup_status"),
        warmup_required=payload.get("warmup_required"),
        warmup_latency_ms=payload.get("warmup_latency_ms"),
        cross_encoder=payload.get("cross_encoder"),
    )


@router.post(
    "/legal-v2/case-similarity/search",
    response_model=CaseSimilarityStage1SearchResponse,
)
async def case_similarity_stage1_search(
    req: CaseSimilarityStage1SearchRequest,
) -> CaseSimilarityStage1SearchResponse:
    endpoint_label = "/api/rag/legal-v2/case-similarity/search"
    from app.rag.legal_v2.retrieve.case_similarity_search import (
        case_similarity_stage1_enabled,
        max_result_limit,
        search_case_similarity_stage1,
        stage1_debug_allowed,
    )
    from app.rag.retrieval.errors import RetrievalConfigurationError

    if not case_similarity_stage1_enabled():
        record_request(endpoint=endpoint_label, status="disabled")
        raise HTTPException(
            status_code=404,
            detail=(
                "Legal v2 Stage 1 case-similarity search is disabled. Set "
                "NALUS_LEGAL_V2_CASE_SIMILARITY_ENABLED=1 (or NALUS_LEGAL_V2_SEARCH_ENABLED=1)."
            ),
        )

    include_debug = bool(req.include_debug) and stage1_debug_allowed()
    if req.limit is not None and req.limit > max_result_limit():
        raise HTTPException(
            status_code=422,
            detail=f"limit must be <= {max_result_limit()}",
        )

    trace_event(
        logger,
        "api.legal_v2.case_similarity.search.start",
        query_length=len(req.query),
        limit=req.limit,
        include_debug=include_debug,
        retrieval_profile=req.retrieval_profile,
    )
    try:
        result = await search_case_similarity_stage1(
            query=req.query,
            limit=req.limit,
            include_debug=include_debug,
            retrieval_profile=req.retrieval_profile,
        )
    except ValueError as exc:
        record_request(endpoint=endpoint_label, status="validation_error")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RetrievalConfigurationError as exc:
        logger.warning(
            "[api] legal v2 stage1 configuration error type=%s",
            exc.__class__.__name__,
        )
        record_request(endpoint=endpoint_label, status="unavailable")
        raise HTTPException(
            status_code=503,
            detail="Legal v2 Stage 1 retrieval dependencies are unavailable.",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[api] legal v2 stage1 search failed type=%s",
            exc.__class__.__name__,
        )
        record_request(endpoint=endpoint_label, status="error")
        raise HTTPException(
            status_code=503,
            detail="Legal v2 Stage 1 search is temporarily unavailable.",
        ) from exc

    record_request(endpoint=endpoint_label, status="success")
    logger.info(
        "[api] legal_v2 stage1 search done result_count=%s collection=%s",
        result.result_count,
        (result.diagnostics or {}).get("collection"),
    )
    return CaseSimilarityStage1SearchResponse(
        query=result.query,
        result_count=result.result_count,
        retrieval_stage=result.retrieval_stage,
        results=[
            CaseSimilarityStage1DocumentResult(
                rank=item.rank,
                document_id=item.document_id,
                canonical_document_id=item.canonical_document_id,
                ecli=item.ecli,
                court=item.court,
                case_number=item.case_number,
                decision_date=item.decision_date,
                document_type=item.document_type,
                score=item.score,
                relevant_passages=[
                    CaseSimilarityStage1PassageResult(
                        text=passage.text,
                        chunk_id=passage.chunk_id,
                        section=passage.section,
                        page=passage.page,
                        score=passage.score,
                    )
                    for passage in item.relevant_passages
                ],
                source_document_id=item.source_document_id,
                dense_rank=item.dense_rank,
                bm25_rank=item.bm25_rank,
                rrf_score=item.rrf_score,
                metadata=item.metadata,
                stage1_rank=item.stage1_rank,
                stage1_score=item.stage1_score,
                ce_rank=item.ce_rank,
                ce_score=item.ce_score,
            )
            for item in result.results
        ],
        diagnostics=result.diagnostics if include_debug else {
            key: value
            for key, value in result.diagnostics.items()
            if key
            in {
                "query_length",
                "generated_query_count",
                "result_count",
                "collection",
                "bm25_index_id",
                "queryspec_latency_ms",
                "dense_latency_ms",
                "bm25_latency_ms",
                "rrf_latency_ms",
                "aggregation_latency_ms",
                "total_latency_ms",
                "dense_candidate_chunks",
                "bm25_candidate_chunks",
                "fused_candidate_chunks",
                "aggregated_documents",
                "retrieval_status",
                "original_query_length",
                "input_processing",
                "rerank",
                "fast_retrieval_profile",
                "fast_retrieval_profile_source",
                "fast_dense_variant",
                "fast_dense_variant_source",
                "fast_dense_variant_applied",
                "use_quantization_search_params",
                "dense_enabled",
                "bm25_enabled",
            }
        },
    )


@router.post("/search-v2", response_model=LegalV2SearchResponse)
def search_v2(
    req: LegalV2SearchRequest,
    runtime_provider: LegalV2RuntimeProvider = Depends(get_legal_v2_runtime_provider),
) -> LegalV2SearchResponse:
    endpoint_label = "/api/rag/search-v2"
    if not legal_v2_search_enabled():
        record_request(endpoint=endpoint_label, status="disabled")
        raise HTTPException(
            status_code=404,
            detail=(
                "Legal Retrieval v2 search is disabled. Set "
                "NALUS_LEGAL_V2_SEARCH_ENABLED=1 to enable the isolated endpoint."
            ),
        )
    requested_sources = _normalize_filter_values(req.sources)
    trace_event(
        logger,
        "api.legal_v2.search.start",
        query_length=len(req.query),
        sources=sorted(requested_sources),
        max_results=req.max_results,
    )
    try:
        runtime = runtime_provider()
        config = runtime.config
        bounded_config = LegalV2RetrieverConfig(
            qdrant_collection=config.qdrant_collection,
            bm25_sidecar_path=config.bm25_sidecar_path,
            bm25_index_id=config.bm25_index_id,
            model_path=config.model_path,
            dense_candidate_chunks=config.dense_candidate_chunks,
            bm25_candidate_chunks=config.bm25_candidate_chunks,
            fused_candidate_chunks=config.fused_candidate_chunks,
            candidate_documents=config.candidate_documents,
            returned_verified_documents=max(1, min(req.max_results, config.returned_verified_documents)),
            evidence_windows_per_constraint=config.evidence_windows_per_constraint,
        )
        bounded_config.validate()
        result = runtime.search(
            query=req.query,
            retriever=runtime.retriever,
            verifier=runtime.verifier,
            thinking_verifier=runtime.thinking_verifier,
            config=bounded_config,
            query_provider=runtime.query_provider,
            source_filter=requested_sources,
            debug=req.debug,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[api] legal v2 search failed type=%s", exc.__class__.__name__)
        record_request(endpoint=endpoint_label, status="error")
        raise HTTPException(status_code=503, detail="Legal Retrieval v2 search is temporarily unavailable.") from exc
    record_request(endpoint=endpoint_label, status=result.status)
    logger.info(
        "[api] legal_v2 search done status=%s interpretation=%s verified=%s related=%s rejected=%s collection=%s",
        result.status,
        result.interpretation_status,
        len(result.verified_documents),
        len(getattr(result, "related_documents", []) or []),
        len(result.rejected_documents),
        getattr(runtime.config, "qdrant_collection", None),
    )
    trace_event(
        logger,
        "api.legal_v2.search.done",
        status=result.status,
        verified_count=len(result.verified_documents),
    )
    return _to_legal_v2_search_response(result)


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
    orchestrator: OrchestratorLike = Depends(get_orchestrator),
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
