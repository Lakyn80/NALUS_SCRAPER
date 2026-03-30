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
from typing import Any
from unittest.mock import MagicMock

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.query_cache import CachedQueryResponse, build_cache_key, query_cache_ttl_seconds
from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.rag.answer.answer_service import AnswerService
from app.rag.execution.execution_service import ExecutionService
from app.rag.orchestration.pipeline import RetrievalPipeline
from app.rag.orchestrator.orchestrator_service import OrchestratorResult, OrchestratorService
from app.rag.planner.planner_service import MockPlannerLLM, PlannerService
from app.rag.retrieval.dense_retriever import DenseRetriever
from app.rag.retrieval.embedder import MockEmbedder
from app.rag.retrieval.keyword_retriever import KeywordRetriever
from app.rag.retrieval.retrieval_service import RetrievalService
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
_SOURCE_FILTER_ALIASES: dict[str, set[str]] = {
    "constitutional": {"constitutional", "nalus"},
    "supreme": {"supreme"},
    "administrative": {"administrative"},
}

# ---------------------------------------------------------------------------
# Dependency providers
# ---------------------------------------------------------------------------


def _collection_name() -> str:
    return os.getenv("QDRANT_COLLECTION_NAME", "nalus")


def get_pipeline() -> RetrievalPipeline:
    """
    Default pipeline wired with stub retrievers.

    Override this dependency in production to inject real Qdrant client
    and embedder:

        app.dependency_overrides[get_pipeline] = lambda: RetrievalPipeline(
            RetrievalService(
                dense=DenseRetriever(client=real_client, collection_name="nalus", embedder=RealEmbedder()),
                keyword=KeywordRetriever(corpus=loaded_corpus),
            )
        )
    """
    mock_client = MagicMock()
    mock_client.search.return_value = []
    dense = DenseRetriever(
        client=mock_client,
        collection_name=_collection_name(),
        embedder=MockEmbedder(),
    )
    keyword = KeywordRetriever(corpus=[])
    service = RetrievalService(dense=dense, keyword=keyword)
    return RetrievalPipeline(service)


def get_answer_service() -> AnswerService:
    return AnswerService()


def get_orchestrator() -> OrchestratorService:
    """
    Returns the live orchestrator (real Qdrant + corpus) if startup succeeded,
    otherwise falls back to a stub with keyword-only retrieval.
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

    # Fallback stub (used when Qdrant is not available or in tests)
    mock_client = MagicMock()
    mock_client.query_points.return_value = MagicMock(points=[])
    dense = DenseRetriever(
        client=mock_client,
        collection_name=_collection_name(),
        embedder=MockEmbedder(),
    )
    keyword = KeywordRetriever(corpus=[])
    retrieval = RetrievalService(dense=dense, keyword=keyword)
    return OrchestratorService(
        planner=PlannerService(llm=MockPlannerLLM()),
        execution=ExecutionService(retrieval_service=retrieval),
        synthesis=SynthesisService(llm=MockSynthesisLLM()),
    )


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
        tags.add(raw_source.lower())
        if raw_source.lower() == "nalus":
            tags.add("constitutional")

    court_name = _normalize_text(metadata.get("court_name"))
    if court_name:
        normalized_court = court_name.lower()
        tags.add(normalized_court)
        if "ústavní" in normalized_court or "ustavni" in normalized_court:
            tags.add("constitutional")
        elif "nejvyšší správní" in normalized_court or "nejvyssi spravni" in normalized_court:
            tags.add("administrative")
        elif "nejvyšší" in normalized_court or "nejvyssi" in normalized_court:
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
        or metadata.get("sp_zn")
    )
    source = _normalize_text(metadata.get("source"))
    court_name = _normalize_text(metadata.get("court_name"))
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    pipeline: RetrievalPipeline = Depends(get_pipeline),
    answer_service: AnswerService = Depends(get_answer_service),
) -> SearchResponse:
    trace_event(logger, "api.rag.start", query=request.query, top_k=request.top_k)

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
    logger.info("[api] retrieve received query=%s", req.query)

    requested_sources = _normalize_filter_values(req.sources)
    fetch_limit = _raw_retrieve_limit(req.top_k, requested_sources)

    trace_event(
        logger,
        "api.retrieve.start",
        query=req.query,
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


@router.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    orchestrator: OrchestratorService = Depends(get_orchestrator),
) -> QueryResponse:
    logger.info("[api] query received query=%s", req.query)
    trace_event(logger, "api.query.start", query=req.query)

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
