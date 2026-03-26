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
from unittest.mock import MagicMock

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

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


# ---------------------------------------------------------------------------
# Live orchestrator — set by startup lifespan, None until Qdrant is ready
# ---------------------------------------------------------------------------

_live_orchestrator: OrchestratorService | None = None
_live_orchestrator_status: str = "pending"
_live_orchestrator_error: str | None = None
_background_ingest_status: str = "idle"
_background_ingest_error: str | None = None

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


@router.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    orchestrator: OrchestratorService = Depends(get_orchestrator),
) -> QueryResponse:
    logger.info("[api] query received query=%s", req.query)
    trace_event(logger, "api.query.start", query=req.query)

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

    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        plan_steps=result.plan_steps,
    )
