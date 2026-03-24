"""
FastAPI router for RAG search.

Usage (mount in your FastAPI app):
    from app.api.rag_router import router as rag_router
    app.include_router(rag_router)

The router expects a RetrievalPipeline and AnswerService to be injected
via FastAPI dependency injection — override `get_pipeline` and
`get_answer_service` for testing or alternative implementations.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.rag.answer.answer_service import AnswerService
from app.rag.orchestration.pipeline import RetrievalPipeline
from app.rag.retrieval.retrieval_service import RetrievalService
from app.rag.retrieval.keyword_retriever import KeywordRetriever
from app.rag.retrieval.dense_retriever import DenseRetriever
from app.rag.retrieval.embedder import MockEmbedder
from unittest.mock import MagicMock

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


# ---------------------------------------------------------------------------
# Dependency providers
# ---------------------------------------------------------------------------


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
        collection_name="nalus",
        embedder=MockEmbedder(),
    )
    keyword = KeywordRetriever(corpus=[])
    service = RetrievalService(dense=dense, keyword=keyword)
    return RetrievalPipeline(service)


def get_answer_service() -> AnswerService:
    return AnswerService()


# ---------------------------------------------------------------------------
# Endpoint
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
