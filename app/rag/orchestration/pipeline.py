"""
Thin orchestration layer for the NALUS RAG pipeline.

Composes existing services — contains no business logic itself.
Designed to be replaced by LangGraph when the time comes, without
touching any of the underlying modules.

Usage:
    from app.rag.orchestration.pipeline import RetrievalPipeline
    from app.rag.retrieval.retrieval_service import RetrievalService
    from app.rag.retrieval.dense_retriever import DenseRetriever
    from app.rag.retrieval.keyword_retriever import KeywordRetriever

    service = RetrievalService(
        dense=DenseRetriever(),
        keyword=KeywordRetriever(corpus),
    )
    pipeline = RetrievalPipeline(service)
    result = pipeline.run("žena cizinka unesla dítě do Ruska", top_k=5)
"""

from dataclasses import dataclass

from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.rag.query.query_processor import ProcessedQuery, process_query
from app.rag.retrieval.models import RetrievedChunk
from app.rag.retrieval.retrieval_service import RetrievalService

logger = get_logger(__name__)


@dataclass
class RetrievalPipelineResult:
    processed_query: ProcessedQuery
    results: list[RetrievedChunk]


class RetrievalPipeline:
    """Orchestrates query processing and retrieval into a single call."""

    def __init__(self, retrieval_service: RetrievalService) -> None:
        self._service = retrieval_service

    def run(self, query: str, top_k: int = 5) -> RetrievalPipelineResult:
        """Execute the full retrieval pipeline for a user query."""
        trace_event(logger, "pipeline.start", query=query, top_k=top_k)

        processed = process_query(query)
        trace_event(
            logger,
            "pipeline.processed",
            normalized_query=processed.normalized_query,
            keywords=processed.keywords,
            concepts=processed.legal_concepts,
        )

        results = self._service.search(query, top_k=top_k)
        trace_event(logger, "pipeline.done", result_count=len(results))

        return RetrievalPipelineResult(processed_query=processed, results=results)
