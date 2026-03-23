"""
Retrieval service — orchestrates query understanding and hybrid retrieval.

Flow:
  raw query
    → process_query        (normalize, keywords, legal concepts)
    → dense retriever      (semantic similarity, stub → Qdrant in prod)
    → keyword retriever    (lexical match)
    → fuse_results         (dedup, rank, top_k)
    → list[RetrievedChunk]

Usage:
    service = RetrievalService(
        dense=DenseRetriever(),
        keyword=KeywordRetriever(corpus),
    )
    results = service.search("žena cizinka unesla dítě do Ruska", top_k=5)
"""

from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.rag.query.query_processor import process_query
from app.rag.retrieval.base import BaseRetriever
from app.rag.retrieval.dense_retriever import DenseRetriever
from app.rag.retrieval.fusion import fuse_results
from app.rag.retrieval.keyword_retriever import KeywordRetriever
from app.rag.retrieval.models import RetrievedChunk

logger = get_logger(__name__)


class RetrievalService:
    """Thin orchestration layer over dense + keyword retrieval."""

    def __init__(
        self,
        dense: DenseRetriever,
        keyword: KeywordRetriever,
    ) -> None:
        self._dense: BaseRetriever = dense
        self._keyword: BaseRetriever = keyword

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Find the top_k most relevant chunks for a free-text query."""
        trace_event(logger, "retrieval.start", query=query, top_k=top_k)

        processed = process_query(query)
        trace_event(
            logger,
            "retrieval.processed",
            keywords=processed.keywords,
            concepts=processed.legal_concepts,
        )

        dense_results = self._dense.retrieve(
            processed.normalized_query, top_k=top_k
        )
        trace_event(logger, "retrieval.dense", count=len(dense_results))

        keyword_query = " ".join(processed.keywords)
        keyword_results = self._keyword.retrieve(keyword_query, top_k=top_k)
        trace_event(logger, "retrieval.keyword", count=len(keyword_results))

        fused = fuse_results(dense_results, keyword_results, top_k=top_k)
        trace_event(logger, "retrieval.fused", count=len(fused))

        return fused
