"""
Abstract base class for rerankers.

A reranker takes a query and a list of already-retrieved chunks and
returns a re-scored, re-ordered subset of them.  The interface is
intentionally minimal so a CrossEncoder or LLM reranker can be plugged
in later without touching callers.
"""

from abc import ABC, abstractmethod

from app.rag.retrieval.models import RetrievedChunk


class BaseReranker(ABC):

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Return up to top_k chunks re-ordered by relevance to query."""
