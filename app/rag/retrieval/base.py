"""
Abstract base class for all retrievers.

Every concrete retriever (dense, keyword, hybrid…) must implement retrieve().
The interface is intentionally minimal so implementations can be swapped
without touching callers.
"""

from abc import ABC, abstractmethod

from app.rag.retrieval.models import RetrievedChunk


class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        """Return up to top_k chunks most relevant to query."""
