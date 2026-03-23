"""
Embedder interface for the dense retrieval layer.

BaseEmbedder is injected into DenseRetriever so the embedding backend
(sentence-transformers, OpenAI, etc.) can be swapped without touching
the retriever.

MockEmbedder returns a fixed vector — useful for tests and local dev
until a real model is configured.
"""

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Return a dense vector representation of query."""


class MockEmbedder(BaseEmbedder):
    """Fixed-vector embedder.  For tests and offline development only."""

    def __init__(self, dim: int = 10) -> None:
        self._dim = dim

    def embed_query(self, query: str) -> list[float]:
        return [0.1] * self._dim
