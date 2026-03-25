"""
Embedder interface for the dense retrieval layer.

BaseEmbedder is injected into DenseRetriever so the embedding backend
(sentence-transformers, OpenAI, etc.) can be swapped without touching
the retriever.

MockEmbedder returns a fixed vector — useful for tests and local dev
until a real model is configured.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEmbedder(ABC):

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Return a dense vector representation of query."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return dense vectors for multiple texts.

        The default implementation keeps the interface backward-compatible and
        simply calls `embed_query()` for each text.
        """
        return [self.embed_query(text) for text in texts]


class MockEmbedder(BaseEmbedder):
    """Fixed-vector embedder.  For tests and offline development only."""

    def __init__(self, dim: int = 10) -> None:
        self._dim = dim

    def embed_query(self, query: str) -> list[float]:
        return [0.1] * self._dim


class SentenceTransformersEmbedder(BaseEmbedder):
    """Real dense embedder backed by sentence-transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model: Any | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        if model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]

            model = SentenceTransformer(model_name)

        self._model_name = model_name
        self._model = model
        self._batch_size = batch_size
        self._normalize_embeddings = normalize_embeddings
        self._dim = self._infer_dim()

    @property
    def dim(self) -> int:
        return self._dim

    def embed_query(self, query: str) -> list[float]:
        vectors = self.embed_documents([query])
        return vectors[0] if vectors else []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        encoded = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize_embeddings,
            show_progress_bar=False,
        )
        return [_to_float_list(vector) for vector in encoded]

    def _infer_dim(self) -> int:
        getter = getattr(self._model, "get_sentence_embedding_dimension", None)
        if callable(getter):
            dim = getter()
            if dim:
                return int(dim)
        return len(self.embed_query("dimension probe"))


def _to_float_list(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]
