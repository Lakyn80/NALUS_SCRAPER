from __future__ import annotations

from pathlib import Path
from typing import Any

from app.rag.retrieval.errors import RetrievalConfigurationError
from app.rag.retrieval.production_profile import ProductionRetrievalConfig


class BgeM3Embedder:
    """Lazy CPU/offline BGE-M3 embedder for production retrieval."""

    def __init__(self, config: ProductionRetrievalConfig, model: Any | None = None) -> None:
        self._config = config
        self._model = model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        if self._config.device.lower() != "cpu":
            raise RetrievalConfigurationError("Production BGE-M3 retrieval is configured for CPU only.")
        if self._config.trust_remote_code:
            raise RetrievalConfigurationError("EMBEDDING_TRUST_REMOTE_CODE must be disabled in production.")
        if not self._config.local_files_only:
            raise RetrievalConfigurationError("EMBEDDING_LOCAL_FILES_ONLY must be enabled in production.")
        model_path = Path(self._config.model_path)
        if not model_path.exists():
            raise RetrievalConfigurationError(f"BGE-M3 model path is missing: {model_path}")

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - depends on optional runtime package
            raise RetrievalConfigurationError(
                "sentence-transformers is required for BGE-M3 query embeddings."
            ) from exc

        self._model = SentenceTransformer(
            self._config.model_path,
            device=self._config.device,
            local_files_only=self._config.local_files_only,
            trust_remote_code=self._config.trust_remote_code,
        )

    def embed_query(self, query: str) -> list[float]:
        vectors = self.embed_texts([query])
        return vectors[0] if vectors else []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self.load()
        encoded = self._model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = [_to_float_list(vector) for vector in encoded]
        for index, vector in enumerate(vectors):
            if len(vector) != self._config.profile.embedding_dimension:
                raise RetrievalConfigurationError(
                    "BGE-M3 embedding dimension mismatch at vector "
                    f"{index}: {len(vector)} != {self._config.profile.embedding_dimension}"
                )
        return vectors


def _to_float_list(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]
