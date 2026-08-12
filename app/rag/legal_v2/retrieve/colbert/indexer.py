"""ColBERT indexer foundation API (no index build in this step)."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from app.rag.legal_v2.retrieve.colbert.backend import ColbertBackend, require_backend
from app.rag.legal_v2.retrieve.colbert.config import (
    COLBERT_PILOT_SOURCE_QDRANT_COLLECTION,
    ColbertConfig,
)
from app.rag.legal_v2.retrieve.colbert.errors import ColbertNotImplementedError


class ColbertIndexer:
    """Index-building facade. Requires an injected backend; does not open indexes.

    First planned corpus (not built here):
    ``COLBERT_PILOT_SOURCE_QDRANT_COLLECTION`` (Slice 4 B contextual).
    """

    def __init__(
        self,
        config: ColbertConfig,
        *,
        backend: ColbertBackend | None = None,
    ) -> None:
        config.validate()
        self._config = config
        self._backend = backend

    @property
    def config(self) -> ColbertConfig:
        return self._config

    @property
    def planned_source_collection(self) -> str:
        return COLBERT_PILOT_SOURCE_QDRANT_COLLECTION

    def build_index(
        self,
        documents: Iterable[Mapping[str, Any]] | None = None,
        *,
        source_collection: str | None = None,
    ) -> None:
        """Build a ColBERT index via the injected backend.

        Foundation: validates backend presence, then refuses — indexing is a
        later explicit step (no model download, no index write here).
        """
        require_backend(self._backend)
        planned = source_collection or self.planned_source_collection
        _ = documents  # reserved for the future indexing step
        raise ColbertNotImplementedError(
            "ColbertIndexer.build_index is foundation-only; indexing is not "
            f"enabled yet (planned source_collection={planned!r}, "
            f"index_path={self._config.index_path.as_posix()!r})."
        )
