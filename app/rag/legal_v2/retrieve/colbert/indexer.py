"""ColBERT indexer (async orchestration; blocking work offloaded in backend)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from app.rag.legal_v2.retrieve.colbert.backend import ColbertBackend, require_backend
from app.rag.legal_v2.retrieve.colbert.config import (
    COLBERT_PILOT_SOURCE_QDRANT_COLLECTION,
    ColbertConfig,
)
from app.rag.legal_v2.retrieve.colbert.errors import ColbertConfigurationError
from app.rag.legal_v2.retrieve.colbert.models import ColbertIndexBuildResult


class ColbertIndexer:
    """Index-building facade over an injected async ColBERT backend."""

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
        return self._config.source_collection or COLBERT_PILOT_SOURCE_QDRANT_COLLECTION

    async def build(
        self,
        documents: Sequence[Mapping[str, Any]] | None = None,
        *,
        source_collection: str | None = None,
    ) -> ColbertIndexBuildResult:
        backend = require_backend(self._backend)
        if documents is None:
            raise ColbertConfigurationError(
                "documents are required for ColbertIndexer.build "
                "(export corpus before calling build)"
            )
        planned = source_collection or self.planned_source_collection
        return await backend.build_index(
            documents,
            source_collection=planned,
        )

    # Backwards-compatible alias used by early foundation callers/tests.
    async def build_index(
        self,
        documents: Sequence[Mapping[str, Any]] | None = None,
        *,
        source_collection: str | None = None,
    ) -> ColbertIndexBuildResult:
        return await self.build(documents, source_collection=source_collection)
