"""Legal v2 ColBERT retrieval foundation (no model/index side effects on import)."""

from __future__ import annotations

from app.rag.legal_v2.retrieve.colbert.backend import (
    ColbertBackend,
    import_colbert_library,
    require_backend,
)
from app.rag.legal_v2.retrieve.colbert.config import (
    COLBERT_PILOT_SOURCE_QDRANT_COLLECTION,
    ColbertConfig,
)
from app.rag.legal_v2.retrieve.colbert.errors import (
    ColbertBackendUnavailableError,
    ColbertConfigurationError,
    ColbertError,
    ColbertNotImplementedError,
)
from app.rag.legal_v2.retrieve.colbert.indexer import ColbertIndexer
from app.rag.legal_v2.retrieve.colbert.models import (
    COLBERT_SOURCE,
    ColbertHit,
    ColbertRetrievalResult,
)
from app.rag.legal_v2.retrieve.colbert.retriever import ColbertRetriever

__all__ = [
    "COLBERT_PILOT_SOURCE_QDRANT_COLLECTION",
    "COLBERT_SOURCE",
    "ColbertBackend",
    "ColbertBackendUnavailableError",
    "ColbertConfig",
    "ColbertConfigurationError",
    "ColbertError",
    "ColbertHit",
    "ColbertIndexer",
    "ColbertNotImplementedError",
    "ColbertRetrievalResult",
    "ColbertRetriever",
    "import_colbert_library",
    "require_backend",
]
