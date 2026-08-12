"""Legal v2 ColBERT retrieval package (async-first; no import-time side effects)."""

from __future__ import annotations

from app.rag.legal_v2.retrieve.colbert.backend import (
    ColbertBackend,
    import_colbert_library,
    require_backend,
)
from app.rag.legal_v2.retrieve.colbert.config import (
    COLBERT_PILOT_EXPECTED_CHUNK_COUNT,
    COLBERT_PILOT_SOURCE_QDRANT_COLLECTION,
    DEFAULT_COLBERT_MODEL,
    DEFAULT_INDEX_NAME,
    ColbertConfig,
)
from app.rag.legal_v2.retrieve.colbert.errors import (
    ColbertBackendUnavailableError,
    ColbertConfigurationError,
    ColbertError,
    ColbertIndexError,
    ColbertMappingError,
    ColbertNotImplementedError,
)
from app.rag.legal_v2.retrieve.colbert.indexer import ColbertIndexer
from app.rag.legal_v2.retrieve.colbert.models import (
    COLBERT_SOURCE,
    ColbertHit,
    ColbertIndexBuildResult,
    ColbertRetrievalResult,
)
from app.rag.legal_v2.retrieve.colbert.pylate_backend import PyLateColbertBackend
from app.rag.legal_v2.retrieve.colbert.retriever import ColbertRetriever

__all__ = [
    "COLBERT_PILOT_EXPECTED_CHUNK_COUNT",
    "COLBERT_PILOT_SOURCE_QDRANT_COLLECTION",
    "COLBERT_SOURCE",
    "DEFAULT_COLBERT_MODEL",
    "DEFAULT_INDEX_NAME",
    "ColbertBackend",
    "ColbertBackendUnavailableError",
    "ColbertConfig",
    "ColbertConfigurationError",
    "ColbertError",
    "ColbertHit",
    "ColbertIndexBuildResult",
    "ColbertIndexError",
    "ColbertIndexer",
    "ColbertMappingError",
    "ColbertNotImplementedError",
    "ColbertRetrievalResult",
    "ColbertRetriever",
    "PyLateColbertBackend",
    "import_colbert_library",
    "require_backend",
]
