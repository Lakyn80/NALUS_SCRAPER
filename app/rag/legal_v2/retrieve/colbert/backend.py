"""ColBERT backend dependency boundary (async Protocol; lazy library import)."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from app.rag.legal_v2.retrieve.colbert.errors import ColbertBackendUnavailableError
from app.rag.legal_v2.retrieve.colbert.models import (
    ColbertHit,
    ColbertIndexBuildResult,
)


class ColbertBackend(Protocol):
    """Injectable async ColBERT backend (search + index build)."""

    async def initialize(self) -> None:
        """Lazy-load model / open index. Safe to call more than once."""
        ...

    async def close(self) -> None:
        """Release resources."""
        ...

    async def search(self, query: str, *, top_k: int) -> Sequence[ColbertHit]:
        """Return ranked ColBERT hits for ``query``."""
        ...

    async def build_index(
        self,
        documents: Sequence[Mapping[str, Any]],
        *,
        source_collection: str | None = None,
    ) -> ColbertIndexBuildResult:
        """Build index + mapping from source chunk rows."""
        ...


def require_backend(backend: ColbertBackend | None) -> ColbertBackend:
    """Fail explicitly when no backend is injected (no silent mock fallback)."""
    if backend is None:
        raise ColbertBackendUnavailableError(
            "ColBERT backend is not configured. Inject a ColbertBackend "
            "implementation; foundation does not load or mock a library."
        )
    return backend


def import_colbert_library() -> Any:
    """Lazy import of the optional PyLate dependency.

    Never called at module import time.
    """
    try:
        import pylate  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ColbertBackendUnavailableError(
            "ColBERT library 'pylate' is not installed. "
            "Install optional deps from requirements-colbert.txt."
        ) from exc
    return pylate
