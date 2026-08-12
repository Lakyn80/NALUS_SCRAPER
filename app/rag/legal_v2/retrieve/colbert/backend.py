"""ColBERT backend dependency boundary (lazy; no import-time loading).

External ColBERT libraries must only be imported behind this boundary and only
when a concrete backend implementation is constructed in a later step.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from app.rag.legal_v2.retrieve.colbert.errors import ColbertBackendUnavailableError
from app.rag.legal_v2.retrieve.colbert.models import ColbertHit


class ColbertBackend(Protocol):
    """Narrow injectable backend for search (and later indexing)."""

    def search(self, query: str, *, top_k: int) -> Sequence[ColbertHit]:
        """Return ranked ColBERT hits for ``query``."""
        ...


def require_backend(backend: ColbertBackend | None) -> ColbertBackend:
    """Fail explicitly when no backend is injected (no silent mock fallback)."""
    if backend is None:
        raise ColbertBackendUnavailableError(
            "ColBERT backend is not configured. Inject a ColbertBackend "
            "implementation; foundation does not load or mock a library."
        )
    return backend


def import_colbert_library() -> None:
    """Lazy library probe for a future concrete backend.

    Intentionally raises: no ColBERT package is installed or wired in this step.
    Safe to call only from backend constructors — never at module import.
    """
    raise ColbertBackendUnavailableError(
        "ColBERT library is not installed or wired yet "
        "(foundation boundary only; no download/install in this step)."
    )
