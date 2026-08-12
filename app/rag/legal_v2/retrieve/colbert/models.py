"""ColBERT hit models compatible with shared retrieval chunk contracts.

Public chunk shape reuses ``RetrievedChunk`` from ``app.rag.retrieval.models``.
ColBERT-specific ranked hits carry document/chunk identity explicitly and can
convert to that shared model without inventing a parallel public result type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.rag.retrieval.models import RetrievedChunk

COLBERT_SOURCE = "colbert"


@dataclass(frozen=True)
class ColbertHit:
    """One ranked ColBERT evidence hit (chunk-level)."""

    document_id: str
    chunk_id: str
    rank: int
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_retrieved_chunk(self) -> RetrievedChunk:
        """Map onto the shared retrieval-layer chunk model."""
        metadata = dict(self.metadata)
        metadata.setdefault("document_id", self.document_id)
        metadata.setdefault("chunk_id", self.chunk_id)
        metadata.setdefault("rank", self.rank)
        return RetrievedChunk(
            id=self.chunk_id,
            text=self.text,
            score=float(self.score),
            source=COLBERT_SOURCE,
            metadata=metadata,
        )


@dataclass(frozen=True)
class ColbertRetrievalResult:
    """Ranked ColBERT hits plus light diagnostics (no fake fallbacks)."""

    hits: tuple[ColbertHit, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_retrieved_chunks(self) -> list[RetrievedChunk]:
        return [hit.to_retrieved_chunk() for hit in self.hits]
