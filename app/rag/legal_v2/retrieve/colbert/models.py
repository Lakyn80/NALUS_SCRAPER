"""Colbert hit models compatible with shared retrieval chunk contracts.

Public chunk shape reuses ``RetrievedChunk`` from ``app.rag.retrieval.models``.
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


@dataclass(frozen=True)
class ColbertIndexBuildResult:
    """Outcome of an index build (integrity fields required for readiness)."""

    status: str
    source_collection: str
    expected_chunk_count: int
    indexed_chunk_count: int
    mapping_row_count: int
    duplicate_chunk_ids: int
    missing_chunk_ids: int
    empty_texts: int
    index_path: str
    mapping_path: str
    model_name: str
    library: str
    library_version: str
    device: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return (
            self.status == "ok"
            and self.indexed_chunk_count == self.expected_chunk_count
            and self.mapping_row_count == self.expected_chunk_count
            and self.duplicate_chunk_ids == 0
            and self.missing_chunk_ids == 0
            and self.empty_texts == 0
        )
