"""Thin wrappers around production and candidate Legal v2 chunkers."""

from __future__ import annotations

from app.rag.legal_v2.ingest.chunkers.contextual_packed_v1 import (
    ContextualPackedConfigV1,
    build_contextual_packed_chunks_v1,
)
from app.rag.legal_v2.ingest.chunkers.names import (
    CHUNKER_A_CURRENT,
    CHUNKER_B_CONTEXTUAL_PACKED_V1,
)
from app.rag.legal_v2.ingest.chunking import (
    HierarchicalChunkConfig,
    HierarchicalChunkingResult,
    build_hierarchical_chunks,
)
from app.rag.legal_v2.models import LegalDocumentStructure

KNOWN_EXPERIMENT_CHUNKERS: tuple[str, ...] = (
    CHUNKER_A_CURRENT,
    CHUNKER_B_CONTEXTUAL_PACKED_V1,
)


def chunk_document_for_experiment(
    document: LegalDocumentStructure,
    *,
    chunker_version: str,
) -> HierarchicalChunkingResult:
    """Dispatch experiment chunking. A always uses production hierarchical code."""
    if chunker_version == CHUNKER_A_CURRENT:
        return build_hierarchical_chunks(document, config=HierarchicalChunkConfig())
    if chunker_version == CHUNKER_B_CONTEXTUAL_PACKED_V1:
        return build_contextual_packed_chunks_v1(
            document, config=ContextualPackedConfigV1()
        )
    known = ", ".join(KNOWN_EXPERIMENT_CHUNKERS)
    raise ValueError(f"unknown experiment chunker_version={chunker_version!r}; known: {known}")
