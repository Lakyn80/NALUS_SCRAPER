"""Chunker policy IDs for Legal v2 experiments."""

from __future__ import annotations

# Production baseline (must match app.rag.legal_v2.audit.CHUNKER_VERSION).
CHUNKER_A_CURRENT = "legal_v2_hierarchical_chunker_v1"

# Candidate experiment policy — frozen knobs live in ContextualPackedConfigV1.
CHUNKER_B_CONTEXTUAL_PACKED_V1 = "legal_contextual_packed_v1"
