"""Evidence passage selectors for Cross-Encoder / future late-interaction rerankers."""

from __future__ import annotations

from app.rag.legal_v2.rerank.selectors.names import (
    DIVERSIFIED_STAGE1_EVIDENCE_V1,
    FIRST_N_STAGE1_ORDER_V1,
)
from app.rag.legal_v2.rerank.selectors.policy import (
    get_evidence_passage_selector,
    resolve_passage_selector_name,
)

__all__ = [
    "DIVERSIFIED_STAGE1_EVIDENCE_V1",
    "FIRST_N_STAGE1_ORDER_V1",
    "get_evidence_passage_selector",
    "resolve_passage_selector_name",
]
