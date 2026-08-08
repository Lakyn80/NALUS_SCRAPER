"""Resolve named passage-selector policies."""

from __future__ import annotations

from app.rag.legal_v2.rerank.selectors.base import EvidencePassageSelector
from app.rag.legal_v2.rerank.selectors.diversified_stage1_evidence_v1 import (
    DiversifiedStage1EvidenceSelectorV1,
)
from app.rag.legal_v2.rerank.selectors.first_n_stage1_order_v1 import (
    FirstNStage1OrderSelectorV1,
)
from app.rag.legal_v2.rerank.selectors.names import (
    DIVERSIFIED_STAGE1_EVIDENCE_V1,
    FIRST_N_STAGE1_ORDER_V1,
)

_REGISTRY: dict[str, type] = {
    FIRST_N_STAGE1_ORDER_V1: FirstNStage1OrderSelectorV1,
    DIVERSIFIED_STAGE1_EVIDENCE_V1: DiversifiedStage1EvidenceSelectorV1,
}


def resolve_passage_selector_name(name: str | None) -> str:
    cleaned = str(name or FIRST_N_STAGE1_ORDER_V1).strip() or FIRST_N_STAGE1_ORDER_V1
    if cleaned not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unknown passage_selector={cleaned!r}; known: {known}")
    return cleaned


def get_evidence_passage_selector(name: str | None = None) -> EvidencePassageSelector:
    policy = resolve_passage_selector_name(name)
    return _REGISTRY[policy]()
