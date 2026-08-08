"""Selector protocol for CE evidence passages."""

from __future__ import annotations

from typing import Protocol, Sequence

from app.rag.legal_v2.rerank.models import RerankCandidate, RerankPassage


class EvidencePassageSelector(Protocol):
    """Narrow abstraction reusable by CE, ColBERT, or other passage scorers."""

    policy_id: str

    def select(
        self,
        candidate: RerankCandidate,
        *,
        limit: int,
    ) -> Sequence[RerankPassage]:
        """Select up to ``limit`` passages for one candidate document.

        Must be deterministic, label-agnostic, and free of randomness.
        """
        ...
