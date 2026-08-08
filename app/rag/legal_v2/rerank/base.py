"""Reranker provider protocol for Legal v2."""

from __future__ import annotations

from typing import Protocol, Sequence

from app.rag.legal_v2.rerank.models import RerankPassage, RerankScore


class Reranker(Protocol):
    """Narrow scoring interface — model details stay in the provider."""

    @property
    def model_id(self) -> str: ...

    @property
    def device(self) -> str: ...

    @property
    def is_loaded(self) -> bool: ...

    def load(self) -> None: ...

    def score(
        self,
        query: str,
        passages: Sequence[RerankPassage],
    ) -> Sequence[RerankScore]: ...
