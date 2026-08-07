"""SearchBrief provider abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.rag.legal_v2.query_input.config import LongInputConfig
from app.rag.legal_v2.query_input.models import SearchBrief


@dataclass(frozen=True)
class CondensationRequest:
    raw_text: str
    normalized_text: str
    config: LongInputConfig


class SearchBriefProvider(Protocol):
    name: str

    def condense(self, request: CondensationRequest) -> SearchBrief:
        ...
