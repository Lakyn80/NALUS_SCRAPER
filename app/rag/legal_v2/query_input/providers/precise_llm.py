"""Future PRECISE LLM SearchBrief provider stub (no external calls)."""

from __future__ import annotations

from app.rag.legal_v2.query_input.errors import UnsupportedCondensationModeError
from app.rag.legal_v2.query_input.models import SearchBrief
from app.rag.legal_v2.query_input.providers.base import CondensationRequest


class PreciseLLMSearchBriefProvider:
    """Interface placeholder for a future paid/offline LLM condensation mode.

    Must never silently call an external LLM from this stub.
    """

    name = "precise"

    def condense(self, request: CondensationRequest) -> SearchBrief:
        raise UnsupportedCondensationModeError(
            "PRECISE SearchBrief provider is not configured. "
            "Use method=extractive or disable long-input condensation."
        )
