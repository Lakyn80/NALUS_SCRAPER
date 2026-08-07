"""Active extractive SearchBrief provider."""

from __future__ import annotations

from app.rag.legal_v2.query_input.extractive import build_extractive_search_brief
from app.rag.legal_v2.query_input.models import SearchBrief
from app.rag.legal_v2.query_input.providers.base import CondensationRequest


class ExtractiveSearchBriefProvider:
    name = "extractive"

    def condense(self, request: CondensationRequest) -> SearchBrief:
        # Prefer already-normalized text when provided.
        text = request.normalized_text or request.raw_text
        return build_extractive_search_brief(text, config=request.config)
