"""
Hybrid Answer Service — routes between rule-based AnswerService and LLMService.

Routing rule:
    chunks < 2  →  AnswerService  (fast, deterministic, no LLM cost)
    chunks >= 2  →  LLMService    (richer answer with context)

Return type varies by route (AnswerResult | LLMOutput).
Designed for future smart routing without changing the underlying services.
"""

from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.rag.answer.answer_service import AnswerResult, AnswerService
from app.rag.llm.models import LLMOutput
from app.rag.llm.service import LLMService
from app.rag.retrieval.models import RetrievedChunk

logger = get_logger(__name__)

_LLM_THRESHOLD = 2  # chunks >= threshold → LLM route


class HybridAnswerService:
    """Routes generation to AnswerService or LLMService based on chunk count."""

    def __init__(self, answer_service: AnswerService, llm_service: LLMService) -> None:
        self._answer_service = answer_service
        self._llm_service = llm_service

    def generate(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> AnswerResult | LLMOutput:
        trace_event(logger, "hybrid.start", query=query, num_chunks=len(chunks))

        if len(chunks) < _LLM_THRESHOLD:
            trace_event(logger, "hybrid.route", route="answer")
            result = self._answer_service.generate(query, chunks)
        else:
            trace_event(logger, "hybrid.route", route="llm")
            result = self._llm_service.generate(query, chunks)

        trace_event(logger, "hybrid.done")
        return result
