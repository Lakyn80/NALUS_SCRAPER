"""
LLM Service — prepares input and delegates generation to an injected BaseLLM.

No business logic beyond input construction and output forwarding.
Swap the injected llm to change the backend (Mock → DeepSeek → OpenAI).
"""

from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.rag.llm.base import BaseLLM
from app.rag.llm.models import LLMInput, LLMOutput
from app.rag.retrieval.models import RetrievedChunk

logger = get_logger(__name__)


class LLMService:
    """Thin service over BaseLLM — stateless, injectable, swappable."""

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> LLMOutput:
        trace_event(logger, "llm_service.start", query=query, num_chunks=len(chunks))

        data = LLMInput(query=query, chunks=chunks)
        output = self._llm.generate(data)

        trace_event(
            logger,
            "llm_service.done",
            num_sources=len(output.sources),
            confidence=output.confidence,
        )

        return output
