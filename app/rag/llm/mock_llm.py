"""
Mock LLM — deterministic stub for tests and local development.

No randomness, no external calls, no model weights.
"""

from app.rag.llm.base import BaseLLM
from app.rag.llm.models import LLMInput, LLMOutput


class MockLLM(BaseLLM):
    """Deterministic LLM implementation used in tests."""

    def generate(self, data: LLMInput) -> LLMOutput:
        return LLMOutput(
            answer=f"Mock odpověď na: {data.query}",
            reasoning="Mock reasoning na základě dodaných chunků",
            sources=[c.id for c in data.chunks[:3]],
            confidence=0.5,
        )
