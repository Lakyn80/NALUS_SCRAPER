"""
Abstract base class for LLM backends.

Extend this to add any provider (OpenAI, DeepSeek, local model, …)
without touching the rest of the system.
"""

from abc import ABC, abstractmethod

from app.rag.llm.models import LLMInput, LLMOutput


class BaseLLM(ABC):
    """Contract that every LLM backend must satisfy."""

    @abstractmethod
    def generate(self, data: LLMInput) -> LLMOutput:
        """Generate an answer from the given input."""
