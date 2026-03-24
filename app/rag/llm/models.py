"""
Data structures for the LLM layer.

No logic — pure containers.
"""

from dataclasses import dataclass

from app.rag.retrieval.models import RetrievedChunk


@dataclass
class LLMInput:
    query: str
    chunks: list[RetrievedChunk]


@dataclass
class LLMOutput:
    answer: str
    reasoning: str
    sources: list[str]
    confidence: float
