"""
Shared data models for the retrieval layer.
"""

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float
    source: str  # "dense" | "keyword"
