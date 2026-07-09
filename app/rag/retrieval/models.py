"""
Shared data models for the retrieval layer.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float
    source: str  # e.g. "dense" | "bm25" | "hybrid" | "keyword" (legacy)
    metadata: dict[str, Any] = field(default_factory=dict)
