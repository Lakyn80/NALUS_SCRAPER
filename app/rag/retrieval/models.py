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
    source: str  # "dense" | "keyword"
    metadata: dict[str, Any] = field(default_factory=dict)
