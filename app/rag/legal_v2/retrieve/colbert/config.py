"""Typed configuration for Legal v2 ColBERT retrieval (foundation only).

Does not load models, open indexes, or download weights.
Future first experiment corpus (not built in this step): Slice 4 B contextual
``nalus_legal_paragraph_chunks_v2_chunk_ab_v8_b_contextual_300``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.rag.legal_v2.retrieve.colbert.errors import ColbertConfigurationError

# Intended source corpus for the first ColBERT experiment (index not built yet).
COLBERT_PILOT_SOURCE_QDRANT_COLLECTION = (
    "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_b_contextual_300"
)

_ALLOWED_DEVICES = frozenset({"cpu", "cuda", "auto"})


@dataclass(frozen=True)
class ColbertConfig:
    """Runtime knobs for a future ColBERT backend.

    Validation is pure and side-effect free. Defaults are conservative placeholders
    only — experiment values are chosen in a later indexing/benchmark step.
    """

    model_name: str
    index_path: Path
    index_name: str = "legal_v2_colbert"
    device: str = "cpu"
    top_k: int = 10
    batch_size: int = 16

    def validate(self) -> None:
        model = str(self.model_name or "").strip()
        if not model:
            raise ColbertConfigurationError("model_name must be a non-empty string")
        name = str(self.index_name or "").strip()
        if not name:
            raise ColbertConfigurationError("index_name must be a non-empty string")
        if self.index_path is None or not str(self.index_path).strip():
            raise ColbertConfigurationError("index_path must be set")
        device = str(self.device or "").strip().lower()
        if device not in _ALLOWED_DEVICES:
            allowed = ", ".join(sorted(_ALLOWED_DEVICES))
            raise ColbertConfigurationError(
                f"device must be one of: {allowed} (got {self.device!r})"
            )
        if int(self.top_k) < 1:
            raise ColbertConfigurationError("top_k must be >= 1")
        if int(self.batch_size) < 1:
            raise ColbertConfigurationError("batch_size must be >= 1")
