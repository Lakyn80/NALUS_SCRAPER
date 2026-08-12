"""Typed configuration for Legal v2 ColBERT retrieval.

Does not load models, open indexes, or download weights on import.
First experiment corpus: Slice 4 B contextual
``nalus_legal_paragraph_chunks_v2_chunk_ab_v8_b_contextual_300`` (4168 chunks).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.rag.legal_v2.retrieve.colbert.errors import ColbertConfigurationError

COLBERT_PILOT_SOURCE_QDRANT_COLLECTION = (
    "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_b_contextual_300"
)
COLBERT_PILOT_EXPECTED_CHUNK_COUNT = 4168
DEFAULT_COLBERT_MODEL = "colbert-ir/colbertv2.0"
DEFAULT_INDEX_NAME = "legal_v2_colbert_b_contextual_300"

_ALLOWED_DEVICES = frozenset({"cpu", "cuda", "auto"})


@dataclass(frozen=True)
class ColbertConfig:
    """Runtime knobs for the ColBERT backend.

    Validation is pure and side-effect free.
    """

    model_name: str
    index_path: Path
    index_name: str = DEFAULT_INDEX_NAME
    device: str = "cpu"
    top_k: int = 10
    batch_size: int = 16
    concurrency_limit: int = 1
    mapping_path: Path | None = None
    allow_download: bool = False
    source_collection: str = COLBERT_PILOT_SOURCE_QDRANT_COLLECTION
    expected_chunk_count: int = COLBERT_PILOT_EXPECTED_CHUNK_COUNT

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
        if int(self.concurrency_limit) < 1:
            raise ColbertConfigurationError("concurrency_limit must be >= 1")
        if int(self.expected_chunk_count) < 1:
            raise ColbertConfigurationError("expected_chunk_count must be >= 1")

    def resolved_mapping_path(self) -> Path:
        if self.mapping_path is not None and str(self.mapping_path).strip():
            return Path(self.mapping_path)
        return Path(self.index_path).parent / "colbert_chunk_mapping.jsonl"
