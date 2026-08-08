"""Domain models for Legal v2 Cross-Encoder reranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RerankPassage:
    ecli: str
    text: str
    chunk_id: str
    stage1_document_rank: int
    passage_index: int = 0


@dataclass(frozen=True)
class RerankScore:
    ecli: str
    chunk_id: str
    score: float
    passage_index: int = 0
    truncated: bool = False


@dataclass(frozen=True)
class RerankCandidate:
    ecli: str
    stage1_rank: int
    stage1_score: float
    passages: tuple[RerankPassage, ...]
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankedDocument:
    ecli: str
    stage1_rank: int
    stage1_score: float
    ce_rank: int
    ce_score: float
    passage_scores: tuple[RerankScore, ...]
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankDiagnostics:
    rerank_enabled: bool
    rerank_applied: bool
    reranker_model: str | None
    reranker_device: str | None
    candidate_document_count: int
    passage_count: int
    pair_count: int
    batch_count: int
    truncated_pair_count: int
    aggregation: str
    rerank_latency_ms: float
    fallback_reason: str | None = None
    warnings: tuple[str, ...] = ()
    model_revision: str | None = None
    dtype: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "rerank_enabled": self.rerank_enabled,
            "rerank_applied": self.rerank_applied,
            "reranker_model": self.reranker_model,
            "reranker_device": self.reranker_device,
            "candidate_document_count": self.candidate_document_count,
            "passage_count": self.passage_count,
            "pair_count": self.pair_count,
            "batch_count": self.batch_count,
            "truncated_pair_count": self.truncated_pair_count,
            "aggregation": self.aggregation,
            "rerank_latency_ms": self.rerank_latency_ms,
            "fallback_reason": self.fallback_reason,
            "warnings": list(self.warnings),
            "model_revision": self.model_revision,
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class RerankResult:
    documents: tuple[RerankedDocument, ...]
    diagnostics: RerankDiagnostics
