"""Domain models for Legal v2 Cross-Encoder reranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvidenceChunkRecord:
    """Stage-1 chunk provenance available to passage selectors."""

    chunk_id: str
    text: str
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_rank: int | None = None
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    retrieval_channels: tuple[str, ...] = ()
    chunk_position: int | None = None
    section: str | None = None
    page: int | None = None


@dataclass(frozen=True)
class RerankPassage:
    ecli: str
    text: str
    chunk_id: str
    stage1_document_rank: int
    passage_index: int = 0
    selection_slot: int | None = None
    selection_reason: str | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_rank: int | None = None
    retrieval_channels: tuple[str, ...] = ()
    chunk_position: int | None = None
    section: str | None = None
    page: int | None = None
    near_duplicate_filtered_count: int = 0
    requested_passages: int | None = None
    selected_passages: int | None = None


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
    evidence_pool: tuple[EvidenceChunkRecord, ...] = ()


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
    passage_selector: str | None = None
    requested_passages_per_document: int | None = None
    mean_selected_passages: float | None = None

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
            "passage_selector": self.passage_selector,
            "requested_passages_per_document": self.requested_passages_per_document,
            "mean_selected_passages": self.mean_selected_passages,
        }


@dataclass(frozen=True)
class RerankResult:
    documents: tuple[RerankedDocument, ...]
    diagnostics: RerankDiagnostics
