from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.rag.legal_v2.chunking import ChunkingDiagnostics
from app.rag.legal_v2.models import ParagraphParsingDiagnostics
from app.rag.legal_v2.query_spec import QuerySpecV2
from app.rag.legal_v2.verifier import VerifierDiagnostics


@dataclass(frozen=True)
class CandidateDiagnostics:
    dense_candidate_count: int = 0
    bm25_candidate_count: int = 0
    rrf_candidate_count: int = 0
    candidate_document_count: int = 0
    rejected_document_reasons: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class LegalV2StageLatency:
    query_spec_ms: float = 0.0
    parsing_ms: float = 0.0
    child_chunking_ms: float = 0.0
    parent_window_ms: float = 0.0
    dense_retrieval_ms: float = 0.0
    bm25_retrieval_ms: float = 0.0
    rrf_ms: float = 0.0
    verifier_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(frozen=True)
class LegalV2RuntimeDiagnostics:
    original_query: str
    normalized_query: str
    retrieval_queries: list[str]
    query_intent: str
    hard_constraint_count: int
    soft_constraint_count: int
    negative_constraint_count: int
    paragraph_parsing: dict[str, Any]
    child_and_parent_chunking: dict[str, Any]
    candidates: CandidateDiagnostics
    evidence_windows_per_constraint: dict[str, int]
    verifier_decisions: dict[str, int]
    latency: LegalV2StageLatency

    def to_safe_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["prometheus_label_safety"] = {
            "raw_queries_in_labels": False,
            "paragraph_ids_in_labels": False,
            "document_ids_in_labels": False,
            "ecli_values_in_labels": False,
            "evidence_text_in_labels": False,
        }
        return payload


def build_legal_v2_diagnostics(
    *,
    query_spec: QuerySpecV2,
    parsing: ParagraphParsingDiagnostics,
    chunking: ChunkingDiagnostics,
    candidates: CandidateDiagnostics | None = None,
    verifier_diagnostics: list[VerifierDiagnostics] | None = None,
    evidence_windows_per_constraint: dict[str, int] | None = None,
    latency: LegalV2StageLatency | None = None,
) -> LegalV2RuntimeDiagnostics:
    decisions: dict[str, int] = {}
    for item in verifier_diagnostics or []:
        decisions[item.decision.value] = decisions.get(item.decision.value, 0) + 1
    return LegalV2RuntimeDiagnostics(
        original_query=query_spec.original_query,
        normalized_query=query_spec.normalized_query,
        retrieval_queries=list(query_spec.retrieval_queries),
        query_intent=query_spec.intent.value,
        hard_constraint_count=len(query_spec.hard_constraints),
        soft_constraint_count=len(query_spec.soft_constraints),
        negative_constraint_count=len(query_spec.negative_constraints),
        paragraph_parsing=asdict(parsing),
        child_and_parent_chunking=asdict(chunking),
        candidates=candidates or CandidateDiagnostics(),
        evidence_windows_per_constraint=dict(evidence_windows_per_constraint or {}),
        verifier_decisions=decisions,
        latency=latency or LegalV2StageLatency(),
    )
