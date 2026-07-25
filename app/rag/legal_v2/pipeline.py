from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.observability.legal_v2_metrics import (
    DOCUMENTS_TOTAL,
    QUERY_INTERPRETATIONS_TOTAL,
    REJECTIONS_TOTAL,
    record_stage_latency,
)
from app.rag.legal_v2.evidence import CandidateEvidenceDocument, select_evidence_windows
from app.rag.legal_v2.interpreter import QuerySpecProvider, interpret_query_spec_v2
from app.rag.legal_v2.query_spec import QuerySpecV2
from app.rag.legal_v2.retriever import LegalV2HybridRetriever, LegalV2RetrieverConfig
from app.rag.legal_v2.verifier import (
    CandidateDocumentForVerification,
    SemanticVerifierProvider,
    VerificationDecision,
    deterministic_verification_gate,
    run_semantic_verifier,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class LegalV2VerifiedDocument:
    document_id: str
    score: float
    status: str
    metadata: dict[str, Any]
    evidence: list[dict[str, Any]]
    constraint_results: list[dict[str, Any]]
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    verification_reason: str = ""
    verifier_diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LegalV2SearchResult:
    status: str
    interpretation_status: str
    query_spec_summary: dict[str, Any] | None
    verified_documents: list[LegalV2VerifiedDocument]
    rejected_documents: list[LegalV2VerifiedDocument] = field(default_factory=list)
    rejection_counts: dict[str, int] = field(default_factory=dict)
    latency_ms_by_stage: dict[str, float] = field(default_factory=dict)
    provider: dict[str, Any] = field(default_factory=dict)
    index: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def legal_v2_search_enabled() -> bool:
    return os.getenv("NALUS_LEGAL_V2_SEARCH_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}


def search_legal_v2(
    *,
    query: str,
    retriever: LegalV2HybridRetriever,
    verifier: SemanticVerifierProvider,
    config: LegalV2RetrieverConfig,
    query_provider: QuerySpecProvider | None = None,
    source_filter: set[str] | None = None,
    debug: bool = False,
) -> LegalV2SearchResult:
    started = time.perf_counter()
    latency: dict[str, float] = {}
    trace_event(logger, "legal_v2.query_interpretation.started", query_length=len(query))
    interpretation = interpret_query_spec_v2(query, provider=query_provider, allow_deterministic_fallback=query_provider is None)
    latency["query_interpretation"] = interpretation.latency_ms
    QUERY_INTERPRETATIONS_TOTAL.labels(status=interpretation.status, provider=interpretation.provider_name).inc()
    record_stage_latency(stage="query_interpretation", status=interpretation.status, latency_ms=interpretation.latency_ms)
    if interpretation.query_spec is None:
        trace_event(logger, "legal_v2.query_interpretation.failed", status=interpretation.status)
        return LegalV2SearchResult(
            status="query_interpretation_error",
            interpretation_status=interpretation.status,
            query_spec_summary=None,
            verified_documents=[],
            latency_ms_by_stage=latency,
            provider={
                "query_interpreter": interpretation.provider_name,
                "reason": interpretation.reason,
                "error": interpretation.provider_error,
            },
        )
    spec = interpretation.query_spec
    trace_event(logger, "legal_v2.query_interpretation.completed", status=interpretation.status)
    if spec.requires_verification and not spec.hard_constraints:
        return LegalV2SearchResult(
            status=VerificationDecision.UNVERIFIABLE_QUERY.value,
            interpretation_status=interpretation.status,
            query_spec_summary=_safe_query_spec_summary(spec),
            verified_documents=[],
            latency_ms_by_stage=latency,
            provider={"query_interpreter": interpretation.provider_name},
        )
    retrieval_started = time.perf_counter()
    try:
        retrieval = retriever.retrieve(spec)
    except Exception as exc:  # noqa: BLE001
        trace_event(logger, "legal_v2.search.failed", stage="retrieval")
        return LegalV2SearchResult(
            status="retrieval_error",
            interpretation_status=interpretation.status,
            query_spec_summary=_safe_query_spec_summary(spec),
            verified_documents=[],
            latency_ms_by_stage={**latency, "retrieval": _elapsed_ms(retrieval_started)},
            provider={"query_interpreter": interpretation.provider_name, "verifier": getattr(verifier, "provider_name", "unknown")},
            diagnostics={"error_type": exc.__class__.__name__},
        )
    latency["retrieval"] = _elapsed_ms(retrieval_started)
    evidence_started = time.perf_counter()
    verified: list[LegalV2VerifiedDocument] = []
    rejected: list[LegalV2VerifiedDocument] = []
    rejection_counts: dict[str, int] = {}
    for candidate in [
        item for item in retrieval.documents if _matches_source_filter(item, source_filter)
    ]:
        windows = select_evidence_windows(
            query_spec=spec,
            candidate=candidate,
            max_windows_per_constraint=config.evidence_windows_per_constraint,
        )
        verifier_result = run_semantic_verifier(
            provider=verifier,
            query_spec=spec,
            candidate_document=CandidateDocumentForVerification(
                document_id=candidate.document_id,
                metadata=candidate.metadata,
                paragraphs=candidate.paragraphs,
            ),
            evidence_windows=windows,
            timeout_seconds=float(os.getenv("NALUS_LEGAL_V2_VERIFIER_TIMEOUT_SECONDS", "20")),
        )
        decision = deterministic_verification_gate(query_spec=spec, verifier_result=verifier_result)
        document_result = _document_result(candidate, decision, verifier_result, windows)
        DOCUMENTS_TOTAL.labels(decision=decision.value).inc()
        if decision == VerificationDecision.VERIFIED_MATCH:
            verified.append(document_result)
        else:
            rejected.append(document_result)
            rejection_counts[decision.value] = rejection_counts.get(decision.value, 0) + 1
            REJECTIONS_TOTAL.labels(reason=decision.value).inc()
            trace_event(logger, "legal_v2.document_rejected", decision=decision.value)
        if len(verified) >= config.returned_verified_documents:
            break
    latency["evidence_selection_and_verification"] = _elapsed_ms(evidence_started)
    latency["total"] = _elapsed_ms(started)
    status = "verified_match" if verified else "no_verified_results"
    trace_event(
        logger,
        "legal_v2.search.completed",
        status=status,
        verified_count=len(verified),
        rejected_count=len(rejected),
    )
    return LegalV2SearchResult(
        status=status,
        interpretation_status=interpretation.status,
        query_spec_summary=_safe_query_spec_summary(spec),
        verified_documents=verified,
        rejected_documents=rejected if debug else [],
        rejection_counts=rejection_counts,
        latency_ms_by_stage=latency,
        provider={"query_interpreter": interpretation.provider_name, "verifier": getattr(verifier, "provider_name", "unknown")},
        index={
            "collection": config.qdrant_collection,
            "bm25_index_id": config.bm25_index_id,
            "bm25_sidecar_path": str(config.bm25_sidecar_path),
        },
        diagnostics=retrieval.diagnostics if debug else {},
    )


def _document_result(
    candidate: CandidateEvidenceDocument,
    decision: VerificationDecision,
    verifier_result: Any,
    windows: list[Any],
) -> LegalV2VerifiedDocument:
    return LegalV2VerifiedDocument(
        document_id=candidate.document_id,
        score=candidate.score,
        status=decision.value,
        metadata=candidate.metadata,
        evidence=[
            {
                "constraint_id": window.constraint_id,
                "paragraph_ids": window.paragraph_ids,
                "section_types": [section.value for section in window.section_types],
                "quote": _bounded_quote(window.text),
                "source_of_claim": window.source_of_claim,
            }
            for window in windows
        ],
        constraint_results=[asdict(result) for result in verifier_result.constraint_results],
        dense_rank=candidate.dense_rank,
        bm25_rank=candidate.bm25_rank,
        rrf_score=candidate.rrf_score,
        verification_reason=str(verifier_result.reason or ""),
        verifier_diagnostics=dict(verifier_result.raw_diagnostics or {}),
    )


def _safe_query_spec_summary(spec: QuerySpecV2) -> dict[str, Any]:
    return {
        "original_query": spec.original_query,
        "normalized_query": spec.normalized_query,
        "intent": spec.intent.value,
        "retrieval_queries": spec.retrieval_queries[:5],
        "hard_constraints": [asdict(item) for item in spec.hard_constraints],
        "soft_constraints": [asdict(item) for item in spec.soft_constraints],
        "negative_constraints": [asdict(item) for item in spec.negative_constraints],
        "ambiguities": spec.ambiguities,
        "requires_verification": spec.requires_verification,
    }


def _bounded_quote(text: str, limit: int = 500) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def _matches_source_filter(
    candidate: CandidateEvidenceDocument,
    source_filter: set[str] | None,
) -> bool:
    if not source_filter:
        return True
    metadata = candidate.metadata
    haystack = " ".join(
        str(metadata.get(key) or "")
        for key in ("source", "court", "court_name", "document_id", "source_document_id", "ecli")
    ).lower()
    aliases = {
        "constitutional": ("constitutional", "nalus", "usoud", "ústav", "ustav", "ecli:cz:us"),
        "supreme": ("supreme", "nsoud", "nejvyšší", "nejvyssi", "ecli:cz:ns"),
    }
    for requested in source_filter:
        allowed = aliases.get(requested, (requested,))
        if any(value in haystack for value in allowed):
            return True
    return False


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
