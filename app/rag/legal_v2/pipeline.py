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
    SemanticVerifierResult,
    VerificationDecision,
    apply_thinking_promotion_policy,
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
    relevance_classification: str = "unknown"


@dataclass(frozen=True)
class LegalV2SearchResult:
    status: str
    interpretation_status: str
    query_spec_summary: dict[str, Any] | None
    verified_documents: list[LegalV2VerifiedDocument]
    rejected_documents: list[LegalV2VerifiedDocument] = field(default_factory=list)
    related_documents: list[LegalV2VerifiedDocument] = field(default_factory=list)
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
    thinking_verifier: SemanticVerifierProvider | None = None,
    source_filter: set[str] | None = None,
    debug: bool = False,
    include_full_document_text: bool = False,
) -> LegalV2SearchResult:
    started = time.perf_counter()
    latency: dict[str, float] = {}
    query_timeout = float(os.getenv("NALUS_LEGAL_V2_QUERYSPEC_TIMEOUT_SECONDS", "120"))
    fast_verifier_timeout = float(os.getenv("NALUS_LEGAL_V2_VERIFIER_TIMEOUT_SECONDS", "30"))
    thinking_verifier_timeout = float(
        os.getenv("NALUS_LEGAL_V2_VERIFIER_THINKING_TIMEOUT_SECONDS", "120")
    )
    thinking_fallback_enabled = os.getenv(
        "NALUS_LEGAL_V2_VERIFIER_THINKING_FALLBACK", "1"
    ).strip().lower() in {"1", "true", "yes", "on"}
    max_thinking_per_query = max(
        0,
        min(2, int(os.getenv("NALUS_LEGAL_V2_VERIFIER_THINKING_FALLBACK_MAX_PER_QUERY", "2"))),
    )
    trace_event(logger, "legal_v2.query_interpretation.started", query_length=len(query))
    interpretation = interpret_query_spec_v2(
        query,
        provider=query_provider,
        timeout_seconds=query_timeout,
        allow_deterministic_fallback=query_provider is None,
        max_provider_attempts=2,
    )
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
    thinking_fallback_calls = 0
    max_fast_verifier_candidates = max(
        1,
        min(8, int(os.getenv("NALUS_LEGAL_V2_VERIFIER_MAX_CANDIDATES_PER_QUERY", "8"))),
    )
    candidates = [
        item for item in retrieval.documents if _matches_source_filter(item, source_filter)
    ][:max_fast_verifier_candidates]
    for candidate in candidates:
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
            timeout_seconds=fast_verifier_timeout,
        )
        fast_result = verifier_result
        if (
            thinking_fallback_enabled
            and thinking_verifier is not None
            and thinking_fallback_calls < max_thinking_per_query
            and _should_escalate_to_thinking_verifier(verifier_result)
        ):
            thinking_fallback_calls += 1
            thinking_result = run_semantic_verifier(
                provider=thinking_verifier,
                query_spec=spec,
                candidate_document=CandidateDocumentForVerification(
                    document_id=candidate.document_id,
                    metadata=candidate.metadata,
                    paragraphs=candidate.paragraphs,
                ),
                evidence_windows=windows,
                timeout_seconds=thinking_verifier_timeout,
            )
            verifier_result = apply_thinking_promotion_policy(
                fast_result=fast_result,
                thinking_result=thinking_result,
                query_spec=spec,
            )
        decision = deterministic_verification_gate(query_spec=spec, verifier_result=verifier_result)
        # Belt-and-suspenders: never treat related/partial as verified hits.
        classification = str(
            (verifier_result.raw_diagnostics or {}).get("classification") or ""
        ).strip().lower()
        if decision == VerificationDecision.VERIFIED_MATCH and classification == "related_only":
            decision = VerificationDecision.NOT_PROVEN
        document_result = _document_result(
            candidate,
            decision,
            verifier_result,
            windows,
            candidate_rank=len(verified) + len(rejected) + 1,
            fast_result=fast_result,
            include_full_document_text=include_full_document_text,
        )
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
    related = _related_documents_from_rejected(rejected)
    status = "verified_match" if verified else "no_verified_results"
    trace_event(
        logger,
        "legal_v2.search.completed",
        status=status,
        verified_count=len(verified),
        rejected_count=len(rejected),
        related_count=len(related),
    )
    return LegalV2SearchResult(
        status=status,
        interpretation_status=interpretation.status,
        query_spec_summary=_safe_query_spec_summary(spec),
        verified_documents=verified,
        rejected_documents=rejected if debug else [],
        related_documents=related,
        rejection_counts=rejection_counts,
        latency_ms_by_stage=latency,
        provider={
            "query_interpreter": interpretation.provider_name,
            "verifier": getattr(verifier, "provider_name", "unknown"),
            "thinking_verifier": getattr(thinking_verifier, "provider_name", None),
            "thinking_fallback_calls": thinking_fallback_calls,
        },
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
    *,
    candidate_rank: int | None = None,
    fast_result: SemanticVerifierResult | None = None,
    include_full_document_text: bool = False,
) -> LegalV2VerifiedDocument:
    diagnostics = dict(verifier_result.raw_diagnostics or {})
    if candidate_rank is not None:
        diagnostics["candidate_rank"] = candidate_rank
    diagnostics["ecli"] = (
        candidate.metadata.get("ecli")
        or candidate.metadata.get("ECLI")
        or candidate.document_id
    )
    diagnostics["final_decision"] = decision.value
    diagnostics["final_rejection_code"] = (
        None if decision == VerificationDecision.VERIFIED_MATCH else decision.value
    )
    if fast_result is not None:
        diagnostics.setdefault("fast_decision", fast_result.decision.value)
        diagnostics.setdefault(
            "fast_classification",
            (fast_result.raw_diagnostics or {}).get("classification"),
        )
        diagnostics["fast_constraint_results"] = [
            {
                "constraint_id": item.constraint_id,
                "status": item.status.value,
                "evidence_paragraph_ids": list(item.evidence_paragraph_ids),
                "source_of_claim": item.source_of_claim,
            }
            for item in fast_result.constraint_results
        ]
    diagnostics["constraint_status_summary"] = {
        "proven": [
            item.constraint_id
            for item in verifier_result.constraint_results
            if item.status.value == "proven"
        ],
        "not_proven": [
            item.constraint_id
            for item in verifier_result.constraint_results
            if item.status.value == "not_proven"
        ],
        "contradicted": [
            item.constraint_id
            for item in verifier_result.constraint_results
            if item.status.value == "contradicted"
        ],
    }
    if include_full_document_text:
        paragraphs = sorted(
            candidate.paragraphs,
            key=lambda paragraph: (paragraph.source_order, paragraph.paragraph_index),
        )
        diagnostics["document_paragraphs"] = [
            {
                "paragraph_id": paragraph.paragraph_id,
                "paragraph_index": paragraph.paragraph_index,
                "section_type": paragraph.section_type.value,
                "text": paragraph.normalized_text or paragraph.original_text,
            }
            for paragraph in paragraphs
        ]
        diagnostics["document_text"] = "\n\n".join(
            str(item["text"]) for item in diagnostics["document_paragraphs"] if item.get("text")
        )
        diagnostics["document_paragraph_count"] = len(diagnostics["document_paragraphs"])
        diagnostics["document_text_char_count"] = len(diagnostics["document_text"])
    return LegalV2VerifiedDocument(
        document_id=candidate.document_id,
        score=candidate.score,
        status=decision.value,
        relevance_classification=_relevance_classification(decision, verifier_result),
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
        verifier_diagnostics=diagnostics,
    )


def _should_escalate_to_thinking_verifier(verifier_result: SemanticVerifierResult) -> bool:
    """Escalate only genuinely difficult fast-verifier outcomes."""
    diagnostics = dict(verifier_result.raw_diagnostics or {})
    reason = str(diagnostics.get("reason") or verifier_result.reason or "").lower()
    if diagnostics.get("failed_closed"):
        # Recoverable provider/structure failures may use one thinking attempt.
        return any(
            token in reason
            for token in (
                "timeout",
                "network_error",
                "empty_message_content",
                "invalid_json",
                "verifier_payload_not_object",
            )
        )
    classification = str(diagnostics.get("classification") or "").strip().lower()
    missing = list(diagnostics.get("mandatory_concepts_missing") or [])
    contradictions = list(diagnostics.get("contradictory_facts") or [])
    if classification in {"partial_match", "insufficient_evidence"}:
        return True
    if contradictions:
        return True
    # Exact/strong with incomplete hard coverage must still get a thinking attempt
    # (compact expand should demote these; this is a safety net).
    if classification in {"exact_match", "strong_match"} and missing and len(missing) <= 2:
        return True
    if classification in {"related_only", "not_relevant"} and missing and len(missing) <= 2:
        return True
    return False


def _related_documents_from_rejected(
    rejected: list[LegalV2VerifiedDocument],
    *,
    limit: int = 5,
) -> list[LegalV2VerifiedDocument]:
    """Surface related_only candidates separately — never as verified matches."""
    related = [
        document
        for document in rejected
        if str(document.relevance_classification or "").strip().lower()
        in {"related_only", "partial_match"}
    ]
    return related[: max(0, limit)]


def _relevance_classification(
    decision: VerificationDecision,
    verifier_result: Any,
) -> str:
    diagnostics = dict(getattr(verifier_result, "raw_diagnostics", {}) or {})
    classification = str(diagnostics.get("classification") or "").strip()
    if classification:
        return classification
    if decision == VerificationDecision.VERIFIED_MATCH:
        return "materially_relevant"
    if decision == VerificationDecision.AMBIGUOUS:
        return "related_only"
    if decision == VerificationDecision.NOT_PROVEN:
        return "insufficient_evidence"
    if decision == VerificationDecision.HARD_MISMATCH:
        return "not_relevant"
    return decision.value


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
