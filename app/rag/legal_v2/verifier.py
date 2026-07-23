from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from app.rag.legal_v2.models import LegalParagraph, SectionType
from app.rag.legal_v2.query_spec import QuerySpecV2


class VerificationDecision(str, Enum):
    VERIFIED_MATCH = "verified_match"
    HARD_MISMATCH = "hard_mismatch"
    NOT_PROVEN = "not_proven"
    AMBIGUOUS = "ambiguous"
    UNVERIFIABLE_QUERY = "unverifiable_query"
    VERIFIER_ERROR = "verifier_error"


class ConstraintVerificationStatus(str, Enum):
    PROVEN = "proven"
    CONTRADICTED = "contradicted"
    NOT_PROVEN = "not_proven"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class EvidenceWindowForConstraint:
    constraint_id: str
    paragraph_ids: list[str]
    text: str
    section_types: list[SectionType]


@dataclass(frozen=True)
class CandidateDocumentForVerification:
    document_id: str
    metadata: dict[str, Any]
    paragraphs: list[LegalParagraph]

    @property
    def paragraph_ids(self) -> set[str]:
        return {paragraph.paragraph_id for paragraph in self.paragraphs}


@dataclass(frozen=True)
class ConstraintVerificationResult:
    constraint_id: str
    status: ConstraintVerificationStatus
    detected_value: str | None = None
    evidence_paragraph_ids: list[str] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0


@dataclass(frozen=True)
class SemanticVerifierResult:
    document_id: str
    decision: VerificationDecision
    constraint_results: list[ConstraintVerificationResult]
    reason: str = ""
    provider_name: str = "unknown"
    latency_ms: float = 0.0
    raw_diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerifierDiagnostics:
    provider_name: str
    decision: VerificationDecision
    constraint_result_count: int
    evidence_window_count: int
    latency_ms: float
    failed_closed_reason: str | None = None


class SemanticVerifierProvider(Protocol):
    provider_name: str

    def verify(
        self,
        *,
        query_spec: QuerySpecV2,
        candidate_document: CandidateDocumentForVerification,
        evidence_windows: list[EvidenceWindowForConstraint],
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Return provider JSON. Runtime code validates it before use."""


class DeterministicFakeVerifier:
    provider_name = "deterministic_fake_verifier"

    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.error = error
        self.calls = 0

    def verify(
        self,
        *,
        query_spec: QuerySpecV2,
        candidate_document: CandidateDocumentForVerification,
        evidence_windows: list[EvidenceWindowForConstraint],
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.payload is not None:
            return self.payload
        evidence_by_constraint = {
            window.constraint_id: window.paragraph_ids for window in evidence_windows
        }
        return {
            "document_id": candidate_document.document_id,
            "decision": VerificationDecision.NOT_PROVEN.value,
            "constraint_results": [
                {
                    "constraint_id": constraint.constraint_id,
                    "status": ConstraintVerificationStatus.NOT_PROVEN.value,
                    "detected_value": None,
                    "evidence_paragraph_ids": evidence_by_constraint.get(
                        constraint.constraint_id, []
                    ),
                    "reason": "Deterministic fake verifier has no configured proof.",
                    "confidence": 0.0,
                }
                for constraint in query_spec.all_constraints()
            ],
        }


def run_semantic_verifier(
    *,
    provider: SemanticVerifierProvider,
    query_spec: QuerySpecV2,
    candidate_document: CandidateDocumentForVerification,
    evidence_windows: list[EvidenceWindowForConstraint],
    timeout_seconds: float | None = None,
) -> SemanticVerifierResult:
    started = time.perf_counter()
    provider_name = getattr(provider, "provider_name", provider.__class__.__name__)
    try:
        payload = provider.verify(
            query_spec=query_spec,
            candidate_document=candidate_document,
            evidence_windows=evidence_windows,
            timeout_seconds=timeout_seconds,
        )
    except TimeoutError:
        return _fail_closed(
            candidate_document.document_id,
            provider_name=provider_name,
            reason="verifier_timeout",
            latency_ms=_elapsed_ms(started),
        )
    except Exception as exc:
        return _fail_closed(
            candidate_document.document_id,
            provider_name=provider_name,
            reason=f"verifier_provider_error:{exc.__class__.__name__}",
            latency_ms=_elapsed_ms(started),
        )
    result = validate_verifier_payload(
        payload=payload,
        query_spec=query_spec,
        candidate_document=candidate_document,
        provider_name=provider_name,
        latency_ms=_elapsed_ms(started),
    )
    return result


def validate_verifier_payload(
    *,
    payload: dict[str, Any],
    query_spec: QuerySpecV2,
    candidate_document: CandidateDocumentForVerification,
    provider_name: str = "unknown",
    latency_ms: float = 0.0,
) -> SemanticVerifierResult:
    if not isinstance(payload, dict):
        return _fail_closed(
            candidate_document.document_id,
            provider_name=provider_name,
            reason="verifier_payload_not_object",
            latency_ms=latency_ms,
        )
    if str(payload.get("document_id") or "") != candidate_document.document_id:
        return _fail_closed(
            candidate_document.document_id,
            provider_name=provider_name,
            reason="verifier_document_id_mismatch",
            latency_ms=latency_ms,
        )

    decision = _parse_provider_decision(payload.get("decision"))
    if decision is None:
        return _fail_closed(
            candidate_document.document_id,
            provider_name=provider_name,
            reason="verifier_unknown_decision",
            latency_ms=latency_ms,
        )

    raw_results = payload.get("constraint_results")
    if not isinstance(raw_results, list):
        return _fail_closed(
            candidate_document.document_id,
            provider_name=provider_name,
            reason="verifier_constraint_results_not_list",
            latency_ms=latency_ms,
        )
    allowed_constraint_ids = {
        constraint.constraint_id for constraint in query_spec.all_constraints()
    }
    known_paragraph_ids = candidate_document.paragraph_ids
    parsed_results: list[ConstraintVerificationResult] = []
    seen_constraint_ids: set[str] = set()
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            return _fail_closed(
                candidate_document.document_id,
                provider_name=provider_name,
                reason="verifier_constraint_result_not_object",
                latency_ms=latency_ms,
            )
        constraint_id = str(raw_result.get("constraint_id") or "")
        if constraint_id not in allowed_constraint_ids:
            return _fail_closed(
                candidate_document.document_id,
                provider_name=provider_name,
                reason="verifier_unknown_constraint_id",
                latency_ms=latency_ms,
            )
        status = _parse_constraint_status(raw_result.get("status"))
        if status is None:
            return _fail_closed(
                candidate_document.document_id,
                provider_name=provider_name,
                reason="verifier_unknown_constraint_status",
                latency_ms=latency_ms,
            )
        evidence_ids = raw_result.get("evidence_paragraph_ids")
        if not isinstance(evidence_ids, list) or not all(
            isinstance(item, str) for item in evidence_ids
        ):
            return _fail_closed(
                candidate_document.document_id,
                provider_name=provider_name,
                reason="verifier_evidence_ids_invalid",
                latency_ms=latency_ms,
            )
        if not set(evidence_ids).issubset(known_paragraph_ids):
            return _fail_closed(
                candidate_document.document_id,
                provider_name=provider_name,
                reason="verifier_unknown_evidence_paragraph_id",
                latency_ms=latency_ms,
            )
        if status in {
            ConstraintVerificationStatus.PROVEN,
            ConstraintVerificationStatus.CONTRADICTED,
        } and not evidence_ids:
            return _fail_closed(
                candidate_document.document_id,
                provider_name=provider_name,
                reason="verifier_evidence_required_for_terminal_status",
                latency_ms=latency_ms,
            )
        confidence = raw_result.get("confidence")
        if not isinstance(confidence, int | float) or not 0.0 <= float(confidence) <= 1.0:
            return _fail_closed(
                candidate_document.document_id,
                provider_name=provider_name,
                reason="verifier_confidence_invalid",
                latency_ms=latency_ms,
            )
        seen_constraint_ids.add(constraint_id)
        parsed_results.append(
            ConstraintVerificationResult(
                constraint_id=constraint_id,
                status=status,
                detected_value=_optional_str(raw_result.get("detected_value")),
                evidence_paragraph_ids=list(evidence_ids),
                reason=str(raw_result.get("reason") or ""),
                confidence=float(confidence),
            )
        )

    return SemanticVerifierResult(
        document_id=candidate_document.document_id,
        decision=decision,
        constraint_results=parsed_results,
        reason=str(payload.get("reason") or ""),
        provider_name=provider_name,
        latency_ms=latency_ms,
        raw_diagnostics={
            "constraint_result_count": len(parsed_results),
            "missing_constraint_result_count": len(
                allowed_constraint_ids.difference(seen_constraint_ids)
            ),
        },
    )


def deterministic_verification_gate(
    *,
    query_spec: QuerySpecV2,
    verifier_result: SemanticVerifierResult,
) -> VerificationDecision:
    if query_spec.requires_verification and not query_spec.hard_constraints:
        return VerificationDecision.UNVERIFIABLE_QUERY
    if verifier_result.decision == VerificationDecision.VERIFIER_ERROR:
        return VerificationDecision.VERIFIER_ERROR

    hard_constraint_ids = {
        constraint.constraint_id for constraint in query_spec.hard_constraints
    }
    hard_results_by_id = {
        result.constraint_id: result
        for result in verifier_result.constraint_results
        if result.constraint_id in hard_constraint_ids
    }
    if hard_constraint_ids.difference(hard_results_by_id):
        return VerificationDecision.NOT_PROVEN
    hard_results = list(hard_results_by_id.values())
    if any(
        result.status == ConstraintVerificationStatus.CONTRADICTED
        for result in hard_results
    ):
        return VerificationDecision.HARD_MISMATCH
    if any(result.status != ConstraintVerificationStatus.PROVEN for result in hard_results):
        return VerificationDecision.NOT_PROVEN
    return VerificationDecision.VERIFIED_MATCH


def verifier_diagnostics(
    *,
    result: SemanticVerifierResult,
    evidence_windows: list[EvidenceWindowForConstraint],
) -> VerifierDiagnostics:
    failed_reason = (
        result.reason if result.decision == VerificationDecision.VERIFIER_ERROR else None
    )
    return VerifierDiagnostics(
        provider_name=result.provider_name,
        decision=result.decision,
        constraint_result_count=len(result.constraint_results),
        evidence_window_count=len(evidence_windows),
        latency_ms=result.latency_ms,
        failed_closed_reason=failed_reason,
    )


def _fail_closed(
    document_id: str,
    *,
    provider_name: str,
    reason: str,
    latency_ms: float,
) -> SemanticVerifierResult:
    return SemanticVerifierResult(
        document_id=document_id,
        decision=VerificationDecision.VERIFIER_ERROR,
        constraint_results=[],
        reason=reason,
        provider_name=provider_name,
        latency_ms=latency_ms,
        raw_diagnostics={"failed_closed": True, "reason": reason},
    )


def _parse_provider_decision(value: object) -> VerificationDecision | None:
    try:
        decision = VerificationDecision(str(value))
    except ValueError:
        return None
    if decision in {
        VerificationDecision.VERIFIED_MATCH,
        VerificationDecision.HARD_MISMATCH,
        VerificationDecision.NOT_PROVEN,
        VerificationDecision.AMBIGUOUS,
    }:
        return decision
    return None


def _parse_constraint_status(value: object) -> ConstraintVerificationStatus | None:
    try:
        return ConstraintVerificationStatus(str(value))
    except ValueError:
        return None


def _optional_str(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
