from __future__ import annotations

import time
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from app.rag.legal_v2.models import LegalParagraph, SectionType
from app.rag.legal_v2.query_spec import QuerySpecV2
from app.rag.llm.config import effective_llm_config_from_env
from app.rag.llm.provider_factory import get_text_llm
from app.rag.llm.providers._base import LLMProviderError


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
    heading_context: list[str] = field(default_factory=list)
    source_of_claim: str = "unknown"
    current_case_classification: str = "current_case"


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
    required_value: str | None = None
    detected_value: str | None = None
    evidence_paragraph_ids: list[str] = field(default_factory=list)
    source_of_claim: str = "unknown"
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
                    "required_value": constraint.value,
                    "detected_value": None,
                    "evidence_paragraph_ids": evidence_by_constraint.get(
                        constraint.constraint_id, []
                    ),
                    "source_of_claim": "unknown",
                    "reason": "Deterministic fake verifier has no configured proof.",
                    "confidence": 0.0,
                }
                for constraint in query_spec.all_constraints()
            ],
        }


class DeepSeekSemanticVerifierProvider:
    provider_name = "deepseek_semantic_verifier_v2"

    def __init__(self, api_key: str, *, model: str | None = None) -> None:
        config = effective_llm_config_from_env()
        self.model = model or config.deepseek_model
        self._llm = get_text_llm(
            "deepseek",
            api_key,
            model=self.model,
            timeout=config.timeout_seconds,
            max_tokens=config.legal_v2_max_tokens,
            max_retries=config.retry_count,
            raise_on_error=True,
            json_response=True,
        )

    def verify(
        self,
        *,
        query_spec: QuerySpecV2,
        candidate_document: CandidateDocumentForVerification,
        evidence_windows: list[EvidenceWindowForConstraint],
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        del timeout_seconds
        text = self._llm.generate_text(
            _verifier_prompt(
                query_spec=query_spec,
                candidate_document=candidate_document,
                evidence_windows=evidence_windows,
            )
        )
        payload = _json_payload(text)
        if payload is None:
            raise ValueError("DeepSeek verifier returned invalid JSON.")
        return _normalize_verifier_payload(payload)


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
    except LLMProviderError as exc:
        return _fail_closed(
            candidate_document.document_id,
            provider_name=provider_name,
            reason=f"verifier_provider_error:{exc.safe_reason}",
            latency_ms=_elapsed_ms(started),
            provider_error=exc.to_safe_dict(),
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
        evidence_windows=evidence_windows,
        provider_name=provider_name,
        latency_ms=_elapsed_ms(started),
    )
    return result


def validate_verifier_payload(
    *,
    payload: dict[str, Any],
    query_spec: QuerySpecV2,
    candidate_document: CandidateDocumentForVerification,
    evidence_windows: list[EvidenceWindowForConstraint] | None = None,
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
    constraints_by_id = {
        constraint.constraint_id: constraint for constraint in query_spec.all_constraints()
    }
    known_paragraph_ids = candidate_document.paragraph_ids
    evidence_windows = evidence_windows or []
    supplied_evidence_ids_by_constraint = _supplied_evidence_ids(evidence_windows)
    supplied_source_by_paragraph = _supplied_source_by_paragraph(evidence_windows)
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
        supplied_ids = supplied_evidence_ids_by_constraint.get(constraint_id, set())
        if not set(evidence_ids).issubset(supplied_ids):
            return _fail_closed(
                candidate_document.document_id,
                provider_name=provider_name,
                reason="verifier_evidence_outside_supplied_windows",
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
        source_of_claim = str(raw_result.get("source_of_claim") or "unknown")
        if status == ConstraintVerificationStatus.PROVEN and _restricted_source_proves_current_case(
            source_of_claim=source_of_claim,
            evidence_ids=evidence_ids,
            source_by_paragraph=supplied_source_by_paragraph,
        ):
            return _fail_closed(
                candidate_document.document_id,
                provider_name=provider_name,
                reason="verifier_restricted_source_claim_used_as_proof",
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
                required_value=_optional_str(raw_result.get("required_value"))
                or constraints_by_id[constraint_id].value,
                detected_value=_optional_str(raw_result.get("detected_value")),
                evidence_paragraph_ids=list(evidence_ids),
                source_of_claim=source_of_claim,
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
    provider_error: dict[str, Any] | None = None,
) -> SemanticVerifierResult:
    raw_diagnostics: dict[str, Any] = {"failed_closed": True, "reason": reason}
    if provider_error is not None:
        raw_diagnostics["provider_error"] = provider_error
    return SemanticVerifierResult(
        document_id=document_id,
        decision=VerificationDecision.VERIFIER_ERROR,
        constraint_results=[],
        reason=reason,
        provider_name=provider_name,
        latency_ms=latency_ms,
        raw_diagnostics=raw_diagnostics,
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


def _supplied_evidence_ids(
    evidence_windows: list[EvidenceWindowForConstraint],
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for window in evidence_windows:
        result.setdefault(window.constraint_id, set()).update(window.paragraph_ids)
    return result


def _supplied_source_by_paragraph(
    evidence_windows: list[EvidenceWindowForConstraint],
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for window in evidence_windows:
        for paragraph_id in window.paragraph_ids:
            result.setdefault(paragraph_id, set()).add(window.source_of_claim)
    return result


def _restricted_source_proves_current_case(
    *,
    source_of_claim: str,
    evidence_ids: list[str],
    source_by_paragraph: dict[str, set[str]],
) -> bool:
    restricted = {"party_claim", "cited_case"}
    if source_of_claim in restricted:
        return True
    return any(source_by_paragraph.get(paragraph_id, set()).issubset(restricted) for paragraph_id in evidence_ids)


def _json_payload(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_verifier_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw_results = payload.get("constraint_results")
    if not isinstance(raw_results, list):
        return payload
    decision = str(payload.get("decision") or "")
    normalized_results: list[Any] = []
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            normalized_results.append(raw_result)
            continue
        result = dict(raw_result)
        if "status" not in result and isinstance(result.get("proven"), bool):
            result["status"] = (
                ConstraintVerificationStatus.PROVEN.value
                if result["proven"]
                else ConstraintVerificationStatus.NOT_PROVEN.value
            )
        if "status" not in result and decision == VerificationDecision.HARD_MISMATCH.value:
            result["status"] = ConstraintVerificationStatus.CONTRADICTED.value
        if "evidence_paragraph_ids" not in result:
            for alias in ("proven_by_paragraph_ids", "paragraph_ids", "evidence_ids"):
                if alias in result:
                    result["evidence_paragraph_ids"] = result.get(alias)
                    break
        result.setdefault("evidence_paragraph_ids", [])
        result.setdefault("source_of_claim", "unknown")
        result.setdefault("reason", str(result.get("explanation") or ""))
        result["confidence"] = _normalized_confidence(result.get("confidence"))
        normalized_results.append(result)
    normalized = dict(payload)
    normalized["constraint_results"] = normalized_results
    return normalized


def _normalized_confidence(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str) and value.strip():
        try:
            parsed = float(value)
        except ValueError:
            return 0.0
        return max(0.0, min(1.0, parsed))
    return 0.0


def _verifier_prompt(
    *,
    query_spec: QuerySpecV2,
    candidate_document: CandidateDocumentForVerification,
    evidence_windows: list[EvidenceWindowForConstraint],
) -> str:
    safe_metadata = {
        key: value
        for key, value in candidate_document.metadata.items()
        if key not in {"text", "full_text", "paragraph_texts", "paragraph_original_texts"}
    }
    evidence_payload = [
        {
            "constraint_id": window.constraint_id,
            "paragraph_ids": window.paragraph_ids,
            "section_types": [section.value for section in window.section_types],
            "source_of_claim": window.source_of_claim,
            "current_case_classification": window.current_case_classification,
            "heading_context": window.heading_context,
            "text": window.text,
        }
        for window in evidence_windows
    ]
    return (
        "You are a strict legal evidence verifier. Return only JSON with document_id, "
        "decision, and constraint_results. decision must be one of verified_match, "
        "hard_mismatch, not_proven, ambiguous. Each constraint_results item must contain "
        "constraint_id, status, required_value, detected_value, evidence_paragraph_ids, "
        "source_of_claim, reason, confidence. status must be one of proven, contradicted, "
        "not_proven, ambiguous. A hard fact is proven only by supplied "
        "paragraph IDs from current-case court findings. Party allegations and cited-case "
        "facts do not prove current-case facts. Evaluate actor, action, object, origin, "
        "destination, and direction as one relation.\n"
        f"QuerySpec JSON: {json.dumps(query_spec.to_dict(), ensure_ascii=False)}\n"
        f"Document ID: {candidate_document.document_id}\n"
        f"Safe metadata: {json.dumps(safe_metadata, ensure_ascii=False)}\n"
        f"Evidence windows: {json.dumps(evidence_payload, ensure_ascii=False)}"
    )
