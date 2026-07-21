"""Deterministic document-level constraint verification."""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

from app.rag.retrieval.constraint_config import ConstraintRetrievalConfig
from app.rag.retrieval.constraint_models import (
    ConstraintCategory,
    ConstraintEvidence,
    ConstraintVerificationResult,
    ConstraintVerificationStatus,
    DocumentDecisionStatus,
    StructuredConstraint,
    StructuredQuery,
    VerificationMethod,
    VerifiedDocument,
)
from app.rag.retrieval.document_retrieval import DocumentSearchResult
from app.rag.retrieval.full_document import FullDocumentChunk, FullDocumentResult

_COUNTRY_PATTERNS: dict[str, tuple[str, ...]] = {
    "CZ": ("ceska republika", "ceske republiky", "ceske republice", "cr", "cesko"),
    "RU": ("ruska federace", "ruske federace", "rusko", "rusku", "rusk", "rusky"),
    "UA": ("ukrajina", "ukrajine", "ukrajinsk"),
    "DE": ("nemecko", "nemecka", "nemeck"),
    "AT": ("rakousko", "rakouska", "rakousk"),
    "US": ("spojene staty", "usa", "americk"),
}


@dataclass(frozen=True)
class _EvidenceMatch:
    chunk_id: str | None
    quote: str
    detected_value: str | None = None
    source_field: str | None = None


def verify_document_constraints(
    *,
    structured_query: StructuredQuery,
    candidate: DocumentSearchResult,
    document: FullDocumentResult | None,
    config: ConstraintRetrievalConfig,
) -> VerifiedDocument:
    if document is None or not document.full_text.strip():
        return VerifiedDocument(
            document_id=candidate.document_id,
            score=candidate.score,
            decision_status=DocumentDecisionStatus.EXCLUDED_INSUFFICIENT_EVIDENCE,
            constraint_results=[
                _not_proven_result(
                    constraint,
                    reason="Full document text is unavailable for verification.",
                )
                for constraint in structured_query.hard_constraints
            ],
            supporting_passages=[],
            metadata=candidate.metadata,
            candidate_chunk_count=candidate.candidate_chunk_count,
        )

    started = time.perf_counter()
    bounded_chunks = document.chunks[: config.max_chunks_per_document_for_verification]
    bounded_text = _bounded_document_text(
        bounded_chunks,
        max_characters=config.max_document_characters_for_verification,
    )
    normalized_text = _normalize(bounded_text)
    normalized_metadata = _normalize_metadata({**candidate.metadata, **document.metadata})

    results: list[ConstraintVerificationResult] = []
    for constraint in structured_query.constraints:
        if _timed_out(started, config.document_verification_timeout_ms):
            results.append(
                ConstraintVerificationResult(
                    constraint_id=constraint.id,
                    category=constraint.category,
                    status=ConstraintVerificationStatus.NOT_PROVEN,
                    required_value=constraint.value,
                    detected_value=None,
                    evidence=[],
                    verification_method=VerificationMethod.NOT_EVALUATED,
                    confidence=0.0,
                    reason="Per-document verification timeout exceeded.",
                )
            )
            continue
        results.append(
            _verify_constraint(
                constraint=constraint,
                document_id=document.document_id,
                chunks=bounded_chunks,
                normalized_text=normalized_text,
                normalized_metadata=normalized_metadata,
            )
        )

    decision_status = _decision_status(results, structured_query, config)
    supporting_evidence = [
        evidence
        for result in results
        if result.status == ConstraintVerificationStatus.MATCHED
        for evidence in result.evidence[:1]
    ][: config.max_supporting_chunks]
    score = _constraint_aware_score(candidate.score, results)

    return VerifiedDocument(
        document_id=candidate.document_id,
        score=score,
        decision_status=decision_status,
        constraint_results=results,
        supporting_passages=supporting_evidence,
        metadata={**candidate.metadata, **document.metadata},
        candidate_chunk_count=candidate.candidate_chunk_count,
    )


def _verify_constraint(
    *,
    constraint: StructuredConstraint,
    document_id: str,
    chunks: list[FullDocumentChunk],
    normalized_text: str,
    normalized_metadata: str,
) -> ConstraintVerificationResult:
    if constraint.category == ConstraintCategory.COURT:
        return _verify_court(constraint, document_id, normalized_metadata)
    if constraint.category == ConstraintCategory.LEGAL_EVENT:
        return _verify_legal_event(constraint, document_id, chunks, normalized_text)
    if constraint.category == ConstraintCategory.NATIONALITY:
        return _verify_nationality(constraint, document_id, chunks, normalized_text)
    if constraint.category == ConstraintCategory.COUNTRY_RELATION:
        return _verify_country_relation(constraint, document_id, chunks, normalized_text)
    if constraint.category == ConstraintCategory.ACTOR_ROLE:
        return _verify_actor_role(constraint, document_id, chunks, normalized_text)
    return ConstraintVerificationResult(
        constraint_id=constraint.id,
        category=constraint.category,
        status=ConstraintVerificationStatus.NOT_APPLICABLE,
        required_value=constraint.value,
        detected_value=None,
        evidence=[],
        verification_method=VerificationMethod.NOT_EVALUATED,
        confidence=0.0,
        reason="Constraint category is not implemented by deterministic verifier.",
    )


def _verify_court(
    constraint: StructuredConstraint,
    document_id: str,
    normalized_metadata: str,
) -> ConstraintVerificationResult:
    haystack = f"{_normalize(document_id)} {normalized_metadata}"
    detected: str | None = None
    if any(token in haystack for token in ("ecli:cz:us", "ustavni soud", "nalus", "usoud")):
        detected = "constitutional_court"
    elif any(token in haystack for token in ("ecli:cz:ns", "nejvyssi soud", "nsoud", "supreme")):
        detected = "supreme_court"

    if detected == constraint.value:
        return ConstraintVerificationResult(
            constraint_id=constraint.id,
            category=constraint.category,
            status=ConstraintVerificationStatus.MATCHED,
            required_value=constraint.value,
            detected_value=detected,
            evidence=[
                ConstraintEvidence(
                    document_id=document_id,
                    chunk_id=None,
                    quote="trusted metadata/document identity",
                    source_field="metadata",
                )
            ],
            verification_method=VerificationMethod.TRUSTED_METADATA,
            confidence=0.95,
            reason="Court matched trusted metadata or document identifier.",
        )
    if detected is not None:
        return ConstraintVerificationResult(
            constraint_id=constraint.id,
            category=constraint.category,
            status=ConstraintVerificationStatus.MISMATCH,
            required_value=constraint.value,
            detected_value=detected,
            evidence=[],
            verification_method=VerificationMethod.TRUSTED_METADATA,
            confidence=0.9,
            reason="Trusted court metadata contradicts requested court.",
        )
    return _not_proven_result(constraint, reason="Court could not be proven from metadata.")


def _verify_legal_event(
    constraint: StructuredConstraint,
    document_id: str,
    chunks: list[FullDocumentChunk],
    normalized_text: str,
) -> ConstraintVerificationResult:
    if constraint.value == "czech_citizenship_application_or_grant":
        patterns = (
            r"zadost.{0,50}udeleni.{0,50}statniho obcanstvi",
            r"neudeleni.{0,40}statniho obcanstvi",
            r"udeleni.{0,50}statniho obcanstvi",
            r"statni obcanstvi ceske republiky",
        )
        match = _find_first_pattern(chunks, patterns)
        if match:
            return _matched_result(
                constraint,
                document_id,
                match,
                detected_value="czech_citizenship_application_or_grant",
                reason="Document text proves Czech citizenship application/grant/refusal context.",
            )
        if "statni obcanstvi" in normalized_text and not _contains_any(
            normalized_text,
            ("udeleni", "neudeleni", "zadost"),
        ):
            return _mismatch_result(
                constraint,
                detected_value="citizenship_topic_without_grant_or_application",
                reason="Text concerns citizenship, but not grant/application/refusal.",
            )
        return _not_proven_result(
            constraint,
            reason="Czech citizenship grant/application event was not proven.",
        )

    if constraint.value == "international_child_abduction":
        patterns = (
            r"mezinarodni.{0,20}unos.{0,20}ditete",
            r"haagsk.{0,30}umluv.{0,80}navraceni.{0,30}ditete",
            r"neopravnene.{0,30}(premisteni|zadrzeni).{0,80}(ditete|nezletil)",
            r"navraceni.{0,30}(ditete|nezletil).{0,80}haagsk",
        )
        match = _find_first_pattern(chunks, patterns)
        if match:
            return _matched_result(
                constraint,
                document_id,
                match,
                detected_value="international_child_abduction",
                reason="Document text proves child abduction/wrongful removal context.",
            )
        return _not_proven_result(
            constraint,
            reason="International child abduction event was not proven.",
        )

    return _not_proven_result(constraint, reason="Unknown legal-event value.")


def _verify_nationality(
    constraint: StructuredConstraint,
    document_id: str,
    chunks: list[FullDocumentChunk],
    normalized_text: str,
) -> ConstraintVerificationResult:
    country_patterns = _COUNTRY_PATTERNS.get(constraint.value, ())
    if not country_patterns:
        return _not_proven_result(constraint, reason="Unsupported nationality country code.")

    country_expr = "|".join(re.escape(pattern) for pattern in country_patterns)
    role_expr = r"(stezovatel|zadatel|zalobce|cizinec|osoba|obcan)"
    patterns = (
        rf"{role_expr}.{{0,90}}(statni )?obcan.{{0,60}}({country_expr})",
        rf"({country_expr}).{{0,60}}(statni )?obcan",
        rf"({country_expr}).{{0,60}}statni prislusn",
    )
    match = _find_first_pattern(chunks, patterns)
    if match:
        return _matched_result(
            constraint,
            document_id,
            match,
            detected_value=constraint.value,
            reason="Requested nationality was proven by document text.",
        )

    if any(word in normalized_text for word in ("obcan ukrajiny", "ukrajinsky obcan")) and constraint.value != "UA":
        return _mismatch_result(
            constraint,
            detected_value="UA",
            reason="Text proves a different applicant nationality.",
        )
    return _not_proven_result(
        constraint,
        reason="Requested applicant/person nationality was not proven.",
    )


def _verify_country_relation(
    constraint: StructuredConstraint,
    document_id: str,
    chunks: list[FullDocumentChunk],
    normalized_text: str,
) -> ConstraintVerificationResult:
    country_patterns = _COUNTRY_PATTERNS.get(constraint.value, ())
    if not country_patterns:
        return _not_proven_result(constraint, reason="Unsupported country relation code.")
    country_expr = "|".join(re.escape(pattern) for pattern in country_patterns)
    event_expr = r"(neopravnene|unos|premisteni|zadrzeni|navraceni|ditete|nezletil)"
    patterns = (
        rf"{event_expr}.{{0,120}}\b(do|v|ve)\s+({country_expr})",
        rf"\b(do|v|ve)\s+({country_expr}).{{0,120}}{event_expr}",
    )
    match = _find_first_pattern(chunks, patterns)
    if match:
        return _matched_result(
            constraint,
            document_id,
            match,
            detected_value=constraint.value,
            reason="Requested country relation was proven by nearby event evidence.",
        )

    if "unos" in normalized_text or "neopravnene premisteni" in normalized_text:
        other = _detect_other_country(normalized_text, constraint.value)
        if other:
            return _mismatch_result(
                constraint,
                detected_value=other,
                reason="Child-abduction context mentions a different country relation.",
            )
    return _not_proven_result(
        constraint,
        reason="Requested country relation was not proven.",
    )


def _verify_actor_role(
    constraint: StructuredConstraint,
    document_id: str,
    chunks: list[FullDocumentChunk],
    normalized_text: str,
) -> ConstraintVerificationResult:
    if constraint.value != "parent":
        return _not_proven_result(constraint, reason="Unsupported actor role.")
    patterns = (
        r"(matka|otec|rodic).{0,120}(neopravnene|premisteni|zadrzeni|unos|navraceni)",
        r"(neopravnene|premisteni|zadrzeni|unos|navraceni).{0,120}(matka|otec|rodic)",
    )
    match = _find_first_pattern(chunks, patterns)
    if match:
        return _matched_result(
            constraint,
            document_id,
            match,
            detected_value="parent",
            reason="Parent actor role was proven near removal/retention evidence.",
        )
    if "matka" in normalized_text or "otec" in normalized_text or "rodic" in normalized_text:
        return _not_proven_result(
            constraint,
            reason="Parent was mentioned, but not proven as removal/retention actor.",
        )
    return _not_proven_result(constraint, reason="Parent actor role was not proven.")


def _matched_result(
    constraint: StructuredConstraint,
    document_id: str,
    match: _EvidenceMatch,
    *,
    detected_value: str,
    reason: str,
) -> ConstraintVerificationResult:
    return ConstraintVerificationResult(
        constraint_id=constraint.id,
        category=constraint.category,
        status=ConstraintVerificationStatus.MATCHED,
        required_value=constraint.value,
        detected_value=detected_value,
        evidence=[
            ConstraintEvidence(
                document_id=document_id,
                chunk_id=match.chunk_id,
                quote=match.quote,
                source_field=match.source_field,
            )
        ],
        verification_method=VerificationMethod.DETERMINISTIC_EVIDENCE,
        confidence=0.85,
        reason=reason,
    )


def _mismatch_result(
    constraint: StructuredConstraint,
    *,
    detected_value: str,
    reason: str,
) -> ConstraintVerificationResult:
    return ConstraintVerificationResult(
        constraint_id=constraint.id,
        category=constraint.category,
        status=ConstraintVerificationStatus.MISMATCH,
        required_value=constraint.value,
        detected_value=detected_value,
        evidence=[],
        verification_method=VerificationMethod.DETERMINISTIC_RELATION,
        confidence=0.75,
        reason=reason,
    )


def _not_proven_result(
    constraint: StructuredConstraint,
    *,
    reason: str,
) -> ConstraintVerificationResult:
    return ConstraintVerificationResult(
        constraint_id=constraint.id,
        category=constraint.category,
        status=ConstraintVerificationStatus.NOT_PROVEN,
        required_value=constraint.value,
        detected_value=None,
        evidence=[],
        verification_method=VerificationMethod.DETERMINISTIC_EVIDENCE,
        confidence=0.0,
        reason=reason,
    )


def _decision_status(
    results: list[ConstraintVerificationResult],
    structured_query: StructuredQuery,
    config: ConstraintRetrievalConfig,
) -> DocumentDecisionStatus:
    hard_ids = {constraint.id for constraint in structured_query.hard_constraints}
    hard_results = [result for result in results if result.constraint_id in hard_ids]
    if any(result.status == ConstraintVerificationStatus.MISMATCH for result in hard_results):
        return DocumentDecisionStatus.EXCLUDED_HARD_MISMATCH
    if config.strict_mode and any(
        result.status == ConstraintVerificationStatus.NOT_PROVEN for result in hard_results
    ):
        return DocumentDecisionStatus.EXCLUDED_NOT_PROVEN
    if hard_results and all(
        result.status == ConstraintVerificationStatus.MATCHED for result in hard_results
    ):
        return DocumentDecisionStatus.VERIFIED_MATCH
    if not hard_results:
        return DocumentDecisionStatus.VERIFIED_MATCH
    return DocumentDecisionStatus.EXCLUDED_NOT_PROVEN


def _constraint_aware_score(
    base_score: float,
    results: list[ConstraintVerificationResult],
) -> float:
    hard_matches = sum(
        1
        for result in results
        if result.status == ConstraintVerificationStatus.MATCHED
        and result.confidence >= 0.75
    )
    hard_mismatches = sum(
        1 for result in results if result.status == ConstraintVerificationStatus.MISMATCH
    )
    return max(0.0, float(base_score) + (0.05 * hard_matches) - (0.2 * hard_mismatches))


def _find_first_pattern(
    chunks: list[FullDocumentChunk],
    patterns: tuple[str, ...],
) -> _EvidenceMatch | None:
    for chunk in chunks:
        normalized = _normalize(chunk.text)
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if match:
                return _EvidenceMatch(
                    chunk_id=chunk.chunk_id,
                    quote=_quote_around_normalized_match(chunk.text, match.start(), match.end()),
                    source_field="full_text_chunk",
                )
    return None


def _quote_around_normalized_match(original_text: str, start: int, end: int) -> str:
    normalized_original = _normalize(original_text)
    center_start = max(0, start - 140)
    center_end = min(len(normalized_original), end + 140)
    quote = normalized_original[center_start:center_end]
    quote = re.sub(r"\s+", " ", quote).strip()
    return quote[:360]


def _bounded_document_text(
    chunks: list[FullDocumentChunk],
    *,
    max_characters: int,
) -> str:
    parts: list[str] = []
    used = 0
    for chunk in chunks:
        if used >= max_characters:
            break
        remaining = max_characters - used
        text = chunk.text[:remaining]
        parts.append(text)
        used += len(text)
    return "\n\n".join(parts)


def _normalize_metadata(metadata: dict[str, Any]) -> str:
    return _normalize(" ".join(str(value) for value in metadata.values() if value is not None))


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", without_marks.lower()).strip()


def _contains_any(normalized: str, needles: tuple[str, ...]) -> bool:
    return any(needle in normalized for needle in needles)


def _detect_other_country(normalized_text: str, requested: str) -> str | None:
    for code, patterns in _COUNTRY_PATTERNS.items():
        if code == requested:
            continue
        if any(pattern in normalized_text for pattern in patterns):
            return code
    return None


def _timed_out(started: float, timeout_ms: int | None) -> bool:
    if timeout_ms is None:
        return False
    return (time.perf_counter() - started) * 1000 > timeout_ms
