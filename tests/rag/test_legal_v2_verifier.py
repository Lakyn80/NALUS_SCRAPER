from __future__ import annotations

from app.rag.legal_v2.parser import parse_legal_document
from app.rag.legal_v2.query_spec import build_query_spec_v2
from app.rag.legal_v2.verifier import (
    CandidateDocumentForVerification,
    ConstraintVerificationStatus,
    DeterministicFakeVerifier,
    EvidenceWindowForConstraint,
    VerificationDecision,
    deterministic_verification_gate,
    run_semantic_verifier,
)


def _candidate() -> CandidateDocumentForVerification:
    document = parse_legal_document(
        document_id="DOC-VERIFY",
        text="\n\n".join(
            [
                "[1] Matka přemístila dítě z Česka do Ruska.",
                "[2] Soud posoudil navrácení dítěte podle Haagské úmluvy.",
            ]
        ),
    )
    return CandidateDocumentForVerification(
        document_id=document.document_id,
        metadata={"court_name": "Ústavní soud"},
        paragraphs=document.paragraphs,
    )


def _evidence(candidate: CandidateDocumentForVerification, constraint_id: str) -> list[EvidenceWindowForConstraint]:
    return [
        EvidenceWindowForConstraint(
            constraint_id=constraint_id,
            paragraph_ids=[candidate.paragraphs[0].paragraph_id],
            text=candidate.paragraphs[0].normalized_text,
            section_types=[candidate.paragraphs[0].section_type],
        )
    ]


def _payload(
    *,
    candidate: CandidateDocumentForVerification,
    spec_query: str,
    status: ConstraintVerificationStatus,
    include_all_hard: bool = True,
    evidence_ids: list[str] | None = None,
) -> dict:
    spec = build_query_spec_v2(spec_query)
    constraints = spec.hard_constraints if include_all_hard else spec.hard_constraints[:-1]
    evidence = (
        evidence_ids
        if evidence_ids is not None
        else [candidate.paragraphs[0].paragraph_id]
    )
    return {
        "document_id": candidate.document_id,
        "decision": VerificationDecision.VERIFIED_MATCH.value,
        "constraint_results": [
            {
                "constraint_id": constraint.constraint_id,
                "status": status.value,
                "detected_value": constraint.value,
                "evidence_paragraph_ids": evidence,
                "reason": "deterministic test payload",
                "confidence": 1.0,
            }
            for constraint in constraints
        ],
    }


def test_precise_query_with_zero_hard_constraints_is_unverifiable() -> None:
    spec = build_query_spec_v2("spravedlivé rozhodnutí ve věci")
    candidate = _candidate()
    provider = DeterministicFakeVerifier()

    result = run_semantic_verifier(
        provider=provider,
        query_spec=spec,
        candidate_document=candidate,
        evidence_windows=[],
    )

    assert provider.calls == 1
    assert spec.hard_constraints == []
    assert (
        deterministic_verification_gate(query_spec=spec, verifier_result=result)
        == VerificationDecision.UNVERIFIABLE_QUERY
    )


def test_invalid_llm_output_fails_closed() -> None:
    spec = build_query_spec_v2("únos dítěte matkou z Česka do Ruska")
    candidate = _candidate()
    provider = DeterministicFakeVerifier(
        {"document_id": candidate.document_id, "decision": "free form"}
    )

    result = run_semantic_verifier(
        provider=provider,
        query_spec=spec,
        candidate_document=candidate,
        evidence_windows=[],
    )

    assert result.decision == VerificationDecision.VERIFIER_ERROR
    assert (
        deterministic_verification_gate(query_spec=spec, verifier_result=result)
        == VerificationDecision.VERIFIER_ERROR
    )


def test_unknown_evidence_paragraph_id_fails_closed() -> None:
    query = "únos dítěte matkou z Česka do Ruska"
    spec = build_query_spec_v2(query)
    candidate = _candidate()
    provider = DeterministicFakeVerifier(
        _payload(
            candidate=candidate,
            spec_query=query,
            status=ConstraintVerificationStatus.PROVEN,
            evidence_ids=["missing-paragraph"],
        )
    )

    result = run_semantic_verifier(
        provider=provider,
        query_spec=spec,
        candidate_document=candidate,
        evidence_windows=[],
    )

    assert result.decision == VerificationDecision.VERIFIER_ERROR


def test_missing_hard_constraint_fails_as_not_proven() -> None:
    query = "únos dítěte matkou z Česka do Ruska"
    spec = build_query_spec_v2(query)
    candidate = _candidate()
    provider = DeterministicFakeVerifier(
        _payload(
            candidate=candidate,
            spec_query=query,
            status=ConstraintVerificationStatus.PROVEN,
            include_all_hard=False,
        )
    )

    result = run_semantic_verifier(
        provider=provider,
        query_spec=spec,
        candidate_document=candidate,
        evidence_windows=_evidence(candidate, spec.hard_constraints[0].constraint_id),
    )

    assert (
        deterministic_verification_gate(query_spec=spec, verifier_result=result)
        == VerificationDecision.NOT_PROVEN
    )


def test_contradicted_hard_constraint_fails_as_hard_mismatch() -> None:
    query = "únos dítěte matkou z Česka do Ruska"
    spec = build_query_spec_v2(query)
    candidate = _candidate()
    provider = DeterministicFakeVerifier(
        _payload(
            candidate=candidate,
            spec_query=query,
            status=ConstraintVerificationStatus.CONTRADICTED,
        )
    )

    result = run_semantic_verifier(
        provider=provider,
        query_spec=spec,
        candidate_document=candidate,
        evidence_windows=[],
    )

    assert (
        deterministic_verification_gate(query_spec=spec, verifier_result=result)
        == VerificationDecision.HARD_MISMATCH
    )


def test_all_proven_hard_constraints_pass_without_external_calls() -> None:
    query = "únos dítěte matkou z Česka do Ruska"
    spec = build_query_spec_v2(query)
    candidate = _candidate()
    provider = DeterministicFakeVerifier(
        _payload(
            candidate=candidate,
            spec_query=query,
            status=ConstraintVerificationStatus.PROVEN,
        )
    )

    result = run_semantic_verifier(
        provider=provider,
        query_spec=spec,
        candidate_document=candidate,
        evidence_windows=[],
    )

    assert provider.calls == 1
    assert result.provider_name == "deterministic_fake_verifier"
    assert (
        deterministic_verification_gate(query_spec=spec, verifier_result=result)
        == VerificationDecision.VERIFIED_MATCH
    )


def test_provider_error_fails_closed() -> None:
    spec = build_query_spec_v2("únos dítěte matkou z Česka do Ruska")
    candidate = _candidate()
    provider = DeterministicFakeVerifier(error=RuntimeError("offline failure"))

    result = run_semantic_verifier(
        provider=provider,
        query_spec=spec,
        candidate_document=candidate,
        evidence_windows=[],
    )

    assert result.decision == VerificationDecision.VERIFIER_ERROR
