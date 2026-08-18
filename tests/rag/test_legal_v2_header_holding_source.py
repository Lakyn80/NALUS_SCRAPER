from __future__ import annotations

from dataclasses import replace

from app.rag.legal_v2.evidence.selection import (
    CandidateEvidenceDocument,
    effective_source_of_claim,
    looks_like_court_holding_text,
    select_evidence_windows,
    source_of_claim_for_section,
)
from app.rag.legal_v2.models import LegalParagraph, MetadataProvenance, SectionType
from app.rag.legal_v2.query_spec import (
    ConstraintCategory,
    ConstraintPolarity,
    QueryConstraint,
    build_query_spec_v2,
)
from app.rag.legal_v2.verifier import (
    CandidateDocumentForVerification,
    DeterministicFakeVerifier,
    EvidenceWindowForConstraint,
    RelevanceClassification,
    VerificationDecision,
    deterministic_verification_gate,
    run_semantic_verifier,
)


_HOLDING = (
    "12. Ústavní soud proto ústavní stížnost mimo ústní jednání bez přítomnosti "
    "účastníků podle § 43 odst. 1 písm. e) zákona o Ústavním soudu pro nepřípustnost odmítl."
)


def _paragraph(
    *,
    document_id: str = "DOC-HEADER",
    index: int = 0,
    text: str,
    section: SectionType,
) -> LegalParagraph:
    return LegalParagraph(
        document_id=document_id,
        paragraph_id=f"{document_id}:p:{index}",
        paragraph_index=index,
        original_text=text,
        normalized_text=text,
        section_type=section,
        start_offset=0,
        end_offset=len(text),
        source_order=index,
        heading_context=[],
        is_boilerplate=False,
        is_citation_block=False,
        language="cs",
        metadata_provenance=MetadataProvenance(
            source="unit_test",
            extraction_method="manual",
        ),
    )


def test_looks_like_court_holding_text_detects_operative_refusal() -> None:
    assert looks_like_court_holding_text(_HOLDING) is True
    assert looks_like_court_holding_text("III. ÚS 2639/24") is False
    assert looks_like_court_holding_text("Ústavní soud") is False


def test_effective_source_upgrades_mislabeled_header_holding() -> None:
    assert source_of_claim_for_section(SectionType.HEADER) == "metadata"
    assert (
        effective_source_of_claim(section=SectionType.HEADER, text=_HOLDING)
        == "court_finding"
    )
    assert (
        effective_source_of_claim(section=SectionType.HEADER, text="Sp. zn. I. ÚS 1/24")
        == "metadata"
    )


def test_select_evidence_windows_upgrades_header_holding_source() -> None:
    query = "kdy Ústavní soud odmítne ústavní stížnost jako nepřípustnou"
    spec = build_query_spec_v2(query)
    assert spec.hard_constraints
    candidate = CandidateEvidenceDocument(
        document_id="DOC-HEADER",
        metadata={},
        paragraphs=[
            _paragraph(index=0, text=_HOLDING, section=SectionType.HEADER),
        ],
    )

    windows = select_evidence_windows(query_spec=spec, candidate=candidate)
    assert windows
    assert all(window.source_of_claim == "court_finding" for window in windows)


def test_compact_exact_with_header_holding_can_pass_gate() -> None:
    query = "kdy Ústavní soud odmítne ústavní stížnost jako nepřípustnou"
    spec = build_query_spec_v2(query)
    paragraph = _paragraph(index=0, text=_HOLDING, section=SectionType.HEADER)
    candidate = CandidateDocumentForVerification(
        document_id="DOC-HEADER",
        metadata={"court_name": "Ústavní soud"},
        paragraphs=[paragraph],
    )
    concept_ids = [f"C{index}" for index in range(1, len(spec.hard_constraints) + 1)]
    evidence_ids = [f"E{index}" for index in range(1, len(spec.hard_constraints) + 1)]
    provider = DeterministicFakeVerifier(
        {
            "document_id": candidate.document_id,
            "classification": RelevanceClassification.EXACT_MATCH.value,
            "confidence": 0.93,
            "jurisdiction_match": True,
            "supported_concept_ids": concept_ids,
            "missing_concept_ids": [],
            "contradiction_ids": [],
            "evidence_ids": evidence_ids,
            "reason_code": "header_holding_supported",
        }
    )
    windows = [
        EvidenceWindowForConstraint(
            constraint_id=constraint.constraint_id,
            paragraph_ids=[paragraph.paragraph_id],
            text=_HOLDING,
            section_types=[SectionType.HEADER],
            source_of_claim=effective_source_of_claim(
                section=SectionType.HEADER,
                text=_HOLDING,
            ),
        )
        for constraint in spec.hard_constraints
    ]

    result = run_semantic_verifier(
        provider=provider,
        query_spec=spec,
        candidate_document=candidate,
        evidence_windows=windows,
    )

    assert result.decision == VerificationDecision.VERIFIED_MATCH
    assert all(
        item.source_of_claim == "court_finding" and item.evidence_paragraph_ids
        for item in result.constraint_results
        if item.constraint_id in {c.constraint_id for c in spec.hard_constraints}
    )
    assert (
        deterministic_verification_gate(query_spec=spec, verifier_result=result)
        == VerificationDecision.VERIFIED_MATCH
    )


def test_compact_strong_with_metadata_only_is_demoted() -> None:
    query = "kdy Ústavní soud odmítne ústavní stížnost jako nepřípustnou"
    spec = build_query_spec_v2(query)
    metadata_text = "Sp. zn. I. ÚS 2639/24 Ústavní soud Brno"
    paragraph = _paragraph(index=0, text=metadata_text, section=SectionType.HEADER)
    candidate = CandidateDocumentForVerification(
        document_id="DOC-META",
        metadata={"court_name": "Ústavní soud"},
        paragraphs=[paragraph],
    )
    concept_ids = [f"C{index}" for index in range(1, len(spec.hard_constraints) + 1)]
    evidence_ids = [f"E{index}" for index in range(1, len(spec.hard_constraints) + 1)]
    provider = DeterministicFakeVerifier(
        {
            "document_id": candidate.document_id,
            "classification": RelevanceClassification.STRONG_MATCH.value,
            "confidence": 0.95,
            "jurisdiction_match": True,
            "supported_concept_ids": concept_ids,
            "missing_concept_ids": [],
            "contradiction_ids": [],
            "evidence_ids": evidence_ids,
            "reason_code": "lying_exact_on_metadata",
        }
    )
    windows = [
        EvidenceWindowForConstraint(
            constraint_id=constraint.constraint_id,
            paragraph_ids=[paragraph.paragraph_id],
            text=metadata_text,
            section_types=[SectionType.HEADER],
            source_of_claim="metadata",
        )
        for constraint in spec.hard_constraints
    ]

    result = run_semantic_verifier(
        provider=provider,
        query_spec=spec,
        candidate_document=candidate,
        evidence_windows=windows,
    )

    assert result.decision == VerificationDecision.AMBIGUOUS
    assert (
        result.raw_diagnostics.get("classification")
        == RelevanceClassification.INSUFFICIENT_EVIDENCE.value
    )
    assert result.raw_diagnostics.get("holding_supports_query") is False
    assert result.reason == "positive_classification_without_holding_proven_constraints"
    assert (
        deterministic_verification_gate(query_spec=spec, verifier_result=result)
        == VerificationDecision.NOT_PROVEN
    )


def test_compact_exact_with_incomplete_hard_proven_is_demoted() -> None:
    query = "únos dítěte matkou z Česka do Ruska"
    spec = build_query_spec_v2(query)
    # Lay origin/destination stay SOFT; inject a second hard concept so this
    # test covers incomplete hard coverage rather than QuerySpec slot polarity.
    extra_hard = QueryConstraint(
        constraint_id="constraint_test_second_hard",
        category=ConstraintCategory.CITED_CASE,
        value="II. ÚS 859/23",
        normalized_value="ii. us 859/23",
        polarity=ConstraintPolarity.HARD,
        attribute="cited_case",
    )
    spec = replace(spec, hard_constraints=[*spec.hard_constraints, extra_hard])
    assert len(spec.hard_constraints) >= 2
    holding = (
        "Soud zjistil, že matka neoprávněně přemístila dítě z Česka do Ruska "
        "a rozhodl o navrácení dítěte podle Haagské úmluvy o mezinárodních únosech."
    )
    paragraph = _paragraph(index=0, text=holding, section=SectionType.COURT_REASONING)
    candidate = CandidateDocumentForVerification(
        document_id="DOC-INCOMPLETE",
        metadata={"court_name": "Ústavní soud"},
        paragraphs=[paragraph],
    )
    # Support all but the last hard concept — mirrors exact_match with one miss.
    supported = [f"C{index}" for index in range(1, len(spec.hard_constraints))]
    evidence_ids = [f"E{index}" for index in range(1, len(spec.hard_constraints))]
    provider = DeterministicFakeVerifier(
        {
            "document_id": candidate.document_id,
            "classification": RelevanceClassification.EXACT_MATCH.value,
            "confidence": 0.93,
            "jurisdiction_match": True,
            "supported_concept_ids": supported,
            "missing_concept_ids": [f"C{len(spec.hard_constraints)}"],
            "contradiction_ids": [],
            "evidence_ids": evidence_ids,
            "reason_code": "exact_missing_one_hard",
        }
    )
    windows = [
        EvidenceWindowForConstraint(
            constraint_id=constraint.constraint_id,
            paragraph_ids=[paragraph.paragraph_id],
            text=holding,
            section_types=[SectionType.COURT_REASONING],
            source_of_claim="court_finding",
        )
        for constraint in spec.hard_constraints
    ]

    result = run_semantic_verifier(
        provider=provider,
        query_spec=spec,
        candidate_document=candidate,
        evidence_windows=windows,
    )

    assert result.decision == VerificationDecision.AMBIGUOUS
    assert (
        result.raw_diagnostics.get("classification")
        == RelevanceClassification.INSUFFICIENT_EVIDENCE.value
    )
    assert result.reason == "positive_classification_with_incomplete_hard_proven_constraints"
    missing = result.raw_diagnostics.get("mandatory_concepts_missing") or []
    assert missing
    assert (
        deterministic_verification_gate(query_spec=spec, verifier_result=result)
        == VerificationDecision.NOT_PROVEN
    )
