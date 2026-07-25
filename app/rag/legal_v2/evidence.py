from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.rag.legal_v2.models import LegalParagraph, SectionType
from app.rag.legal_v2.query_spec import QuerySpecV2
from app.rag.legal_v2.verifier import EvidenceWindowForConstraint

PREFERRED_SECTIONS = (
    SectionType.FACTS,
    SectionType.PROCEDURAL_HISTORY,
    SectionType.COURT_REASONING,
    SectionType.OPERATIVE_PART,
)
RESTRICTED_SECTIONS = (
    SectionType.PARTY_ARGUMENTS,
    SectionType.LEGAL_FRAMEWORK,
    SectionType.CITED_CASE,
    SectionType.HEADER,
    SectionType.INSTRUCTION,
)


@dataclass(frozen=True)
class CandidateEvidenceDocument:
    document_id: str
    metadata: dict[str, Any]
    paragraphs: list[LegalParagraph]
    score: float = 0.0
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    chunk_ids: list[str] = field(default_factory=list)


def select_evidence_windows(
    *,
    query_spec: QuerySpecV2,
    candidate: CandidateEvidenceDocument,
    max_windows_per_constraint: int = 2,
    max_window_chars: int = 1400,
) -> list[EvidenceWindowForConstraint]:
    windows: list[EvidenceWindowForConstraint] = []
    for constraint in query_spec.hard_constraints:
        ranked = sorted(
            candidate.paragraphs,
            key=lambda paragraph: (
                -_paragraph_score(paragraph, constraint.normalized_value),
                paragraph.paragraph_index,
            ),
        )
        selected = [paragraph for paragraph in ranked if _paragraph_score(paragraph, constraint.normalized_value) > 0]
        if not selected:
            selected = ranked[:1]
        for paragraph in selected[:max_windows_per_constraint]:
            context = _neighbor_context(candidate.paragraphs, paragraph)
            text = "\n\n".join(item.normalized_text for item in context)[:max_window_chars]
            windows.append(
                EvidenceWindowForConstraint(
                    constraint_id=constraint.constraint_id,
                    paragraph_ids=[item.paragraph_id for item in context],
                    text=text,
                    section_types=[item.section_type for item in context],
                    heading_context=paragraph.heading_context,
                    source_of_claim=source_of_claim_for_section(paragraph.section_type),
                    current_case_classification=(
                        "cited_case" if paragraph.section_type == SectionType.CITED_CASE else "current_case"
                    ),
                )
            )
    return windows


def source_of_claim_for_section(section: SectionType) -> str:
    if section in {SectionType.FACTS, SectionType.PROCEDURAL_HISTORY, SectionType.COURT_REASONING, SectionType.OPERATIVE_PART}:
        return "court_finding"
    if section == SectionType.PARTY_ARGUMENTS:
        return "party_claim"
    if section == SectionType.CITED_CASE:
        return "cited_case"
    if section == SectionType.HEADER:
        return "metadata"
    return "unknown"


def _paragraph_score(paragraph: LegalParagraph, normalized_value: str) -> float:
    text = paragraph.normalized_text.lower()
    score = 0.0
    for token in normalized_value.split():
        if len(token) >= 3 and token in text:
            score += 1.0
    if paragraph.section_type in PREFERRED_SECTIONS:
        score += 0.4
    if paragraph.section_type in RESTRICTED_SECTIONS:
        score -= 0.6
    if paragraph.is_citation_block:
        score -= 0.4
    if paragraph.is_boilerplate:
        score -= 0.2
    return score


def _neighbor_context(paragraphs: list[LegalParagraph], anchor: LegalParagraph) -> list[LegalParagraph]:
    start = max(0, anchor.paragraph_index - 1)
    end = min(len(paragraphs), anchor.paragraph_index + 2)
    return paragraphs[start:end]
