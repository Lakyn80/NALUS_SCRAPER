from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from unicodedata import combining, normalize

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
    for constraint in [*query_spec.hard_constraints, *query_spec.soft_constraints]:
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
                    source_of_claim=effective_source_of_claim(
                        section=paragraph.section_type,
                        text=text,
                    ),
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


def effective_source_of_claim(*, section: SectionType, text: str) -> str:
    """Resolve claim source, repairing mislabeled judgment-body paragraphs.

    Some indexed chunks store operative/reasoning paragraphs as ``header``.
    Those must not be treated as metadata-only for holding verification.
    """
    source = source_of_claim_for_section(section)
    if source in {"metadata", "unknown"} and looks_like_court_holding_text(text):
        return "court_finding"
    return source


def looks_like_court_holding_text(text: str) -> bool:
    raw = str(text or "").strip()
    if len(raw) < 80:
        return False
    folded = " ".join(
        "".join(
            character
            for character in normalize("NFKD", raw.casefold())
            if not combining(character)
        ).split()
    )
    numbered = bool(re.match(r"^\d+\.\s+\S", raw))
    has_court = "ustavni soud" in folded or folded.startswith("soud ")
    disposition_markers = (
        "odmitl",
        "odmita",
        "odmitnout",
        "nepripustn",
        "vyhovel",
        "zrusil",
        "zamitl",
        "pro nepripustnost",
    )
    has_disposition = any(marker in folded for marker in disposition_markers)
    has_statute = "§ 43" in raw or "§ 75" in raw or "paragrafu 43" in folded
    if numbered and (has_court or has_disposition or has_statute):
        return True
    if has_court and (has_disposition or has_statute) and len(folded) >= 120:
        return True
    return False


def _paragraph_score(paragraph: LegalParagraph, normalized_value: str) -> float:
    text = paragraph.normalized_text.lower()
    score = 0.0
    for token in normalized_value.split():
        if len(token) >= 3 and token in text:
            score += 1.0
    if paragraph.section_type in PREFERRED_SECTIONS:
        score += 0.4
    if paragraph.section_type in RESTRICTED_SECTIONS:
        # Mislabeled operative/reasoning paragraphs often land in HEADER.
        # Do not bury lexical holding matches behind the restricted penalty.
        if paragraph.section_type == SectionType.HEADER and looks_like_court_holding_text(
            paragraph.normalized_text or paragraph.original_text
        ):
            score += 0.2
        else:
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
