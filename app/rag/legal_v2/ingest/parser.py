from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.rag.legal_v2.models import (
    LegalDocumentStructure,
    LegalParagraph,
    MetadataProvenance,
    ParagraphParsingDiagnostics,
    SectionType,
    normalize_legal_text,
    stable_paragraph_id,
)

_NUMBERED_RE = re.compile(r"^\s*(?:\[(\d{1,4})\]|(\d{1,4})[.)])\s+")
_ROMAN_RE = re.compile(r"^\s*(I{1,3}|IV|V|VI{0,3}|IX|X)[.)]\s+")
_HEADING_RE = re.compile(r"^\s*(I{1,3}|IV|V|VI{0,3}|IX|X)?\.?\s*([A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ][A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ\s]{3,})\s*$")
_HEADING_PREFIX_RE = re.compile(r"^\s*(?:(?:I{1,3}|IV|V|VI{0,3}|IX|X)[.)]\s+)?(.+?)\s*:?\s*$", re.IGNORECASE)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ0-9])")

_SECTION_KEYWORDS: tuple[tuple[SectionType, tuple[str, ...]], ...] = (
    (SectionType.HEADER, ("ústavní soud", "nejvyšší soud", "česká republika")),
    (SectionType.PARTICIPANTS, ("účastní", "stěžovatel", "navrhovatel", "odpůrce")),
    (SectionType.PROCEDURAL_HISTORY, ("řízení", "dosavadní průběh", "napadeným rozhodnutím")),
    (SectionType.FACTS, ("skutkov", "zjistil", "vyplývá", "stalo")),
    (SectionType.PARTY_ARGUMENTS, ("namít", "tvrdí", "argument", "vyjádření")),
    (SectionType.LEGAL_FRAMEWORK, ("zákon", "ustanovení", "čl.", "§", "úmluv")),
    (SectionType.CITED_CASE, ("judikatur", "srov.", "nález", "rozsudek")),
    (SectionType.COURT_REASONING, ("ústavní soud dospěl", "soud shledal", "posoudil")),
    (SectionType.OPERATIVE_PART, ("takto", "výrok", "návrh se", "ústavní stížnost se")),
    (SectionType.INSTRUCTION, ("poučení", "opravný prostředek")),
)

_HEADING_SECTION_HINTS: tuple[tuple[SectionType, tuple[str, ...]], ...] = (
    (SectionType.PARTICIPANTS, ("účastníci", "účastníků")),
    (SectionType.PROCEDURAL_HISTORY, ("řízení", "průběh")),
    (SectionType.FACTS, ("skutkový stav", "skutková")),
    (SectionType.PARTY_ARGUMENTS, ("argumentace", "námitky", "vyjádření")),
    (SectionType.LEGAL_FRAMEWORK, ("právní úprava", "relevantní právo")),
    (SectionType.CITED_CASE, ("judikatura", "citovaná")),
    (SectionType.COURT_REASONING, ("odůvodnění", "posouzení", "hodnocení")),
    (SectionType.OPERATIVE_PART, ("výrok", "takto")),
    (SectionType.INSTRUCTION, ("poučení",)),
)
_EXACT_HEADING_TITLES: tuple[tuple[SectionType, tuple[str, ...]], ...] = (
    (SectionType.PARTICIPANTS, ("účastníci řízení", "účastníci", "účastníků řízení")),
    (SectionType.PROCEDURAL_HISTORY, ("průběh řízení", "dosavadní průběh řízení")),
    (SectionType.FACTS, ("skutkový stav", "skutková zjištění")),
    (SectionType.PARTY_ARGUMENTS, ("argumentace", "námitky", "vyjádření")),
    (SectionType.LEGAL_FRAMEWORK, ("právní úprava", "relevantní právo")),
    (SectionType.CITED_CASE, ("judikatura", "citovaná judikatura")),
    (
        SectionType.COURT_REASONING,
        (
            "odůvodnění",
            "posouzení",
            "posouzení ústavního soudu",
            "právní posouzení",
            "hodnocení",
        ),
    ),
    (SectionType.OPERATIVE_PART, ("výrok", "takto")),
    (SectionType.INSTRUCTION, ("poučení",)),
)

_BOILERPLATE_RE = re.compile(
    r"(ústavní soud rozhodl|takto:|odůvodnění:|poučení:|za účasti|soudce zpravodaj)",
    re.IGNORECASE,
)
_CITATION_RE = re.compile(
    r"(\bsp\.\s*zn\.|\bč\.\s*j\.|\bECLI:|§\s*\d+|čl\.\s*\d+|Sb\.|ÚS\s*\d+/\d+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _ParagraphCandidate:
    text: str
    start: int
    end: int
    numbering: str | None = None
    heading: bool = False


class _LineKind(str, Enum):
    NUMBERED_PARAGRAPH_START = "numbered_paragraph_start"
    ROMAN_BOUNDARY = "roman_boundary"
    NUMBERED_PARAGRAPH_CONTINUATION = "numbered_paragraph_continuation"
    HEADING = "heading"
    PROSE = "prose"


def parse_legal_document(
    *,
    document_id: str,
    text: str,
    metadata: dict[str, Any] | None = None,
    language: str = "cs",
    provenance: MetadataProvenance | None = None,
) -> LegalDocumentStructure:
    normalized_source = _normalize_line_endings(text)
    provenance = provenance or MetadataProvenance(
        source="runtime",
        extraction_method="deterministic_legal_v2_parser",
        document_version=(metadata or {}).get("document_version"),
        source_url=(metadata or {}).get("source_url") or (metadata or {}).get("text_url"),
    )
    candidates = _paragraph_candidates(normalized_source)
    fallback_count = 0
    if not candidates and normalized_source.strip():
        candidates = _fallback_candidates(normalized_source)
        fallback_count = len(candidates)

    heading_context: list[str] = []
    current_section = SectionType.OTHER
    paragraphs: list[LegalParagraph] = []
    heading_count = 0
    numbered_count = 0
    boilerplate_count = 0
    citation_count = 0
    damaged_formatting = _damaged_formatting_detected(normalized_source)

    for index, candidate in enumerate(candidates):
        original_text = candidate.text.strip()
        normalized_text = normalize_legal_text(original_text)
        if not normalized_text:
            continue

        is_heading = candidate.heading or _is_heading(original_text)
        if candidate.numbering is not None and not is_heading:
            numbered_count += 1
        if is_heading:
            heading_count += 1
            heading_context = [normalized_text]
            current_section = _section_from_heading(normalized_text)
            section_type = current_section
        else:
            inferred = _section_from_text(normalized_text, current_section)
            if inferred != SectionType.OTHER:
                current_section = inferred
            section_type = current_section

        is_boilerplate = bool(_BOILERPLATE_RE.search(normalized_text))
        is_citation_block = _is_citation_block(normalized_text)
        boilerplate_count += int(is_boilerplate)
        citation_count += int(is_citation_block)
        paragraph_id = stable_paragraph_id(
            document_id=document_id,
            paragraph_index=len(paragraphs),
            normalized_text=normalized_text,
            document_version=provenance.document_version,
        )
        paragraphs.append(
            LegalParagraph(
                document_id=document_id,
                paragraph_id=paragraph_id,
                paragraph_index=len(paragraphs),
                original_text=original_text,
                normalized_text=normalized_text,
                section_type=section_type,
                start_offset=candidate.start,
                end_offset=candidate.end,
                source_order=len(paragraphs),
                heading_context=list(heading_context),
                is_boilerplate=is_boilerplate,
                is_citation_block=is_citation_block,
                language=language,
                metadata_provenance=provenance,
                numbering=candidate.numbering,
            )
        )

    section_counts: dict[str, int] = {}
    for paragraph in paragraphs:
        section_counts[paragraph.section_type.value] = (
            section_counts.get(paragraph.section_type.value, 0) + 1
        )
    warnings: list[str] = []
    if damaged_formatting:
        warnings.append("damaged_blank_line_formatting_detected")
    if fallback_count:
        warnings.append("fallback_paragraph_segmentation_used")

    return LegalDocumentStructure(
        document_id=document_id,
        normalized_text=normalized_source,
        paragraphs=paragraphs,
        diagnostics=ParagraphParsingDiagnostics(
            paragraph_count=len(paragraphs),
            numbered_paragraph_count=numbered_count,
            heading_count=heading_count,
            boilerplate_count=boilerplate_count,
            citation_block_count=citation_count,
            damaged_formatting_detected=damaged_formatting,
            fallback_paragraphs_created=fallback_count,
            section_counts=section_counts,
            warnings=warnings,
        ),
        metadata=dict(metadata or {}),
    )


def _normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _paragraph_candidates(text: str) -> list[_ParagraphCandidate]:
    line_candidates = _line_based_candidates(text)
    if len(line_candidates) >= 2:
        return line_candidates
    return _fallback_candidates(text)


def _line_based_candidates(text: str) -> list[_ParagraphCandidate]:
    candidates: list[_ParagraphCandidate] = []
    current: list[str] = []
    current_start: int | None = None
    current_end = 0
    current_numbering: str | None = None
    offset = 0
    for line in text.splitlines(keepends=True):
        raw = line.rstrip("\n")
        stripped = raw.strip()
        line_start = offset
        line_end = offset + len(raw)
        offset += len(line)
        if not stripped:
            _flush_candidate(candidates, current, current_start, current_end)
            current = []
            current_start = None
            current_numbering = None
            continue
        kind = _classify_line(stripped, active_numbering=current_numbering)
        starts_new = kind in {
            _LineKind.NUMBERED_PARAGRAPH_START,
            _LineKind.ROMAN_BOUNDARY,
            _LineKind.HEADING,
        }
        if starts_new and current:
            _flush_candidate(candidates, current, current_start, current_end)
            current = []
            current_start = None
            current_numbering = None
        if current_start is None:
            current_start = line_start
        current.append(stripped)
        current_end = line_end
        if kind is _LineKind.NUMBERED_PARAGRAPH_START:
            current_numbering = _extract_numbering(stripped)
        elif kind in {_LineKind.HEADING, _LineKind.ROMAN_BOUNDARY}:
            _flush_candidate(candidates, current, current_start, current_end)
            current = []
            current_start = None
            current_numbering = None
    _flush_candidate(candidates, current, current_start, current_end)
    return candidates


def _fallback_candidates(text: str) -> list[_ParagraphCandidate]:
    stripped = text.strip()
    if not stripped:
        return []
    pieces = _SENTENCE_BOUNDARY_RE.split(stripped)
    candidates: list[_ParagraphCandidate] = []
    search_from = 0
    buffer: list[str] = []
    buffer_start: int | None = None
    buffer_end = 0
    for piece in pieces:
        value = piece.strip()
        if not value:
            continue
        start = text.find(value, search_from)
        if start < 0:
            start = search_from
        end = start + len(value)
        search_from = end
        if buffer_start is None:
            buffer_start = start
        buffer.append(value)
        buffer_end = end
        if len(" ".join(buffer).split()) >= 80 or _NUMBERED_RE.match(value):
            _flush_candidate(candidates, buffer, buffer_start, buffer_end)
            buffer = []
            buffer_start = None
    _flush_candidate(candidates, buffer, buffer_start, buffer_end)
    return candidates


def _flush_candidate(
    candidates: list[_ParagraphCandidate],
    lines: list[str],
    start: int | None,
    end: int,
) -> None:
    if start is None or not lines:
        return
    text = " ".join(line.strip() for line in lines if line.strip()).strip()
    if not text:
        return
    numbering = _extract_numbering(text)
    candidates.append(
        _ParagraphCandidate(
            text=text,
            start=start,
            end=end,
            numbering=numbering,
            heading=_is_heading(text),
        )
    )


def _extract_numbering(text: str) -> str | None:
    number_match = _NUMBERED_RE.match(text)
    if number_match:
        return next((group for group in number_match.groups() if group), None)
    roman_match = _ROMAN_RE.match(text)
    return roman_match.group(1) if roman_match else None


def _classify_line(text: str, *, active_numbering: str | None) -> _LineKind:
    if _NUMBERED_RE.match(text):
        return _LineKind.NUMBERED_PARAGRAPH_START
    if active_numbering is not None:
        if _is_heading(text):
            return _LineKind.HEADING
        return _LineKind.NUMBERED_PARAGRAPH_CONTINUATION
    if _ROMAN_RE.match(text):
        return _LineKind.ROMAN_BOUNDARY
    if _is_heading(text):
        return _LineKind.HEADING
    return _LineKind.PROSE


def _is_heading(text: str) -> bool:
    stripped = text.strip()
    if _NUMBERED_RE.match(stripped):
        return False
    if len(stripped) > 120 or len(stripped.split()) > 10:
        return False
    if stripped.endswith(".") and len(stripped.split()) > 3:
        return False
    if _HEADING_RE.match(stripped):
        return True
    return _heading_section_from_text(stripped) is not None


def _section_from_heading(text: str) -> SectionType:
    lowered = text.lower()
    exact = _heading_section_from_text(text)
    if exact is not None:
        return exact
    for section, keywords in _HEADING_SECTION_HINTS:
        if any(keyword in lowered for keyword in keywords):
            return section
    return SectionType.HEADER if "soud" in lowered else SectionType.OTHER


def _heading_section_from_text(text: str) -> SectionType | None:
    match = _HEADING_PREFIX_RE.match(text)
    if not match:
        return None
    title = normalize_legal_text(match.group(1)).casefold()
    for section, titles in _EXACT_HEADING_TITLES:
        if title in titles:
            return section
    return None


def _section_from_text(text: str, current: SectionType) -> SectionType:
    lowered = text.lower()
    for section, keywords in _SECTION_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return section
    return current


def _is_citation_block(text: str) -> bool:
    matches = len(_CITATION_RE.findall(text))
    if matches >= 2:
        return True
    lowered = text.lower()
    return "srov." in lowered and ("nález" in lowered or "rozsudek" in lowered)


def _damaged_formatting_detected(text: str) -> bool:
    non_empty_lines = [line for line in text.splitlines() if line.strip()]
    if len(non_empty_lines) < 3:
        return True
    blank_lines = len([line for line in text.splitlines() if not line.strip()])
    return blank_lines == 0 and len(non_empty_lines) > 5
