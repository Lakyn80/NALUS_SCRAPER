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
_ROMAN_ONLY_RE = re.compile(r"^\s*(I{1,3}|IV|V|VI{0,3}|IX|X)[.)]?\s*$", re.IGNORECASE)
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
    (
        SectionType.HEADER,
        (
            "česká republika",
            "ústavní soud",
            "vrchní soud v praze",
            "vrchní soud v olomouci",
            "nález",
            "usnesení",
            "rozsudek",
            "jménem republiky",
            "rozsudek jménem republiky",
        ),
    ),
    (SectionType.PARTICIPANTS, ("účastníci řízení", "účastníci", "účastníků řízení")),
    (
        SectionType.PROCEDURAL_HISTORY,
        (
            "průběh řízení",
            "dosavadní průběh řízení",
            "procesní předpoklady řízení před ústavním soudem",
            "průběh řízení před ústavním soudem",
        ),
    ),
    (
        SectionType.FACTS,
        (
            "skutkový stav",
            "skutková zjištění",
            "vymezení věci a obsah napadeného rozhodnutí",
        ),
    ),
    (SectionType.PARTY_ARGUMENTS, ("argumentace", "argumentace stěžovatele", "námitky", "vyjádření")),
    (SectionType.LEGAL_FRAMEWORK, ("právní úprava", "relevantní právo")),
    (SectionType.CITED_CASE, ("judikatura", "citovaná judikatura")),
    (
        SectionType.COURT_REASONING,
        (
            "odůvodnění",
            "posouzení",
            "posouzení důvodnosti ústavní stížnosti",
            "posouzení ústavního soudu",
            "právní posouzení",
            "hodnocení",
            "závěr",
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
_NALUS_US_HEADER_RE = re.compile(r"^NALUS\s*-\s*databáze rozhodnutí Ústavního soudu$", re.IGNORECASE)
_US_CASE_DATE_RE = re.compile(
    r"^[IVXLCDM]+\.?\s*ÚS\s+\d+/\d+\s+ze dne\s+\d{1,2}\.\s*\d{1,2}\.\s*\d{4}$",
    re.IGNORECASE,
)
_US_STATE_RE = re.compile(r"^Česká republika$", re.IGNORECASE)
_US_DECISION_TYPE_RE = re.compile(r"^(?:USNESENÍ|NÁLEZ)$", re.IGNORECASE)
_US_COURT_TITLE_RE = re.compile(r"^Ústavního soudu$", re.IGNORECASE)
_US_DECISION_FORMULA_RE = re.compile(r"^Ústavní soud rozhodl\b.*takto:\s*$", re.IGNORECASE)
_US_DECISION_FORMULA_START_RE = re.compile(r"^Ústavní soud rozhodl\b", re.IGNORECASE)
_REASONING_HEADING_RE = re.compile(r"^Odůvodnění:?\s*$", re.IGNORECASE)
_INSTRUCTION_START_RE = re.compile(r"^Poučení:\s+", re.IGNORECASE)
_BRNO_DATE_RE = re.compile(r"^V Brně dne\s+\d{1,2}\.\s*\d{1,2}\.\s*\d{4}$", re.IGNORECASE)
_SIGNATURE_ROLE_RE = re.compile(r"^(?:soudce zpravodaj|soudkyně zpravodajka|předseda senátu|předsedkyně senátu)$", re.IGNORECASE)
_SIGNATURE_NAME_RE = re.compile(r"\bv\.\s*r\.\s*$", re.IGNORECASE)
_REPUBLIC_TITLE_RE = re.compile(r"^Jménem republiky$", re.IGNORECASE)
_SIMPLE_HEADING_RE = re.compile(r"^(?:Výrok|Odůvodnění|Odůvodnění:|Poučení)$", re.IGNORECASE)
_DASH_BULLET_RE = re.compile(r"^-+\)")
_PLAIN_DASH_BULLET_RE = re.compile(r"^[-–—]\s+\S")
_LETTER_ITEM_RE = re.compile(r"^[a-z]\)")
_SEMICOLON_TABLE_RE = re.compile(r";")
_CONSTITUTIONAL_COMPACT_HEADING_RE = re.compile(
    r"^\s*(?:I{1,3}|IV|V|VI{0,3}|IX|X)(?:\.\d+)?[.)]?\s+"
    r"([A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ][\wÁČĎÉĚÍŇÓŘŠŤÚŮÝŽáčďéěíňóřšťúůýž\s,\-]{1,90})$",
    re.UNICODE,
)
_PRAGUE_OPENING_START_RE = re.compile(
    r"^Vrchní soud v Praze(?:\s+jako soud odvolací)?\s+rozhodl\b|^Vrchní soud v Praze jako soud odvolací\b",
    re.IGNORECASE,
)
_OLOMOUC_OPENING_START_RE = re.compile(r"^Vrchní soud v Olomouci\b.*\brozhodl\b", re.IGNORECASE)
_CIVIL_CASE_CUE_RE = re.compile(r"\b\d+\s*co\s+\d+|\bo\.\s*s\.\s*ř|\bosř\b", re.IGNORECASE)
_CRIMINAL_CASE_CUE_RE = re.compile(r"\b(?:to|tmo|nt)\s+\d+|\btr\.\s*ř|\btrestn", re.IGNORECASE)


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
    ROMAN_SECTION_MARKER = "roman_section_marker"
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
    court = str((metadata or {}).get("court") or "")
    candidates = _paragraph_candidates(normalized_source, court=court)
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
            if court == "constitutional_court":
                inferred = _constitutional_section_from_text(normalized_text, current_section)
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


def _paragraph_candidates(text: str, *, court: str = "") -> list[_ParagraphCandidate]:
    if court == "constitutional_court":
        line_candidates = _constitutional_court_line_candidates(text)
    elif court == "high_court_prague":
        line_candidates = _high_court_prague_line_candidates(text)
    elif court == "high_court_olomouc":
        line_candidates = _high_court_olomouc_line_candidates(text)
    else:
        line_candidates = _line_based_candidates(text)
    if len(line_candidates) >= 2:
        return line_candidates
    return _fallback_candidates(text)


def _constitutional_court_line_candidates(text: str) -> list[_ParagraphCandidate]:
    entries = _line_entries(text)
    candidates: list[_ParagraphCandidate] = []
    i = 0
    while i < len(entries):
        stripped = entries[i][0]
        if _US_DECISION_TYPE_RE.match(stripped) and i + 1 < len(entries) and _US_COURT_TITLE_RE.match(entries[i + 1][0]):
            end_index = i + 2
            if end_index < len(entries) and _REPUBLIC_TITLE_RE.match(entries[end_index][0]):
                end_index += 1
            candidates.append(_candidate_from_entries(entries[i:end_index], heading=True))
            i = end_index
            continue
        if _ROMAN_ONLY_RE.match(stripped) and i + 1 < len(entries) and _looks_like_section_caption(entries[i + 1][0]):
            candidates.append(_candidate_from_entries(entries[i : i + 2], heading=True))
            i += 2
            continue
        if _is_constitutional_compact_heading(stripped):
            candidates.append(_candidate_from_entries(entries[i : i + 1], heading=True))
            i += 1
            continue
        if _is_constitutional_singleton(stripped):
            candidates.append(_candidate_from_entries(entries[i : i + 1], heading=_is_constitutional_heading(stripped)))
            i += 1
            continue
        if _US_DECISION_FORMULA_START_RE.match(stripped):
            end_index = i + 1
            while end_index < len(entries) and not _US_DECISION_FORMULA_RE.match(" ".join(entry[0] for entry in entries[i:end_index])):
                next_text = entries[end_index][0]
                if (
                    _is_constitutional_singleton(next_text)
                    or _NUMBERED_RE.match(next_text)
                    or _is_constitutional_compact_heading(next_text)
                ):
                    break
                end_index += 1
            candidates.append(_candidate_from_entries(entries[i:end_index], heading=False))
            i = end_index
            continue
        if _SIGNATURE_NAME_RE.search(stripped) and i + 1 < len(entries) and _SIGNATURE_ROLE_RE.match(entries[i + 1][0]):
            candidates.append(_candidate_from_entries(entries[i : i + 2], heading=False))
            i += 2
            continue
        if _NUMBERED_RE.match(stripped):
            end_index = i + 1
            while end_index < len(entries):
                next_text = entries[end_index][0]
                if (
                    _NUMBERED_RE.match(next_text)
                    or _is_constitutional_singleton(next_text)
                    or _is_heading(next_text)
                    or _is_constitutional_compact_heading(next_text)
                    or (_ROMAN_ONLY_RE.match(next_text) and end_index + 1 < len(entries) and _looks_like_section_caption(entries[end_index + 1][0]))
                ):
                    break
                end_index += 1
            candidates.append(_candidate_from_entries(entries[i:end_index], heading=False))
            i = end_index
            continue
        kind = _classify_line(stripped, active_numbering=None)
        if kind in {_LineKind.ROMAN_BOUNDARY, _LineKind.ROMAN_SECTION_MARKER, _LineKind.HEADING}:
            candidates.append(_candidate_from_entries(entries[i : i + 1], heading=True))
        else:
            candidates.append(_candidate_from_entries(entries[i : i + 1], heading=False))
        i += 1
    return candidates


def _high_court_prague_line_candidates(text: str) -> list[_ParagraphCandidate]:
    entries = _line_entries(text)
    candidates: list[_ParagraphCandidate] = []
    i = 0
    if entries and _PRAGUE_OPENING_START_RE.match(entries[0][0]):
        end_index = 1
        while end_index < len(entries) and not _SIMPLE_HEADING_RE.match(entries[end_index][0]):
            end_index += 1
        candidates.append(_candidate_from_entries(entries[:end_index], heading=False))
        i = end_index
    while i < len(entries):
        stripped = entries[i][0]
        if _SIMPLE_HEADING_RE.match(stripped):
            candidates.append(_candidate_from_entries(entries[i : i + 1], heading=True))
            i += 1
            continue
        if _NUMBERED_RE.match(stripped):
            end_index = i + 1
            expected_nested = 1
            while end_index < len(entries):
                next_text = entries[end_index][0]
                if _SIMPLE_HEADING_RE.match(next_text):
                    break
                next_number = _leading_arabic_number(next_text)
                if next_number is not None:
                    if next_number == expected_nested and _opens_nested_list(stripped):
                        expected_nested += 1
                        end_index += 1
                        continue
                    break
                end_index += 1
            candidates.append(_candidate_from_entries(entries[i:end_index], heading=False))
            i = end_index
            continue
        candidates.append(_candidate_from_entries(entries[i : i + 1], heading=False))
        i += 1
    return candidates


def _high_court_olomouc_line_candidates(text: str) -> list[_ParagraphCandidate]:
    entries = _line_entries(text)
    reasoning_index = next((idx for idx, entry in enumerate(entries) if entry[0].casefold() == "odůvodnění"), len(entries))
    if _olomouc_is_civil_structure(entries, reasoning_index):
        return _olomouc_civil_line_candidates(entries, reasoning_index)
    return _olomouc_criminal_line_candidates(entries, reasoning_index)


def _olomouc_criminal_line_candidates(
    entries: list[tuple[str, int, int]],
    reasoning_index: int,
) -> list[_ParagraphCandidate]:
    candidates: list[_ParagraphCandidate] = []
    i = 0
    while i < reasoning_index:
        stripped = entries[i][0]
        if _SIMPLE_HEADING_RE.match(stripped) or _ROMAN_ONLY_RE.match(stripped):
            candidates.append(_candidate_from_entries(entries[i : i + 1], heading=True))
            i += 1
            continue
        end_index = i + 1
        while end_index < reasoning_index:
            next_text = entries[end_index][0]
            if _SIMPLE_HEADING_RE.match(next_text) or _ROMAN_ONLY_RE.match(next_text):
                break
            if _DASH_BULLET_RE.match(next_text) or _leading_arabic_number(next_text) is not None:
                break
            end_index += 1
        candidates.append(_candidate_from_entries(entries[i:end_index], heading=False))
        i = end_index
    if i < len(entries) and i == reasoning_index:
        candidates.append(_candidate_from_entries(entries[i : i + 1], heading=True))
        i += 1
    expected = 1
    while i < len(entries):
        stripped = entries[i][0]
        if _olomouc_top_level_reasoning_start(stripped, expected, civil=False):
            end_index = i + 1
            expected += 1
            while end_index < len(entries):
                if _olomouc_top_level_reasoning_start(entries[end_index][0], expected, civil=False):
                    break
                end_index += 1
            candidates.append(_candidate_from_entries(entries[i:end_index], heading=False))
            i = end_index
            continue
        candidates.append(_candidate_from_entries(entries[i : i + 1], heading=False))
        i += 1
    return candidates


def _olomouc_civil_line_candidates(
    entries: list[tuple[str, int, int]],
    reasoning_index: int,
) -> list[_ParagraphCandidate]:
    candidates: list[_ParagraphCandidate] = []
    i = 0
    if entries and _OLOMOUC_OPENING_START_RE.match(entries[0][0]):
        end_index = 1
        while end_index < reasoning_index and not _SIMPLE_HEADING_RE.match(entries[end_index][0]):
            end_index += 1
        candidates.append(_candidate_from_entries(entries[:end_index], heading=False))
        i = end_index
    while i < reasoning_index:
        stripped = entries[i][0]
        if _SIMPLE_HEADING_RE.match(stripped):
            candidates.append(_candidate_from_entries(entries[i : i + 1], heading=True))
            i += 1
            continue
        if _ROMAN_RE.match(stripped) or _ROMAN_ONLY_RE.match(stripped):
            # Civil operative Roman clauses are independent blocks, not nested lists.
            candidates.append(_candidate_from_entries(entries[i : i + 1], heading=False))
            i += 1
            continue
        end_index = i + 1
        while end_index < reasoning_index:
            next_text = entries[end_index][0]
            if _SIMPLE_HEADING_RE.match(next_text) or _ROMAN_RE.match(next_text) or _ROMAN_ONLY_RE.match(next_text):
                break
            end_index += 1
        candidates.append(_candidate_from_entries(entries[i:end_index], heading=False))
        i = end_index
    if i < len(entries) and i == reasoning_index:
        candidates.append(_candidate_from_entries(entries[i : i + 1], heading=True))
        i += 1
    expected = 1
    while i < len(entries):
        stripped = entries[i][0]
        if _olomouc_top_level_reasoning_start(stripped, expected, civil=True):
            end_index = i + 1
            expected += 1
            while end_index < len(entries):
                next_text = entries[end_index][0]
                if _olomouc_top_level_reasoning_start(next_text, expected, civil=True):
                    break
                end_index += 1
            candidates.append(_candidate_from_entries(entries[i:end_index], heading=False))
            i = end_index
            continue
        candidates.append(_candidate_from_entries(entries[i : i + 1], heading=False))
        i += 1
    return candidates


def _line_entries(text: str) -> list[tuple[str, int, int]]:
    entries: list[tuple[str, int, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        raw = line.rstrip("\n")
        stripped = raw.strip()
        line_start = offset
        line_end = offset + len(raw)
        offset += len(line)
        if stripped:
            entries.append((stripped, line_start, line_end))
    return entries


def _candidate_from_entries(entries: list[tuple[str, int, int]], *, heading: bool) -> _ParagraphCandidate:
    text = " ".join(entry[0] for entry in entries).strip()
    return _ParagraphCandidate(
        text=text,
        start=entries[0][1],
        end=entries[-1][2],
        numbering=_extract_numbering(text),
        heading=heading,
    )


def _is_constitutional_singleton(text: str) -> bool:
    return any(
        pattern.match(text)
        for pattern in (
            _NALUS_US_HEADER_RE,
            _US_CASE_DATE_RE,
            _US_STATE_RE,
            _US_DECISION_TYPE_RE,
            _REASONING_HEADING_RE,
            _INSTRUCTION_START_RE,
            _BRNO_DATE_RE,
            _SIGNATURE_ROLE_RE,
        )
    )


def _is_constitutional_heading(text: str) -> bool:
    return bool(_US_DECISION_TYPE_RE.match(text) or _US_COURT_TITLE_RE.match(text) or _REASONING_HEADING_RE.match(text))


def _looks_like_section_caption(text: str) -> bool:
    stripped = text.strip()
    if not stripped or _NUMBERED_RE.match(stripped):
        return False
    if len(stripped.split()) > 12:
        return False
    return bool(stripped[0].isupper())


def _leading_arabic_number(text: str) -> int | None:
    match = _NUMBERED_RE.match(text)
    if not match:
        return None
    value = next((group for group in match.groups() if group), None)
    return int(value) if value is not None else None


def _opens_nested_list(text: str) -> bool:
    stripped = text.strip()
    return stripped.endswith(":") or stripped.endswith("že") or "obsahuje:" in stripped.casefold()


def _is_constitutional_compact_heading(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 120:
        return False
    if stripped.endswith((".", ",", ";")):
        return False
    if _NUMBERED_RE.match(stripped) or _ROMAN_ONLY_RE.match(stripped):
        return False
    match = _CONSTITUTIONAL_COMPACT_HEADING_RE.match(stripped)
    if not match:
        return False
    caption = match.group(1).strip()
    if len(caption.split()) > 12:
        return False
    # Reject operative-style Roman clauses and long sentence continuations.
    if re.search(r"\b(?:se |je |bylo |byly |byl |byla )\b", caption.casefold()):
        return False
    if re.match(
        r"^(?:Usnesením|Rozsudkem|Rozsudek|Návrh|Žalob|Ústavní stížnost)\b",
        caption,
    ):
        return False
    return True


def _olomouc_is_civil_structure(entries: list[tuple[str, int, int]], reasoning_index: int) -> bool:
    vyrok_index = next((idx for idx, entry in enumerate(entries) if entry[0].casefold() == "výrok"), None)
    if vyrok_index is None or reasoning_index <= vyrok_index:
        return False
    between = entries[vyrok_index + 1 : reasoning_index]
    if not between or len(between) > 8:
        return False
    roman_text_clauses = [entry for entry in between if _ROMAN_RE.match(entry[0])]
    if not roman_text_clauses or len(roman_text_clauses) != len(between):
        return False
    opening = " ".join(entry[0] for entry in entries[:vyrok_index])
    if _CRIMINAL_CASE_CUE_RE.search(opening) and not _CIVIL_CASE_CUE_RE.search(opening):
        return False
    if _CIVIL_CASE_CUE_RE.search(opening):
        return True
    # Compact Roman+text operative block between Výrok and Odůvodnění is the civil shape.
    return all(len(entry[0]) < 600 for entry in roman_text_clauses)


def _olomouc_top_level_reasoning_start(text: str, expected_number: int, *, civil: bool = False) -> bool:
    match = _NUMBERED_RE.match(text)
    if not match:
        return False
    value = next((group for group in match.groups() if group), None)
    if value is None or int(value) != expected_number:
        return False
    rest = text[match.end() :].strip()
    if not rest or not rest[0].isupper():
        return False
    if _DASH_BULLET_RE.match(rest) or _PLAIN_DASH_BULLET_RE.match(rest) or _LETTER_ITEM_RE.match(rest):
        return False
    # Criminal nested/table rows may use semicolon-delimited cells; civil statutory paragraphs
    # can contain multiple semicolons without being tables.
    if not civil and _SEMICOLON_TABLE_RE.search(text) and text.count(";") >= 2:
        return False
    return True


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
            _LineKind.ROMAN_SECTION_MARKER,
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
        elif kind in {
            _LineKind.HEADING,
            _LineKind.ROMAN_BOUNDARY,
            _LineKind.ROMAN_SECTION_MARKER,
        }:
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
    if _ROMAN_ONLY_RE.match(text):
        return _LineKind.ROMAN_SECTION_MARKER
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
    if _ROMAN_ONLY_RE.match(stripped):
        return True
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


def _constitutional_section_from_text(text: str, current: SectionType) -> SectionType:
    if _US_DECISION_FORMULA_START_RE.match(text):
        return SectionType.OPERATIVE_PART
    if current == SectionType.OPERATIVE_PART:
        if _REASONING_HEADING_RE.match(text):
            return SectionType.COURT_REASONING
        return SectionType.OPERATIVE_PART
    if current == SectionType.COURT_REASONING:
        if _INSTRUCTION_START_RE.match(text):
            return SectionType.INSTRUCTION
        if _BRNO_DATE_RE.match(text) or _SIGNATURE_NAME_RE.search(text) or _SIGNATURE_ROLE_RE.match(text):
            return SectionType.HEADER
        return SectionType.COURT_REASONING
    if _INSTRUCTION_START_RE.match(text):
        return SectionType.INSTRUCTION
    if _BRNO_DATE_RE.match(text) or _SIGNATURE_NAME_RE.search(text) or _SIGNATURE_ROLE_RE.match(text):
        return SectionType.HEADER
    if _NALUS_US_HEADER_RE.match(text) or _US_CASE_DATE_RE.match(text) or _US_STATE_RE.match(text):
        return SectionType.HEADER
    return _section_from_text(text, current)


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
