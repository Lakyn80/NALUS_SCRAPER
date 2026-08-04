from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.rag.legal_v2.ingest.parser import parse_legal_document

from scripts.legal_v2.court_format_study import Candidate, _source_lines

from .manifest import ReviewDocument, base_manifest_payload, load_design_documents
from .models import (
    DEFAULT_REVIEW_DIR,
    PROJECT_ROOT,
    REVIEW_SCHEMA_VERSION,
    REVIEW_TOOL_VERSION,
    boundary_review_id,
    line_review_id,
    read_jsonl,
    utc_now,
    write_json,
    write_jsonl,
)
from .progress import write_progress_files

CASE_REF_RE = re.compile(r"\b(?:sp\.\s*zn\.|č\.\s*j\.|c\.\s*j\.|ECLI:)", re.IGNORECASE)
NALUS_US_HEADER_RE = re.compile(r"^NALUS\s*-\s*databáze rozhodnutí Ústavního soudu$", re.IGNORECASE)
US_CASE_DATE_RE = re.compile(r"^[IVXLCDM]+\.?\s*ÚS\s+\d+/\d+\s+ze dne\s+\d{1,2}\.\s*\d{1,2}\.\s*\d{4}$", re.IGNORECASE)
US_STATE_RE = re.compile(r"^Česká republika$", re.IGNORECASE)
US_DECISION_TYPE_RE = re.compile(r"^(?:USNESENÍ|NÁLEZ)$", re.IGNORECASE)
US_COURT_TITLE_RE = re.compile(r"^Ústavního soudu$", re.IGNORECASE)
US_DECISION_FORMULA_RE = re.compile(r"^Ústavní soud rozhodl\b.*takto:\s*$", re.IGNORECASE)
REASONING_HEADING_RE = re.compile(r"^Odůvodnění:?\s*$", re.IGNORECASE)
INSTRUCTION_START_RE = re.compile(r"^Poučení:\s+", re.IGNORECASE)
BRNO_DATE_RE = re.compile(r"^V Brně dne\s+\d{1,2}\.\s*(?:\d{1,2}\.|[a-zá-ž]+)\s+\d{4}$", re.IGNORECASE)
SIGNATURE_ROLE_RE = re.compile(r"^(?:soudce zpravodaj|soudkyně zpravodajka|předseda senátu|předsedkyně senátu)$", re.IGNORECASE)
SIGNATURE_NAME_RE = re.compile(r"\bv\.\s*r\.\s*$", re.IGNORECASE)
REPUBLIC_TITLE_RE = re.compile(r"^Jménem republiky$", re.IGNORECASE)
SIMPLE_HEADING_RE = re.compile(r"^(?:Výrok|Odůvodnění|Odůvodnění:|Poučení)$", re.IGNORECASE)
ARABIC_NUMBER_RE = re.compile(r"^\s*(\d{1,4})[.)]\s+")
ROMAN_NUMBER_RE = re.compile(r"^\s*(?:I{1,3}|IV|V|VI{0,3}|IX|X)[.)]?\s*")
DASH_BULLET_RE = re.compile(r"^-+\)")
PLAIN_DASH_BULLET_RE = re.compile(r"^[-–—]\s+\S")
LETTER_ITEM_RE = re.compile(r"^[a-z]\)")
SEMICOLON_TABLE_RE = re.compile(r";")
CONSTITUTIONAL_COMPACT_HEADING_RE = re.compile(
    r"^\s*(?:I{1,3}|IV|V|VI{0,3}|IX|X)(?:\.\d+)?[.)]?\s+"
    r"([A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ][\wÁČĎÉĚÍŇÓŘŠŤÚŮÝŽáčďéěíňóřšťúůýž\s,\-]{1,90})$",
    re.UNICODE,
)


def build_snapshot(
    *,
    review_dir: Path = DEFAULT_REVIEW_DIR,
    document_filter: str | None = None,
    validate_only: bool = False,
) -> dict[str, Any]:
    source_manifest, documents = load_design_documents()
    if document_filter:
        documents = [
            item
            for item in documents
            if document_filter in {item.review_id, item.source_id, str(item.review_number)}
        ]
        if not documents:
            raise ValueError(f"Unknown design document filter: {document_filter}")
    prev_lines = _load_previous_line_annotations()
    prev_boundaries = _load_previous_boundary_annotations()
    manifest = base_manifest_payload(source_manifest, documents)
    manifest.update(
        {
            "review_tool_version": REVIEW_TOOL_VERSION,
            "generated_at": utc_now(),
            "output_dir": str(review_dir.relative_to(PROJECT_ROOT)),
        }
    )
    doc_records: list[dict[str, Any]] = []
    line_records: list[dict[str, Any]] = []
    boundary_records: list[dict[str, Any]] = []
    for document in documents:
        raw_lines = _raw_lines(document)
        text = "\n".join(raw_lines)
        parsed = parse_legal_document(
            document_id=document.source_id,
            text=text,
            metadata={"court": document.court, "source_url": document.source_url},
        )
        _assert_text_conserved(text, parsed.reconstruct_text(), document.source_id)
        line_offsets = _line_offsets(raw_lines)
        block_for_line = [_paragraph_for_line(parsed.paragraphs, start, end) for start, end in line_offsets]
        doc_record = _document_record(document, raw_lines, parsed)
        doc_records.append(doc_record)
        document_line_records: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(raw_lines, start=1):
            if not raw_line.strip():
                continue
            paragraph = block_for_line[line_number - 1]
            previous = prev_lines.get((document.source_id, line_number))
            item_id = line_review_id(
                document_id=document.review_id,
                raw_line_number=line_number,
                source_checksum=document.source_checksum,
            )
            document_line_records.append(
                {
                    "schema_version": REVIEW_SCHEMA_VERSION,
                    "item_type": "line",
                    "item_id": item_id,
                    "document_id": document.review_id,
                    "source_document_id": document.source_id,
                    "document_review_number": document.review_number,
                    "court": document.court,
                    "source_checksum": document.source_checksum,
                    "raw_line_number": line_number,
                    "source_page": None,
                    "raw_text": raw_line,
                    "normalized_display_text": _normalize(raw_line),
                    "parser_block_id": paragraph.paragraph_id if paragraph else None,
                    "parser_block_index": paragraph.paragraph_index if paragraph else None,
                    "parser_proposed_line_class": _line_class(raw_line, paragraph, parsed, line_number, block_for_line),
                    "parser_proposed_boundary_before": _boundary_before(line_number, block_for_line),
                    "parser_proposed_boundary_after": _boundary_after(line_number, raw_lines, block_for_line),
                    "parser_reason_code": _line_reason(raw_line, paragraph, line_number, block_for_line),
                    "previous_automated_annotation": previous.get("structural_class") if previous else None,
                    "previous_annotation_reason_code": previous.get("classification_reason_code") if previous else None,
                    "suspicious_reason_codes": _line_suspicious(raw_line, paragraph, previous),
                    "manual_decision_status": "pending",
                }
            )
        line_records.extend(document_line_records)
        boundary_records.extend(_boundary_records(document, document_line_records, prev_boundaries, block_for_line))
        if not validate_only:
            _write_document_files(review_dir, document, raw_lines, parsed, document_line_records, boundary_records)
    if not validate_only:
        review_dir.mkdir(parents=True, exist_ok=True)
        write_json(review_dir / "review_manifest.json", manifest)
        write_jsonl(review_dir / "review_documents.jsonl", doc_records)
        write_jsonl(review_dir / "review_lines.jsonl", line_records)
        write_jsonl(review_dir / "review_boundaries.jsonl", boundary_records)
        _ensure_manual_store(review_dir)
        write_json(review_dir / "server_state.json", {"schema_version": REVIEW_SCHEMA_VERSION, "status": "ready", "generated_at": utc_now()})
        write_progress_files(review_dir)
    return {
        "manifest": manifest,
        "documents": len(doc_records),
        "lines": len(line_records),
        "boundaries": len(boundary_records),
    }


def _load_previous_line_annotations() -> dict[tuple[str, int], dict[str, Any]]:
    rows = read_jsonl(PROJECT_ROOT / "artifacts" / "legal_v2" / "court_format_study" / "design_line_annotations.jsonl")
    return {(str(row["document_id"]), int(row["source_line_number"])): row for row in rows}


def _load_previous_boundary_annotations() -> dict[tuple[str, int, int], dict[str, Any]]:
    rows = read_jsonl(PROJECT_ROOT / "artifacts" / "legal_v2" / "court_format_study" / "design_boundary_annotations.jsonl")
    return {(str(row["document_id"]), int(row["left_source_line_number"]), int(row["right_source_line_number"])): row for row in rows}


def _raw_lines(document: ReviewDocument) -> list[str]:
    item = dict(document.manifest_item)
    item["raw_path"] = str(document.raw_path.relative_to(PROJECT_ROOT))
    lines = _source_lines(Candidate(**item))
    if not lines:
        raise ValueError(f"No extracted lines for {document.source_id}")
    return lines


def _line_offsets(lines: list[str]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    offset = 0
    for index, line in enumerate(lines):
        start = offset
        end = start + len(line)
        offsets.append((start, end))
        offset = end + (1 if index < len(lines) - 1 else 0)
    return offsets


def _paragraph_for_line(paragraphs: list[Any], start: int, end: int) -> Any | None:
    for paragraph in paragraphs:
        if paragraph.start_offset <= start and end <= paragraph.end_offset:
            return paragraph
    for paragraph in paragraphs:
        if max(start, paragraph.start_offset) < min(end, paragraph.end_offset):
            return paragraph
    return None


def _line_class(raw_line: str, paragraph: Any | None, parsed: Any, line_number: int, block_for_line: list[Any | None]) -> str:
    if paragraph is None:
        return "unmapped"
    is_first = _boundary_before(line_number, block_for_line)
    court = parsed.metadata.get("court")
    if court == "constitutional_court":
        constitutional_class = _constitutional_line_class(raw_line, paragraph, is_first)
        if constitutional_class:
            return constitutional_class
    if court == "high_court_prague":
        return _prague_line_class(raw_line, is_first, line_number, block_for_line, parsed)
    if court == "high_court_olomouc":
        return _olomouc_line_class(raw_line, paragraph, is_first, line_number, block_for_line, parsed)
    if paragraph.numbering:
        if is_first:
            return "numbered_paragraph_start"
        return "numbered_paragraph_continuation"
    section = paragraph.section_type.value
    if paragraph.normalized_text in paragraph.heading_context or (is_first and len(paragraph.original_text.split()) <= 10 and section in {"header", "court_reasoning", "operative_part", "instruction"}):
        return "heading"
    if section in {"header", "participants", "procedural_history"}:
        return "metadata"
    if section == "instruction":
        return "instruction"
    return "prose_start" if is_first else "prose_continuation"


def _constitutional_line_class(raw_line: str, paragraph: Any, is_first: bool) -> str | None:
    stripped = _normalize(raw_line)
    if NALUS_US_HEADER_RE.match(stripped):
        return "layout_noise"
    if US_CASE_DATE_RE.match(stripped) or US_STATE_RE.match(stripped) or BRNO_DATE_RE.match(stripped):
        return "metadata"
    if US_DECISION_TYPE_RE.match(stripped) or US_COURT_TITLE_RE.match(stripped) or REPUBLIC_TITLE_RE.match(stripped) or REASONING_HEADING_RE.match(stripped):
        return "heading"
    if US_DECISION_FORMULA_RE.match(stripped):
        return "prose_start"
    if INSTRUCTION_START_RE.match(stripped):
        return "instruction"
    if SIGNATURE_NAME_RE.search(stripped) or SIGNATURE_ROLE_RE.match(stripped):
        return "signature"
    if _is_constitutional_compact_heading(stripped):
        return "heading"
    if _is_roman_section_line(stripped) or (
        paragraph.normalized_text in paragraph.heading_context and _looks_like_short_caption(stripped)
    ):
        return "heading"
    if is_first and _looks_like_short_caption(stripped) and not stripped.endswith((".", ",", ";", ":")):
        return "heading"
    if paragraph.numbering:
        return "numbered_paragraph_start" if is_first else "numbered_paragraph_continuation"
    if paragraph.section_type.value == "court_reasoning":
        return "prose_start" if is_first else "prose_continuation"
    if paragraph.section_type.value == "operative_part":
        return "prose_start" if is_first else "prose_continuation"
    return None


def _prague_line_class(
    raw_line: str,
    is_first: bool,
    line_number: int,
    block_for_line: list[Any | None],
    parsed: Any,
) -> str:
    stripped = _normalize(raw_line)
    if SIMPLE_HEADING_RE.match(stripped):
        return "heading"
    # Participant numbers inside the case-opening formula remain prose, not nested lists.
    if _prague_in_opening(line_number, block_for_line, parsed):
        return "prose_start" if is_first else "prose_continuation"
    if _is_nested_or_table(stripped):
        return "list_or_table"
    if _is_numbered_or_roman_item(stripped):
        return "numbered_paragraph_start" if is_first else "list_or_table"
    return "prose_start" if is_first else "prose_continuation"


def _prague_in_opening(line_number: int, block_for_line: list[Any | None], parsed: Any) -> bool:
    current = block_for_line[line_number - 1] if 0 <= line_number - 1 < len(block_for_line) else None
    if current is None:
        return False
    for paragraph in parsed.paragraphs:
        if str(paragraph.normalized_text or "").strip().casefold() == "výrok":
            return int(current.start_offset) < int(paragraph.start_offset)
    return False


_MONTH_CONTINUATION_RE = re.compile(
    r"^(?:ledna|února|března|dubna|května|června|července|srpna|září|října|listopadu|prosince)\b",
    re.IGNORECASE,
)


def _olomouc_line_class(
    raw_line: str,
    paragraph: Any,
    is_first: bool,
    line_number: int,
    block_for_line: list[Any | None],
    parsed: Any,
) -> str:
    stripped = _normalize(raw_line)
    if SIMPLE_HEADING_RE.match(stripped) or _is_roman_section_line(stripped):
        return "heading"
    in_reasoning = _olomouc_in_reasoning(line_number, block_for_line, parsed)
    if in_reasoning:
        if is_first:
            return "numbered_paragraph_start"
        if _is_nested_or_table(stripped) or _is_genuine_nested_marker(stripped):
            return "list_or_table"
        return "numbered_paragraph_continuation"
    # Pre-reasoning.
    if _is_nested_or_table(stripped):
        return "list_or_table"
    if _is_numbered_or_roman_item(stripped):
        # Civil Roman operative clauses are independent numbered blocks.
        if is_first and paragraph.numbering and re.fullmatch(
            r"(?:I{1,3}|IV|V|VI{0,3}|IX|X)",
            str(paragraph.numbering),
            flags=re.IGNORECASE,
        ):
            return "numbered_paragraph_start"
        return "list_or_table"
    return "prose_start" if is_first else "prose_continuation"


def _olomouc_in_reasoning(line_number: int, block_for_line: list[Any | None], parsed: Any) -> bool:
    current = block_for_line[line_number - 1] if 0 <= line_number - 1 < len(block_for_line) else None
    if current is None:
        return False
    for paragraph in parsed.paragraphs:
        normalized = str(paragraph.normalized_text or "").strip().casefold()
        if normalized == "odůvodnění":
            return int(current.start_offset) > int(paragraph.start_offset)
    return False


def _is_constitutional_compact_heading(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 120 or ARABIC_NUMBER_RE.match(stripped) or _is_roman_section_line(stripped):
        return False
    if stripped.endswith((".", ",", ";")):
        return False
    match = CONSTITUTIONAL_COMPACT_HEADING_RE.match(stripped)
    if not match:
        return False
    caption = match.group(1).strip()
    if len(caption.split()) > 12:
        return False
    if re.search(r"\b(?:se |je |bylo |byly |byl |byla )\b", caption.casefold()):
        return False
    if re.match(
        r"^(?:Usnesením|Rozsudkem|Rozsudek|Návrh|Žalob|Ústavní stížnost)\b",
        caption,
    ):
        return False
    return True


def _is_numbered_or_roman_item(text: str) -> bool:
    return bool(ARABIC_NUMBER_RE.match(text) or re.match(r"^\s*(?:I{1,3}|IV|V|VI{0,3}|IX|X)[.)]\s+", text, re.IGNORECASE))


def _is_roman_section_line(text: str) -> bool:
    return bool(re.match(r"^\s*(?:I{1,3}|IV|V|VI{0,3}|IX|X)[.)]?\s*$", text, re.IGNORECASE))


def _looks_like_short_caption(text: str) -> bool:
    return bool(text and not ARABIC_NUMBER_RE.match(text) and len(text.split()) <= 12 and text[0].isupper())


def _is_nested_or_table(text: str) -> bool:
    if DASH_BULLET_RE.match(text) or PLAIN_DASH_BULLET_RE.match(text) or LETTER_ITEM_RE.match(text):
        return True
    if SEMICOLON_TABLE_RE.search(text) and text.count(";") >= 2:
        return True
    if text.casefold().startswith(("celkem", "; celkem")):
        return True
    return False


def _is_genuine_nested_marker(text: str) -> bool:
    if _is_nested_or_table(text):
        return True
    match = ARABIC_NUMBER_RE.match(text) or re.match(
        r"^\s*(?:I{1,3}|IV|V|VI{0,3}|IX|X)[.)]\s+",
        text,
        re.IGNORECASE,
    )
    if not match:
        return False
    rest = text[match.end() :].strip()
    if not rest:
        return False
    # Date-like continuation such as "1. července 2014..." remains a paragraph continuation.
    if _MONTH_CONTINUATION_RE.match(rest):
        return False
    return True


def _boundary_before(line_number: int, block_for_line: list[Any | None]) -> bool:
    if line_number <= 1:
        return True
    return block_for_line[line_number - 1] is not block_for_line[line_number - 2]


def _boundary_after(line_number: int, lines: list[str], block_for_line: list[Any | None]) -> bool:
    if line_number >= len(lines):
        return True
    return block_for_line[line_number - 1] is not block_for_line[line_number]


def _line_reason(raw_line: str, paragraph: Any | None, line_number: int, block_for_line: list[Any | None]) -> str:
    if paragraph is None:
        return "parser_unmapped_line"
    if paragraph.numbering and _boundary_before(line_number, block_for_line):
        return "parser_numbered_paragraph_start"
    if paragraph.numbering:
        return "parser_numbered_paragraph_continuation"
    if CASE_REF_RE.search(raw_line):
        return "parser_case_reference"
    return f"parser_section_{paragraph.section_type.value}"


def _line_suspicious(raw_line: str, paragraph: Any | None, previous: dict[str, Any] | None) -> list[str]:
    reasons: list[str] = []
    if paragraph is None:
        reasons.append("parser_unmapped_line")
    if CASE_REF_RE.match(raw_line.strip()) and (paragraph is None or not paragraph.numbering):
        reasons.append("orphan_case_reference_candidate")
    if previous is None:
        reasons.append("missing_previous_annotation")
    return reasons


def _boundary_records(
    document: ReviewDocument,
    line_records: list[dict[str, Any]],
    prev_boundaries: dict[tuple[str, int, int], dict[str, Any]],
    block_for_line: list[Any | None],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for previous, following in zip(line_records, line_records[1:], strict=False):
        left = int(previous["raw_line_number"])
        right = int(following["raw_line_number"])
        parser_boundary = block_for_line[left - 1] is not block_for_line[right - 1]
        previous_annotation = prev_boundaries.get((document.source_id, left, right))
        boundary_id = boundary_review_id(
            document_id=document.review_id,
            previous_line_id=str(previous["item_id"]),
            next_line_id=str(following["item_id"]),
            source_checksum=document.source_checksum,
        )
        suspicious: list[str] = []
        if previous_annotation and bool(previous_annotation.get("boundary")) != parser_boundary:
            suspicious.append("parser_boundary_disagrees_previous_annotation")
        records.append(
            {
                "schema_version": REVIEW_SCHEMA_VERSION,
                "item_type": "boundary",
                "item_id": boundary_id,
                "document_id": document.review_id,
                "source_document_id": document.source_id,
                "source_checksum": document.source_checksum,
                "previous_line_id": previous["item_id"],
                "next_line_id": following["item_id"],
                "previous_line_number": left,
                "next_line_number": right,
                "parser_proposed_boundary": parser_boundary,
                "parser_proposed_boundary_type": "parser_block_boundary" if parser_boundary else "parser_same_block",
                "previous_automated_boundary_annotation": previous_annotation.get("boundary") if previous_annotation else None,
                "parser_reason_code": "parser_block_change" if parser_boundary else "parser_same_block",
                "suspicious_reason_codes": suspicious,
                "manual_decision_status": "pending",
            }
        )
    return records


def _document_record(document: ReviewDocument, raw_lines: list[str], parsed: Any) -> dict[str, Any]:
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "document_id": document.review_id,
        "review_number": document.review_number,
        "source_id": document.source_id,
        "court": document.court,
        "source_checksum": document.source_checksum,
        "normalized_content_checksum": document.normalized_content_checksum,
        "source_format": document.source_format,
        "raw_path": str(document.raw_path.relative_to(PROJECT_ROOT)),
        "source_url": document.source_url,
        "case_number": document.case_number,
        "decision_date": document.decision_date,
        "document_type": document.document_type,
        "raw_line_count": len(raw_lines),
        "parser_block_count": len(parsed.paragraphs),
        "parser_heading_count": parsed.diagnostics.heading_count,
        "parser_numbered_paragraph_count": parsed.diagnostics.numbered_paragraph_count,
    }


def _write_document_files(
    review_dir: Path,
    document: ReviewDocument,
    raw_lines: list[str],
    parsed: Any,
    line_records: list[dict[str, Any]],
    all_boundary_records: list[dict[str, Any]],
) -> None:
    doc_dir = review_dir / "documents" / document.review_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "raw_numbered.txt").write_text(
        "\n".join(f"{index:05d}: {line}" for index, line in enumerate(raw_lines, start=1)) + "\n",
        encoding="utf-8",
    )
    (doc_dir / "parser_blocks.txt").write_text(
        "\n\n".join(
            f"[{paragraph.paragraph_index:05d}] section={paragraph.section_type.value} numbering={paragraph.numbering or ''}\n{paragraph.original_text}"
            for paragraph in parsed.paragraphs
        )
        + "\n",
        encoding="utf-8",
    )
    write_jsonl(
        doc_dir / "parser_trace.jsonl",
        (
            {
                "line": row["raw_line_number"],
                "item_id": row["item_id"],
                "parser_block_id": row["parser_block_id"],
                "parser_class": row["parser_proposed_line_class"],
                "reason": row["parser_reason_code"],
                "suspicious": row["suspicious_reason_codes"],
            }
            for row in line_records
        ),
    )
    boundaries = [row for row in all_boundary_records if row["document_id"] == document.review_id]
    (doc_dir / "boundary_table.tsv").write_text(
        "boundary_id\tprevious_line\tnext_line\tparser_boundary\tprevious_annotation\treason\n"
        + "".join(
            f"{row['item_id']}\t{row['previous_line_number']}\t{row['next_line_number']}\t{row['parser_proposed_boundary']}\t{row['previous_automated_boundary_annotation']}\t{row['parser_reason_code']}\n"
            for row in boundaries
        ),
        encoding="utf-8",
    )
    write_json(doc_dir / "document_summary.json", _document_record(document, raw_lines, parsed))


def _ensure_manual_store(review_dir: Path) -> None:
    for name in ("manual_review_decisions.jsonl", "manual_review_history.jsonl"):
        path = review_dir / name
        if not path.exists():
            path.write_text("", encoding="utf-8")


def _assert_text_conserved(source: str, reconstructed: str, source_id: str) -> None:
    if re.sub(r"\s+", "", source) != re.sub(r"\s+", "", reconstructed):
        raise ValueError(f"Parser text conservation failed for {source_id}")


def _normalize(value: str) -> str:
    return " ".join(value.split()).strip()
