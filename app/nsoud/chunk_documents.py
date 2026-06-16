from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("pandas is required for this script.") from exc

try:
    import pyarrow  # noqa: F401
except ImportError:
    pyarrow = None

from app.nsoud.structure.confidence import calculate_structure_confidence
from app.nsoud.structure.section_detector import detect_ns_document_structure


METADATA_FIELDS = [
    "source",
    "court",
    "authority_level",
    "case_number",
    "ecli",
    "decision_date",
    "publication_date",
    "document_type",
    "legal_area",
    "title",
    "url",
    "source_attribution",
    "content_hash",
]
FIXED_VALUES = {
    "source": "nsoud",
    "court": "Nejvyšší soud",
    "authority_level": "supreme",
}
TARGET_CHUNK_SIZE = 1800
SOFT_MAX_CHUNK_SIZE = 2500
HARD_MAX_CHUNK_SIZE = 4000
CASE_NUMBER_START_RE = re.compile(r"^\s*\d{1,3}\s+[A-Z][A-Za-z]{1,6}\s+\d+/\d{4}")
BLANK_LINE_SEPARATOR_RE = re.compile(r"(?:\r\n|\r|\n)[ \t]*(?:\r\n|\r|\n)+")
SECTION_LABEL_PATTERNS = {
    "takto:": re.compile(r"takto:", re.IGNORECASE),
    "Odůvodnění:": re.compile(r"Odůvodnění:"),
    "Poučení:": re.compile(r"Poučení:"),
    "P o u č e n í:": re.compile(r"P\s*o\s*u\s*[čc]\s*e\s*n\s*[íi]\s*:", re.IGNORECASE),
    "V Brně dne": re.compile(r"V Brně dne"),
}
NUMBERED_DOT_RE = re.compile(r"\d{1,3}\.")
NUMBERED_SLASH_RE = re.compile(r"\d{1,3}/")
BRACKETED_NUMBER_RE = re.compile(r"\[\d{1,3}\]")
NUMBERED_PAREN_RE = re.compile(r"\d{1,3}\)")
ROMAN_SECTION_RE = re.compile(r"(?:XX|XIX|XVIII|XVII|XVI|XV|XIV|XIII|XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\.")
ALLOWED_BOUNDARY_PREVIOUS_CHARS = ".:;!?)]}\"'"
ALLOWED_PARAGRAPH_NEXT_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ0123456789§„\"'([{")
DISALLOWED_PRECEDING_SUFFIXES = (
    "odst.",
    "písm.",
    "sp. zn.",
    "č. j.",
    "čl.",
    "bodem",
    "bodu",
    "bodech",
    "body",
)
SECTION_TYPE_MAP = {
    "header": "header",
    "vyrok": "operative_part",
    "oduvodneni": "reasoning",
    "pouceni": "appeal_instruction",
    "closing/signature": "signature",
}
SECTION_HINT_MAP = {
    "header": "header",
    "vyrok": "vyrok",
    "oduvodneni": "oduvodneni",
    "pouceni": "pouceni",
    "closing/signature": "closing",
}


@dataclass(frozen=True)
class ParagraphSpan:
    start: int
    end: int
    text: str
    start_pattern: str


@dataclass(frozen=True)
class StructureAnalysisSummary:
    total_documents: int
    marker_counts: dict[str, int]
    section_order_counts: dict[str, int]
    strong_count: int
    medium_count: int
    weak_count: int
    needs_review_count: int
    report_path: Path


@dataclass(frozen=True)
class ChunkingSummary:
    structure_analysis_status: str
    chunking_status: str
    validation_status: str
    total_documents: int
    total_chunks: int
    documents_with_zero_chunks: int
    empty_chunk_count: int
    duplicate_chunk_id_count: int
    paragraph_preservation_passed: int
    paragraph_preservation_failed: int
    reconstruction_validation_passed: int
    reconstruction_validation_failed: int
    section_reconstruction_passed: int
    section_reconstruction_failed: int
    unresolved_boundary_issue_count: int
    strong_structure_count: int
    medium_structure_count: int
    weak_structure_count: int
    needs_review_count: int
    output_parquet_path: Path
    output_jsonl_path: Path
    validation_report_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk NSoud Parquet documents into deterministic NS structural chunks.")
    parser.add_argument("--input", type=Path, required=True, help="Input Parquet path with NSoud documents.")
    parser.add_argument("--out", type=Path, required=True, help="Output Parquet path for chunked records.")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value)


def normalize_whitespace(text: str) -> str:
    return " ".join(normalize_text(text).split())


def validation_path_for_output(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}_validation.md")


def jsonl_path_for_output(out_path: Path) -> Path:
    return out_path.with_suffix(".jsonl")


def structure_report_path_for_input(input_path: Path) -> Path:
    stem = input_path.stem
    prefix = "nsoud_documents_"
    suffix = stem[len(prefix) :] if stem.startswith(prefix) else stem
    return input_path.parent / f"nsoud_structure_patterns_{suffix}.md"


def load_documents(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def is_official_nsoud_url(value: Any) -> bool:
    url = normalize_text(value).strip().lower()
    return url.startswith("http://nsoud.cz") or url.startswith("https://nsoud.cz") or ".nsoud.cz/" in url


def compute_document_id(record: dict[str, Any]) -> str:
    ecli = normalize_text(record.get("ecli")).strip()
    if ecli:
        return ecli
    return normalize_text(record.get("content_hash")).strip()


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    left = start
    right = end
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    return left, right


def previous_non_space_index(text: str, start: int) -> int:
    position = start
    while position >= 0 and text[position].isspace():
        position -= 1
    return position


def next_non_space_index(text: str, start: int) -> int:
    position = start
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def previous_context(text: str, start: int, width: int = 24) -> str:
    return text[max(0, start - width) : start].lower().rstrip()


def has_valid_boundary_prefix(text: str, marker_start: int) -> bool:
    if marker_start == 0:
        return True
    if not text[marker_start - 1].isspace():
        return False

    previous_index = previous_non_space_index(text, marker_start - 1)
    if previous_index < 0:
        return True

    previous_char = text[previous_index]
    if previous_char not in ALLOWED_BOUNDARY_PREVIOUS_CHARS:
        return False

    context = previous_context(text, marker_start)
    return not any(context.endswith(suffix) for suffix in DISALLOWED_PRECEDING_SUFFIXES)


def starts_with_date_like_sequence(text: str, start: int) -> bool:
    return bool(re.match(r"\d{1,2}\.\s+\d{1,2}\.\s+\d{2,4}", text[start:]))


def starts_with_uppercase_citation(text: str, start: int) -> bool:
    return bool(re.match(r"[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]{1,4}\s+\d+/\d{4}", text[start:]))


def is_probable_numbered_paragraph_boundary(text: str, marker_start: int, marker_end: int, marker_text: str) -> bool:
    if not has_valid_boundary_prefix(text, marker_start):
        return False

    marker_value = marker_text.rstrip("./")
    if not marker_value.isdigit():
        return False
    if not 1 <= int(marker_value) <= 200:
        return False

    next_index = next_non_space_index(text, marker_end)
    if next_index >= len(text):
        return False

    next_char = text[next_index]
    if next_char not in ALLOWED_PARAGRAPH_NEXT_CHARS:
        return False

    if next_char.isdigit() and starts_with_date_like_sequence(text, next_index):
        return False

    return True


def is_probable_bracketed_paragraph_boundary(text: str, marker_start: int, marker_end: int) -> bool:
    if not has_valid_boundary_prefix(text, marker_start):
        return False

    next_index = next_non_space_index(text, marker_end)
    if next_index >= len(text):
        return False

    next_char = text[next_index]
    return next_char in ALLOWED_PARAGRAPH_NEXT_CHARS


def is_probable_parenthesized_paragraph_boundary(text: str, marker_start: int, marker_end: int, marker_text: str) -> bool:
    if not has_valid_boundary_prefix(text, marker_start):
        return False

    marker_value = marker_text.rstrip(")")
    if not marker_value.isdigit():
        return False
    if not 1 <= int(marker_value) <= 200:
        return False

    next_index = next_non_space_index(text, marker_end)
    if next_index >= len(text):
        return False

    next_char = text[next_index]
    return next_char in ALLOWED_PARAGRAPH_NEXT_CHARS and not next_char.isdigit()


def is_probable_roman_section_boundary(text: str, marker_start: int, marker_end: int) -> bool:
    if not has_valid_boundary_prefix(text, marker_start):
        return False

    next_index = next_non_space_index(text, marker_end)
    if next_index >= len(text):
        return False

    next_char = text[next_index]
    if next_char not in ALLOWED_PARAGRAPH_NEXT_CHARS or next_char.isdigit():
        return False

    if starts_with_uppercase_citation(text, next_index):
        return False

    return True


def is_probable_section_label_boundary(text: str, marker_start: int) -> bool:
    return has_valid_boundary_prefix(text, marker_start)


def detect_ns_structural_boundaries(text: str) -> list[tuple[int, str]]:
    boundaries: dict[int, str] = {0: "header"}

    for label, pattern in SECTION_LABEL_PATTERNS.items():
        for match in pattern.finditer(text):
            marker_start = match.start()
            if marker_start > 0 and marker_start not in boundaries and is_probable_section_label_boundary(text, marker_start):
                boundaries[marker_start] = match.group(0)

    for match in ROMAN_SECTION_RE.finditer(text):
        marker_start = match.start()
        marker_end = match.end()
        marker_text = match.group(0)
        if marker_start > 0 and marker_start not in boundaries and is_probable_roman_section_boundary(text, marker_start, marker_end):
            boundaries[marker_start] = marker_text

    for regex in (NUMBERED_DOT_RE, NUMBERED_SLASH_RE):
        for match in regex.finditer(text):
            marker_start = match.start()
            marker_end = match.end()
            marker_text = match.group(0)
            if marker_start > 0 and marker_start not in boundaries and is_probable_numbered_paragraph_boundary(
                text,
                marker_start,
                marker_end,
                marker_text,
            ):
                boundaries[marker_start] = marker_text

    for match in NUMBERED_PAREN_RE.finditer(text):
        marker_start = match.start()
        marker_end = match.end()
        marker_text = match.group(0)
        if marker_start > 0 and marker_start not in boundaries and is_probable_parenthesized_paragraph_boundary(
            text,
            marker_start,
            marker_end,
            marker_text,
        ):
            boundaries[marker_start] = marker_text

    for match in BRACKETED_NUMBER_RE.finditer(text):
        marker_start = match.start()
        marker_end = match.end()
        if marker_start > 0 and marker_start not in boundaries and is_probable_bracketed_paragraph_boundary(
            text,
            marker_start,
            marker_end,
        ):
            boundaries[marker_start] = match.group(0)

    return sorted(boundaries.items(), key=lambda item: item[0])


def split_on_blank_lines(text: str) -> list[ParagraphSpan]:
    separators = list(BLANK_LINE_SEPARATOR_RE.finditer(text))
    if not separators:
        return []

    spans: list[ParagraphSpan] = []
    cursor = 0
    for separator in separators:
        start, end = trim_span(text, cursor, separator.start())
        if start < end:
            spans.append(ParagraphSpan(start=start, end=end, text=text[start:end], start_pattern="blank_line_block"))
        cursor = separator.end()

    start, end = trim_span(text, cursor, len(text))
    if start < end:
        spans.append(ParagraphSpan(start=start, end=end, text=text[start:end], start_pattern="blank_line_block"))
    return spans


def split_text_by_ns_boundaries(text: str) -> list[ParagraphSpan]:
    boundaries = detect_ns_structural_boundaries(text)
    spans: list[ParagraphSpan] = []
    for index, (boundary_start, pattern_name) in enumerate(boundaries):
        next_start = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(text)
        start, end = trim_span(text, boundary_start, next_start)
        if start < end:
            spans.append(ParagraphSpan(start=start, end=end, text=text[start:end], start_pattern=pattern_name))
    return spans


def extract_ns_paragraphs(text: str) -> list[ParagraphSpan]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    blank_line_spans = split_on_blank_lines(normalized)
    if blank_line_spans:
        return blank_line_spans
    return split_text_by_ns_boundaries(normalized)


def deduplicate_messages(messages: list[str]) -> list[str]:
    deduplicated: list[str] = []
    seen: set[str] = set()
    for message in messages:
        if message in seen:
            continue
        seen.add(message)
        deduplicated.append(message)
    return deduplicated


def sanitize_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower() or "section"


def section_type_from_raw(raw_section: str) -> str:
    return SECTION_TYPE_MAP.get(raw_section, "unknown")


def ns_section_hint_from_raw(raw_section: str) -> str:
    return SECTION_HINT_MAP.get(raw_section, "unknown")


def should_append_unit(current_start: int, current_end: int, next_unit: dict[str, Any]) -> bool:
    current_length = current_end - current_start
    prospective_length = int(next_unit["end"]) - current_start
    if prospective_length <= TARGET_CHUNK_SIZE:
        return True
    if current_length < TARGET_CHUNK_SIZE and prospective_length <= SOFT_MAX_CHUNK_SIZE:
        return True
    return False


def build_exact_structural_spans(text: str, *, absolute_offset: int = 0) -> list[dict[str, Any]]:
    boundaries = detect_ns_structural_boundaries(text)
    spans: list[dict[str, Any]] = []
    for index, (boundary_start, pattern_name) in enumerate(boundaries):
        next_start = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(text)
        if boundary_start >= next_start:
            continue
        spans.append(
            {
                "start": absolute_offset + boundary_start,
                "end": absolute_offset + next_start,
                "start_pattern": pattern_name,
            }
        )
    if not spans and text:
        spans.append({"start": absolute_offset, "end": absolute_offset + len(text), "start_pattern": "header"})
    return spans


def build_section_spans(full_text: str, structure: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = list(structure.get("section_candidates") or [{"section": "header", "position": 0}])
    if not candidates:
        candidates = [{"section": "header", "position": 0}]

    spans: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        start = int(candidate["position"])
        end = int(candidates[index + 1]["position"]) if index + 1 < len(candidates) else len(full_text)
        if start >= end:
            continue
        raw_section = str(candidate["section"])
        spans.append(
            {
                "section_raw": raw_section,
                "section_type": section_type_from_raw(raw_section),
                "ns_section_hint": ns_section_hint_from_raw(raw_section),
                "start": start,
                "end": end,
            }
        )
    return spans


def build_document_chunks(record: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    full_text = normalize_text(record.get("full_text"))
    metadata = {
        "case_number": normalize_text(record.get("case_number")),
        "ecli": normalize_text(record.get("ecli")),
        "document_type": normalize_text(record.get("document_type")),
        "legal_area": normalize_text(record.get("legal_area")),
    }
    structure = detect_ns_document_structure(full_text=full_text, metadata=metadata)
    confidence = calculate_structure_confidence(structure)
    document_id = compute_document_id(record)

    section_spans = build_section_spans(full_text, structure)
    section_chunks: list[list[dict[str, Any]]] = []

    for section_index, section in enumerate(section_spans):
        units = build_exact_structural_spans(
            full_text[section["start"] : section["end"]],
            absolute_offset=int(section["start"]),
        )
        if not units:
            units = [
                {
                    "start": int(section["start"]),
                    "end": int(section["end"]),
                    "start_pattern": section["ns_section_hint"],
                }
            ]

        raw_section_chunks: list[dict[str, Any]] = []
        current_units: list[dict[str, Any]] = []
        current_start = 0
        current_end = 0

        def finalize_current_chunk() -> None:
            nonlocal current_units, current_start, current_end
            if not current_units:
                return

            chunk_start = int(current_units[0]["start"])
            chunk_end = int(current_units[-1]["end"])
            chunk_text = full_text[chunk_start:chunk_end]
            chunk_warning = ""
            if len(current_units) == 1 and len(chunk_text) > HARD_MAX_CHUNK_SIZE:
                chunk_warning = "overlong_ns_paragraph"

            raw_section_chunks.append(
                {
                    "chunk_text": chunk_text,
                    "chunk_char_start": chunk_start,
                    "chunk_char_end": chunk_end,
                    "chunk_text_length": len(chunk_text),
                    "paragraph_count": len(current_units),
                    "chunk_warning": chunk_warning,
                    "section_char_start": int(section["start"]),
                    "section_char_end": int(section["end"]),
                    "section_type": section["section_type"],
                    "section_raw": section["section_raw"],
                    "ns_section_hint": section["ns_section_hint"],
                }
            )
            current_units = []
            current_start = 0
            current_end = 0

        for unit in units:
            unit_length = int(unit["end"]) - int(unit["start"])
            if unit_length > HARD_MAX_CHUNK_SIZE:
                finalize_current_chunk()
                current_units = [unit]
                current_start = int(unit["start"])
                current_end = int(unit["end"])
                finalize_current_chunk()
                continue

            if not current_units:
                current_units = [unit]
                current_start = int(unit["start"])
                current_end = int(unit["end"])
                continue

            if should_append_unit(current_start, current_end, unit):
                current_units.append(unit)
                current_end = int(unit["end"])
                continue

            finalize_current_chunk()
            current_units = [unit]
            current_start = int(unit["start"])
            current_end = int(unit["end"])

        finalize_current_chunk()

        section_id = f"{document_id}__section_{section_index:02d}_{sanitize_label(section['ns_section_hint'])}"
        total_chunks_in_section = len(raw_section_chunks)
        width_in_section = max(2, len(str(max(0, total_chunks_in_section - 1))))
        for chunk_index_in_section, chunk in enumerate(raw_section_chunks):
            chunk["section_id"] = section_id
            chunk["section_index"] = section_index
            chunk["chunk_index_in_section"] = chunk_index_in_section
            chunk["total_chunks_in_section"] = total_chunks_in_section
            chunk["previous_section_chunk_id"] = None
            chunk["next_section_chunk_id"] = None
            chunk["section_width"] = width_in_section
        section_chunks.append(raw_section_chunks)

    flat_chunks = [chunk for chunk_group in section_chunks for chunk in chunk_group]
    total_chunks_in_document = len(flat_chunks)
    width = max(4, len(str(max(0, total_chunks_in_document - 1))))
    detected_markers_json = json.dumps(structure["detected_markers"], ensure_ascii=False)
    detected_section_order_json = json.dumps(structure["detected_section_order"], ensure_ascii=False)

    for chunk_index, chunk in enumerate(flat_chunks):
        chunk_id = f"{document_id}__chunk_{chunk_index:0{width}d}"
        chunk["document_id"] = document_id
        chunk["chunk_id"] = chunk_id
        chunk["chunk_index"] = chunk_index
        chunk["total_chunks_in_document"] = total_chunks_in_document
        chunk["previous_chunk_id"] = None
        chunk["next_chunk_id"] = None
        chunk["structure_confidence"] = float(confidence["structure_confidence"])
        chunk["structure_status"] = str(confidence["structure_status"])
        chunk["structure_needs_review"] = bool(confidence["needs_review"])
        chunk["detected_markers"] = detected_markers_json
        chunk["detected_section_order"] = detected_section_order_json
        chunk["section_source"] = "nsoud.structure"
        chunk["chunking_strategy"] = "document_section_aware"

    for chunk_index, chunk in enumerate(flat_chunks):
        if chunk_index > 0:
            chunk["previous_chunk_id"] = flat_chunks[chunk_index - 1]["chunk_id"]
        if chunk_index + 1 < len(flat_chunks):
            chunk["next_chunk_id"] = flat_chunks[chunk_index + 1]["chunk_id"]

    for chunk_group in section_chunks:
        for index, chunk in enumerate(chunk_group):
            if index > 0:
                chunk["previous_section_chunk_id"] = chunk_group[index - 1]["chunk_id"]
            if index + 1 < len(chunk_group):
                chunk["next_section_chunk_id"] = chunk_group[index + 1]["chunk_id"]

    return flat_chunks, structure, confidence


def build_chunk_records(documents_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    chunk_rows: list[dict[str, Any]] = []
    document_analyses: list[dict[str, Any]] = []

    for _, row in documents_df.iterrows():
        record = row.to_dict()
        document_id = compute_document_id(record)
        full_text = normalize_text(record.get("full_text"))
        produced_chunks, structure, confidence = build_document_chunks(record)

        document_analyses.append(
            {
                "document_id": document_id,
                "case_number": normalize_text(record.get("case_number")),
                "full_text": full_text,
                "structure": structure,
                "confidence": confidence,
            }
        )

        if full_text and not produced_chunks:
            failures.append(f"Document `{document_id}` has non-empty full_text but produced zero chunks.")
        if confidence["needs_review"]:
            failures.append(f"Document `{document_id}` has weak structure confidence and needs review.")

        for chunk in produced_chunks:
            row_payload = {field: record.get(field) for field in METADATA_FIELDS}
            row_payload.update(chunk)
            chunk_rows.append(row_payload)

    ordered_columns = METADATA_FIELDS + [
        "document_id",
        "chunk_id",
        "chunk_index",
        "total_chunks_in_document",
        "section_id",
        "section_type",
        "section_index",
        "chunk_index_in_section",
        "total_chunks_in_section",
        "previous_chunk_id",
        "next_chunk_id",
        "previous_section_chunk_id",
        "next_section_chunk_id",
        "chunk_text",
        "chunk_text_length",
        "chunk_char_start",
        "chunk_char_end",
        "section_char_start",
        "section_char_end",
        "paragraph_count",
        "chunk_warning",
        "ns_section_hint",
        "structure_confidence",
        "structure_status",
        "structure_needs_review",
        "detected_section_order",
        "detected_markers",
        "section_source",
        "chunking_strategy",
    ]
    chunk_df = pd.DataFrame(chunk_rows)
    if chunk_df.empty:
        chunk_df = pd.DataFrame(columns=ordered_columns)
    else:
        chunk_df = chunk_df.reindex(columns=ordered_columns)

    overlong_count = int((chunk_df["chunk_warning"] == "overlong_ns_paragraph").sum()) if not chunk_df.empty else 0
    if overlong_count > 0:
        warnings.append("Overlong NS paragraphs were preserved as standalone chunks.")

    return chunk_df, document_analyses, failures, warnings


def distribution_counts(df: pd.DataFrame, column_name: str) -> dict[str, int]:
    if df.empty:
        return {}
    series = df[column_name].fillna("").map(lambda value: str(value).strip() or "<missing>")
    counts = series.value_counts(dropna=False).to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def render_distribution_table(title: str, counts: dict[str, int]) -> list[str]:
    lines = [f"## {title}", "", "| Value | Count |", "| --- | ---: |"]
    if not counts:
        lines.append("| - | 0 |")
    else:
        for value, count in counts.items():
            lines.append(f"| {value} | {count} |")
    lines.append("")
    return lines


def analyze_structure(
    documents_df: pd.DataFrame,
    document_analyses: list[dict[str, Any]],
    report_path: Path,
) -> StructureAnalysisSummary:
    marker_counts = {label: 0 for label in detect_ns_document_structure(full_text="", metadata={}).get("detected_markers", {}).keys()}
    section_order_counts: Counter[str] = Counter()
    strong_count = 0
    medium_count = 0
    weak_count = 0
    needs_review_count = 0

    for analysis in document_analyses:
        structure = analysis["structure"]
        confidence = analysis["confidence"]
        for label, marker_data in structure["detected_markers"].items():
            if marker_data["present"]:
                marker_counts[label] = marker_counts.get(label, 0) + 1
        section_order_counts[" > ".join(structure["detected_section_order"]["observed_sections"])] += 1

        status = confidence["structure_status"]
        if status == "strong":
            strong_count += 1
        elif status == "medium":
            medium_count += 1
        else:
            weak_count += 1
        if confidence["needs_review"]:
            needs_review_count += 1

    lines = [
        "# NSoud Structure Pattern Analysis",
        "",
        f"- Total documents: **{len(documents_df)}**",
        f"- Strong structure count: **{strong_count}**",
        f"- Medium structure count: **{medium_count}**",
        f"- Weak structure count: **{weak_count}**",
        f"- Needs review count: **{needs_review_count}**",
        "",
        "## Marker Coverage",
        "",
        "| Marker | Document Count |",
        "| --- | ---: |",
    ]
    for label, count in marker_counts.items():
        lines.append(f"| {label} | {count} |")

    lines.extend(
        [
            "",
            "## Detected Section Order",
            "",
            "| Order | Document Count |",
            "| --- | ---: |",
        ]
    )
    for order, count in section_order_counts.most_common(15):
        lines.append(f"| {order} | {count} |")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return StructureAnalysisSummary(
        total_documents=len(documents_df),
        marker_counts=marker_counts,
        section_order_counts=dict(section_order_counts),
        strong_count=strong_count,
        medium_count=medium_count,
        weak_count=weak_count,
        needs_review_count=needs_review_count,
        report_path=report_path,
    )


def normalize_paragraph_for_compare(paragraph_text: str) -> str:
    return paragraph_text.strip()


def validate_paragraph_preservation(
    documents_df: pd.DataFrame,
    chunk_df: pd.DataFrame,
) -> tuple[list[str], int, int]:
    failures: list[str] = []
    passed_documents = 0
    failed_documents = 0
    grouped_chunks = {
        str(document_id): group.sort_values("chunk_index")
        for document_id, group in chunk_df.groupby("document_id", sort=False)
    }

    for _, row in documents_df.iterrows():
        record = row.to_dict()
        document_id = compute_document_id(record)
        full_text = normalize_text(record.get("full_text"))
        if not full_text:
            passed_documents += 1
            continue

        chunk_group = grouped_chunks.get(document_id)
        if chunk_group is None or chunk_group.empty:
            failed_documents += 1
            failures.append(f"Paragraph preservation failed for document `{document_id}`.")
            continue

        structure = detect_ns_document_structure(
            full_text=full_text,
            metadata={
                "case_number": normalize_text(record.get("case_number")),
                "ecli": normalize_text(record.get("ecli")),
                "document_type": normalize_text(record.get("document_type")),
                "legal_area": normalize_text(record.get("legal_area")),
            },
        )
        section_spans = build_section_spans(full_text, structure)
        structural_units: list[tuple[int, int]] = []
        for section in section_spans:
            structural_units.extend(
                (int(unit["start"]), int(unit["end"]))
                for unit in build_exact_structural_spans(
                    full_text[int(section["start"]) : int(section["end"])],
                    absolute_offset=int(section["start"]),
                )
            )

        allowed_starts = {start for start, _ in structural_units}
        allowed_ends = {end for _, end in structural_units}
        boundary_violation = False
        for _, chunk_row in chunk_group.iterrows():
            chunk_start = int(chunk_row["chunk_char_start"])
            chunk_end = int(chunk_row["chunk_char_end"])
            if chunk_start not in allowed_starts or chunk_end not in allowed_ends:
                boundary_violation = True
                break

        if boundary_violation:
            failed_documents += 1
            failures.append(f"Paragraph preservation failed for document `{document_id}`.")
            continue

        passed_documents += 1

    return failures, passed_documents, failed_documents


def validate_document_reconstruction(
    documents_df: pd.DataFrame,
    chunk_df: pd.DataFrame,
) -> tuple[list[str], int, int]:
    failures: list[str] = []
    passed_documents = 0
    failed_documents = 0
    grouped_chunks = {
        str(document_id): group.sort_values("chunk_index")
        for document_id, group in chunk_df.groupby("document_id", sort=False)
    }

    for _, row in documents_df.iterrows():
        record = row.to_dict()
        document_id = compute_document_id(record)
        full_text = normalize_text(record.get("full_text"))
        if not full_text:
            passed_documents += 1
            continue

        chunk_group = grouped_chunks.get(document_id)
        if chunk_group is None or chunk_group.empty:
            failed_documents += 1
            failures.append(f"Document reconstruction failed for `{document_id}` because no chunks were found.")
            continue

        reconstructed = "".join(normalize_text(value) for value in chunk_group["chunk_text"].tolist())
        if normalize_whitespace(full_text) != normalize_whitespace(reconstructed):
            failed_documents += 1
            failures.append(f"Document reconstruction failed for `{document_id}`.")
            continue

        expected_start = 0
        contiguous = True
        for _, chunk_row in chunk_group.iterrows():
            start = int(chunk_row["chunk_char_start"])
            end = int(chunk_row["chunk_char_end"])
            if start != expected_start:
                contiguous = False
                break
            expected_start = end
        if not contiguous or expected_start != len(full_text):
            failed_documents += 1
            failures.append(f"Document span continuity failed for `{document_id}`.")
            continue

        passed_documents += 1

    return failures, passed_documents, failed_documents


def validate_section_reconstruction(
    documents_df: pd.DataFrame,
    chunk_df: pd.DataFrame,
) -> tuple[list[str], int, int]:
    failures: list[str] = []
    passed_sections = 0
    failed_sections = 0
    text_by_document = {
        compute_document_id(row.to_dict()): normalize_text(row.to_dict().get("full_text"))
        for _, row in documents_df.iterrows()
    }

    if chunk_df.empty:
        return failures, 0, 0

    for (_, _), group in chunk_df.groupby(["document_id", "section_id"], sort=False):
        sorted_group = group.sort_values("chunk_index_in_section")
        first_row = sorted_group.iloc[0]
        document_id = normalize_text(first_row["document_id"])
        full_text = text_by_document.get(document_id, "")
        section_start = int(first_row["section_char_start"])
        section_end = int(first_row["section_char_end"])
        original_section_text = full_text[section_start:section_end]
        reconstructed = "".join(normalize_text(value) for value in sorted_group["chunk_text"].tolist())

        if normalize_whitespace(original_section_text) != normalize_whitespace(reconstructed):
            failed_sections += 1
            failures.append(
                f"Section reconstruction failed for `{normalize_text(first_row['section_id'])}` in document `{document_id}`."
            )
            continue

        expected_start = section_start
        contiguous = True
        for _, chunk_row in sorted_group.iterrows():
            start = int(chunk_row["chunk_char_start"])
            end = int(chunk_row["chunk_char_end"])
            if start != expected_start:
                contiguous = False
                break
            expected_start = end
        if not contiguous or expected_start != section_end:
            failed_sections += 1
            failures.append(
                f"Section span continuity failed for `{normalize_text(first_row['section_id'])}` in document `{document_id}`."
            )
            continue

        passed_sections += 1

    return failures, passed_sections, failed_sections


def validate_chunk_metadata(chunk_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    if chunk_df.empty:
        return failures, warnings

    required_nonempty_fields = [
        "document_id",
        "chunk_id",
        "case_number",
        "document_type",
        "section_id",
        "section_type",
        "chunk_text",
        "section_source",
        "chunking_strategy",
    ]
    for field_name in required_nonempty_fields:
        if int(chunk_df[field_name].map(lambda value: normalize_text(value).strip() == "").sum()) > 0:
            failures.append(f"Required field `{field_name}` contains empty values.")

    for document_id, group in chunk_df.groupby("document_id", sort=False):
        sorted_group = group.sort_values("chunk_index").reset_index(drop=True)
        expected_sequence = list(range(len(sorted_group)))
        actual_sequence = sorted_group["chunk_index"].astype(int).tolist()
        if actual_sequence != expected_sequence:
            failures.append(f"Document `{document_id}` has a non-continuous chunk_index sequence.")

        total_chunks_values = sorted_group["total_chunks_in_document"].astype(int).unique().tolist()
        if total_chunks_values != [len(sorted_group)]:
            failures.append(f"Document `{document_id}` has inconsistent total_chunks_in_document metadata.")

        for index, row in sorted_group.iterrows():
            expected_previous = sorted_group.iloc[index - 1]["chunk_id"] if index > 0 else None
            expected_next = sorted_group.iloc[index + 1]["chunk_id"] if index + 1 < len(sorted_group) else None
            actual_previous = row["previous_chunk_id"] if pd.notna(row["previous_chunk_id"]) else None
            actual_next = row["next_chunk_id"] if pd.notna(row["next_chunk_id"]) else None
            if expected_previous != actual_previous:
                failures.append(f"Document `{document_id}` has invalid previous_chunk_id metadata.")
                break
            if expected_next != actual_next:
                failures.append(f"Document `{document_id}` has invalid next_chunk_id metadata.")
                break

    for (document_id, section_id), group in chunk_df.groupby(["document_id", "section_id"], sort=False):
        sorted_group = group.sort_values("chunk_index_in_section").reset_index(drop=True)
        expected_sequence = list(range(len(sorted_group)))
        actual_sequence = sorted_group["chunk_index_in_section"].astype(int).tolist()
        if actual_sequence != expected_sequence:
            failures.append(f"Section `{section_id}` in document `{document_id}` has a non-continuous chunk_index_in_section sequence.")

        total_chunks_values = sorted_group["total_chunks_in_section"].astype(int).unique().tolist()
        if total_chunks_values != [len(sorted_group)]:
            failures.append(f"Section `{section_id}` in document `{document_id}` has inconsistent total_chunks_in_section metadata.")

        if len(sorted_group["section_type"].astype(str).unique().tolist()) != 1:
            failures.append(f"Section `{section_id}` in document `{document_id}` mixes multiple section_type values.")

        for index, row in sorted_group.iterrows():
            expected_previous = sorted_group.iloc[index - 1]["chunk_id"] if index > 0 else None
            expected_next = sorted_group.iloc[index + 1]["chunk_id"] if index + 1 < len(sorted_group) else None
            actual_previous = row["previous_section_chunk_id"] if pd.notna(row["previous_section_chunk_id"]) else None
            actual_next = row["next_section_chunk_id"] if pd.notna(row["next_section_chunk_id"]) else None
            if expected_previous != actual_previous:
                failures.append(f"Section `{section_id}` in document `{document_id}` has invalid previous_section_chunk_id metadata.")
                break
            if expected_next != actual_next:
                failures.append(f"Section `{section_id}` in document `{document_id}` has invalid next_section_chunk_id metadata.")
                break

    return deduplicate_messages(failures), warnings


def validate_chunks(
    documents_df: pd.DataFrame,
    chunk_df: pd.DataFrame,
    structure_summary: StructureAnalysisSummary,
) -> tuple[str, list[str], list[str], dict[str, Any]]:
    failures: list[str] = []
    warnings: list[str] = []
    non_empty_document_count = int(documents_df["full_text"].map(lambda value: bool(normalize_text(value))).sum())
    empty_chunk_count = int(chunk_df["chunk_text"].map(lambda value: normalize_text(value) == "").sum()) if not chunk_df.empty else 0
    duplicate_chunk_id_count = int(chunk_df["chunk_id"].duplicated(keep=False).sum()) if not chunk_df.empty else 0
    documents_with_zero_chunks: list[str] = []

    if empty_chunk_count > 0:
        failures.append("One or more chunk_text values are empty.")
    if duplicate_chunk_id_count > 0:
        failures.append("Duplicate chunk_id values detected.")

    chunk_counts: dict[str, int] = {}
    if not chunk_df.empty:
        chunk_counts = {str(key): int(value) for key, value in chunk_df["document_id"].value_counts().to_dict().items()}

    for _, row in documents_df.iterrows():
        record = row.to_dict()
        full_text = normalize_text(record.get("full_text"))
        document_id = compute_document_id(record)
        if full_text and chunk_counts.get(document_id, 0) == 0:
            documents_with_zero_chunks.append(document_id)

    if documents_with_zero_chunks:
        failures.append("One or more documents with non-empty full_text produced zero chunks.")

    if structure_summary.weak_count > 0 or structure_summary.needs_review_count > 0:
        failures.append("One or more documents have weak structure confidence or need review.")

    if not chunk_df.empty:
        for field_name, expected_value in FIXED_VALUES.items():
            invalid_count = int((chunk_df[field_name].fillna("").map(str) != expected_value).sum())
            if invalid_count > 0:
                failures.append(f"Field `{field_name}` contains invalid fixed values.")

        invalid_url_count = int(chunk_df["url"].map(lambda value: not is_official_nsoud_url(value)).sum())
        if invalid_url_count > 0:
            failures.append("One or more chunk URLs do not point to an official nsoud.cz domain.")

    paragraph_failures, paragraph_preservation_passed, paragraph_preservation_failed = validate_paragraph_preservation(
        documents_df,
        chunk_df,
    )
    reconstruction_failures, reconstruction_passed, reconstruction_failed = validate_document_reconstruction(
        documents_df,
        chunk_df,
    )
    section_failures, section_reconstruction_passed, section_reconstruction_failed = validate_section_reconstruction(
        documents_df,
        chunk_df,
    )
    metadata_failures, metadata_warnings = validate_chunk_metadata(chunk_df)

    failures.extend(paragraph_failures)
    failures.extend(reconstruction_failures)
    failures.extend(section_failures)
    failures.extend(metadata_failures)
    warnings.extend(metadata_warnings)

    overlong_paragraph_chunk_count = int((chunk_df["chunk_warning"] == "overlong_ns_paragraph").sum()) if not chunk_df.empty else 0
    if overlong_paragraph_chunk_count > 0:
        warnings.append("Overlong NS paragraphs were preserved as standalone chunks.")

    unresolved_boundary_issue_count = 0
    if not chunk_df.empty:
        overlong_texts = chunk_df.loc[chunk_df["chunk_warning"] == "overlong_ns_paragraph", "chunk_text"].tolist()
        for chunk_text in overlong_texts:
            if len(detect_ns_structural_boundaries(normalize_text(chunk_text))) > 1:
                unresolved_boundary_issue_count += 1
    if unresolved_boundary_issue_count > 0:
        failures.append("One or more overlong chunks still contain unresolved internal structural boundaries.")

    failures = deduplicate_messages(failures)
    warnings = deduplicate_messages(warnings)

    if failures:
        status = "FAIL"
    elif warnings:
        status = "WARN"
    else:
        status = "PASS"

    chunk_lengths = chunk_df["chunk_text_length"].tolist() if not chunk_df.empty else []
    chunks_per_document = list(chunk_counts.values()) if chunk_counts else []
    metrics = {
        "non_empty_document_count": non_empty_document_count,
        "total_chunks": int(len(chunk_df)),
        "chunks_per_document_min": min(chunks_per_document) if chunks_per_document else 0,
        "chunks_per_document_max": max(chunks_per_document) if chunks_per_document else 0,
        "chunks_per_document_avg": mean(chunks_per_document) if chunks_per_document else 0.0,
        "chunk_length_min": min(chunk_lengths) if chunk_lengths else 0,
        "chunk_length_max": max(chunk_lengths) if chunk_lengths else 0,
        "chunk_length_avg": mean(chunk_lengths) if chunk_lengths else 0.0,
        "empty_chunk_count": empty_chunk_count,
        "duplicate_chunk_id_count": duplicate_chunk_id_count,
        "documents_with_zero_chunks": documents_with_zero_chunks,
        "overlong_paragraph_chunk_count": overlong_paragraph_chunk_count,
        "paragraph_preservation_passed": paragraph_preservation_passed,
        "paragraph_preservation_failed": paragraph_preservation_failed,
        "reconstruction_validation_passed": reconstruction_passed,
        "reconstruction_validation_failed": reconstruction_failed,
        "section_reconstruction_passed": section_reconstruction_passed,
        "section_reconstruction_failed": section_reconstruction_failed,
        "unresolved_boundary_issue_count": unresolved_boundary_issue_count,
        "strong_structure_count": structure_summary.strong_count,
        "medium_structure_count": structure_summary.medium_count,
        "weak_structure_count": structure_summary.weak_count,
        "needs_review_count": structure_summary.needs_review_count,
        "marker_coverage": structure_summary.marker_counts,
    }
    return status, failures, warnings, metrics


def build_validation_report(
    *,
    input_path: Path,
    output_parquet_path: Path,
    output_jsonl_path: Path,
    structure_report_path: Path,
    documents_df: pd.DataFrame,
    chunk_df: pd.DataFrame,
    validation_status: str,
    failures: list[str],
    warnings: list[str],
    metrics: dict[str, Any],
) -> str:
    status_items = failures + warnings if failures or warnings else ["Chunking and validation passed."]
    lines = [
        "# NSoud Chunk Validation",
        "",
        f"- Input: `{input_path}`",
        f"- Output Parquet: `{output_parquet_path}`",
        f"- Output JSONL: `{output_jsonl_path}`",
        f"- Structure report: `{structure_report_path}`",
        f"- Validation status: **{validation_status}**",
        f"- Total documents: **{len(documents_df)}**",
        f"- Total chunks: **{len(chunk_df)}**",
        "",
        "## Status",
    ]
    lines.extend(f"- {item}" for item in status_items)
    lines.extend(
        [
            "",
            "## Chunk Metrics",
            f"- chunks per document min: {metrics['chunks_per_document_min']}",
            f"- chunks per document max: {metrics['chunks_per_document_max']}",
            f"- chunks per document avg: {metrics['chunks_per_document_avg']:.2f}",
            f"- chunk_text_length min: {metrics['chunk_length_min']}",
            f"- chunk_text_length max: {metrics['chunk_length_max']}",
            f"- chunk_text_length avg: {metrics['chunk_length_avg']:.2f}",
            f"- empty chunk count: {metrics['empty_chunk_count']}",
            f"- duplicate chunk_id count: {metrics['duplicate_chunk_id_count']}",
            f"- documents with zero chunks: {len(metrics['documents_with_zero_chunks'])}",
            f"- overlong NS paragraph chunk count: {metrics['overlong_paragraph_chunk_count']}",
            "",
            "## Reconstruction Validation",
            f"- paragraph preservation passed/failed: {metrics['paragraph_preservation_passed']}/{metrics['paragraph_preservation_failed']}",
            f"- document reconstruction passed/failed: {metrics['reconstruction_validation_passed']}/{metrics['reconstruction_validation_failed']}",
            f"- section reconstruction passed/failed: {metrics['section_reconstruction_passed']}/{metrics['section_reconstruction_failed']}",
            f"- unresolved boundary issue count: {metrics['unresolved_boundary_issue_count']}",
            "",
            "## Structure Confidence Summary",
            f"- strong structure count: {metrics['strong_structure_count']}",
            f"- medium structure count: {metrics['medium_structure_count']}",
            f"- weak structure count: {metrics['weak_structure_count']}",
            f"- needs_review count: {metrics['needs_review_count']}",
            "",
            "## Marker Coverage",
            "",
            "| Marker | Document Count |",
            "| --- | ---: |",
        ]
    )
    for marker, count in metrics["marker_coverage"].items():
        lines.append(f"| {marker} | {count} |")

    lines.extend(["", "## Documents With Zero Chunks"])
    if metrics["documents_with_zero_chunks"]:
        lines.extend(f"- {document_id}" for document_id in metrics["documents_with_zero_chunks"])
    else:
        lines.append("- none")
    lines.append("")

    lines.extend(render_distribution_table("Source Distribution", distribution_counts(chunk_df, "source")))
    lines.extend(render_distribution_table("Document Type Distribution", distribution_counts(chunk_df, "document_type")))
    lines.extend(render_distribution_table("Legal Area Distribution", distribution_counts(chunk_df, "legal_area")))
    lines.extend(render_distribution_table("Section Type Distribution", distribution_counts(chunk_df, "section_type")))
    lines.extend(render_distribution_table("NS Section Hint Distribution", distribution_counts(chunk_df, "ns_section_hint")))
    return "\n".join(lines)


def write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", index=False)


def write_jsonl(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in df.to_dict(orient="records"):
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def main() -> int:
    args = parse_args()

    if pyarrow is None:
        print("structure analysis status: FAIL")
        print("chunking status: FAIL")
        print("error: pyarrow is required for Parquet output.")
        print("install command: pip install pyarrow")
        return 1

    validation_path = validation_path_for_output(args.out)
    output_jsonl_path = jsonl_path_for_output(args.out)
    structure_report_path = structure_report_path_for_input(args.input)

    try:
        documents_df = load_documents(args.input)
    except Exception as exc:
        print("structure analysis status: FAIL")
        print("chunking status: FAIL")
        print(f"error: {exc}")
        return 1

    try:
        chunk_df, document_analyses, build_failures, build_warnings = build_chunk_records(documents_df)
        structure_summary = analyze_structure(documents_df, document_analyses, structure_report_path)
        write_parquet(chunk_df, args.out)
        write_jsonl(chunk_df, output_jsonl_path)
    except Exception as exc:
        print("structure analysis status: FAIL")
        print("chunking status: FAIL")
        print(f"error: {exc}")
        return 1

    validation_status, validation_failures, validation_warnings, metrics = validate_chunks(
        documents_df,
        chunk_df,
        structure_summary,
    )
    failures = deduplicate_messages(build_failures + validation_failures)
    warnings = deduplicate_messages(build_warnings + validation_warnings)
    if failures:
        validation_status = "FAIL"
    elif warnings:
        validation_status = "WARN"
    else:
        validation_status = "PASS"

    report = build_validation_report(
        input_path=args.input,
        output_parquet_path=args.out,
        output_jsonl_path=output_jsonl_path,
        structure_report_path=structure_report_path,
        documents_df=documents_df,
        chunk_df=chunk_df,
        validation_status=validation_status,
        failures=failures,
        warnings=warnings,
        metrics=metrics,
    )
    validation_path.write_text(report, encoding="utf-8")

    summary = ChunkingSummary(
        structure_analysis_status="PASS",
        chunking_status="PASS",
        validation_status=validation_status,
        total_documents=len(documents_df),
        total_chunks=len(chunk_df),
        documents_with_zero_chunks=len(metrics["documents_with_zero_chunks"]),
        empty_chunk_count=metrics["empty_chunk_count"],
        duplicate_chunk_id_count=metrics["duplicate_chunk_id_count"],
        paragraph_preservation_passed=metrics["paragraph_preservation_passed"],
        paragraph_preservation_failed=metrics["paragraph_preservation_failed"],
        reconstruction_validation_passed=metrics["reconstruction_validation_passed"],
        reconstruction_validation_failed=metrics["reconstruction_validation_failed"],
        section_reconstruction_passed=metrics["section_reconstruction_passed"],
        section_reconstruction_failed=metrics["section_reconstruction_failed"],
        unresolved_boundary_issue_count=metrics["unresolved_boundary_issue_count"],
        strong_structure_count=metrics["strong_structure_count"],
        medium_structure_count=metrics["medium_structure_count"],
        weak_structure_count=metrics["weak_structure_count"],
        needs_review_count=metrics["needs_review_count"],
        output_parquet_path=args.out,
        output_jsonl_path=output_jsonl_path,
        validation_report_path=validation_path,
    )
    print(f"structure analysis status: {summary.structure_analysis_status}")
    print(f"chunking status: {summary.chunking_status}")
    print(f"validation status: {summary.validation_status}")
    print(f"total documents: {summary.total_documents}")
    print(f"total chunks: {summary.total_chunks}")
    print(f"documents with zero chunks: {summary.documents_with_zero_chunks}")
    print(f"empty chunk count: {summary.empty_chunk_count}")
    print(f"duplicate chunk_id count: {summary.duplicate_chunk_id_count}")
    print(
        f"paragraph preservation passed/failed: {summary.paragraph_preservation_passed}/{summary.paragraph_preservation_failed}"
    )
    print(
        "reconstruction validation passed/failed: "
        f"{summary.reconstruction_validation_passed}/{summary.reconstruction_validation_failed}"
    )
    print(
        "section reconstruction passed/failed: "
        f"{summary.section_reconstruction_passed}/{summary.section_reconstruction_failed}"
    )
    print(f"unresolved boundary issue count: {summary.unresolved_boundary_issue_count}")
    print(f"strong structure count: {summary.strong_structure_count}")
    print(f"medium structure count: {summary.medium_structure_count}")
    print(f"weak structure count: {summary.weak_structure_count}")
    print(f"needs_review count: {summary.needs_review_count}")
    print(f"structure report path: {structure_report_path}")
    print(f"output parquet path: {summary.output_parquet_path}")
    print(f"output jsonl path: {summary.output_jsonl_path}")
    print(f"validation report path: {summary.validation_report_path}")
    return 1 if summary.validation_status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
