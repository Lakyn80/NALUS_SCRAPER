from __future__ import annotations

import argparse
import re
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
ROMAN_SECTION_RE = re.compile(r"(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\.")
DOCUMENT_TYPE_MARKERS = ["ROZSUDEK", "USNESENÍ", "STANOVISKO"]
SECTION_MARKERS = ["takto:", "Odůvodnění:", "I.", "II.", "III.", "IV.", "V.", "Poučení:", "V Brně dne"]
CLOSING_MARKERS = ["Poučení:", "V Brně dne", "předseda senátu", "předsedkyně senátu"]
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
    "bodu",
    "body",
)


@dataclass(frozen=True)
class ParagraphSpan:
    start: int
    end: int
    text: str
    start_pattern: str


@dataclass(frozen=True)
class StructureAnalysisSummary:
    total_documents: int
    header_counts: dict[str, int]
    section_marker_counts: dict[str, int]
    closing_marker_counts: dict[str, int]
    numbered_paragraph_doc_count: int
    top_paragraph_start_patterns: list[tuple[str, int]]
    report_path: Path


@dataclass(frozen=True)
class ChunkingSummary:
    structure_analysis_status: str
    chunking_status: str
    validation_status: str
    total_documents: int
    total_chunks: int
    overlong_paragraph_chunk_count: int
    structure_report_path: Path
    output_parquet_path: Path
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


def validation_path_for_output(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}_validation.md")


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
    # NS one-line exports still preserve real paragraph starts as "space + marker".
    # We require a sentence/section ending before that space to avoid splitting dates,
    # citations, and inline statutory references such as "odst. 1".
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
    if not 1 <= int(marker_value) <= 150:
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
    # NS decisions often inline list legal issues as "že: 1) ... 2) ...".
    # We only split these markers after a sentence/list separator, never after a word
    # like "povinný 1)", which keeps party labels and citations intact.
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

    # Explicit NS section labels are the safest anchors and should win early.
    for label, pattern in SECTION_LABEL_PATTERNS.items():
        for match in pattern.finditer(text):
            marker_start = match.start()
            if marker_start > 0 and marker_start not in boundaries and is_probable_section_label_boundary(text, marker_start):
                boundaries[marker_start] = match.group(0)

    # Roman sections and numbered paragraphs are added only after context checks.
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
        marker_text = match.group(0)
        if marker_start > 0 and marker_start not in boundaries and is_probable_bracketed_paragraph_boundary(
            text,
            marker_start,
            marker_end,
        ):
            boundaries[marker_start] = marker_text

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


def should_append_paragraph(current_start: int, current_end: int, next_paragraph: ParagraphSpan) -> bool:
    current_length = current_end - current_start
    prospective_length = next_paragraph.end - current_start
    if prospective_length <= TARGET_CHUNK_SIZE:
        return True
    if current_length < TARGET_CHUNK_SIZE and prospective_length <= SOFT_MAX_CHUNK_SIZE:
        return True
    return False


def deduplicate_messages(messages: list[str]) -> list[str]:
    deduplicated: list[str] = []
    seen: set[str] = set()
    for message in messages:
        if message in seen:
            continue
        seen.add(message)
        deduplicated.append(message)
    return deduplicated


def infer_section_zone(text: str) -> dict[str, int]:
    marker_positions = {
        "takto": normalize_text(text).find("takto:"),
        "oduvodneni": normalize_text(text).find("Odůvodnění:"),
        "pouceni": normalize_text(text).find("Poučení:"),
        "closing": normalize_text(text).find("V Brně dne"),
    }
    return marker_positions


def classify_paragraph_section(start: int, start_pattern: str, zone_positions: dict[str, int]) -> str:
    if start_pattern == "header":
        takto_pos = zone_positions.get("takto", -1)
        oduvodneni_pos = zone_positions.get("oduvodneni", -1)
        if takto_pos > 0 and start < takto_pos:
            return "header"
        if oduvodneni_pos > 0 and start < oduvodneni_pos:
            return "header"
        return "unknown"

    if start_pattern == "V Brně dne":
        return "closing"
    if start_pattern == "Poučení:":
        return "pouceni"
    if start_pattern == "Odůvodnění:":
        return "oduvodneni"
    if start_pattern == "takto:":
        return "vyrok"

    normalized_pattern = start_pattern.strip()
    if normalized_pattern.startswith("[") and normalized_pattern.endswith("]"):
        normalized_pattern = normalized_pattern[1:-1]
    normalized_pattern = normalized_pattern.rstrip("./")

    if normalized_pattern.isdigit():
        oduvodneni_pos = zone_positions.get("oduvodneni", -1)
        pouceni_pos = zone_positions.get("pouceni", -1)
        closing_pos = zone_positions.get("closing", -1)
        if oduvodneni_pos >= 0 and start >= oduvodneni_pos and (pouceni_pos < 0 or start < pouceni_pos):
            return "oduvodneni"
        if pouceni_pos >= 0 and start >= pouceni_pos and (closing_pos < 0 or start < closing_pos):
            return "pouceni"

    if re.fullmatch(r"[IVXLCDM]+\.", start_pattern):
        takto_pos = zone_positions.get("takto", -1)
        oduvodneni_pos = zone_positions.get("oduvodneni", -1)
        pouceni_pos = zone_positions.get("pouceni", -1)
        closing_pos = zone_positions.get("closing", -1)
        if takto_pos >= 0 and start >= takto_pos and (oduvodneni_pos < 0 or start < oduvodneni_pos):
            return "vyrok"
        if oduvodneni_pos >= 0 and start >= oduvodneni_pos and (pouceni_pos < 0 or start < pouceni_pos):
            return "oduvodneni"
        if pouceni_pos >= 0 and start >= pouceni_pos and (closing_pos < 0 or start < closing_pos):
            return "pouceni"

    takto_pos = zone_positions.get("takto", -1)
    pouceni_pos = zone_positions.get("pouceni", -1)
    closing_pos = zone_positions.get("closing", -1)
    oduvodneni_pos = zone_positions.get("oduvodneni", -1)
    if closing_pos >= 0 and start >= closing_pos:
        return "closing"
    if pouceni_pos >= 0 and start >= pouceni_pos:
        return "pouceni"
    if oduvodneni_pos >= 0 and start >= oduvodneni_pos:
        return "oduvodneni"
    if takto_pos >= 0 and start >= takto_pos:
        return "vyrok"
    if takto_pos > 0 or oduvodneni_pos > 0:
        return "header"
    return "unknown"


def build_document_chunks(full_text: str) -> list[dict[str, Any]]:
    text = normalize_text(full_text)
    paragraphs = extract_ns_paragraphs(text)
    chunks: list[dict[str, Any]] = []
    zone_positions = infer_section_zone(text)

    if not paragraphs:
        return chunks

    current_indices: list[int] = []
    current_start = 0
    current_end = 0
    current_section = ""

    def finalize_current_chunk() -> None:
        nonlocal current_indices, current_start, current_end, current_section
        if not current_indices:
            return

        first_paragraph = paragraphs[current_indices[0]]
        last_paragraph = paragraphs[current_indices[-1]]
        chunk_text = text[first_paragraph.start:last_paragraph.end]
        chunk_warning = ""
        if len(current_indices) == 1 and len(first_paragraph.text) > HARD_MAX_CHUNK_SIZE:
            chunk_warning = "overlong_ns_paragraph"

        chunks.append(
            {
                "chunk_text": chunk_text,
                "chunk_char_start": first_paragraph.start,
                "chunk_char_end": last_paragraph.end,
                "paragraph_count": len(current_indices),
                "chunk_warning": chunk_warning,
                "ns_section_hint": current_section or classify_paragraph_section(
                    first_paragraph.start,
                    first_paragraph.start_pattern,
                    zone_positions,
                ),
            }
        )
        current_indices = []
        current_start = 0
        current_end = 0
        current_section = ""

    for paragraph_index, paragraph in enumerate(paragraphs):
        paragraph_length = len(paragraph.text)
        paragraph_section = classify_paragraph_section(paragraph.start, paragraph.start_pattern, zone_positions)

        if paragraph_length > HARD_MAX_CHUNK_SIZE:
            finalize_current_chunk()
            current_indices = [paragraph_index]
            current_start = paragraph.start
            current_end = paragraph.end
            current_section = paragraph_section
            finalize_current_chunk()
            continue

        if not current_indices:
            current_indices = [paragraph_index]
            current_start = paragraph.start
            current_end = paragraph.end
            current_section = paragraph_section
            continue

        if paragraph_section != current_section:
            finalize_current_chunk()
            current_indices = [paragraph_index]
            current_start = paragraph.start
            current_end = paragraph.end
            current_section = paragraph_section
            continue

        if should_append_paragraph(current_start, current_end, paragraph):
            current_indices.append(paragraph_index)
            current_end = paragraph.end
            continue

        finalize_current_chunk()
        current_indices = [paragraph_index]
        current_start = paragraph.start
        current_end = paragraph.end
        current_section = paragraph_section

    finalize_current_chunk()
    return chunks


def build_chunk_records(documents_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    chunk_rows: list[dict[str, Any]] = []

    for _, row in documents_df.iterrows():
        record = row.to_dict()
        document_id = compute_document_id(record)
        full_text = normalize_text(record.get("full_text"))
        produced_chunks = build_document_chunks(full_text)

        if full_text and not produced_chunks:
            failures.append(f"Document `{document_id}` has non-empty full_text but produced zero chunks.")

        width = max(4, len(str(max(0, len(produced_chunks) - 1))))
        for chunk_index, chunk in enumerate(produced_chunks):
            chunk_text_value = str(chunk["chunk_text"])
            chunk_id = f"{document_id}__chunk_{chunk_index:0{width}d}"
            row_payload = {field: record.get(field) for field in METADATA_FIELDS}
            row_payload.update(
                {
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text_value,
                    "chunk_text_length": len(chunk_text_value),
                    "chunk_char_start": int(chunk["chunk_char_start"]),
                    "chunk_char_end": int(chunk["chunk_char_end"]),
                    "paragraph_count": int(chunk["paragraph_count"]),
                    "chunk_warning": str(chunk["chunk_warning"]),
                    "ns_section_hint": str(chunk["ns_section_hint"]),
                }
            )
            chunk_rows.append(row_payload)

    ordered_columns = METADATA_FIELDS + [
        "document_id",
        "chunk_id",
        "chunk_index",
        "chunk_text",
        "chunk_text_length",
        "chunk_char_start",
        "chunk_char_end",
        "paragraph_count",
        "chunk_warning",
        "ns_section_hint",
    ]
    chunk_df = pd.DataFrame(chunk_rows)
    if chunk_df.empty:
        chunk_df = pd.DataFrame(columns=ordered_columns)
    else:
        chunk_df = chunk_df.reindex(columns=ordered_columns)

    overlong_count = int((chunk_df["chunk_warning"] == "overlong_ns_paragraph").sum()) if not chunk_df.empty else 0
    if overlong_count > 0:
        warnings.append("Overlong NS paragraphs were preserved as standalone chunks.")

    return chunk_df, failures, warnings


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


def analyze_structure(documents_df: pd.DataFrame, report_path: Path) -> StructureAnalysisSummary:
    header_counts = {
        "case_number_at_beginning": 0,
        "ecli_in_metadata": 0,
        "ROZSUDEK": 0,
        "USNESENÍ": 0,
        "STANOVISKO": 0,
    }
    section_marker_counts = {marker: 0 for marker in SECTION_MARKERS}
    closing_marker_counts = {marker: 0 for marker in CLOSING_MARKERS}
    paragraph_start_counts: dict[str, int] = {}
    numbered_paragraph_doc_count = 0

    for _, row in documents_df.iterrows():
        record = row.to_dict()
        text = normalize_text(record.get("full_text"))
        if CASE_NUMBER_START_RE.search(text):
            header_counts["case_number_at_beginning"] += 1
        if normalize_text(record.get("ecli")).strip():
            header_counts["ecli_in_metadata"] += 1

        for marker in DOCUMENT_TYPE_MARKERS:
            if marker in text:
                header_counts[marker] += 1

        has_numbered_paragraph = False
        for marker in SECTION_MARKERS:
            if marker in text:
                section_marker_counts[marker] += 1

        lowered_text = text.lower()
        for marker in CLOSING_MARKERS:
            if marker.lower() in lowered_text:
                closing_marker_counts[marker] += 1

        for paragraph in extract_ns_paragraphs(text):
            paragraph_start_counts[paragraph.start_pattern] = paragraph_start_counts.get(paragraph.start_pattern, 0) + 1
            normalized_pattern = paragraph.start_pattern.strip()
            if normalized_pattern.startswith("[") and normalized_pattern.endswith("]"):
                normalized_pattern = normalized_pattern[1:-1]
            normalized_pattern = normalized_pattern.rstrip("./")
            if normalized_pattern.isdigit():
                has_numbered_paragraph = True

        if has_numbered_paragraph:
            numbered_paragraph_doc_count += 1

    top_paragraph_start_patterns = sorted(
        paragraph_start_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[:15]

    lines = [
        "# NSoud Structure Pattern Analysis",
        "",
        f"- Total documents: **{len(documents_df)}**",
        f"- Numbered paragraph documents: **{numbered_paragraph_doc_count}**",
        "",
        "## Header Patterns",
        "",
        "| Pattern | Document Count |",
        "| --- | ---: |",
    ]
    for pattern_name, count in header_counts.items():
        lines.append(f"| {pattern_name} | {count} |")

    lines.extend(
        [
            "",
            "## Section Markers",
            "",
            "| Marker | Document Count |",
            "| --- | ---: |",
        ]
    )
    for marker, count in section_marker_counts.items():
        lines.append(f"| {marker} | {count} |")

    lines.extend(
        [
            "",
            "## Closing and Signature Patterns",
            "",
            "| Marker | Document Count |",
            "| --- | ---: |",
        ]
    )
    for marker, count in closing_marker_counts.items():
        lines.append(f"| {marker} | {count} |")

    lines.extend(
        [
            "",
            "## Top Paragraph-Start Patterns",
            "",
            "| Pattern | Count |",
            "| --- | ---: |",
        ]
    )
    for pattern_name, count in top_paragraph_start_patterns:
        lines.append(f"| {pattern_name} | {count} |")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return StructureAnalysisSummary(
        total_documents=len(documents_df),
        header_counts=header_counts,
        section_marker_counts=section_marker_counts,
        closing_marker_counts=closing_marker_counts,
        numbered_paragraph_doc_count=numbered_paragraph_doc_count,
        top_paragraph_start_patterns=top_paragraph_start_patterns,
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

        original_paragraphs = [
            normalize_paragraph_for_compare(paragraph.text)
            for paragraph in extract_ns_paragraphs(full_text)
        ]
        chunk_group = grouped_chunks.get(document_id)
        reconstructed_paragraphs: list[str] = []

        if chunk_group is not None:
            for chunk_text in chunk_group["chunk_text"].tolist():
                reconstructed_paragraphs.extend(
                    normalize_paragraph_for_compare(paragraph.text)
                    for paragraph in extract_ns_paragraphs(normalize_text(chunk_text))
                )

        if original_paragraphs != reconstructed_paragraphs:
            failed_documents += 1
            failures.append(f"Paragraph preservation failed for document `{document_id}`.")
            continue

        passed_documents += 1

    return failures, passed_documents, failed_documents


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
    failures.extend(paragraph_failures)

    overlong_paragraph_chunk_count = int((chunk_df["chunk_warning"] == "overlong_ns_paragraph").sum()) if not chunk_df.empty else 0
    if overlong_paragraph_chunk_count > 0:
        warnings.append("Overlong NS paragraphs were preserved as standalone chunks.")

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
    overlong_lengths = (
        chunk_df.loc[chunk_df["chunk_warning"] == "overlong_ns_paragraph", "chunk_text_length"].tolist()
        if not chunk_df.empty
        else []
    )
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
        "max_overlong_paragraph_length": max(overlong_lengths) if overlong_lengths else 0,
        "paragraph_preservation_passed": paragraph_preservation_passed,
        "paragraph_preservation_failed": paragraph_preservation_failed,
        "section_marker_coverage": structure_summary.section_marker_counts,
    }
    return status, failures, warnings, metrics


def build_validation_report(
    *,
    input_path: Path,
    output_path: Path,
    documents_df: pd.DataFrame,
    chunk_df: pd.DataFrame,
    structure_summary: StructureAnalysisSummary,
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
        f"- Output Parquet: `{output_path}`",
        f"- Structure report: `{structure_summary.report_path}`",
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
            f"- max overlong paragraph length: {metrics['max_overlong_paragraph_length']}",
            "",
            "## Paragraph Preservation Check",
            f"- documents passed: {metrics['paragraph_preservation_passed']}",
            f"- documents failed: {metrics['paragraph_preservation_failed']}",
            "",
            "## Section Marker Coverage",
            "",
            "| Marker | Document Count |",
            "| --- | ---: |",
        ]
    )
    for marker, count in metrics["section_marker_coverage"].items():
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
    return "\n".join(lines)


def write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", index=False)


def main() -> int:
    args = parse_args()

    if pyarrow is None:
        print("structure analysis status: FAIL")
        print("chunking status: FAIL")
        print("error: pyarrow is required for Parquet output.")
        print("install command: pip install pyarrow")
        return 1

    validation_path = validation_path_for_output(args.out)
    structure_report_path = structure_report_path_for_input(args.input)

    try:
        documents_df = load_documents(args.input)
    except Exception as exc:
        print("structure analysis status: FAIL")
        print("chunking status: FAIL")
        print(f"error: {exc}")
        return 1

    try:
        structure_summary = analyze_structure(documents_df, structure_report_path)
    except Exception as exc:
        print("structure analysis status: FAIL")
        print("chunking status: FAIL")
        print(f"error: {exc}")
        return 1

    try:
        chunk_df, build_failures, build_warnings = build_chunk_records(documents_df)
        write_parquet(chunk_df, args.out)
    except Exception as exc:
        print("structure analysis status: PASS")
        print("chunking status: FAIL")
        print(f"error: {exc}")
        print(f"structure report path: {structure_report_path}")
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
        output_path=args.out,
        documents_df=documents_df,
        chunk_df=chunk_df,
        structure_summary=structure_summary,
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
        overlong_paragraph_chunk_count=metrics["overlong_paragraph_chunk_count"],
        structure_report_path=structure_report_path,
        output_parquet_path=args.out,
        validation_report_path=validation_path,
    )
    print(f"structure analysis status: {summary.structure_analysis_status}")
    print(f"chunking status: {summary.chunking_status}")
    print(f"total documents: {summary.total_documents}")
    print(f"total chunks: {summary.total_chunks}")
    print(f"overlong NS paragraph chunk count: {summary.overlong_paragraph_chunk_count}")
    print(f"structure report path: {summary.structure_report_path}")
    print(f"output parquet path: {summary.output_parquet_path}")
    print(f"validation report path: {summary.validation_report_path}")
    print(f"validation status: {summary.validation_status}")
    return 1 if summary.validation_status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
