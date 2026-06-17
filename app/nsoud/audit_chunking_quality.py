from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("pandas is required for this script.") from exc

try:
    import pyarrow  # noqa: F401
except ImportError:
    pyarrow = None

from app.nsoud.structure.section_detector import detect_ns_document_structure


HARD_MAX_CHUNK_SIZE = 4000
PREVIEW_LIMIT = 10
SECTION_NAME_MAP = {
    "header": "header",
    "operative_part": "operative_part",
    "oduvodneni": "reasoning",
    "pouceni": "appeal_instruction",
    "closing/signature": "signature",
}
REQUIRED_CHUNK_FIELDS = [
    "chunk_id",
    "document_id",
    "case_number",
    "ecli",
    "document_type",
    "legal_area",
    "chunk_text",
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
    "structure_confidence",
    "structure_status",
    "structure_needs_review",
    "detected_section_order",
    "detected_markers",
    "section_source",
    "chunking_strategy",
]
NON_EMPTY_REQUIRED_CHUNK_FIELDS = [
    "chunk_id",
    "document_id",
    "chunk_text",
    "chunk_index",
    "total_chunks_in_document",
    "section_id",
    "section_type",
    "section_index",
    "chunk_index_in_section",
    "total_chunks_in_section",
    "structure_confidence",
    "structure_status",
    "structure_needs_review",
    "detected_section_order",
    "detected_markers",
    "section_source",
    "chunking_strategy",
]
NULLABLE_LINK_FIELDS = {
    "previous_chunk_id",
    "next_chunk_id",
    "previous_section_chunk_id",
    "next_section_chunk_id",
}
REQUIRED_SECTION_FIELDS = [
    "section_id",
    "section_type",
    "section_index",
    "section_char_start",
    "section_char_end",
]
CHANGED_FILES = [
    "app/nsoud/audit_chunking_quality.py",
    "app/artifacts/nsoud/rag_ready/nsoud_chunking_quality_audit_section_2025_01_03.md",
]


def sanitize_human_excerpt(*, text: str, section_type: str) -> str:
    normalized = normalize_text(text)
    if normalize_text(section_type) == "operative_part":
        return re.sub(r"^\s*takto\s*:\s*", "", normalized, count=1, flags=re.IGNORECASE)
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exhaustively audit NSoud section-aware chunk quality.")
    parser.add_argument("--documents", type=Path, required=True, help="Input Parquet path with NSoud documents.")
    parser.add_argument("--chunks", type=Path, required=True, help="Input Parquet path with NSoud chunks.")
    parser.add_argument("--out-json", type=Path, required=True, help="Output JSON path for the audit report.")
    parser.add_argument("--out-md", type=Path, required=True, help="Output Markdown path for the audit report.")
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


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if is_missing(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def parse_json_field(value: Any) -> Any:
    text = normalize_text(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def compute_document_id(record: dict[str, Any]) -> str:
    ecli = normalize_text(record.get("ecli")).strip()
    if ecli:
        return ecli
    return normalize_text(record.get("content_hash")).strip()


def determine_index_base(values: list[Any]) -> tuple[int | None, bool]:
    if not values:
        return None, False
    normalized_values = [int(value) for value in values if not is_missing(value)]
    if len(normalized_values) != len(values):
        return None, False
    sorted_values = sorted(normalized_values)
    index_base = sorted_values[0]
    if index_base not in (0, 1):
        return index_base, False
    expected = list(range(index_base, index_base + len(sorted_values)))
    return index_base, sorted_values == expected


def excerpt(text: str, size: int) -> str:
    normalized = normalize_text(text)
    if len(normalized) <= size:
        return normalized
    return normalized[:size]


def excerpt_tail(text: str, size: int) -> str:
    normalized = normalize_text(text)
    if len(normalized) <= size:
        return normalized
    return normalized[-size:]


def format_float(value: float) -> float:
    return round(float(value), 6)


def load_documents(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_chunks(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def build_document_index(documents_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    documents_by_id: dict[str, dict[str, Any]] = {}
    for _, row in documents_df.iterrows():
        record = row.to_dict()
        documents_by_id[compute_document_id(record)] = record
    return documents_by_id


def build_document_groups(chunk_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    grouped: dict[str, pd.DataFrame] = {}
    for document_id, group in chunk_df.groupby("document_id", sort=False, dropna=False):
        grouped[normalize_text(document_id)] = group.copy()
    return grouped


def build_section_groups(chunk_df: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    grouped: dict[tuple[str, str], pd.DataFrame] = {}
    for keys, group in chunk_df.groupby(["document_id", "section_id"], sort=False, dropna=False):
        document_id, section_id = keys
        grouped[(normalize_text(document_id), normalize_text(section_id))] = group.copy()
    return grouped


def build_structure_boundaries(document_record: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    full_text = normalize_text(document_record.get("full_text"))
    metadata = {
        "case_number": normalize_text(document_record.get("case_number")),
        "ecli": normalize_text(document_record.get("ecli")),
        "document_type": normalize_text(document_record.get("document_type")),
        "legal_area": normalize_text(document_record.get("legal_area")),
    }
    structure = detect_ns_document_structure(full_text=full_text, metadata=metadata)
    candidates = list(structure.get("section_candidates") or [])
    boundaries: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates[:-1]):
        next_candidate = candidates[index + 1]
        boundaries.append(
            {
                "from_section_raw": normalize_text(candidate.get("section")),
                "from_section_type": SECTION_NAME_MAP.get(normalize_text(candidate.get("section")), "unknown"),
                "to_section_raw": normalize_text(next_candidate.get("section")),
                "to_section_type": SECTION_NAME_MAP.get(normalize_text(next_candidate.get("section")), "unknown"),
                "position": int(next_candidate.get("position", 0)),
            }
        )
    return structure, boundaries


def detect_broken_word_fragment(
    *,
    full_text: str,
    chunk_start: int,
    chunk_end: int,
    section_start: int,
    section_end: int,
) -> tuple[bool, bool]:
    broken_start = False
    broken_end = False

    if 0 < chunk_start < len(full_text) and chunk_start > section_start:
        previous_char = full_text[chunk_start - 1]
        current_char = full_text[chunk_start]
        broken_start = previous_char.isalnum() and current_char.isalnum()

    if 0 < chunk_end < len(full_text) and chunk_end < section_end:
        previous_char = full_text[chunk_end - 1]
        next_char = full_text[chunk_end]
        broken_end = previous_char.isalnum() and next_char.isalnum()

    return broken_start, broken_end


def render_markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |", "| none |"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [normalize_text(value).replace("\n", " ").replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def summarize_messages(results: list[dict[str, Any]], key: str) -> list[str]:
    values: list[str] = []
    for result in results:
        if result.get(key):
            values.append(normalize_text(result.get(key)))
    return values


def audit_chunk_metadata(
    *,
    chunk_df: pd.DataFrame,
    documents_by_id: dict[str, dict[str, Any]],
    boundary_violations_by_chunk_id: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    duplicated_chunk_ids = {
        normalize_text(chunk_id)
        for chunk_id, count in Counter(normalize_text(value) for value in chunk_df["chunk_id"].tolist()).items()
        if chunk_id and count > 1
    }
    missing_schema_fields = [field_name for field_name in REQUIRED_CHUNK_FIELDS if field_name not in chunk_df.columns]
    chunk_results: list[dict[str, Any]] = []
    metadata_passed = 0
    metadata_failed = 0
    failed_chunk_ids: list[str] = []
    empty_chunk_count = 0
    broken_word_warning_count = 0

    for _, row in chunk_df.iterrows():
        chunk = row.to_dict()
        chunk_id = normalize_text(chunk.get("chunk_id"))
        document_id = normalize_text(chunk.get("document_id"))
        section_id = normalize_text(chunk.get("section_id"))
        document_record = documents_by_id.get(document_id)
        full_text = normalize_text(document_record.get("full_text")) if document_record else ""
        chunk_start = int(chunk.get("chunk_char_start", 0))
        chunk_end = int(chunk.get("chunk_char_end", 0))
        section_start = int(chunk.get("section_char_start", 0))
        section_end = int(chunk.get("section_char_end", 0))

        metadata_messages: list[str] = []
        validation_messages: list[str] = []
        missing_fields = [
            field_name
            for field_name in NON_EMPTY_REQUIRED_CHUNK_FIELDS
            if field_name not in NULLABLE_LINK_FIELDS and is_missing(chunk.get(field_name))
        ]
        if missing_schema_fields:
            metadata_messages.append(f"Missing required chunk columns: {', '.join(missing_schema_fields)}.")
        if missing_fields:
            metadata_messages.append(f"Missing required fields: {', '.join(missing_fields)}.")

        if chunk_id in duplicated_chunk_ids:
            metadata_messages.append("chunk_id is duplicated.")

        chunk_text = normalize_text(chunk.get("chunk_text"))
        if not chunk_text:
            empty_chunk_count += 1
            metadata_messages.append("chunk_text is empty.")

        if normalize_text(chunk.get("section_source")) != "nsoud.structure":
            metadata_messages.append("section_source must equal `nsoud.structure`.")

        if normalize_text(chunk.get("chunking_strategy")) != "document_section_aware":
            metadata_messages.append("chunking_strategy must equal `document_section_aware`.")

        if document_record is None:
            validation_messages.append("Chunk references an unknown document_id.")
            chunk_text_matches_span = False
        else:
            expected_chunk_text = full_text[chunk_start:chunk_end]
            chunk_text_matches_span = chunk_text == expected_chunk_text
            if not chunk_text_matches_span:
                validation_messages.append(
                    "chunk_text does not match the document slice defined by chunk_char_start/chunk_char_end."
                )

        boundary_violations = boundary_violations_by_chunk_id.get(chunk_id, [])
        if boundary_violations:
            validation_messages.append("Chunk crosses a known legal section boundary.")

        broken_start = False
        broken_end = False
        if document_record is not None and 0 <= chunk_start <= chunk_end <= len(full_text):
            broken_start, broken_end = detect_broken_word_fragment(
                full_text=full_text,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                section_start=section_start,
                section_end=section_end,
            )
            if broken_start or broken_end:
                broken_word_warning_count += 1

        metadata_ok = not metadata_messages
        if metadata_ok:
            metadata_passed += 1
        else:
            metadata_failed += 1
            failed_chunk_ids.append(chunk_id)

        all_messages = metadata_messages + validation_messages

        chunk_results.append(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "section_id": section_id,
                "section_type": normalize_text(chunk.get("section_type")),
                "chunk_index": None if is_missing(chunk.get("chunk_index")) else int(chunk.get("chunk_index")),
                "chunk_index_in_section": None
                if is_missing(chunk.get("chunk_index_in_section"))
                else int(chunk.get("chunk_index_in_section")),
                "chunk_text_length": len(chunk_text),
                "metadata_ok": metadata_ok,
                "missing_fields": missing_fields,
                "unique_chunk_id_ok": chunk_id not in duplicated_chunk_ids,
                "non_empty_text_ok": bool(chunk_text),
                "chunk_text_matches_span_ok": chunk_text_matches_span,
                "section_source_ok": normalize_text(chunk.get("section_source")) == "nsoud.structure",
                "chunking_strategy_ok": normalize_text(chunk.get("chunking_strategy")) == "document_section_aware",
                "cross_section_boundary_violation": bool(boundary_violations),
                "broken_word_start_warning": broken_start,
                "broken_word_end_warning": broken_end,
                "status": "FAIL" if all_messages else ("WARN" if broken_start or broken_end else "PASS"),
                "messages": all_messages,
            }
        )

    return chunk_results, {
        "missing_schema_fields": missing_schema_fields,
        "duplicated_chunk_ids": sorted(duplicated_chunk_ids),
        "duplicate_chunk_id_count": len(duplicated_chunk_ids),
        "chunk_metadata_validation_passed": metadata_passed,
        "chunk_metadata_validation_failed": metadata_failed,
        "failed_chunk_ids": sorted(set(failed_chunk_ids)),
        "empty_chunk_count": empty_chunk_count,
        "broken_word_warning_count": broken_word_warning_count,
    }


def audit_documents(
    *,
    documents_df: pd.DataFrame,
    document_groups: dict[str, pd.DataFrame],
    structure_info_by_document_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    document_results: list[dict[str, Any]] = []
    boundary_violations: list[dict[str, Any]] = []
    boundary_violations_by_chunk_id: dict[str, list[dict[str, Any]]] = defaultdict(list)

    document_reconstruction_passed = 0
    document_reconstruction_failed = 0
    document_sequence_validation_passed = 0
    document_sequence_validation_failed = 0
    document_neighbor_validation_passed = 0
    document_neighbor_validation_failed = 0
    documents_with_zero_chunks: list[str] = []
    medium_structure_documents: list[str] = []
    weak_structure_documents: list[str] = []
    needs_review_documents: list[str] = []
    failed_document_ids: list[str] = []

    for _, row in documents_df.iterrows():
        document_record = row.to_dict()
        document_id = compute_document_id(document_record)
        full_text = normalize_text(document_record.get("full_text"))
        full_text_length = len(full_text)
        chunk_group = document_groups.get(document_id)
        structure_entry = structure_info_by_document_id[document_id]
        structure = structure_entry["structure"]
        boundaries = structure_entry["boundaries"]

        messages: list[str] = []
        if chunk_group is None or chunk_group.empty:
            documents_with_zero_chunks.append(document_id)
            messages.append("Document does not exist in chunks or has zero chunks.")
            document_sequence_validation_failed += 1
            document_neighbor_validation_failed += 1
            document_reconstruction_failed += 1
            failed_document_ids.append(document_id)
            document_results.append(
                {
                    "document_id": document_id,
                    "case_number": normalize_text(document_record.get("case_number")),
                    "ecli": normalize_text(document_record.get("ecli")),
                    "document_type": normalize_text(document_record.get("document_type")),
                    "legal_area": normalize_text(document_record.get("legal_area")),
                    "full_text_length": full_text_length,
                    "chunk_count": 0,
                    "section_count": 0,
                    "document_exists_in_chunks": False,
                    "has_at_least_one_chunk": False,
                    "chunk_index_base": None,
                    "document_sequence_ok": False,
                    "neighbor_links_ok": False,
                    "reconstruction_ok": False,
                    "char_span_coverage_ok": False,
                    "structure_confidence": None,
                    "structure_status": "",
                    "structure_needs_review": False,
                    "boundary_violation_count": 0,
                    "status": "FAIL",
                    "messages": messages,
                }
            )
            continue

        sorted_group = chunk_group.sort_values("chunk_index").reset_index(drop=True)
        chunk_count = len(sorted_group)
        section_count = int(sorted_group["section_id"].nunique(dropna=False))
        chunk_index_values = sorted_group["chunk_index"].tolist()
        chunk_index_base, chunk_index_sequence_ok = determine_index_base(chunk_index_values)
        total_chunks_match = sorted_group["total_chunks_in_document"].nunique(dropna=False) == 1 and int(
            sorted_group["total_chunks_in_document"].iloc[0]
        ) == chunk_count
        document_sequence_ok = chunk_index_sequence_ok and total_chunks_match

        if document_sequence_ok:
            document_sequence_validation_passed += 1
        else:
            document_sequence_validation_failed += 1
            if not chunk_index_sequence_ok:
                messages.append("chunk_index sequence is not continuous according to the current project convention.")
            if not total_chunks_match:
                messages.append("total_chunks_in_document does not match the actual number of chunks.")

        neighbor_links_ok = True
        for index, chunk_row in sorted_group.iterrows():
            expected_previous = normalize_text(sorted_group.iloc[index - 1]["chunk_id"]) if index > 0 else ""
            expected_next = normalize_text(sorted_group.iloc[index + 1]["chunk_id"]) if index + 1 < len(sorted_group) else ""
            actual_previous = normalize_text(chunk_row.get("previous_chunk_id"))
            actual_next = normalize_text(chunk_row.get("next_chunk_id"))
            if actual_previous != expected_previous:
                neighbor_links_ok = False
                messages.append("previous_chunk_id links are invalid for this document.")
                break
            if actual_next != expected_next:
                neighbor_links_ok = False
                messages.append("next_chunk_id links are invalid for this document.")
                break

        if neighbor_links_ok:
            document_neighbor_validation_passed += 1
        else:
            document_neighbor_validation_failed += 1

        all_chunks_share_document_id = bool((sorted_group["document_id"].map(normalize_text) == document_id).all())
        if not all_chunks_share_document_id:
            messages.append("Not all chunks in the document group share the same document_id.")

        char_span_coverage_ok = True
        expected_start = 0
        for _, chunk_row in sorted_group.iterrows():
            chunk_start = int(chunk_row.get("chunk_char_start", 0))
            chunk_end = int(chunk_row.get("chunk_char_end", 0))
            if chunk_start != expected_start:
                char_span_coverage_ok = False
                messages.append("Chunk char spans do not reconstruct the document contiguously.")
                break
            expected_start = chunk_end
        if char_span_coverage_ok and expected_start != full_text_length:
            char_span_coverage_ok = False
            messages.append("Chunk char spans do not cover the full document.")

        reconstructed = "".join(normalize_text(value) for value in sorted_group["chunk_text"].tolist())
        reconstruction_ok = reconstructed == full_text and char_span_coverage_ok
        if reconstruction_ok:
            document_reconstruction_passed += 1
        else:
            document_reconstruction_failed += 1
            if reconstructed != full_text:
                messages.append("Chunks ordered by chunk_index do not exactly reconstruct the document text.")

        structure_status_values = sorted(
            {normalize_text(value) for value in sorted_group["structure_status"].tolist() if normalize_text(value)}
        )
        structure_needs_review_values = {
            to_bool(value) for value in sorted_group["structure_needs_review"].tolist() if not is_missing(value)
        }
        structure_confidence_values = {
            format_float(value) for value in sorted_group["structure_confidence"].tolist() if not is_missing(value)
        }
        structure_status = structure_status_values[0] if len(structure_status_values) == 1 else ""
        structure_needs_review = next(iter(structure_needs_review_values)) if len(structure_needs_review_values) == 1 else False
        structure_confidence = next(iter(structure_confidence_values)) if len(structure_confidence_values) == 1 else None

        if len(structure_status_values) != 1:
            messages.append("structure_status is inconsistent across chunks of the same document.")
        if len(structure_needs_review_values) != 1:
            messages.append("structure_needs_review is inconsistent across chunks of the same document.")
        if len(structure_confidence_values) != 1:
            messages.append("structure_confidence is inconsistent across chunks of the same document.")

        if structure_status == "medium":
            medium_structure_documents.append(document_id)
        if structure_status == "weak":
            weak_structure_documents.append(document_id)
            messages.append("Document has weak structure status.")
        if structure_needs_review:
            needs_review_documents.append(document_id)
            messages.append("Document is marked as structure_needs_review.")

        detected_markers = parse_json_field(sorted_group.iloc[0].get("detected_markers"))
        detected_section_order = parse_json_field(sorted_group.iloc[0].get("detected_section_order"))
        chunk_records = sorted_group.to_dict(orient="records")
        for boundary in boundaries:
            boundary_position = int(boundary["position"])
            for chunk_row in chunk_records:
                chunk_start = int(chunk_row.get("chunk_char_start", 0))
                chunk_end = int(chunk_row.get("chunk_char_end", 0))
                if chunk_start < boundary_position < chunk_end:
                    violation = {
                        "document_id": document_id,
                        "chunk_id": normalize_text(chunk_row.get("chunk_id")),
                        "chunk_index": int(chunk_row.get("chunk_index")),
                        "boundary_position": boundary_position,
                        "from_section_raw": boundary["from_section_raw"],
                        "from_section_type": boundary["from_section_type"],
                        "to_section_raw": boundary["to_section_raw"],
                        "to_section_type": boundary["to_section_type"],
                        "message": (
                            "Known cross-section transition occurs inside a single chunk: "
                            f"{boundary['from_section_type']} -> {boundary['to_section_type']}."
                        ),
                    }
                    boundary_violations.append(violation)
                    boundary_violations_by_chunk_id[normalize_text(chunk_row.get("chunk_id"))].append(violation)

        document_boundary_violations = [violation for violation in boundary_violations if violation["document_id"] == document_id]

        if any(
            boundary["from_section_type"] == "operative_part" and boundary["to_section_type"] == "reasoning"
            for boundary in boundaries
        ) and any(
            violation["document_id"] == document_id
            and violation["from_section_type"] == "operative_part"
            and violation["to_section_type"] == "reasoning"
            for violation in boundary_violations
        ):
            messages.append("operative_part is merged with reasoning inside a chunk.")
        if any(
            boundary["from_section_type"] == "reasoning" and boundary["to_section_type"] == "appeal_instruction"
            for boundary in boundaries
        ) and any(
            violation["document_id"] == document_id
            and violation["from_section_type"] == "reasoning"
            and violation["to_section_type"] == "appeal_instruction"
            for violation in boundary_violations
        ):
            messages.append("reasoning is merged with pouceni inside a chunk.")
        if any(
            boundary["from_section_type"] == "appeal_instruction" and boundary["to_section_type"] == "signature"
            for boundary in boundaries
        ) and any(
            violation["document_id"] == document_id
            and violation["from_section_type"] == "appeal_instruction"
            and violation["to_section_type"] == "signature"
            for violation in boundary_violations
        ):
            messages.append("pouceni is merged with signature inside a chunk.")

        status = "FAIL" if messages or document_boundary_violations else "PASS"
        if status == "FAIL":
            failed_document_ids.append(document_id)

        section_list = []
        for _, section_row in (
            sorted_group[["section_id", "section_type", "section_index"]]
            .drop_duplicates()
            .sort_values("section_index")
            .iterrows()
        ):
            section_list.append(
                {
                    "section_id": normalize_text(section_row.get("section_id")),
                    "section_type": normalize_text(section_row.get("section_type")),
                    "section_index": int(section_row.get("section_index")),
                }
            )

        document_results.append(
            {
                "document_id": document_id,
                "case_number": normalize_text(document_record.get("case_number")),
                "ecli": normalize_text(document_record.get("ecli")),
                "document_type": normalize_text(document_record.get("document_type")),
                "legal_area": normalize_text(document_record.get("legal_area")),
                "full_text_length": full_text_length,
                "chunk_count": chunk_count,
                "section_count": section_count,
                "document_exists_in_chunks": True,
                "has_at_least_one_chunk": True,
                "all_chunks_share_same_document_id": all_chunks_share_document_id,
                "chunk_index_base": chunk_index_base,
                "document_sequence_ok": document_sequence_ok,
                "neighbor_links_ok": neighbor_links_ok,
                "reconstruction_ok": reconstruction_ok,
                "char_span_coverage_ok": char_span_coverage_ok,
                "structure_confidence": structure_confidence,
                "structure_status": structure_status,
                "structure_needs_review": structure_needs_review,
                "boundary_violation_count": len(document_boundary_violations),
                "detected_section_order": detected_section_order,
                "detected_markers": detected_markers,
                "sections": section_list,
                "status": status,
                "messages": sorted(set(messages)),
            }
        )

    return document_results, boundary_violations, {
        "boundary_violations_by_chunk_id": boundary_violations_by_chunk_id,
        "document_reconstruction_passed": document_reconstruction_passed,
        "document_reconstruction_failed": document_reconstruction_failed,
        "document_sequence_validation_passed": document_sequence_validation_passed,
        "document_sequence_validation_failed": document_sequence_validation_failed,
        "document_neighbor_validation_passed": document_neighbor_validation_passed,
        "document_neighbor_validation_failed": document_neighbor_validation_failed,
        "documents_with_zero_chunks": sorted(set(documents_with_zero_chunks)),
        "medium_structure_documents": sorted(set(medium_structure_documents)),
        "weak_structure_documents": sorted(set(weak_structure_documents)),
        "needs_review_documents": sorted(set(needs_review_documents)),
        "failed_document_ids": sorted(set(failed_document_ids)),
    }


def audit_sections(
    *,
    section_groups: dict[tuple[str, str], pd.DataFrame],
    documents_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    section_results: list[dict[str, Any]] = []
    section_reconstruction_passed = 0
    section_reconstruction_failed = 0
    section_sequence_validation_passed = 0
    section_sequence_validation_failed = 0
    section_neighbor_validation_passed = 0
    section_neighbor_validation_failed = 0
    failed_section_ids: list[str] = []

    for (document_id, section_id), group in section_groups.items():
        sorted_group = group.sort_values("chunk_index_in_section").reset_index(drop=True)
        document_record = documents_by_id.get(document_id)
        full_text = normalize_text(document_record.get("full_text")) if document_record else ""
        messages: list[str] = []

        missing_section_fields = [
            field_name
            for field_name in REQUIRED_SECTION_FIELDS
            if int(sorted_group[field_name].map(is_missing).sum()) > 0
        ]
        if missing_section_fields:
            messages.append(f"Missing required section fields: {', '.join(missing_section_fields)}.")

        section_index_values = sorted_group["chunk_index_in_section"].tolist()
        section_index_base, chunk_index_sequence_ok = determine_index_base(section_index_values)
        total_chunks_match = sorted_group["total_chunks_in_section"].nunique(dropna=False) == 1 and int(
            sorted_group["total_chunks_in_section"].iloc[0]
        ) == len(sorted_group)
        consistent_section_type = sorted_group["section_type"].map(normalize_text).nunique(dropna=False) == 1
        consistent_section_index = sorted_group["section_index"].nunique(dropna=False) == 1
        consistent_section_start = sorted_group["section_char_start"].nunique(dropna=False) == 1
        consistent_section_end = sorted_group["section_char_end"].nunique(dropna=False) == 1
        section_sequence_ok = (
            chunk_index_sequence_ok
            and total_chunks_match
            and consistent_section_type
            and consistent_section_index
            and consistent_section_start
            and consistent_section_end
        )

        if section_sequence_ok:
            section_sequence_validation_passed += 1
        else:
            section_sequence_validation_failed += 1
            if not chunk_index_sequence_ok:
                messages.append("chunk_index_in_section sequence is not continuous.")
            if not total_chunks_match:
                messages.append("total_chunks_in_section does not match the actual section chunk count.")
            if not consistent_section_type:
                messages.append("section_type is inconsistent within the section.")
            if not consistent_section_index:
                messages.append("section_index is inconsistent within the section.")
            if not consistent_section_start or not consistent_section_end:
                messages.append("section char boundaries are inconsistent within the section.")

        neighbor_links_ok = True
        for index, chunk_row in sorted_group.iterrows():
            expected_previous = normalize_text(sorted_group.iloc[index - 1]["chunk_id"]) if index > 0 else ""
            expected_next = normalize_text(sorted_group.iloc[index + 1]["chunk_id"]) if index + 1 < len(sorted_group) else ""
            actual_previous = normalize_text(chunk_row.get("previous_section_chunk_id"))
            actual_next = normalize_text(chunk_row.get("next_section_chunk_id"))
            if actual_previous != expected_previous:
                neighbor_links_ok = False
                messages.append("previous_section_chunk_id links are invalid for this section.")
                break
            if actual_next != expected_next:
                neighbor_links_ok = False
                messages.append("next_section_chunk_id links are invalid for this section.")
                break

        if neighbor_links_ok:
            section_neighbor_validation_passed += 1
        else:
            section_neighbor_validation_failed += 1

        if document_record is None:
            reconstruction_ok = False
            messages.append("Section references an unknown document_id.")
            section_reconstruction_failed += 1
            section_start = 0
            section_end = 0
        else:
            section_start = int(sorted_group.iloc[0].get("section_char_start", 0))
            section_end = int(sorted_group.iloc[0].get("section_char_end", 0))
            original_section_text = full_text[section_start:section_end]
            reconstructed = "".join(normalize_text(value) for value in sorted_group["chunk_text"].tolist())
            contiguous = True
            expected_start = section_start
            for _, chunk_row in sorted_group.iterrows():
                chunk_start = int(chunk_row.get("chunk_char_start", 0))
                chunk_end = int(chunk_row.get("chunk_char_end", 0))
                if chunk_start != expected_start or chunk_start < section_start or chunk_end > section_end:
                    contiguous = False
                    break
                expected_start = chunk_end
            if expected_start != section_end:
                contiguous = False
            reconstruction_ok = contiguous and reconstructed == original_section_text
            if reconstruction_ok:
                section_reconstruction_passed += 1
            else:
                section_reconstruction_failed += 1
                messages.append("Section chunks do not exactly reconstruct the section text.")

        status = "FAIL" if messages else "PASS"
        if status == "FAIL":
            failed_section_ids.append(section_id)

        section_results.append(
            {
                "document_id": document_id,
                "section_id": section_id,
                "section_type": normalize_text(sorted_group.iloc[0].get("section_type")),
                "section_index": None if is_missing(sorted_group.iloc[0].get("section_index")) else int(sorted_group.iloc[0].get("section_index")),
                "chunk_count": len(sorted_group),
                "chunk_index_base": section_index_base,
                "section_sequence_ok": section_sequence_ok,
                "neighbor_links_ok": neighbor_links_ok,
                "reconstruction_ok": reconstruction_ok,
                "status": status,
                "messages": sorted(set(messages)),
            }
        )

    return section_results, {
        "section_reconstruction_passed": section_reconstruction_passed,
        "section_reconstruction_failed": section_reconstruction_failed,
        "section_sequence_validation_passed": section_sequence_validation_passed,
        "section_sequence_validation_failed": section_sequence_validation_failed,
        "section_neighbor_validation_passed": section_neighbor_validation_passed,
        "section_neighbor_validation_failed": section_neighbor_validation_failed,
        "failed_section_ids": sorted(set(failed_section_ids)),
    }


def build_overlong_chunk_audit(
    *,
    chunk_df: pd.DataFrame,
    document_results_by_id: dict[str, dict[str, Any]],
    section_results_by_key: dict[tuple[str, str], dict[str, Any]],
    boundary_violations_by_chunk_id: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    overlong_chunks: list[dict[str, Any]] = []
    for _, row in chunk_df.iterrows():
        chunk = row.to_dict()
        chunk_id = normalize_text(chunk.get("chunk_id"))
        chunk_text = normalize_text(chunk.get("chunk_text"))
        is_overlong = normalize_text(chunk.get("chunk_warning")) == "overlong_ns_paragraph" or len(chunk_text) > HARD_MAX_CHUNK_SIZE
        if not is_overlong:
            continue

        document_id = normalize_text(chunk.get("document_id"))
        section_id = normalize_text(chunk.get("section_id"))
        document_result = document_results_by_id.get(document_id, {})
        section_result = section_results_by_key.get((document_id, section_id), {})
        overlong_chunks.append(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "case_number": normalize_text(chunk.get("case_number")),
                "section_type": normalize_text(chunk.get("section_type")),
                "chunk_text_length": len(chunk_text),
                "is_standalone": int(chunk.get("paragraph_count", 0)) == 1,
                "document_reconstruction_ok": bool(document_result.get("reconstruction_ok")),
                "section_reconstruction_ok": bool(section_result.get("reconstruction_ok")),
                "crosses_section_boundary": bool(boundary_violations_by_chunk_id.get(chunk_id)),
                "first_300_chars": excerpt(
                    sanitize_human_excerpt(text=chunk_text, section_type=normalize_text(chunk.get("section_type"))),
                    300,
                ),
                "last_300_chars": excerpt_tail(
                    sanitize_human_excerpt(text=chunk_text, section_type=normalize_text(chunk.get("section_type"))),
                    300,
                ),
            }
        )
    return overlong_chunks


def build_preview_documents(
    *,
    documents_df: pd.DataFrame,
    document_groups: dict[str, pd.DataFrame],
    document_results_by_id: dict[str, dict[str, Any]],
    overlong_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    preview_reasons: dict[str, list[str]] = defaultdict(list)
    documents_df = documents_df.copy()
    documents_df["document_id"] = [compute_document_id(row.to_dict()) for _, row in documents_df.iterrows()]

    for document_id in (
        documents_df.sort_values("full_text_length", ascending=False)["document_id"].head(PREVIEW_LIMIT).tolist()
    ):
        preview_reasons[normalize_text(document_id)].append("top_10_longest_documents")

    chunk_count_pairs = sorted(
        ((document_id, len(group)) for document_id, group in document_groups.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    for document_id, _ in chunk_count_pairs[:PREVIEW_LIMIT]:
        preview_reasons[normalize_text(document_id)].append("top_10_documents_by_chunk_count")

    for item in overlong_chunks:
        preview_reasons[normalize_text(item["document_id"])].append("document_with_overlong_chunk")

    preview_documents: list[dict[str, Any]] = []
    for document_id in sorted(preview_reasons):
        chunk_group = document_groups.get(document_id)
        document_result = document_results_by_id.get(document_id, {})
        if chunk_group is None or not document_result:
            continue

        sorted_group = chunk_group.sort_values("chunk_index")
        chunk_table = []
        for _, chunk_row in sorted_group.iterrows():
            chunk_text = normalize_text(chunk_row.get("chunk_text"))
            sanitized_chunk_text = sanitize_human_excerpt(
                text=chunk_text,
                section_type=normalize_text(chunk_row.get("section_type")),
            )
            chunk_table.append(
                {
                    "chunk_index": int(chunk_row.get("chunk_index")),
                    "chunk_id": normalize_text(chunk_row.get("chunk_id")),
                    "section_type": normalize_text(chunk_row.get("section_type")),
                    "section_index": int(chunk_row.get("section_index")),
                    "chunk_index_in_section": int(chunk_row.get("chunk_index_in_section")),
                    "text_length": len(chunk_text),
                    "previous_chunk_id": normalize_text(chunk_row.get("previous_chunk_id")),
                    "next_chunk_id": normalize_text(chunk_row.get("next_chunk_id")),
                    "first_250_chars": excerpt(sanitized_chunk_text, 250),
                    "last_250_chars": excerpt_tail(sanitized_chunk_text, 250),
                }
            )

        preview_documents.append(
            {
                "preview_reasons": sorted(set(preview_reasons[document_id])),
                "document_id": document_id,
                "case_number": normalize_text(document_result.get("case_number")),
                "ecli": normalize_text(document_result.get("ecli")),
                "document_type": normalize_text(document_result.get("document_type")),
                "legal_area": normalize_text(document_result.get("legal_area")),
                "full_text_length": int(document_result.get("full_text_length", 0)),
                "total_chunks": int(document_result.get("chunk_count", 0)),
                "structure_status": normalize_text(document_result.get("structure_status")),
                "sections": document_result.get("sections", []),
                "chunk_table": chunk_table,
            }
        )

    return preview_documents


def decide_status(summary: dict[str, Any]) -> str:
    if (
        summary["document_reconstruction_failed"] > 0
        or summary["section_reconstruction_failed"] > 0
        or summary["documents_with_zero_chunks_count"] > 0
        or summary["chunk_metadata_validation_failed"] > 0
        or summary["duplicate_chunk_id_count"] > 0
        or summary["document_sequence_validation_failed"] > 0
        or summary["section_sequence_validation_failed"] > 0
        or summary["neighbor_link_validation_failed"] > 0
        or summary["section_boundary_violation_count"] > 0
        or summary["weak_structure_document_count"] > 0
        or summary["needs_review_document_count"] > 0
    ):
        return "FAIL"

    if summary["overlong_chunk_count"] > 0 or summary["medium_structure_document_count"] > 0:
        return "WARN"

    return "PASS"


def build_summary(
    *,
    documents_df: pd.DataFrame,
    chunk_df: pd.DataFrame,
    section_results: list[dict[str, Any]],
    document_metrics: dict[str, Any],
    section_metrics: dict[str, Any],
    chunk_metrics: dict[str, Any],
    overlong_chunks: list[dict[str, Any]],
    boundary_violations: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = {
        "total_documents_validated": int(len(documents_df)),
        "total_chunks_validated": int(len(chunk_df)),
        "total_sections_validated": int(len(section_results)),
        "document_reconstruction_passed": int(document_metrics["document_reconstruction_passed"]),
        "document_reconstruction_failed": int(document_metrics["document_reconstruction_failed"]),
        "section_reconstruction_passed": int(section_metrics["section_reconstruction_passed"]),
        "section_reconstruction_failed": int(section_metrics["section_reconstruction_failed"]),
        "chunk_metadata_validation_passed": int(chunk_metrics["chunk_metadata_validation_passed"]),
        "chunk_metadata_validation_failed": int(chunk_metrics["chunk_metadata_validation_failed"]),
        "document_sequence_validation_passed": int(document_metrics["document_sequence_validation_passed"]),
        "document_sequence_validation_failed": int(document_metrics["document_sequence_validation_failed"]),
        "section_sequence_validation_passed": int(section_metrics["section_sequence_validation_passed"]),
        "section_sequence_validation_failed": int(section_metrics["section_sequence_validation_failed"]),
        "document_neighbor_link_validation_passed": int(document_metrics["document_neighbor_validation_passed"]),
        "document_neighbor_link_validation_failed": int(document_metrics["document_neighbor_validation_failed"]),
        "section_neighbor_link_validation_passed": int(section_metrics["section_neighbor_validation_passed"]),
        "section_neighbor_link_validation_failed": int(section_metrics["section_neighbor_validation_failed"]),
        "neighbor_link_validation_passed": int(
            document_metrics["document_neighbor_validation_passed"] + section_metrics["section_neighbor_validation_passed"]
        ),
        "neighbor_link_validation_failed": int(
            document_metrics["document_neighbor_validation_failed"] + section_metrics["section_neighbor_validation_failed"]
        ),
        "section_boundary_violation_count": int(len(boundary_violations)),
        "overlong_chunk_count": int(len(overlong_chunks)),
        "medium_structure_document_count": int(len(document_metrics["medium_structure_documents"])),
        "weak_structure_document_count": int(len(document_metrics["weak_structure_documents"])),
        "needs_review_document_count": int(len(document_metrics["needs_review_documents"])),
        "documents_with_zero_chunks_count": int(len(document_metrics["documents_with_zero_chunks"])),
        "duplicate_chunk_id_count": int(chunk_metrics["duplicate_chunk_id_count"]),
        "empty_chunk_count": int(chunk_metrics["empty_chunk_count"]),
        "broken_word_warning_count": int(chunk_metrics["broken_word_warning_count"]),
        "failed_document_ids": document_metrics["failed_document_ids"],
        "failed_section_ids": section_metrics["failed_section_ids"],
        "failed_chunk_ids": chunk_metrics["failed_chunk_ids"],
        "medium_structure_documents": document_metrics["medium_structure_documents"],
        "weak_structure_documents": document_metrics["weak_structure_documents"],
        "needs_review_documents": document_metrics["needs_review_documents"],
        "documents_with_zero_chunks": document_metrics["documents_with_zero_chunks"],
        "duplicated_chunk_ids": chunk_metrics["duplicated_chunk_ids"],
    }
    summary["status"] = decide_status(summary)
    return summary


def build_markdown_report(
    *,
    summary: dict[str, Any],
    out_json_path: Path,
    documents_path: Path,
    chunks_path: Path,
    document_results: list[dict[str, Any]],
    section_results: list[dict[str, Any]],
    chunk_results: list[dict[str, Any]],
    overlong_chunks: list[dict[str, Any]],
    boundary_violations: list[dict[str, Any]],
    preview_documents: list[dict[str, Any]],
) -> str:
    failed_documents = [item for item in document_results if item["status"] == "FAIL"]
    failed_sections = [item for item in section_results if item["status"] == "FAIL"]
    failed_chunks = [item for item in chunk_results if item["status"] == "FAIL"]

    lines = [
        "# NSoud Chunking Quality Audit",
        "",
        f"- Audit status: **{summary['status']}**",
        f"- Documents input: `{documents_path}`",
        f"- Chunks input: `{chunks_path}`",
        f"- JSON output: `{out_json_path}`",
        "",
        "## Summary",
        f"- total documents validated: {summary['total_documents_validated']}",
        f"- total chunks validated: {summary['total_chunks_validated']}",
        f"- total sections validated: {summary['total_sections_validated']}",
        f"- document reconstruction passed/failed: {summary['document_reconstruction_passed']}/{summary['document_reconstruction_failed']}",
        f"- section reconstruction passed/failed: {summary['section_reconstruction_passed']}/{summary['section_reconstruction_failed']}",
        f"- chunk metadata validation passed/failed: {summary['chunk_metadata_validation_passed']}/{summary['chunk_metadata_validation_failed']}",
        f"- document sequence validation passed/failed: {summary['document_sequence_validation_passed']}/{summary['document_sequence_validation_failed']}",
        f"- section sequence validation passed/failed: {summary['section_sequence_validation_passed']}/{summary['section_sequence_validation_failed']}",
        f"- neighbor link validation passed/failed: {summary['neighbor_link_validation_passed']}/{summary['neighbor_link_validation_failed']}",
        f"- section boundary violation count: {summary['section_boundary_violation_count']}",
        f"- overlong chunk count: {summary['overlong_chunk_count']}",
        f"- medium structure document count: {summary['medium_structure_document_count']}",
        f"- weak structure document count: {summary['weak_structure_document_count']}",
        f"- needs_review document count: {summary['needs_review_document_count']}",
        f"- duplicate chunk_id count: {summary['duplicate_chunk_id_count']}",
        f"- empty chunk_text count: {summary['empty_chunk_count']}",
        "",
        "## Failed Documents",
    ]

    if failed_documents:
        lines.extend(f"- {item['document_id']}: {'; '.join(item['messages'])}" for item in failed_documents)
    else:
        lines.append("- none")

    lines.extend(["", "## Failed Sections"])
    if failed_sections:
        lines.extend(
            f"- {item['section_id']} ({item['document_id']}): {'; '.join(item['messages'])}" for item in failed_sections
        )
    else:
        lines.append("- none")

    lines.extend(["", "## Failed Chunks"])
    if failed_chunks:
        lines.extend(f"- {item['chunk_id']}: {'; '.join(item['messages'])}" for item in failed_chunks)
    else:
        lines.append("- none")

    lines.extend(["", "## Boundary Violations"])
    if boundary_violations:
        lines.extend(
            f"- {item['document_id']} / {item['chunk_id']}: {item['message']}" for item in boundary_violations
        )
    else:
        lines.append("- none")

    lines.extend(["", "## Overlong Chunks"])
    if overlong_chunks:
        for item in overlong_chunks:
            lines.extend(
                [
                    f"### {item['chunk_id']}",
                    f"- document_id: {item['document_id']}",
                    f"- case_number: {item['case_number']}",
                    f"- section_type: {item['section_type']}",
                    f"- chunk_text_length: {item['chunk_text_length']}",
                    f"- is_standalone: {item['is_standalone']}",
                    f"- document_reconstruction_ok: {item['document_reconstruction_ok']}",
                    f"- section_reconstruction_ok: {item['section_reconstruction_ok']}",
                    f"- crosses_section_boundary: {item['crosses_section_boundary']}",
                    "",
                    "```text",
                    item["first_300_chars"],
                    "...",
                    item["last_300_chars"],
                    "```",
                    "",
                ]
            )
    else:
        lines.append("- none")

    lines.extend(["## Medium Structure Documents"])
    if summary["medium_structure_documents"]:
        lines.extend(f"- {document_id}" for document_id in summary["medium_structure_documents"])
    else:
        lines.append("- none")

    lines.extend(["", "## Preview Documents"])
    if not preview_documents:
        lines.append("- none")
    else:
        for document in preview_documents:
            lines.extend(
                [
                    f"### {document['document_id']}",
                    f"- preview_reasons: {', '.join(document['preview_reasons'])}",
                    f"- case_number: {document['case_number']}",
                    f"- ecli: {document['ecli']}",
                    f"- document_type: {document['document_type']}",
                    f"- legal_area: {document['legal_area']}",
                    f"- full_text_length: {document['full_text_length']}",
                    f"- total_chunks: {document['total_chunks']}",
                    f"- structure_status: {document['structure_status']}",
                    "",
                    "Sections:",
                ]
            )
            if document["sections"]:
                lines.extend(
                    f"- {section['section_index']}: {section['section_type']} ({section['section_id']})"
                    for section in document["sections"]
                )
            else:
                lines.append("- none")
            lines.append("")
            lines.extend(
                render_markdown_table(
                    [
                        "chunk_index",
                        "chunk_id",
                        "section_type",
                        "section_index",
                        "chunk_index_in_section",
                        "text length",
                        "previous_chunk_id",
                        "next_chunk_id",
                        "first 250 chars",
                        "last 250 chars",
                    ],
                    [
                        [
                            item["chunk_index"],
                            item["chunk_id"],
                            item["section_type"],
                            item["section_index"],
                            item["chunk_index_in_section"],
                            item["text_length"],
                            item["previous_chunk_id"],
                            item["next_chunk_id"],
                            item["first_250_chars"],
                            item["last_250_chars"],
                        ]
                        for item in document["chunk_table"]
                    ],
                )
            )
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()

    if pyarrow is None:
        print("audit status: FAIL")
        print("error: pyarrow is required for Parquet input support.")
        return 1

    try:
        documents_df = load_documents(args.documents)
        chunk_df = load_chunks(args.chunks)
    except Exception as exc:
        print("audit status: FAIL")
        print(f"error: {exc}")
        return 1

    documents_by_id = build_document_index(documents_df)
    document_groups = build_document_groups(chunk_df)
    section_groups = build_section_groups(chunk_df)
    orphan_chunk_document_ids = sorted(set(document_groups) - set(documents_by_id))

    structure_info_by_document_id: dict[str, dict[str, Any]] = {}
    for document_id, record in documents_by_id.items():
        structure, boundaries = build_structure_boundaries(record)
        structure_info_by_document_id[document_id] = {"structure": structure, "boundaries": boundaries}

    document_results, boundary_violations, document_metrics = audit_documents(
        documents_df=documents_df,
        document_groups=document_groups,
        structure_info_by_document_id=structure_info_by_document_id,
    )
    section_results, section_metrics = audit_sections(
        section_groups=section_groups,
        documents_by_id=documents_by_id,
    )
    chunk_results, chunk_metrics = audit_chunk_metadata(
        chunk_df=chunk_df,
        documents_by_id=documents_by_id,
        boundary_violations_by_chunk_id=document_metrics["boundary_violations_by_chunk_id"],
    )

    if orphan_chunk_document_ids:
        document_metrics["failed_document_ids"] = sorted(
            set(document_metrics["failed_document_ids"]) | set(orphan_chunk_document_ids)
        )

    document_results_by_id = {item["document_id"]: item for item in document_results}
    section_results_by_key = {(item["document_id"], item["section_id"]): item for item in section_results}
    overlong_chunks = build_overlong_chunk_audit(
        chunk_df=chunk_df,
        document_results_by_id=document_results_by_id,
        section_results_by_key=section_results_by_key,
        boundary_violations_by_chunk_id=document_metrics["boundary_violations_by_chunk_id"],
    )
    preview_documents = build_preview_documents(
        documents_df=documents_df,
        document_groups=document_groups,
        document_results_by_id=document_results_by_id,
        overlong_chunks=overlong_chunks,
    )

    summary = build_summary(
        documents_df=documents_df,
        chunk_df=chunk_df,
        section_results=section_results,
        document_metrics=document_metrics,
        section_metrics=section_metrics,
        chunk_metrics=chunk_metrics,
        overlong_chunks=overlong_chunks,
        boundary_violations=boundary_violations,
    )
    if orphan_chunk_document_ids:
        summary["status"] = "FAIL"
        summary["orphan_chunk_document_ids"] = orphan_chunk_document_ids
    else:
        summary["orphan_chunk_document_ids"] = []

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = {
        "status": summary["status"],
        "created_at": created_at,
        "inputs": {
            "documents": str(args.documents),
            "chunks": str(args.chunks),
        },
        "summary": summary,
        "document_results": document_results,
        "section_results": section_results,
        "chunk_results": chunk_results,
        "overlong_chunks": overlong_chunks,
        "boundary_violations": boundary_violations,
        "preview_documents": preview_documents,
    }

    markdown_report = build_markdown_report(
        summary=summary,
        out_json_path=args.out_json,
        documents_path=args.documents,
        chunks_path=args.chunks,
        document_results=document_results,
        section_results=section_results,
        chunk_results=chunk_results,
        overlong_chunks=overlong_chunks,
        boundary_violations=boundary_violations,
        preview_documents=preview_documents,
    )

    write_json(args.out_json, payload)
    write_markdown(args.out_md, markdown_report)

    print(f"audit status: {summary['status']}")
    print(f"total documents validated: {summary['total_documents_validated']}")
    print(f"total chunks validated: {summary['total_chunks_validated']}")
    print(f"total sections validated: {summary['total_sections_validated']}")
    print(
        "document reconstruction passed/failed: "
        f"{summary['document_reconstruction_passed']}/{summary['document_reconstruction_failed']}"
    )
    print(
        "section reconstruction passed/failed: "
        f"{summary['section_reconstruction_passed']}/{summary['section_reconstruction_failed']}"
    )
    print(
        "chunk metadata validation passed/failed: "
        f"{summary['chunk_metadata_validation_passed']}/{summary['chunk_metadata_validation_failed']}"
    )
    print(
        "document sequence validation passed/failed: "
        f"{summary['document_sequence_validation_passed']}/{summary['document_sequence_validation_failed']}"
    )
    print(
        "section sequence validation passed/failed: "
        f"{summary['section_sequence_validation_passed']}/{summary['section_sequence_validation_failed']}"
    )
    print(
        "neighbor link validation passed/failed: "
        f"{summary['neighbor_link_validation_passed']}/{summary['neighbor_link_validation_failed']}"
    )
    print(f"section boundary violation count: {summary['section_boundary_violation_count']}")
    print(f"overlong chunk count: {summary['overlong_chunk_count']}")
    print(f"medium structure document count: {summary['medium_structure_document_count']}")
    print(f"output json path: {args.out_json}")
    print(f"output markdown path: {args.out_md}")
    print(f"changed files: {', '.join(CHANGED_FILES)}")
    return 1 if summary["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
