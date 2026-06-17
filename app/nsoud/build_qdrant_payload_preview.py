from __future__ import annotations

import argparse
import hashlib
import json
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


PAYLOAD_FIELDS = [
    "point_id",
    "text",
    "source",
    "provider",
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
    "chunk_text_length",
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
REQUIRED_COLUMNS = {
    "source",
    "court",
    "authority_level",
    "case_number",
    "url",
    "source_attribution",
    "content_hash",
    "document_id",
    "chunk_id",
    "chunk_index",
    "total_chunks_in_document",
    "section_id",
    "section_type",
    "section_index",
    "chunk_index_in_section",
    "total_chunks_in_section",
    "chunk_text",
    "chunk_text_length",
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
}
REQUIRED_NONEMPTY_FIELDS = [
    "point_id",
    "text",
    "source",
    "provider",
    "court",
    "authority_level",
    "case_number",
    "url",
    "source_attribution",
    "content_hash",
    "document_id",
    "chunk_id",
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


@dataclass(frozen=True)
class PayloadPreviewSummary:
    payload_preview_status: str
    validation_status: str
    total_rows: int
    duplicate_point_id_count: int
    duplicate_chunk_id_count: int
    empty_text_count: int
    missing_required_metadata_count: int
    document_sequence_validation_passed: int
    document_sequence_validation_failed: int
    section_sequence_validation_passed: int
    section_sequence_validation_failed: int
    document_neighbor_validation_passed: int
    document_neighbor_validation_failed: int
    section_neighbor_validation_passed: int
    section_neighbor_validation_failed: int
    output_parquet_path: Path
    output_jsonl_path: Path
    validation_report_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NSoud Qdrant payload preview from chunk parquet.")
    parser.add_argument("--input", type=Path, required=True, help="Input NSoud chunks parquet path.")
    parser.add_argument("--out-parquet", type=Path, required=True, help="Output payload preview parquet path.")
    parser.add_argument("--out-jsonl", type=Path, required=True, help="Output payload preview JSONL path.")
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


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def validation_path_for_output(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}_validation.md")


def load_chunks(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def stable_point_id(chunk_id: str) -> str:
    return hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()


def validate_input_columns(chunks_df: pd.DataFrame) -> list[str]:
    missing_columns = sorted(column_name for column_name in REQUIRED_COLUMNS if column_name not in chunks_df.columns)
    if not missing_columns:
        return []
    return [f"Missing required chunk columns: {', '.join(missing_columns)}."]


def build_payload_dataframe(chunks_df: pd.DataFrame) -> pd.DataFrame:
    payload_df = pd.DataFrame(
        {
            "point_id": chunks_df["chunk_id"].map(lambda value: stable_point_id(normalize_text(value))),
            "text": chunks_df["chunk_text"].map(normalize_text),
            "source": chunks_df["source"],
            "provider": "nsoud",
            "court": chunks_df["court"],
            "authority_level": chunks_df["authority_level"],
            "case_number": chunks_df["case_number"],
            "ecli": chunks_df["ecli"],
            "decision_date": chunks_df["decision_date"],
            "publication_date": chunks_df["publication_date"],
            "document_type": chunks_df["document_type"],
            "legal_area": chunks_df["legal_area"],
            "title": chunks_df["title"],
            "url": chunks_df["url"],
            "source_attribution": chunks_df["source_attribution"],
            "content_hash": chunks_df["content_hash"],
            "document_id": chunks_df["document_id"],
            "chunk_id": chunks_df["chunk_id"],
            "chunk_index": chunks_df["chunk_index"],
            "total_chunks_in_document": chunks_df["total_chunks_in_document"],
            "section_id": chunks_df["section_id"],
            "section_type": chunks_df["section_type"],
            "section_index": chunks_df["section_index"],
            "chunk_index_in_section": chunks_df["chunk_index_in_section"],
            "total_chunks_in_section": chunks_df["total_chunks_in_section"],
            "previous_chunk_id": chunks_df["previous_chunk_id"],
            "next_chunk_id": chunks_df["next_chunk_id"],
            "previous_section_chunk_id": chunks_df["previous_section_chunk_id"],
            "next_section_chunk_id": chunks_df["next_section_chunk_id"],
            "chunk_text_length": chunks_df["chunk_text_length"],
            "paragraph_count": chunks_df["paragraph_count"],
            "chunk_warning": chunks_df["chunk_warning"],
            "ns_section_hint": chunks_df["ns_section_hint"],
            "structure_confidence": chunks_df["structure_confidence"],
            "structure_status": chunks_df["structure_status"],
            "structure_needs_review": chunks_df["structure_needs_review"],
            "detected_section_order": chunks_df["detected_section_order"],
            "detected_markers": chunks_df["detected_markers"],
            "section_source": chunks_df["section_source"],
            "chunking_strategy": chunks_df["chunking_strategy"],
        }
    )
    return payload_df.reindex(columns=PAYLOAD_FIELDS)


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


def write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", index=False)


def write_jsonl(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in df.to_dict(orient="records"):
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def validate_document_sequences(df: pd.DataFrame) -> tuple[int, int, int, int, list[str]]:
    sequence_passed = 0
    sequence_failed = 0
    neighbor_passed = 0
    neighbor_failed = 0
    failures: list[str] = []

    for document_id, group in df.groupby("document_id", sort=False):
        sorted_group = group.sort_values("chunk_index").reset_index(drop=True)
        actual_sequence = sorted_group["chunk_index"].astype(int).tolist()
        expected_sequence = list(range(len(sorted_group)))
        total_chunks_values = sorted_group["total_chunks_in_document"].astype(int).unique().tolist()
        if actual_sequence == expected_sequence and total_chunks_values == [len(sorted_group)]:
            sequence_passed += 1
        else:
            sequence_failed += 1
            failures.append(f"Document `{document_id}` has invalid chunk_index or total_chunks_in_document metadata.")

        links_ok = True
        for index, row in sorted_group.iterrows():
            expected_previous = normalize_text(sorted_group.iloc[index - 1]["chunk_id"]) if index > 0 else ""
            expected_next = normalize_text(sorted_group.iloc[index + 1]["chunk_id"]) if index + 1 < len(sorted_group) else ""
            actual_previous = normalize_text(row["previous_chunk_id"])
            actual_next = normalize_text(row["next_chunk_id"])
            if actual_previous != expected_previous or actual_next != expected_next:
                links_ok = False
                failures.append(f"Document `{document_id}` has invalid previous_chunk_id/next_chunk_id links.")
                break

        if links_ok:
            neighbor_passed += 1
        else:
            neighbor_failed += 1

    return sequence_passed, sequence_failed, neighbor_passed, neighbor_failed, failures


def validate_section_sequences(df: pd.DataFrame) -> tuple[int, int, int, int, list[str]]:
    sequence_passed = 0
    sequence_failed = 0
    neighbor_passed = 0
    neighbor_failed = 0
    failures: list[str] = []

    for (document_id, section_id), group in df.groupby(["document_id", "section_id"], sort=False):
        sorted_group = group.sort_values("chunk_index_in_section").reset_index(drop=True)
        actual_sequence = sorted_group["chunk_index_in_section"].astype(int).tolist()
        expected_sequence = list(range(len(sorted_group)))
        total_chunks_values = sorted_group["total_chunks_in_section"].astype(int).unique().tolist()
        if actual_sequence == expected_sequence and total_chunks_values == [len(sorted_group)]:
            sequence_passed += 1
        else:
            sequence_failed += 1
            failures.append(
                f"Section `{section_id}` in document `{document_id}` has invalid chunk_index_in_section or total_chunks_in_section metadata."
            )

        links_ok = True
        for index, row in sorted_group.iterrows():
            expected_previous = normalize_text(sorted_group.iloc[index - 1]["chunk_id"]) if index > 0 else ""
            expected_next = normalize_text(sorted_group.iloc[index + 1]["chunk_id"]) if index + 1 < len(sorted_group) else ""
            actual_previous = normalize_text(row["previous_section_chunk_id"])
            actual_next = normalize_text(row["next_section_chunk_id"])
            if actual_previous != expected_previous or actual_next != expected_next:
                links_ok = False
                failures.append(
                    f"Section `{section_id}` in document `{document_id}` has invalid previous_section_chunk_id/next_section_chunk_id links."
                )
                break

        if links_ok:
            neighbor_passed += 1
        else:
            neighbor_failed += 1

    return sequence_passed, sequence_failed, neighbor_passed, neighbor_failed, failures


def validate_payload_dataframe(
    df: pd.DataFrame,
) -> tuple[str, dict[str, int], list[str], int, int, int, int, int, int, int, int, int, int]:
    failures: list[str] = []
    missing_field_counts: dict[str, int] = {}

    duplicate_point_id_count = int(df["point_id"].duplicated(keep=False).sum()) if not df.empty else 0
    duplicate_chunk_id_count = int(df["chunk_id"].duplicated(keep=False).sum()) if not df.empty else 0
    empty_text_count = int(df["text"].map(lambda value: normalize_text(value).strip() == "").sum()) if not df.empty else 0

    if duplicate_point_id_count > 0:
        failures.append("Duplicate point_id values detected.")
    if duplicate_chunk_id_count > 0:
        failures.append("Duplicate chunk_id values detected.")
    if empty_text_count > 0:
        failures.append("One or more payload rows have empty text.")

    required_missing_rows = pd.Series([False] * len(df))
    for field_name in REQUIRED_NONEMPTY_FIELDS:
        field_missing = df[field_name].map(is_missing)
        missing_field_counts[field_name] = int(field_missing.sum())
        required_missing_rows = required_missing_rows | field_missing
        if int(field_missing.sum()) > 0:
            failures.append(f"Required payload field `{field_name}` contains missing values.")

    for field_name in NULLABLE_LINK_FIELDS:
        missing_field_counts[field_name] = int(df[field_name].isna().sum())

    for field_name in ("ecli", "decision_date", "publication_date", "document_type", "legal_area", "title", "chunk_warning"):
        missing_field_counts[field_name] = int(df[field_name].map(is_missing).sum())

    missing_required_metadata_count = int(required_missing_rows.sum())

    invalid_chunking_strategy_count = int(
        (df["chunking_strategy"].map(normalize_text) != "document_section_aware").sum()
    ) if not df.empty else 0
    if invalid_chunking_strategy_count > 0:
        failures.append("One or more payload rows have invalid chunking_strategy values.")

    document_sequence_passed, document_sequence_failed, document_neighbor_passed, document_neighbor_failed, document_failures = (
        validate_document_sequences(df)
    )
    section_sequence_passed, section_sequence_failed, section_neighbor_passed, section_neighbor_failed, section_failures = (
        validate_section_sequences(df)
    )
    failures.extend(document_failures)
    failures.extend(section_failures)

    validation_status = "FAIL" if failures else "PASS"
    return (
        validation_status,
        missing_field_counts,
        sorted(set(failures)),
        duplicate_point_id_count,
        duplicate_chunk_id_count,
        empty_text_count,
        missing_required_metadata_count,
        document_sequence_passed,
        document_sequence_failed,
        section_sequence_passed,
        section_sequence_failed,
        document_neighbor_passed,
        document_neighbor_failed,
        section_neighbor_passed,
        section_neighbor_failed,
    )


def build_validation_report(
    df: pd.DataFrame,
    *,
    input_path: Path,
    output_parquet_path: Path,
    output_jsonl_path: Path,
    validation_status: str,
    missing_field_counts: dict[str, int],
    failures: list[str],
    duplicate_point_id_count: int,
    duplicate_chunk_id_count: int,
    empty_text_count: int,
    missing_required_metadata_count: int,
    document_sequence_validation_passed: int,
    document_sequence_validation_failed: int,
    section_sequence_validation_passed: int,
    section_sequence_validation_failed: int,
    document_neighbor_validation_passed: int,
    document_neighbor_validation_failed: int,
    section_neighbor_validation_passed: int,
    section_neighbor_validation_failed: int,
) -> str:
    text_lengths = df["chunk_text_length"].astype(int).tolist() if not df.empty else []
    lines = [
        "# NSoud Qdrant Payload Preview Validation",
        "",
        f"- Input: `{input_path}`",
        f"- Output Parquet: `{output_parquet_path}`",
        f"- Output JSONL: `{output_jsonl_path}`",
        f"- Validation status: **{validation_status}**",
        f"- Total payload rows: **{len(df)}**",
        f"- Duplicate point_id count: **{duplicate_point_id_count}**",
        f"- Duplicate chunk_id count: **{duplicate_chunk_id_count}**",
        f"- Empty text count: **{empty_text_count}**",
        f"- Missing required metadata count: **{missing_required_metadata_count}**",
        f"- Document sequence validation passed/failed: **{document_sequence_validation_passed}/{document_sequence_validation_failed}**",
        f"- Section sequence validation passed/failed: **{section_sequence_validation_passed}/{section_sequence_validation_failed}**",
        f"- Document neighbor validation passed/failed: **{document_neighbor_validation_passed}/{document_neighbor_validation_failed}**",
        f"- Section neighbor validation passed/failed: **{section_neighbor_validation_passed}/{section_neighbor_validation_failed}**",
        "",
        "## Status",
    ]
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- Payload preview validation passed.")

    lines.extend(
        [
            "",
            "## Missing Field Counts",
            "",
            "| Field | Missing Count |",
            "| --- | ---: |",
        ]
    )
    for field_name in PAYLOAD_FIELDS:
        lines.append(f"| `{field_name}` | {missing_field_counts.get(field_name, 0)} |")

    lines.extend(
        [
            "",
            "## Text Lengths",
            f"- min: {min(text_lengths) if text_lengths else 0}",
            f"- max: {max(text_lengths) if text_lengths else 0}",
            f"- avg: {mean(text_lengths):.2f}" if text_lengths else "- avg: 0.00",
            "",
        ]
    )
    lines.extend(render_distribution_table("Document Type Distribution", distribution_counts(df, "document_type")))
    lines.extend(render_distribution_table("Legal Area Distribution", distribution_counts(df, "legal_area")))
    lines.extend(render_distribution_table("Section Type Distribution", distribution_counts(df, "section_type")))
    lines.extend(render_distribution_table("Structure Status Distribution", distribution_counts(df, "structure_status")))
    lines.extend(render_distribution_table("Chunk Warning Distribution", distribution_counts(df, "chunk_warning")))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    if pyarrow is None:
        print("payload preview status: FAIL")
        print("error: pyarrow is required for Parquet output.")
        print("install command: pip install pyarrow")
        return 1

    validation_path = validation_path_for_output(args.out_parquet)

    try:
        chunks_df = load_chunks(args.input)
    except Exception as exc:
        print("payload preview status: FAIL")
        print(f"error: {exc}")
        return 1

    input_errors = validate_input_columns(chunks_df)
    if input_errors:
        report = build_validation_report(
            pd.DataFrame(columns=PAYLOAD_FIELDS),
            input_path=args.input,
            output_parquet_path=args.out_parquet,
            output_jsonl_path=args.out_jsonl,
            validation_status="FAIL",
            missing_field_counts={field_name: 0 for field_name in PAYLOAD_FIELDS},
            failures=input_errors,
            duplicate_point_id_count=0,
            duplicate_chunk_id_count=0,
            empty_text_count=0,
            missing_required_metadata_count=0,
            document_sequence_validation_passed=0,
            document_sequence_validation_failed=0,
            section_sequence_validation_passed=0,
            section_sequence_validation_failed=0,
            document_neighbor_validation_passed=0,
            document_neighbor_validation_failed=0,
            section_neighbor_validation_passed=0,
            section_neighbor_validation_failed=0,
        )
        validation_path.write_text(report, encoding="utf-8")
        print("payload preview status: FAIL")
        print("error: input chunk parquet is missing required columns.")
        print(f"validation report path: {validation_path}")
        return 1

    try:
        payload_df = build_payload_dataframe(chunks_df)
        write_parquet(payload_df, args.out_parquet)
        write_jsonl(payload_df, args.out_jsonl)
    except Exception as exc:
        print("payload preview status: FAIL")
        print(f"error: {exc}")
        return 1

    (
        validation_status,
        missing_field_counts,
        failures,
        duplicate_point_id_count,
        duplicate_chunk_id_count,
        empty_text_count,
        missing_required_metadata_count,
        document_sequence_passed,
        document_sequence_failed,
        section_sequence_passed,
        section_sequence_failed,
        document_neighbor_passed,
        document_neighbor_failed,
        section_neighbor_passed,
        section_neighbor_failed,
    ) = validate_payload_dataframe(payload_df)

    report = build_validation_report(
        payload_df,
        input_path=args.input,
        output_parquet_path=args.out_parquet,
        output_jsonl_path=args.out_jsonl,
        validation_status=validation_status,
        missing_field_counts=missing_field_counts,
        failures=failures,
        duplicate_point_id_count=duplicate_point_id_count,
        duplicate_chunk_id_count=duplicate_chunk_id_count,
        empty_text_count=empty_text_count,
        missing_required_metadata_count=missing_required_metadata_count,
        document_sequence_validation_passed=document_sequence_passed,
        document_sequence_validation_failed=document_sequence_failed,
        section_sequence_validation_passed=section_sequence_passed,
        section_sequence_validation_failed=section_sequence_failed,
        document_neighbor_validation_passed=document_neighbor_passed,
        document_neighbor_validation_failed=document_neighbor_failed,
        section_neighbor_validation_passed=section_neighbor_passed,
        section_neighbor_validation_failed=section_neighbor_failed,
    )
    validation_path.write_text(report, encoding="utf-8")

    summary = PayloadPreviewSummary(
        payload_preview_status="PASS",
        validation_status=validation_status,
        total_rows=len(payload_df),
        duplicate_point_id_count=duplicate_point_id_count,
        duplicate_chunk_id_count=duplicate_chunk_id_count,
        empty_text_count=empty_text_count,
        missing_required_metadata_count=missing_required_metadata_count,
        document_sequence_validation_passed=document_sequence_passed,
        document_sequence_validation_failed=document_sequence_failed,
        section_sequence_validation_passed=section_sequence_passed,
        section_sequence_validation_failed=section_sequence_failed,
        document_neighbor_validation_passed=document_neighbor_passed,
        document_neighbor_validation_failed=document_neighbor_failed,
        section_neighbor_validation_passed=section_neighbor_passed,
        section_neighbor_validation_failed=section_neighbor_failed,
        output_parquet_path=args.out_parquet,
        output_jsonl_path=args.out_jsonl,
        validation_report_path=validation_path,
    )
    print(f"payload preview status: {summary.payload_preview_status}")
    print(f"validation status: {summary.validation_status}")
    print(f"total rows: {summary.total_rows}")
    print(f"duplicate point_id count: {summary.duplicate_point_id_count}")
    print(f"duplicate chunk_id count: {summary.duplicate_chunk_id_count}")
    print(f"empty text count: {summary.empty_text_count}")
    print(f"missing required metadata count: {summary.missing_required_metadata_count}")
    print(
        "document sequence validation passed/failed: "
        f"{summary.document_sequence_validation_passed}/{summary.document_sequence_validation_failed}"
    )
    print(
        "section sequence validation passed/failed: "
        f"{summary.section_sequence_validation_passed}/{summary.section_sequence_validation_failed}"
    )
    print(
        "document neighbor validation passed/failed: "
        f"{summary.document_neighbor_validation_passed}/{summary.document_neighbor_validation_failed}"
    )
    print(
        "section neighbor validation passed/failed: "
        f"{summary.section_neighbor_validation_passed}/{summary.section_neighbor_validation_failed}"
    )
    print(f"output parquet path: {summary.output_parquet_path}")
    print(f"output jsonl path: {summary.output_jsonl_path}")
    print(f"validation report path: {summary.validation_report_path}")
    print("changed files: app/nsoud/build_qdrant_payload_preview.py")
    return 1 if summary.validation_status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
