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
    "chunk_text_length",
    "paragraph_count",
    "chunk_warning",
    "ns_section_hint",
]
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
    "ns_section_hint",
]
OPTIONAL_METADATA_FIELDS = [
    "ecli",
    "decision_date",
    "publication_date",
    "document_type",
    "legal_area",
    "title",
]


@dataclass(frozen=True)
class PayloadPreviewSummary:
    payload_preview_status: str
    validation_status: str
    total_rows: int
    duplicate_point_id_count: int
    duplicate_chunk_id_count: int
    empty_text_count: int
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


def validation_path_for_output(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}_validation.md")


def load_chunks(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def stable_point_id(chunk_id: str) -> str:
    return hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()


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
            "chunk_text_length": chunks_df["chunk_text_length"],
            "paragraph_count": chunks_df["paragraph_count"],
            "chunk_warning": chunks_df["chunk_warning"],
            "ns_section_hint": chunks_df["ns_section_hint"],
        }
    )
    return payload_df.reindex(columns=PAYLOAD_FIELDS)


def count_missing_values(series: pd.Series, *, treat_empty_string_as_missing: bool) -> int:
    if treat_empty_string_as_missing:
        return int(series.map(lambda value: normalize_text(value).strip() == "").sum())
    return int(series.isna().sum())


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


def validate_payload_dataframe(df: pd.DataFrame) -> tuple[str, dict[str, int], list[str], list[str], int, int, int]:
    failures: list[str] = []
    warnings: list[str] = []

    missing_required_counts: dict[str, int] = {}
    for field_name in REQUIRED_NONEMPTY_FIELDS:
        missing_count = count_missing_values(df[field_name], treat_empty_string_as_missing=True)
        missing_required_counts[field_name] = missing_count
        if missing_count > 0:
            failures.append(f"Required payload field `{field_name}` contains missing values.")

    for field_name in ("chunk_index", "chunk_text_length", "paragraph_count"):
        missing_count = count_missing_values(df[field_name], treat_empty_string_as_missing=False)
        missing_required_counts[field_name] = missing_count
        if missing_count > 0:
            failures.append(f"Required payload field `{field_name}` contains missing values.")

    for field_name in ("chunk_warning",):
        missing_required_counts[field_name] = count_missing_values(df[field_name], treat_empty_string_as_missing=False)

    empty_text_count = count_missing_values(df["text"], treat_empty_string_as_missing=True)
    if empty_text_count > 0:
        failures.append("One or more payload rows have empty text.")

    duplicate_point_id_count = int(df["point_id"].duplicated(keep=False).sum())
    if duplicate_point_id_count > 0:
        failures.append("Duplicate point_id values detected.")

    duplicate_chunk_id_count = int(df["chunk_id"].duplicated(keep=False).sum())
    if duplicate_chunk_id_count > 0:
        failures.append("Duplicate chunk_id values detected.")

    optional_missing_total = 0
    for field_name in OPTIONAL_METADATA_FIELDS:
        missing_count = count_missing_values(df[field_name], treat_empty_string_as_missing=True)
        missing_required_counts[field_name] = missing_count
        optional_missing_total += missing_count
    if optional_missing_total > 0:
        warnings.append("Some optional metadata fields are missing.")

    validation_status = "FAIL" if failures else "WARN" if warnings else "PASS"
    return (
        validation_status,
        missing_required_counts,
        failures,
        warnings,
        duplicate_point_id_count,
        duplicate_chunk_id_count,
        empty_text_count,
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
    warnings: list[str],
    duplicate_point_id_count: int,
    duplicate_chunk_id_count: int,
    empty_text_count: int,
) -> str:
    status_items = failures + warnings if failures or warnings else ["Payload preview validation passed."]
    text_lengths = df["chunk_text_length"].tolist() if not df.empty else []

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
        "",
        "## Status",
    ]
    lines.extend(f"- {item}" for item in status_items)
    lines.extend(
        [
            "",
            "## Missing Required Field Counts",
            "",
            "| Field | Missing Count |",
            "| --- | ---: |",
        ]
    )
    for field_name in PAYLOAD_FIELDS:
        count = missing_field_counts.get(field_name, 0)
        lines.append(f"| `{field_name}` | {count} |")

    lines.extend(
        [
            "",
            "## Chunk Text Lengths",
            f"- min: {min(text_lengths) if text_lengths else 0}",
            f"- max: {max(text_lengths) if text_lengths else 0}",
            f"- avg: {mean(text_lengths):.2f}" if text_lengths else "- avg: 0.00",
            "",
        ]
    )
    lines.extend(render_distribution_table("Source Distribution", distribution_counts(df, "source")))
    lines.extend(render_distribution_table("Authority Level Distribution", distribution_counts(df, "authority_level")))
    lines.extend(render_distribution_table("Document Type Distribution", distribution_counts(df, "document_type")))
    lines.extend(render_distribution_table("Legal Area Distribution", distribution_counts(df, "legal_area")))
    lines.extend(render_distribution_table("Chunk Warning Distribution", distribution_counts(df, "chunk_warning")))
    lines.extend(render_distribution_table("NS Section Hint Distribution", distribution_counts(df, "ns_section_hint")))
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
        warnings,
        duplicate_point_id_count,
        duplicate_chunk_id_count,
        empty_text_count,
    ) = validate_payload_dataframe(payload_df)
    report = build_validation_report(
        payload_df,
        input_path=args.input,
        output_parquet_path=args.out_parquet,
        output_jsonl_path=args.out_jsonl,
        validation_status=validation_status,
        missing_field_counts=missing_field_counts,
        failures=failures,
        warnings=warnings,
        duplicate_point_id_count=duplicate_point_id_count,
        duplicate_chunk_id_count=duplicate_chunk_id_count,
        empty_text_count=empty_text_count,
    )
    validation_path.write_text(report, encoding="utf-8")

    summary = PayloadPreviewSummary(
        payload_preview_status="PASS",
        validation_status=validation_status,
        total_rows=len(payload_df),
        duplicate_point_id_count=duplicate_point_id_count,
        duplicate_chunk_id_count=duplicate_chunk_id_count,
        empty_text_count=empty_text_count,
        output_parquet_path=args.out_parquet,
        output_jsonl_path=args.out_jsonl,
        validation_report_path=validation_path,
    )
    print(f"payload preview status: {summary.payload_preview_status}")
    print(f"total rows: {summary.total_rows}")
    print(f"duplicate point_id count: {summary.duplicate_point_id_count}")
    print(f"duplicate chunk_id count: {summary.duplicate_chunk_id_count}")
    print(f"empty text count: {summary.empty_text_count}")
    print(f"output parquet path: {summary.output_parquet_path}")
    print(f"output jsonl path: {summary.output_jsonl_path}")
    print(f"validation report path: {summary.validation_report_path}")
    print(f"validation status: {summary.validation_status}")
    return 1 if summary.validation_status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
