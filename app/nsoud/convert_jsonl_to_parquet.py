from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
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


REQUIRED_FIELDS = [
    "source",
    "court",
    "authority_level",
    "case_number",
    "url",
    "full_text",
    "source_attribution",
    "scraped_at",
    "content_hash",
]
OPTIONAL_METADATA_FIELDS = [
    "ecli",
    "decision_date",
    "publication_date",
    "document_type",
    "legal_area",
    "title",
]
FIXED_VALUES = {
    "source": "nsoud",
    "court": "Nejvyšší soud",
    "authority_level": "supreme",
}
DERIVED_FIELDS = [
    "full_text_length",
    "has_ecli",
    "has_decision_date",
    "has_publication_date",
    "has_legal_area",
]


@dataclass(frozen=True)
class ConversionSummary:
    conversion_status: str
    validation_status: str
    total_records: int
    output_parquet_path: Path
    schema_path: Path
    validation_report_path: Path
    duplicate_content_hash_count: int
    duplicate_url_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert consolidated NSoud JSONL documents to Parquet.")
    parser.add_argument("--input", type=Path, required=True, help="Input consolidated JSONL path.")
    parser.add_argument("--out", type=Path, required=True, help="Output Parquet path.")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def count_missing_values(series: pd.Series) -> int:
    return int(series.map(lambda value: normalize_text(value).strip() == "").sum())


def is_official_nsoud_url(value: Any) -> bool:
    url = normalize_text(value).strip().lower()
    return url.startswith("http://nsoud.cz") or url.startswith("https://nsoud.cz") or ".nsoud.cz/" in url


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} on line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {path} on line {line_number}.")
            records.append(payload)
    return records


def schema_path_for_output(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}_schema.json")


def validation_path_for_output(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}_validation.md")


def build_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    base_columns = list(records[0].keys())
    df = pd.DataFrame(records)
    df = df.reindex(columns=base_columns)

    full_text_series = df["full_text"].map(lambda value: len(normalize_text(value)))
    df["full_text_length"] = full_text_series.astype("int64")
    df["has_ecli"] = df["ecli"].map(lambda value: bool(str(value).strip()) if pd.notna(value) else False)
    df["has_decision_date"] = df["decision_date"].map(lambda value: bool(str(value).strip()) if pd.notna(value) else False)
    df["has_publication_date"] = df["publication_date"].map(
        lambda value: bool(str(value).strip()) if pd.notna(value) else False
    )
    df["has_legal_area"] = df["legal_area"].map(lambda value: bool(str(value).strip()) if pd.notna(value) else False)
    return df


def write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", index=False)


def write_schema_json(df: pd.DataFrame, input_path: Path, output_path: Path, schema_path: Path) -> None:
    columns = []
    for column_name in df.columns:
        columns.append(
            {
                "name": column_name,
                "dtype": str(df[column_name].dtype),
                "nullable": bool(df[column_name].isna().any()),
            }
        )

    payload = {
        "columns": columns,
        "record_count": int(len(df)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
    }
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def distribution_counts(df: pd.DataFrame, column_name: str) -> dict[str, int]:
    series = df[column_name].fillna("").map(lambda value: str(value).strip() or "<missing>")
    counts = series.value_counts(dropna=False).to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def render_distribution_table(title: str, counts: dict[str, int]) -> list[str]:
    lines = [f"## {title}", "", "| Value | Count |", "| --- | ---: |"]
    for value, count in counts.items():
        lines.append(f"| {value} | {count} |")
    lines.append("")
    return lines


def build_validation_report(
    df: pd.DataFrame,
    *,
    input_path: Path,
    output_path: Path,
    duplicate_content_hash_count: int,
    duplicate_url_count: int,
    validation_status: str,
    failures: list[str],
    warnings: list[str],
) -> str:
    missing_counts = {column: count_missing_values(df[column]) for column in df.columns}

    full_text_lengths = df["full_text_length"].tolist()
    report_lines = [
        "# NSoud JSONL to Parquet Validation",
        "",
        f"- Input: `{input_path}`",
        f"- Output Parquet: `{output_path}`",
        f"- Validation status: **{validation_status}**",
        f"- Total records: **{len(df)}**",
        f"- Duplicate content_hash count: **{duplicate_content_hash_count}**",
        f"- Duplicate URL count: **{duplicate_url_count}**",
        "",
        "## Status",
    ]

    status_items = failures + warnings if failures or warnings else ["Conversion and validation passed."]
    report_lines.extend(f"- {item}" for item in status_items)
    report_lines.extend(
        [
            "",
            "## Columns",
            ", ".join(df.columns),
            "",
            "## Missing Value Counts",
            "",
            "| Column | Missing Count |",
            "| --- | ---: |",
        ]
    )
    for column_name, count in missing_counts.items():
        report_lines.append(f"| `{column_name}` | {count} |")

    report_lines.extend(
        [
            "",
            "## Full Text Lengths",
            f"- min: {min(full_text_lengths) if full_text_lengths else 0}",
            f"- max: {max(full_text_lengths) if full_text_lengths else 0}",
            f"- avg: {mean(full_text_lengths):.2f}" if full_text_lengths else "- avg: 0.00",
            "",
        ]
    )
    report_lines.extend(render_distribution_table("Source Distribution", distribution_counts(df, "source")))
    report_lines.extend(render_distribution_table("Document Type Distribution", distribution_counts(df, "document_type")))
    report_lines.extend(render_distribution_table("Legal Area Distribution", distribution_counts(df, "legal_area")))
    return "\n".join(report_lines)


def validate_dataframe(df: pd.DataFrame) -> tuple[str, int, int, list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []

    required_field_missing = 0
    for field_name in REQUIRED_FIELDS:
        missing_count = count_missing_values(df[field_name])
        if missing_count > 0:
            required_field_missing += missing_count

    if required_field_missing > 0:
        failures.append("Required fields contain missing values.")

    empty_full_text_count = count_missing_values(df["full_text"])
    if empty_full_text_count > 0:
        failures.append("One or more records have empty full_text.")

    duplicate_content_hash_count = int(df["content_hash"].duplicated(keep=False).sum())
    if duplicate_content_hash_count > 0:
        failures.append("Duplicate content_hash values detected.")

    duplicate_url_count = int(df["url"].duplicated(keep=False).sum())

    for field_name, expected_value in FIXED_VALUES.items():
        invalid_count = int((df[field_name].fillna("").map(str) != expected_value).sum())
        if invalid_count > 0:
            failures.append(f"Field `{field_name}` contains invalid fixed values.")

    invalid_url_count = int(df["url"].map(lambda value: not is_official_nsoud_url(value)).sum())
    if invalid_url_count > 0:
        failures.append("One or more URLs do not point to an official nsoud.cz domain.")

    optional_missing = 0
    for field_name in OPTIONAL_METADATA_FIELDS:
        optional_missing += count_missing_values(df[field_name])
    if optional_missing > 0:
        warnings.append("Some optional metadata fields are missing.")

    validation_status = "FAIL" if failures else "WARN" if warnings else "PASS"
    return validation_status, duplicate_content_hash_count, duplicate_url_count, failures, warnings


def main() -> int:
    args = parse_args()

    if pyarrow is None:
        print("conversion status: FAIL")
        print("error: pyarrow is required for Parquet output.")
        print("install command: pip install pyarrow")
        return 1

    schema_path = schema_path_for_output(args.out)
    validation_path = validation_path_for_output(args.out)

    try:
        records = load_jsonl_records(args.input)
    except Exception as exc:
        print("conversion status: FAIL")
        print(f"error: {exc}")
        return 1

    try:
        df = build_dataframe(records)
        write_parquet(df, args.out)
        write_schema_json(df, args.input, args.out, schema_path)
    except Exception as exc:
        print("conversion status: FAIL")
        print(f"error: {exc}")
        return 1

    validation_status, duplicate_content_hash_count, duplicate_url_count, failures, warnings = validate_dataframe(df)
    report = build_validation_report(
        df,
        input_path=args.input,
        output_path=args.out,
        duplicate_content_hash_count=duplicate_content_hash_count,
        duplicate_url_count=duplicate_url_count,
        validation_status=validation_status,
        failures=failures,
        warnings=warnings,
    )
    validation_path.write_text(report, encoding="utf-8")

    summary = ConversionSummary(
        conversion_status="PASS",
        validation_status=validation_status,
        total_records=len(df),
        output_parquet_path=args.out,
        schema_path=schema_path,
        validation_report_path=validation_path,
        duplicate_content_hash_count=duplicate_content_hash_count,
        duplicate_url_count=duplicate_url_count,
    )
    print(f"conversion status: {summary.conversion_status}")
    print(f"total records: {summary.total_records}")
    print(f"output parquet path: {summary.output_parquet_path}")
    print(f"schema path: {summary.schema_path}")
    print(f"validation report path: {summary.validation_report_path}")
    print(f"validation status: {summary.validation_status}")
    return 1 if summary.validation_status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
