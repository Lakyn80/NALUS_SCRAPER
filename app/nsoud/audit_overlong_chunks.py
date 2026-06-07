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


OVERLONG_WARNING = "overlong_ns_paragraph"
PREVIEW_LENGTH = 500

NUMBERED_BOUNDARY_RE = re.compile(r"(?=(?:^|\s)(\d{1,3}\.)(?=\s+\S))")
ROMAN_BOUNDARY_RE = re.compile(r"(?=(?:^|\s)((?:I|II|III|IV|V|VI|VII|VIII|IX|X)\.)(?=\s+\S))")
SECTION_BOUNDARY_RE = re.compile(r"(?=(?:^|\s)(Odůvodnění:|Poučení:|takto:|V Brně dne))")


@dataclass(frozen=True)
class AuditSummary:
    audit_status: str
    total_chunks: int
    overlong_chunk_count: int
    suspicious_possible_missed_boundary_count: int
    markdown_output_path: Path
    csv_output_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit overlong NSoud structural chunks.")
    parser.add_argument("--input", type=Path, required=True, help="Input NSoud chunks Parquet path.")
    parser.add_argument("--out-md", type=Path, required=True, help="Output Markdown audit report path.")
    parser.add_argument("--out-csv", type=Path, required=True, help="Output CSV audit export path.")
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


def load_chunks(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def find_internal_positions(pattern: re.Pattern[str], text: str) -> list[int]:
    positions: list[int] = []
    for match in pattern.finditer(text):
        position = match.start(1) if match.lastindex else match.start()
        if position > 0:
            positions.append(position)
    return positions


def classify_overlong_chunk(chunk_text: str) -> tuple[str, bool, int, int, int]:
    text = normalize_text(chunk_text)
    numbered_positions = find_internal_positions(NUMBERED_BOUNDARY_RE, text)
    roman_positions = find_internal_positions(ROMAN_BOUNDARY_RE, text)
    section_positions = find_internal_positions(SECTION_BOUNDARY_RE, text)

    if section_positions:
        return "possible missed section marker", True, len(numbered_positions), len(roman_positions), len(section_positions)
    if roman_positions:
        return (
            "possible missed roman section boundary",
            True,
            len(numbered_positions),
            len(roman_positions),
            len(section_positions),
        )
    if numbered_positions:
        return (
            "possible missed numbered paragraph boundary",
            True,
            len(numbered_positions),
            len(roman_positions),
            len(section_positions),
        )
    return "real long paragraph", False, 0, 0, 0


def preview_start(text: str) -> str:
    return normalize_text(text)[:PREVIEW_LENGTH]


def preview_end(text: str) -> str:
    normalized = normalize_text(text)
    return normalized[-PREVIEW_LENGTH:] if normalized else ""


def build_audit_dataframe(chunks_df: pd.DataFrame) -> pd.DataFrame:
    overlong_df = chunks_df.loc[chunks_df["chunk_warning"] == OVERLONG_WARNING].copy()
    if overlong_df.empty:
        return pd.DataFrame(
            columns=[
                "chunk_id",
                "document_id",
                "case_number",
                "ecli",
                "decision_date",
                "document_type",
                "legal_area",
                "url",
                "ns_section_hint",
                "chunk_text_length",
                "paragraph_count",
                "audit_classification",
                "suspicious_possible_missed_boundary",
                "internal_numbered_marker_count",
                "internal_roman_marker_count",
                "internal_section_marker_count",
                "chunk_text_first_500",
                "chunk_text_last_500",
            ]
        )

    classifications = overlong_df["chunk_text"].map(classify_overlong_chunk)
    overlong_df["audit_classification"] = classifications.map(lambda item: item[0])
    overlong_df["suspicious_possible_missed_boundary"] = classifications.map(lambda item: item[1])
    overlong_df["internal_numbered_marker_count"] = classifications.map(lambda item: item[2]).astype("int64")
    overlong_df["internal_roman_marker_count"] = classifications.map(lambda item: item[3]).astype("int64")
    overlong_df["internal_section_marker_count"] = classifications.map(lambda item: item[4]).astype("int64")
    overlong_df["chunk_text_first_500"] = overlong_df["chunk_text"].map(preview_start)
    overlong_df["chunk_text_last_500"] = overlong_df["chunk_text"].map(preview_end)

    ordered_columns = [
        "chunk_id",
        "document_id",
        "case_number",
        "ecli",
        "decision_date",
        "document_type",
        "legal_area",
        "url",
        "ns_section_hint",
        "chunk_text_length",
        "paragraph_count",
        "audit_classification",
        "suspicious_possible_missed_boundary",
        "internal_numbered_marker_count",
        "internal_roman_marker_count",
        "internal_section_marker_count",
        "chunk_text_first_500",
        "chunk_text_last_500",
    ]
    return overlong_df.reindex(columns=ordered_columns)


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


def escape_markdown_cell(value: Any) -> str:
    text = normalize_text(value)
    text = text.replace("|", "\\|").replace("\r", " ").replace("\n", " ")
    return text


def render_overlong_table(audit_df: pd.DataFrame) -> list[str]:
    lines = [
        "## Overlong Chunks",
        "",
        "| Chunk ID | Case Number | Section Hint | Length | Paragraphs | Classification | Suspicious | First 500 | Last 500 |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    if audit_df.empty:
        lines.append("| - | - | - | 0 | 0 | - | - | - | - |")
        lines.append("")
        return lines

    for _, row in audit_df.iterrows():
        lines.append(
            f"| {escape_markdown_cell(row['chunk_id'])} | {escape_markdown_cell(row['case_number'])} | "
            f"{escape_markdown_cell(row['ns_section_hint'])} | {int(row['chunk_text_length'])} | "
            f"{int(row['paragraph_count'])} | {escape_markdown_cell(row['audit_classification'])} | "
            f"{'yes' if bool(row['suspicious_possible_missed_boundary']) else 'no'} | "
            f"{escape_markdown_cell(row['chunk_text_first_500'])} | {escape_markdown_cell(row['chunk_text_last_500'])} |"
        )
    lines.append("")
    return lines


def build_markdown_report(chunks_df: pd.DataFrame, audit_df: pd.DataFrame, input_path: Path) -> str:
    total_chunks = len(chunks_df)
    overlong_chunk_count = len(audit_df)
    overlong_lengths = audit_df["chunk_text_length"].tolist() if not audit_df.empty else []
    suspicious_count = int(audit_df["suspicious_possible_missed_boundary"].sum()) if not audit_df.empty else 0
    overlong_percentage = (overlong_chunk_count / total_chunks * 100.0) if total_chunks else 0.0

    lines = [
        "# NSoud Overlong Chunks Audit",
        "",
        f"- Input: `{input_path}`",
        f"- Total chunks: **{total_chunks}**",
        f"- Overlong chunk count: **{overlong_chunk_count}**",
        f"- Overlong percentage: **{overlong_percentage:.2f}%**",
        f"- Max overlong length: **{max(overlong_lengths) if overlong_lengths else 0}**",
        f"- Avg overlong length: **{mean(overlong_lengths):.2f}**" if overlong_lengths else "- Avg overlong length: **0.00**",
        f"- Suspicious possible missed boundary count: **{suspicious_count}**",
        "",
    ]
    lines.extend(render_distribution_table("Distribution by NS Section Hint", distribution_counts(audit_df, "ns_section_hint")))
    lines.extend(render_distribution_table("Distribution by Document Type", distribution_counts(audit_df, "document_type")))
    lines.extend(render_distribution_table("Audit Classification Distribution", distribution_counts(audit_df, "audit_classification")))
    lines.extend(render_overlong_table(audit_df))
    return "\n".join(lines)


def write_csv(audit_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(out_path, index=False, encoding="utf-8")


def write_markdown(report: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()

    if pyarrow is None:
        print("audit status: FAIL")
        print("error: pyarrow is required for Parquet input.")
        print("install command: pip install pyarrow")
        return 1

    try:
        chunks_df = load_chunks(args.input)
        audit_df = build_audit_dataframe(chunks_df)
        write_csv(audit_df, args.out_csv)
        report = build_markdown_report(chunks_df, audit_df, args.input)
        write_markdown(report, args.out_md)
    except Exception as exc:
        print("audit status: FAIL")
        print(f"error: {exc}")
        return 1

    summary = AuditSummary(
        audit_status="PASS",
        total_chunks=len(chunks_df),
        overlong_chunk_count=len(audit_df),
        suspicious_possible_missed_boundary_count=int(audit_df["suspicious_possible_missed_boundary"].sum())
        if not audit_df.empty
        else 0,
        markdown_output_path=args.out_md,
        csv_output_path=args.out_csv,
    )
    print(f"audit status: {summary.audit_status}")
    print(f"total chunks: {summary.total_chunks}")
    print(f"overlong chunk count: {summary.overlong_chunk_count}")
    print(f"suspicious possible missed boundary count: {summary.suspicious_possible_missed_boundary_count}")
    print(f"markdown output path: {summary.markdown_output_path}")
    print(f"csv output path: {summary.csv_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
