from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import urlparse


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
LEGAL_KEYWORDS = [
    "Nejvyšší soud",
    "rozhodl",
    "usnesení",
    "rozsudek",
    "odůvodnění",
    "dovolání",
]
NAVIGATION_MARKERS = [
    "Nové hledání",
    "Zpět na list",
    "Stáhnout vše",
    "Stáhnout vybrané",
    "Vyhledávání - Nejvyšší soud",
]
METADATA_OPTIONAL_FIELDS = [
    "ecli",
    "decision_date",
    "publication_date",
    "document_type",
    "legal_area",
    "title",
]
OFFICIAL_DOMAIN_SUFFIXES = ("nsoud.cz",)


@dataclass
class SuspiciousRecord:
    index: int
    case_number: str
    reasons: list[str] = field(default_factory=list)
    url: str = ""
    full_text_length: int = 0


@dataclass
class ValidationSummary:
    status: str
    total_records: int
    suspicious_records: list[SuspiciousRecord]
    required_field_failures: list[str]
    fixed_value_failures: list[str]
    url_failures: list[str]
    duplicate_hash_count: int
    duplicate_hash_values: list[str]
    missing_metadata_counts: dict[str, int]
    full_text_min: int
    full_text_max: int
    full_text_avg: float
    first_three_summaries: list[dict[str, Any]]
    warnings: list[str]
    failures: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Czech Supreme Court sample JSONL output.")
    parser.add_argument("--input", type=Path, required=True, help="Input NS sample JSONL path.")
    parser.add_argument("--out", type=Path, required=True, help="Output Markdown report path.")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


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
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object on line {line_number}.")
            records.append(payload)
    return records


def is_official_nsoud_url(url: str) -> bool:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    return bool(hostname) and any(hostname == suffix or hostname.endswith(f".{suffix}") for suffix in OFFICIAL_DOMAIN_SUFFIXES)


def looks_like_navigation_only(full_text: str) -> bool:
    lowered = full_text.lower()
    nav_hits = sum(1 for marker in NAVIGATION_MARKERS if marker.lower() in lowered)
    legal_hits = sum(1 for keyword in LEGAL_KEYWORDS if keyword.lower() in lowered)
    return nav_hits >= 2 and legal_hits == 0


def validate_records(records: list[dict[str, Any]]) -> ValidationSummary:
    required_field_failures: list[str] = []
    fixed_value_failures: list[str] = []
    url_failures: list[str] = []
    warnings: list[str] = []
    failures: list[str] = []
    suspicious_records: list[SuspiciousRecord] = []
    hash_counts: dict[str, int] = {}
    missing_metadata_counts = {field: 0 for field in METADATA_OPTIONAL_FIELDS}
    full_text_lengths: list[int] = []

    for index, record in enumerate(records, start=1):
        case_number = normalize_text(record.get("case_number")) or f"record_{index}"
        suspicious = SuspiciousRecord(
            index=index,
            case_number=case_number,
            url=normalize_text(record.get("url")),
        )

        for field_name in REQUIRED_FIELDS:
            value = record.get(field_name)
            if normalize_text(value) == "":
                required_field_failures.append(f"Record {index} ({case_number}) missing required field `{field_name}`.")

        if record.get("source") != "nsoud":
            fixed_value_failures.append(f"Record {index} ({case_number}) has invalid `source`: {record.get('source')!r}.")
        if record.get("court") != "Nejvyšší soud":
            fixed_value_failures.append(f"Record {index} ({case_number}) has invalid `court`: {record.get('court')!r}.")
        if record.get("authority_level") != "supreme":
            fixed_value_failures.append(
                f"Record {index} ({case_number}) has invalid `authority_level`: {record.get('authority_level')!r}."
            )

        url = normalize_text(record.get("url"))
        if not url:
            url_failures.append(f"Record {index} ({case_number}) has empty `url`.")
        elif not is_official_nsoud_url(url):
            url_failures.append(f"Record {index} ({case_number}) points to a non-official domain: {url}.")

        content_hash = normalize_text(record.get("content_hash"))
        if content_hash:
            hash_counts[content_hash] = hash_counts.get(content_hash, 0) + 1

        for metadata_field in METADATA_OPTIONAL_FIELDS:
            if normalize_text(record.get(metadata_field)) == "":
                missing_metadata_counts[metadata_field] += 1

        full_text = normalize_text(record.get("full_text"))
        full_text_length = len(full_text)
        suspicious.full_text_length = full_text_length
        full_text_lengths.append(full_text_length)

        if full_text_length == 0:
            suspicious.reasons.append("empty full_text")
        elif full_text_length < 1000:
            suspicious.reasons.append("full_text shorter than 1000 chars")

        if looks_like_navigation_only(full_text):
            suspicious.reasons.append("full_text looks like navigation/menu text")

        if full_text and not any(keyword.lower() in full_text.lower() for keyword in LEGAL_KEYWORDS):
            suspicious.reasons.append("full_text lacks legal-looking keywords")

        if suspicious.reasons:
            suspicious_records.append(suspicious)

    duplicate_hash_values = sorted(hash_value for hash_value, count in hash_counts.items() if count > 1)
    duplicate_hash_count = sum(hash_counts[hash_value] - 1 for hash_value in duplicate_hash_values)

    if required_field_failures:
        failures.append("Required field validation failed.")
    if fixed_value_failures:
        failures.append("Fixed value validation failed.")
    if url_failures:
        failures.append("URL validation failed.")
    if duplicate_hash_count > 0:
        failures.append("Duplicate content hashes detected.")
    if any("empty full_text" in suspicious.reasons for suspicious in suspicious_records):
        failures.append("One or more records have empty full_text.")
    if any("full_text looks like navigation/menu text" in suspicious.reasons for suspicious in suspicious_records):
        failures.append("One or more records appear to contain navigation/menu text instead of decision text.")
    if any("full_text lacks legal-looking keywords" in suspicious.reasons for suspicious in suspicious_records):
        failures.append("One or more records do not look like usable legal texts.")

    if any("full_text shorter than 1000 chars" in suspicious.reasons for suspicious in suspicious_records):
        warnings.append("One or more records have full_text shorter than 1000 characters.")
    if any(count > 0 for count in missing_metadata_counts.values()):
        warnings.append("Some non-required metadata fields are missing.")

    if failures:
        status = "FAIL"
    elif warnings:
        status = "WARN"
    else:
        status = "PASS"

    first_three_summaries = []
    for record in records[:3]:
        full_text = normalize_text(record.get("full_text"))
        first_three_summaries.append(
            {
                "case_number": normalize_text(record.get("case_number")),
                "ecli": normalize_text(record.get("ecli")),
                "decision_date": normalize_text(record.get("decision_date")),
                "document_type": normalize_text(record.get("document_type")),
                "url": normalize_text(record.get("url")),
                "full_text_length": len(full_text),
            }
        )

    if full_text_lengths:
        full_text_min = min(full_text_lengths)
        full_text_max = max(full_text_lengths)
        full_text_avg = mean(full_text_lengths)
    else:
        full_text_min = 0
        full_text_max = 0
        full_text_avg = 0.0

    return ValidationSummary(
        status=status,
        total_records=len(records),
        suspicious_records=suspicious_records,
        required_field_failures=required_field_failures,
        fixed_value_failures=fixed_value_failures,
        url_failures=url_failures,
        duplicate_hash_count=duplicate_hash_count,
        duplicate_hash_values=duplicate_hash_values,
        missing_metadata_counts=missing_metadata_counts,
        full_text_min=full_text_min,
        full_text_max=full_text_max,
        full_text_avg=full_text_avg,
        first_three_summaries=first_three_summaries,
        warnings=warnings,
        failures=failures,
    )


def render_bullet_list(items: list[str]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- {item}" for item in items)


def render_missing_metadata_table(missing_counts: dict[str, int]) -> str:
    lines = [
        "| Field | Missing Count |",
        "| --- | ---: |",
    ]
    for field_name, count in missing_counts.items():
        lines.append(f"| `{field_name}` | {count} |")
    return "\n".join(lines)


def render_suspicious_records_table(records: list[SuspiciousRecord]) -> str:
    lines = [
        "| # | case_number | reasons | full_text_length | url |",
        "| ---: | --- | --- | ---: | --- |",
    ]
    if not records:
        lines.append("| - | - | none | - | - |")
        return "\n".join(lines)

    for record in records:
        reasons = "; ".join(record.reasons)
        lines.append(
            f"| {record.index} | {record.case_number} | {reasons} | {record.full_text_length} | {record.url} |"
        )
    return "\n".join(lines)


def render_first_three_table(records: list[dict[str, Any]]) -> str:
    lines = [
        "| # | case_number | ecli | decision_date | document_type | url | full_text_length |",
        "| ---: | --- | --- | --- | --- | --- | ---: |",
    ]
    if not records:
        lines.append("| - | - | - | - | - | - | - |")
        return "\n".join(lines)

    for index, record in enumerate(records, start=1):
        lines.append(
            "| "
            f"{index} | {record['case_number'] or '-'} | {record['ecli'] or '-'} | "
            f"{record['decision_date'] or '-'} | {record['document_type'] or '-'} | "
            f"{record['url'] or '-'} | {record['full_text_length']} |"
        )
    return "\n".join(lines)


def build_markdown_report(input_path: Path, summary: ValidationSummary) -> str:
    sections = [
        "# NSoud Sample Validation",
        "",
        f"- Input: `{input_path}`",
        f"- Total records: **{summary.total_records}**",
        f"- Validation status: **{summary.status}**",
        f"- Duplicate hash count: **{summary.duplicate_hash_count}**",
        f"- Suspicious records count: **{len(summary.suspicious_records)}**",
        "",
        "## Status",
        render_bullet_list(summary.failures + summary.warnings if summary.failures or summary.warnings else ["All checks passed."]),
        "",
        "## Required Field Failures",
        render_bullet_list(summary.required_field_failures),
        "",
        "## Fixed Value Failures",
        render_bullet_list(summary.fixed_value_failures),
        "",
        "## URL Failures",
        render_bullet_list(summary.url_failures),
        "",
        "## Duplicate Hashes",
        render_bullet_list(summary.duplicate_hash_values),
        "",
        "## Full Text Lengths",
        f"- min: {summary.full_text_min}",
        f"- max: {summary.full_text_max}",
        f"- avg: {summary.full_text_avg:.2f}",
        "",
        "## Missing Metadata Counts",
        render_missing_metadata_table(summary.missing_metadata_counts),
        "",
        "## Suspicious Records",
        render_suspicious_records_table(summary.suspicious_records),
        "",
        "## First 3 Records Summary",
        render_first_three_table(summary.first_three_summaries),
        "",
    ]
    return "\n".join(sections)


def write_report(path: Path, report: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()

    try:
        records = load_jsonl_records(args.input)
    except Exception as exc:
        report = "\n".join(
            [
                "# NSoud Sample Validation",
                "",
                f"- Input: `{args.input}`",
                "- Validation status: **FAIL**",
                "",
                "## Status",
                f"- JSONL could not be loaded: {exc}",
                "",
            ]
        )
        write_report(args.out, report)
        print("validation status: FAIL")
        print("total records: 0")
        print("suspicious records count: 0")
        print(f"output path: {args.out}")
        return 1

    summary = validate_records(records)
    report = build_markdown_report(args.input, summary)
    write_report(args.out, report)

    print(f"validation status: {summary.status}")
    print(f"total records: {summary.total_records}")
    print(f"suspicious records count: {len(summary.suspicious_records)}")
    print(f"output path: {args.out}")
    return 1 if summary.status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
