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
OPTIONAL_METADATA_FIELDS = [
    "ecli",
    "decision_date",
    "publication_date",
    "document_type",
    "legal_area",
    "title",
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
OFFICIAL_DOMAIN_SUFFIXES = ("nsoud.cz",)


@dataclass
class LoadedMonth:
    year: int
    month: int
    date_from: str
    date_to: str
    output_path: str
    manifest_records_written: int
    duplicates_skipped: int
    pages_visited: int
    status: str
    error_message: str | None
    loaded_records: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DuplicateEntry:
    kind: str
    value: str
    count: int
    months: list[str]
    case_numbers: list[str]


@dataclass
class SuspiciousRecord:
    index: int
    month_key: str
    case_number: str
    url: str
    full_text_length: int
    reasons: list[str] = field(default_factory=list)


@dataclass
class AggregateValidationSummary:
    status: str
    total_records: int
    months_validated: int
    duplicate_content_hash_count: int
    duplicate_url_count: int
    suspicious_records: list[SuspiciousRecord]
    required_field_failures: list[str]
    fixed_value_failures: list[str]
    url_failures: list[str]
    duplicate_entries: list[DuplicateEntry]
    missing_metadata_counts: dict[str, int]
    full_text_min: int
    full_text_max: int
    full_text_avg: float
    first_five_records: list[dict[str, Any]]
    monthly_rows: list[dict[str, Any]]
    warnings: list[str]
    failures: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate aggregate Czech Supreme Court monthly batch outputs.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to nsoud monthly manifest JSON.")
    parser.add_argument("--out", type=Path, required=True, help="Output Markdown report path.")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Manifest file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifest is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Manifest root must be a JSON object.")
    return payload


def resolve_data_path(raw_path: str, manifest_path: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate

    if candidate.exists():
        return candidate

    repo_relative = manifest_path.parents[3] / candidate if len(manifest_path.parents) >= 4 else manifest_path.parent / candidate
    if repo_relative.exists():
        return repo_relative

    manifest_relative = manifest_path.parent / candidate.name
    if manifest_relative.exists():
        return manifest_relative

    return candidate


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


def is_official_nsoud_url(url: str) -> bool:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    return bool(hostname) and any(hostname == suffix or hostname.endswith(f".{suffix}") for suffix in OFFICIAL_DOMAIN_SUFFIXES)


def looks_like_navigation_only(full_text: str) -> bool:
    lowered = full_text.lower()
    nav_hits = sum(1 for marker in NAVIGATION_MARKERS if marker.lower() in lowered)
    legal_hits = sum(1 for keyword in LEGAL_KEYWORDS if keyword.lower() in lowered)
    return nav_hits >= 2 and legal_hits == 0


def month_key(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def load_successful_months(manifest_path: Path) -> list[LoadedMonth]:
    payload = load_json(manifest_path)
    batches = payload.get("batches")
    if not isinstance(batches, list):
        raise ValueError("Manifest field `batches` must be a list.")

    loaded_months: list[LoadedMonth] = []
    for entry in batches:
        if not isinstance(entry, dict):
            raise ValueError("Every manifest batch entry must be an object.")

        status = normalize_text(entry.get("status"))
        loaded = LoadedMonth(
            year=int(entry.get("year")),
            month=int(entry.get("month")),
            date_from=normalize_text(entry.get("date_from")),
            date_to=normalize_text(entry.get("date_to")),
            output_path=normalize_text(entry.get("output_path")),
            manifest_records_written=int(entry.get("records_written", 0)),
            duplicates_skipped=int(entry.get("duplicates_skipped", 0)),
            pages_visited=int(entry.get("pages_visited", 0)),
            status=status,
            error_message=normalize_text(entry.get("error_message")) or None,
        )

        if status == "success":
            output_path = resolve_data_path(loaded.output_path, manifest_path)
            if not output_path.exists():
                raise ValueError(f"Successful monthly output is missing: {loaded.output_path}")
            loaded.loaded_records = load_jsonl_records(output_path)

        loaded_months.append(loaded)

    return loaded_months


def validate_loaded_months(months: list[LoadedMonth]) -> AggregateValidationSummary:
    failures: list[str] = []
    warnings: list[str] = []
    required_field_failures: list[str] = []
    fixed_value_failures: list[str] = []
    url_failures: list[str] = []
    suspicious_records: list[SuspiciousRecord] = []
    missing_metadata_counts = {field: 0 for field in OPTIONAL_METADATA_FIELDS}
    full_text_lengths: list[int] = []
    hash_map: dict[str, list[dict[str, str]]] = {}
    url_map: dict[str, list[dict[str, str]]] = {}
    first_five_records: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []

    global_index = 0
    for loaded_month in months:
        month_label = month_key(loaded_month.year, loaded_month.month)
        monthly_rows.append(
            {
                "month": month_label,
                "date_from": loaded_month.date_from,
                "date_to": loaded_month.date_to,
                "status": loaded_month.status,
                "pages_visited": loaded_month.pages_visited,
                "manifest_records_written": loaded_month.manifest_records_written,
                "loaded_records": len(loaded_month.loaded_records),
                "duplicates_skipped": loaded_month.duplicates_skipped,
                "output_path": loaded_month.output_path,
                "error_message": loaded_month.error_message or "",
            }
        )

        if loaded_month.status != "success":
            continue

        for record in loaded_month.loaded_records:
            global_index += 1
            case_number = normalize_text(record.get("case_number")) or f"record_{global_index}"
            record_url = normalize_text(record.get("url"))
            suspicious = SuspiciousRecord(
                index=global_index,
                month_key=month_label,
                case_number=case_number,
                url=record_url,
                full_text_length=0,
            )

            for field_name in REQUIRED_FIELDS:
                if normalize_text(record.get(field_name)) == "":
                    required_field_failures.append(
                        f"Record {global_index} ({month_label}, {case_number}) missing required field `{field_name}`."
                    )

            if record.get("source") != "nsoud":
                fixed_value_failures.append(
                    f"Record {global_index} ({month_label}, {case_number}) has invalid `source`: {record.get('source')!r}."
                )
            if record.get("court") != "Nejvyšší soud":
                fixed_value_failures.append(
                    f"Record {global_index} ({month_label}, {case_number}) has invalid `court`: {record.get('court')!r}."
                )
            if record.get("authority_level") != "supreme":
                fixed_value_failures.append(
                    f"Record {global_index} ({month_label}, {case_number}) has invalid `authority_level`: {record.get('authority_level')!r}."
                )

            if not record_url:
                url_failures.append(f"Record {global_index} ({month_label}, {case_number}) has empty `url`.")
            elif not is_official_nsoud_url(record_url):
                url_failures.append(
                    f"Record {global_index} ({month_label}, {case_number}) points to a non-official domain: {record_url}."
                )

            content_hash = normalize_text(record.get("content_hash"))
            if content_hash:
                hash_map.setdefault(content_hash, []).append({"month": month_label, "case_number": case_number})
            if record_url:
                url_map.setdefault(record_url, []).append({"month": month_label, "case_number": case_number})

            for metadata_field in OPTIONAL_METADATA_FIELDS:
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

            if len(first_five_records) < 5:
                first_five_records.append(
                    {
                        "month": month_label,
                        "case_number": case_number,
                        "ecli": normalize_text(record.get("ecli")),
                        "decision_date": normalize_text(record.get("decision_date")),
                        "document_type": normalize_text(record.get("document_type")),
                        "url": record_url,
                        "full_text_length": full_text_length,
                    }
                )

    duplicate_entries: list[DuplicateEntry] = []
    duplicate_content_hash_count = 0
    duplicate_url_count = 0

    for content_hash, items in sorted(hash_map.items()):
        if len(items) > 1:
            duplicate_content_hash_count += len(items) - 1
            duplicate_entries.append(
                DuplicateEntry(
                    kind="content_hash",
                    value=content_hash,
                    count=len(items),
                    months=[item["month"] for item in items],
                    case_numbers=[item["case_number"] for item in items],
                )
            )

    for record_url, items in sorted(url_map.items()):
        if len(items) > 1:
            duplicate_url_count += len(items) - 1
            duplicate_entries.append(
                DuplicateEntry(
                    kind="url",
                    value=record_url,
                    count=len(items),
                    months=[item["month"] for item in items],
                    case_numbers=[item["case_number"] for item in items],
                )
            )

    if required_field_failures:
        failures.append("Required field validation failed.")
    if fixed_value_failures:
        failures.append("Fixed value validation failed.")
    if url_failures:
        failures.append("URL validation failed.")
    if duplicate_content_hash_count > 0:
        failures.append("Duplicate content hashes detected across monthly files.")
    if any("empty full_text" in record.reasons for record in suspicious_records):
        failures.append("One or more records have empty full_text.")
    if any("full_text looks like navigation/menu text" in record.reasons for record in suspicious_records):
        failures.append("One or more records appear to contain navigation/menu text instead of decision text.")
    if any("full_text lacks legal-looking keywords" in record.reasons for record in suspicious_records):
        failures.append("One or more records do not look like usable legal texts.")

    if any("full_text shorter than 1000 chars" in record.reasons for record in suspicious_records):
        warnings.append("One or more records have full_text shorter than 1000 characters.")
    if any(count > 0 for count in missing_metadata_counts.values()):
        warnings.append("Some non-required metadata fields are missing.")
    if duplicate_url_count > 0:
        warnings.append("Duplicate URLs were found across monthly files.")

    if failures:
        status = "FAIL"
    elif warnings:
        status = "WARN"
    else:
        status = "PASS"

    if full_text_lengths:
        full_text_min = min(full_text_lengths)
        full_text_max = max(full_text_lengths)
        full_text_avg = mean(full_text_lengths)
    else:
        full_text_min = 0
        full_text_max = 0
        full_text_avg = 0.0

    months_validated = sum(1 for month in months if month.status == "success")
    total_records = sum(len(month.loaded_records) for month in months if month.status == "success")

    return AggregateValidationSummary(
        status=status,
        total_records=total_records,
        months_validated=months_validated,
        duplicate_content_hash_count=duplicate_content_hash_count,
        duplicate_url_count=duplicate_url_count,
        suspicious_records=suspicious_records,
        required_field_failures=required_field_failures,
        fixed_value_failures=fixed_value_failures,
        url_failures=url_failures,
        duplicate_entries=duplicate_entries,
        missing_metadata_counts=missing_metadata_counts,
        full_text_min=full_text_min,
        full_text_max=full_text_max,
        full_text_avg=full_text_avg,
        first_five_records=first_five_records,
        monthly_rows=monthly_rows,
        warnings=warnings,
        failures=failures,
    )


def render_bullet_list(items: list[str]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- {item}" for item in items)


def render_monthly_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Month | Status | Date From | Date To | Manifest Records | Loaded Records | Duplicates Skipped | Pages Visited | Output Path | Error |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['month']} | {row['status']} | {row['date_from']} | {row['date_to']} | "
            f"{row['manifest_records_written']} | {row['loaded_records']} | {row['duplicates_skipped']} | "
            f"{row['pages_visited']} | {row['output_path']} | {row['error_message'] or '-'} |"
        )
    return "\n".join(lines)


def render_duplicate_table(entries: list[DuplicateEntry]) -> str:
    lines = [
        "| Type | Value | Count | Months | Case Numbers |",
        "| --- | --- | ---: | --- | --- |",
    ]
    if not entries:
        lines.append("| - | none | 0 | - | - |")
        return "\n".join(lines)

    for entry in entries:
        lines.append(
            f"| {entry.kind} | {entry.value} | {entry.count} | "
            f"{', '.join(entry.months)} | {', '.join(entry.case_numbers)} |"
        )
    return "\n".join(lines)


def render_suspicious_records_table(records: list[SuspiciousRecord]) -> str:
    lines = [
        "| # | Month | Case Number | Reasons | Full Text Length | URL |",
        "| ---: | --- | --- | --- | ---: | --- |",
    ]
    if not records:
        lines.append("| - | - | - | none | - | - |")
        return "\n".join(lines)

    for record in records:
        lines.append(
            f"| {record.index} | {record.month_key} | {record.case_number} | "
            f"{'; '.join(record.reasons)} | {record.full_text_length} | {record.url} |"
        )
    return "\n".join(lines)


def render_metadata_table(counts: dict[str, int]) -> str:
    lines = [
        "| Field | Missing Count |",
        "| --- | ---: |",
    ]
    for field_name, count in counts.items():
        lines.append(f"| `{field_name}` | {count} |")
    return "\n".join(lines)


def render_first_five_table(records: list[dict[str, Any]]) -> str:
    lines = [
        "| # | Month | Case Number | ECLI | Decision Date | Document Type | URL | Full Text Length |",
        "| ---: | --- | --- | --- | --- | --- | --- | ---: |",
    ]
    if not records:
        lines.append("| - | - | - | - | - | - | - | - |")
        return "\n".join(lines)

    for index, record in enumerate(records, start=1):
        lines.append(
            f"| {index} | {record['month']} | {record['case_number'] or '-'} | {record['ecli'] or '-'} | "
            f"{record['decision_date'] or '-'} | {record['document_type'] or '-'} | {record['url'] or '-'} | "
            f"{record['full_text_length']} |"
        )
    return "\n".join(lines)


def build_markdown_report(
    manifest_path: Path,
    summary: AggregateValidationSummary,
    months: list[LoadedMonth],
) -> str:
    total_manifest_months = len(months)
    total_successful_months = sum(1 for month in months if month.status == "success")
    total_failed_months = total_manifest_months - total_successful_months

    sections = [
        "# NSoud Monthly Batch Validation",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Validation status: **{summary.status}**",
        f"- Total records: **{summary.total_records}**",
        f"- Months validated: **{summary.months_validated}**",
        f"- Duplicate content_hash count: **{summary.duplicate_content_hash_count}**",
        f"- Duplicate URL count: **{summary.duplicate_url_count}**",
        f"- Suspicious records count: **{len(summary.suspicious_records)}**",
        "",
        "## Status",
        render_bullet_list(summary.failures + summary.warnings if summary.failures or summary.warnings else ["All checks passed."]),
        "",
        "## Manifest Summary",
        f"- total months in manifest: {total_manifest_months}",
        f"- successful months in manifest: {total_successful_months}",
        f"- failed months in manifest: {total_failed_months}",
        "",
        "## Monthly Table",
        render_monthly_table(summary.monthly_rows),
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
        "## Duplicate Table",
        render_duplicate_table(summary.duplicate_entries),
        "",
        "## Full Text Lengths",
        f"- min: {summary.full_text_min}",
        f"- max: {summary.full_text_max}",
        f"- avg: {summary.full_text_avg:.2f}",
        "",
        "## Suspicious Records",
        render_suspicious_records_table(summary.suspicious_records),
        "",
        "## Metadata Completeness",
        render_metadata_table(summary.missing_metadata_counts),
        "",
        "## First 5 Records Summary",
        render_first_five_table(summary.first_five_records),
        "",
    ]
    return "\n".join(sections)


def write_report(path: Path, report: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def build_failure_report(manifest_path: Path, error_message: str) -> str:
    return "\n".join(
        [
            "# NSoud Monthly Batch Validation",
            "",
            f"- Manifest: `{manifest_path}`",
            "- Validation status: **FAIL**",
            "",
            "## Status",
            f"- Validation could not proceed: {error_message}",
            "",
        ]
    )


def main() -> int:
    args = parse_args()

    try:
        loaded_months = load_successful_months(args.manifest)
        summary = validate_loaded_months(loaded_months)
        report = build_markdown_report(args.manifest, summary, loaded_months)
        write_report(args.out, report)
    except Exception as exc:
        report = build_failure_report(args.manifest, str(exc))
        write_report(args.out, report)
        print("validation status: FAIL")
        print("total records: 0")
        print("months validated: 0")
        print("duplicate content_hash count: 0")
        print("suspicious records count: 0")
        print(f"output path: {args.out}")
        return 1

    print(f"validation status: {summary.status}")
    print(f"total records: {summary.total_records}")
    print(f"months validated: {summary.months_validated}")
    print(f"duplicate content_hash count: {summary.duplicate_content_hash_count}")
    print(f"suspicious records count: {len(summary.suspicious_records)}")
    print(f"output path: {args.out}")
    return 1 if summary.status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
