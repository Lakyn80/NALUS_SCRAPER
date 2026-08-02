from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.rag.legal_v2.sources import DecisionDateRange, parse_decision_date


_IDENTITY_KEYS = ("ecli", "source_document_id", "document_id", "case_reference", "spisova_znacka", "result_id")
_DATE_KEYS = ("decision_date", "date", "publication_date", "scraped_at")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


@dataclass
class SourceFileIssue:
    path: str
    issue: str


@dataclass
class SourceBucket:
    source_type: str
    court: str
    source_file_count: int = 0
    raw_record_count: int = 0
    discovered_document_count: int = 0
    missing_stable_document_identifier_count: int = 0
    missing_complete_text_count: int = 0
    duplicate_source_document_identifier_count: int = 0
    duplicate_source_document_identifiers: list[str] = field(default_factory=list)
    date_coverage: dict[str, Any] = field(default_factory=dict)
    decision_date_range: dict[str, Any] = field(default_factory=dict)
    unreadable_files: list[SourceFileIssue] = field(default_factory=list)
    unsupported_formats: list[SourceFileIssue] = field(default_factory=list)


@dataclass
class SourceInventoryReport:
    schema: str
    generated_at: str
    total_discovered_source_documents: int
    source_file_count: int
    document_count_per_adapter: dict[str, int]
    documents_missing_stable_document_identifiers: int
    documents_missing_complete_text: int
    duplicate_source_document_identifiers: int
    unreadable_file_count: int
    unsupported_format_count: int
    sources: list[SourceBucket]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_source_inventory(
    *,
    batches_dir: Path,
    nsoud_chunks_path: Path,
    decision_date_range: DecisionDateRange | None = None,
) -> SourceInventoryReport:
    buckets = [
        _inventory_nalus_batches(batches_dir, decision_date_range=decision_date_range),
        _inventory_nsoud_chunks(nsoud_chunks_path, decision_date_range=decision_date_range),
    ]
    return SourceInventoryReport(
        schema="legal_v2_source_inventory_v1",
        generated_at=_utc_now(),
        total_discovered_source_documents=sum(bucket.discovered_document_count for bucket in buckets),
        source_file_count=sum(bucket.source_file_count for bucket in buckets),
        document_count_per_adapter={
            bucket.source_type: bucket.discovered_document_count for bucket in buckets
        },
        documents_missing_stable_document_identifiers=sum(
            bucket.missing_stable_document_identifier_count for bucket in buckets
        ),
        documents_missing_complete_text=sum(bucket.missing_complete_text_count for bucket in buckets),
        duplicate_source_document_identifiers=sum(
            bucket.duplicate_source_document_identifier_count for bucket in buckets
        ),
        unreadable_file_count=sum(len(bucket.unreadable_files) for bucket in buckets),
        unsupported_format_count=sum(len(bucket.unsupported_formats) for bucket in buckets),
        sources=buckets,
    )


def write_source_inventory(report: SourceInventoryReport, json_path: Path, markdown_path: Path) -> tuple[Path, Path]:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return json_path, markdown_path


def _inventory_nalus_batches(
    batches_dir: Path,
    *,
    decision_date_range: DecisionDateRange | None,
) -> SourceBucket:
    bucket = SourceBucket(source_type="constitutional", court="Ustavni soud")
    files = _nalus_batch_files(batches_dir, bucket)
    bucket.source_file_count = len(files)
    seen: set[str] = set()
    duplicate_ids: set[str] = set()
    years: list[int] = []
    date_range_counter = _date_range_counter(decision_date_range)
    for file_path in files:
        payload = _load_json(file_path, bucket)
        if payload is None:
            continue
        if not isinstance(payload, list):
            bucket.unsupported_formats.append(SourceFileIssue(str(file_path), "json_root_not_list"))
            continue
        for item in payload:
            if not isinstance(item, dict):
                bucket.unsupported_formats.append(SourceFileIssue(str(file_path), "record_not_object"))
                continue
            bucket.raw_record_count += 1
            document_id = _document_identity(item)
            text = str(item.get("full_text") or "").strip()
            if not document_id:
                bucket.missing_stable_document_identifier_count += 1
            if not text:
                bucket.missing_complete_text_count += 1
            if document_id:
                if document_id in seen:
                    duplicate_ids.add(document_id)
                elif text:
                    bucket.discovered_document_count += 1
                    _count_document_date(date_range_counter, item)
                seen.add(document_id)
            years.extend(_years_from_metadata(item))
    _finish_bucket(bucket, years, duplicate_ids)
    bucket.decision_date_range = date_range_counter
    return bucket


def _inventory_nsoud_chunks(
    path: Path,
    *,
    decision_date_range: DecisionDateRange | None,
) -> SourceBucket:
    bucket = SourceBucket(source_type="supreme", court="Nejvyssi soud")
    bucket.source_file_count = int(path.exists())
    chunks_by_document: dict[str, int] = {}
    metadata_by_document: dict[str, dict[str, Any]] = {}
    years: list[int] = []
    date_range_counter = _date_range_counter(decision_date_range)
    if not path.exists():
        bucket.unreadable_files.append(SourceFileIssue(str(path), "file_missing"))
        return bucket
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    bucket.unsupported_formats.append(SourceFileIssue(f"{path}:{line_number}", "invalid_jsonl"))
                    continue
                if not isinstance(item, dict):
                    bucket.unsupported_formats.append(SourceFileIssue(f"{path}:{line_number}", "record_not_object"))
                    continue
                bucket.raw_record_count += 1
                document_id = _document_identity(item)
                text = str(item.get("text") or item.get("chunk_text") or "").strip()
                if not document_id:
                    bucket.missing_stable_document_identifier_count += 1
                if not text:
                    bucket.missing_complete_text_count += 1
                if document_id and text:
                    chunks_by_document[document_id] = chunks_by_document.get(document_id, 0) + 1
                    metadata_by_document.setdefault(document_id, item)
                years.extend(_years_from_metadata(item))
    except OSError as exc:
        bucket.unreadable_files.append(SourceFileIssue(str(path), exc.__class__.__name__))
    bucket.discovered_document_count = len(chunks_by_document)
    for item in metadata_by_document.values():
        _count_document_date(date_range_counter, item)
    _finish_bucket(bucket, years, set())
    bucket.decision_date_range = date_range_counter
    return bucket


def _nalus_batch_files(batches_dir: Path, bucket: SourceBucket) -> list[Path]:
    if not batches_dir.exists():
        bucket.unreadable_files.append(SourceFileIssue(str(batches_dir), "directory_missing"))
        return []
    manifest_path = batches_dir / "manifest.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path, bucket)
        if isinstance(manifest, dict):
            files = [
                batches_dir / str(entry.get("file"))
                for entry in manifest.get("batches", [])
                if isinstance(entry, dict) and entry.get("file")
            ]
            existing = [path for path in files if path.exists() and path.name != "manifest.json"]
            if existing:
                missing = [path for path in files if not path.exists()]
                bucket.unreadable_files.extend(SourceFileIssue(str(path), "manifest_file_missing") for path in missing)
                return existing
        elif manifest is not None:
            bucket.unsupported_formats.append(SourceFileIssue(str(manifest_path), "manifest_root_not_object"))
    return sorted(path for path in batches_dir.glob("*.json") if path.name != "manifest.json")


def _load_json(path: Path, bucket: SourceBucket) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        bucket.unreadable_files.append(SourceFileIssue(str(path), exc.__class__.__name__))
    except json.JSONDecodeError:
        bucket.unsupported_formats.append(SourceFileIssue(str(path), "invalid_json"))
    return None


def _finish_bucket(bucket: SourceBucket, years: list[int], duplicate_ids: set[str]) -> None:
    bucket.duplicate_source_document_identifiers = sorted(duplicate_ids)[:100]
    bucket.duplicate_source_document_identifier_count = len(duplicate_ids)
    bucket.date_coverage = {
        "min_year": min(years) if years else None,
        "max_year": max(years) if years else None,
        "year_count": len(set(years)),
    }


def _date_range_counter(decision_date_range: DecisionDateRange | None) -> dict[str, Any]:
    summary = (decision_date_range or DecisionDateRange()).as_summary()
    return {
        **summary,
        "document_count_with_valid_decision_date": 0,
        "document_count_in_range": 0,
        "document_count_out_of_range": 0,
        "document_count_missing_or_invalid_decision_date": 0,
    }


def _count_document_date(counter: dict[str, Any], item: dict[str, Any]) -> None:
    parsed = parse_decision_date(item.get("decision_date") or item.get("date"))
    date_from = parse_decision_date(counter.get("decision_date_from"))
    date_to = parse_decision_date(counter.get("decision_date_to"))
    if parsed is None:
        counter["document_count_missing_or_invalid_decision_date"] += 1
        return
    counter["document_count_with_valid_decision_date"] += 1
    if date_from is not None and parsed < date_from:
        counter["document_count_out_of_range"] += 1
        return
    if date_to is not None and parsed > date_to:
        counter["document_count_out_of_range"] += 1
        return
    counter["document_count_in_range"] += 1


def _document_identity(item: dict[str, Any]) -> str:
    for key in _IDENTITY_KEYS:
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return ""


def _years_from_metadata(item: dict[str, Any]) -> list[int]:
    years: list[int] = []
    for key in _DATE_KEYS:
        value = str(item.get(key) or "")
        years.extend(int(match.group(1)) for match in _YEAR_RE.finditer(value))
    return years


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Legal Retrieval v2 source inventory",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Total discovered source documents: {payload['total_discovered_source_documents']}",
        f"- Source file count: {payload['source_file_count']}",
        f"- Missing stable document identifiers: {payload['documents_missing_stable_document_identifiers']}",
        f"- Missing complete text: {payload['documents_missing_complete_text']}",
        f"- Duplicate source-document identifiers: {payload['duplicate_source_document_identifiers']}",
        f"- Unreadable files: {payload['unreadable_file_count']}",
        f"- Unsupported formats: {payload['unsupported_format_count']}",
        "",
        "## Sources",
        "",
        "| Source type | Court | Files | Raw records | Documents | Missing IDs | Missing text | Duplicate IDs | Date coverage |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for source in payload["sources"]:
        coverage = source["date_coverage"]
        date_coverage = f"{coverage.get('min_year')}..{coverage.get('max_year')}"
        lines.append(
            "| {source_type} | {court} | {files} | {raw} | {docs} | {missing_ids} | "
            "{missing_text} | {dupes} | {coverage} |".format(
                source_type=source["source_type"],
                court=source["court"],
                files=source["source_file_count"],
                raw=source["raw_record_count"],
                docs=source["discovered_document_count"],
                missing_ids=source["missing_stable_document_identifier_count"],
                missing_text=source["missing_complete_text_count"],
                dupes=source["duplicate_source_document_identifier_count"],
                coverage=date_coverage,
            )
        )
    lines.extend(["", "## Issues", ""])
    for source in payload["sources"]:
        unreadable = source["unreadable_files"]
        unsupported = source["unsupported_formats"]
        lines.append(f"### {source['source_type']}")
        if not unreadable and not unsupported:
            lines.append("- None")
        for item in unreadable[:100]:
            lines.append(f"- Unreadable `{item['path']}`: {item['issue']}")
        for item in unsupported[:100]:
            lines.append(f"- Unsupported `{item['path']}`: {item['issue']}")
        lines.append("")
    lines.extend(["## Decision Date Range", ""])
    for source in payload["sources"]:
        summary = source.get("decision_date_range") or {}
        lines.append(f"### {source['source_type']}")
        lines.append(f"- From: `{summary.get('decision_date_from')}`")
        lines.append(f"- To: `{summary.get('decision_date_to')}`")
        lines.append(f"- In range: {summary.get('document_count_in_range')}")
        lines.append(f"- Out of range: {summary.get('document_count_out_of_range')}")
        lines.append(f"- Missing/invalid date: {summary.get('document_count_missing_or_invalid_decision_date')}")
        lines.append("")
    return "\n".join(lines)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
