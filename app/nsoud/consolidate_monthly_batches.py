from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ConsolidationStats:
    total_input_records: int
    total_output_records: int
    duplicate_content_hash_skipped: int
    duplicate_url_skipped: int
    source_months: list[str]
    sidecar_manifest_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate Czech Supreme Court monthly JSONL batches.")
    parser.add_argument("--manifest", type=Path, required=True, help="Monthly manifest JSON path.")
    parser.add_argument("--out", type=Path, required=True, help="Consolidated JSONL output path.")
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


def month_key(year: Any, month: Any) -> str:
    return f"{int(year):04d}-{int(month):02d}"


def load_successful_month_records(manifest_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    payload = load_json(manifest_path)
    batches = payload.get("batches")
    if not isinstance(batches, list):
        raise ValueError("Manifest field `batches` must be a list.")

    all_records: list[dict[str, Any]] = []
    source_months: list[str] = []

    for entry in batches:
        if not isinstance(entry, dict):
            raise ValueError("Every manifest batch entry must be an object.")

        if normalize_text(entry.get("status")) != "success":
            continue

        output_path_raw = normalize_text(entry.get("output_path"))
        if not output_path_raw:
            raise ValueError(f"Successful manifest entry is missing output_path: {entry}")

        output_path = resolve_data_path(output_path_raw, manifest_path)
        if not output_path.exists():
            raise ValueError(f"Successful monthly output is missing: {output_path_raw}")

        records = load_jsonl_records(output_path)
        all_records.extend(records)
        source_months.append(month_key(entry.get("year"), entry.get("month")))

    return all_records, source_months


def sort_key(record: dict[str, Any]) -> tuple[tuple[int, str], tuple[int, str], str, str]:
    publication_date = normalize_text(record.get("publication_date"))
    decision_date = normalize_text(record.get("decision_date"))
    case_number = normalize_text(record.get("case_number")).lower()
    url = normalize_text(record.get("url")).lower()

    publication_part = (0, publication_date) if publication_date else (1, "")
    decision_part = (0, decision_date) if decision_date else (1, "")
    return publication_part, decision_part, case_number, url


def sidecar_manifest_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}_manifest.json")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def write_sidecar_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def consolidate_records(
    manifest_path: Path,
    out_path: Path,
) -> ConsolidationStats:
    input_records, source_months = load_successful_month_records(manifest_path)

    seen_content_hashes: set[str] = set()
    seen_urls: set[str] = set()
    duplicate_content_hash_skipped = 0
    duplicate_url_skipped = 0
    deduplicated: list[dict[str, Any]] = []

    for record in input_records:
        content_hash = normalize_text(record.get("content_hash"))
        url = normalize_text(record.get("url"))

        if content_hash and content_hash in seen_content_hashes:
            duplicate_content_hash_skipped += 1
            continue

        if url and url in seen_urls:
            duplicate_url_skipped += 1
            continue

        if content_hash:
            seen_content_hashes.add(content_hash)
        if url:
            seen_urls.add(url)
        deduplicated.append(record)

    deduplicated.sort(key=sort_key)
    write_jsonl(out_path, deduplicated)

    sidecar_path = sidecar_manifest_path(out_path)
    sidecar_payload = {
        "input_manifest": str(manifest_path),
        "output_path": str(out_path),
        "total_input_records": len(input_records),
        "total_output_records": len(deduplicated),
        "duplicate_content_hash_skipped": duplicate_content_hash_skipped,
        "duplicate_url_skipped": duplicate_url_skipped,
        "source_months": source_months,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_sidecar_manifest(sidecar_path, sidecar_payload)

    return ConsolidationStats(
        total_input_records=len(input_records),
        total_output_records=len(deduplicated),
        duplicate_content_hash_skipped=duplicate_content_hash_skipped,
        duplicate_url_skipped=duplicate_url_skipped,
        source_months=source_months,
        sidecar_manifest_path=sidecar_path,
    )


def main() -> int:
    args = parse_args()

    try:
        stats = consolidate_records(args.manifest, args.out)
    except Exception as exc:
        print(f"consolidation failed: {exc}")
        return 1

    print(f"total input records: {stats.total_input_records}")
    print(f"total output records: {stats.total_output_records}")
    print(f"duplicate content_hash skipped: {stats.duplicate_content_hash_skipped}")
    print(f"duplicate URL skipped: {stats.duplicate_url_skipped}")
    print(f"output path: {args.out}")
    print(f"sidecar manifest path: {stats.sidecar_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
