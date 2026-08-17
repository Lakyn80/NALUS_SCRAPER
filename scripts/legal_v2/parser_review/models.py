from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REVIEW_SCHEMA_VERSION = "parser-visual-review.v1"
REVIEW_TOOL_VERSION = "visual-parser-review-tool.v1"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STUDY_DIR = PROJECT_ROOT / "artifacts" / "legal_v2" / "court_format_study"
DEFAULT_REVIEW_DIR = PROJECT_ROOT / "artifacts" / "legal_v2" / "visual_parser_review"

MANUAL_STATUSES = {"pending", "accepted", "overridden", "unresolved"}
DECISION_STATUSES = {"accepted", "overridden", "unresolved"}
ITEM_TYPES = {"line", "boundary", "document_approval"}
INTERFACES = {"powershell", "html", "validator", "assisted_html", "assisted_cli", "parser_profile_migration"}
MANUAL_LINE_CLASSES = {
    "metadata",
    "heading",
    "numbered_paragraph_start",
    "numbered_paragraph_continuation",
    "prose_start",
    "prose_continuation",
    "citation_continuation",
    "list_or_table",
    "signature",
    "instruction",
    "layout_noise",
    "unresolved",
}
MANUAL_BOUNDARY_DECISIONS = {"split", "merge", "preserve_parser", "unresolved"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def digest_text(*parts: object, length: int = 20) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def document_review_id(*, source_id: str, source_checksum: str, manifest_checksum: str) -> str:
    return f"doc-{digest_text(REVIEW_SCHEMA_VERSION, manifest_checksum, source_id, source_checksum, length=16)}"


def line_review_id(*, document_id: str, raw_line_number: int, source_checksum: str) -> str:
    return f"{document_id}:line:{raw_line_number:05d}:{digest_text(document_id, raw_line_number, source_checksum, length=12)}"


def boundary_review_id(
    *,
    document_id: str,
    previous_line_id: str,
    next_line_id: str,
    source_checksum: str,
) -> str:
    return f"{document_id}:boundary:{digest_text(previous_line_id, next_line_id, source_checksum, length=16)}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL in {path} at line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Invalid JSONL record in {path} at line {line_number}: expected object")
        records.append(value)
    return records


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
