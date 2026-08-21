"""Build the document-level judgment archive index from NALUS batch records.

Source of truth for Constitutional Court metadata is the document-level batch
JSON (one decision per record), not Qdrant chunks. Full judgment text is never
copied into the archive index.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from app.rag.legal_v2.archive.courts import COURT_CONSTITUTIONAL, normalize_court_id
from app.rag.legal_v2.archive.models import ArchiveDecision
from app.rag.legal_v2.archive.store import write_archive_index
from app.rag.legal_v2.identity import is_valid_ecli, normalize_ecli
from app.rag.legal_v2.ingest.sources import parse_decision_date

_CASE_REF_FRAGMENT_RE = re.compile(r"\s+#\d+\s*$")
_ECLI_YEAR_RE = re.compile(r"^ECLI:CZ:[A-Z]{2,8}:(\d{4}):", re.IGNORECASE)


def normalize_archive_document(raw: dict[str, Any]) -> ArchiveDecision | None:
    """Map a source record to archive metadata; never invent a title."""
    ecli = _resolve_ecli(raw)
    if not ecli:
        return None

    court = _resolve_court(raw, ecli)
    if not court:
        return None

    parsed_date = parse_decision_date(
        raw.get("decision_date") or raw.get("date") or raw.get("publication_date")
    )
    if parsed_date is not None:
        year = parsed_date.year
        month = parsed_date.month
        decision_date = parsed_date.isoformat()
    else:
        year = _year_from_ecli(ecli) or _year_from_raw(raw)
        if year is None:
            return None
        month = 1
        decision_date = None

    case_number = _real_optional_text(
        raw.get("case_number"),
        raw.get("case_reference"),
        raw.get("spisova_znacka"),
    )
    if case_number:
        case_number = _CASE_REF_FRAGMENT_RE.sub("", case_number).strip() or None

    document_type = _real_optional_text(
        raw.get("document_type"),
        raw.get("decision_form"),
        raw.get("decision_type"),
        raw.get("form_decision"),
    )
    # Real titles only (NALUS popular_name). Never synthesize from topics/keywords.
    title = _real_optional_text(raw.get("title"), raw.get("popular_name"))

    return ArchiveDecision(
        canonical_document_id=ecli,
        ecli=ecli,
        case_number=case_number,
        court=court,
        decision_date=decision_date,
        year=int(year),
        month=int(month),
        document_type=document_type,
        title=title,
    )


def build_archive_index_from_batches(
    *,
    batches_dir: Path,
    sqlite_path: Path,
    court_id: str = COURT_CONSTITUTIONAL,
) -> dict[str, Any]:
    court = normalize_court_id(court_id) or COURT_CONSTITUTIONAL
    files, manifest = _manifest_files(batches_dir)
    documents = list(
        _iter_unique_documents_from_batch_files(files, default_court=court)
    )
    built_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    count = write_archive_index(
        sqlite_path=sqlite_path,
        documents=documents,
        source_kind="nalus_batches_document_metadata",
        built_at=built_at,
    )
    return {
        "schema": "judgment_archive_index_v1",
        "sqlite_path": str(sqlite_path.resolve()),
        "source_kind": "nalus_batches_document_metadata",
        "batches_dir": str(batches_dir.resolve()),
        "manifest_batch_count": len(manifest.get("batches") or []),
        "batch_files_read": len(files),
        "unique_documents": count,
        "court": court,
        "built_at": built_at,
        "notes": (
            "Document-level metadata only; full_text is never stored. "
            "Deduplicated by canonical ECLI document identity."
        ),
    }


def build_archive_index_from_records(
    *,
    records: Iterable[dict[str, Any]],
    sqlite_path: Path,
    source_kind: str = "records",
) -> int:
    built_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    documents = [
        document
        for document in (normalize_archive_document(raw) for raw in records)
        if document is not None
    ]
    return write_archive_index(
        sqlite_path=sqlite_path,
        documents=documents,
        source_kind=source_kind,
        built_at=built_at,
    )


def _iter_unique_documents_from_batch_files(
    files: list[Path],
    *,
    default_court: str,
) -> Iterator[ArchiveDecision]:
    seen: set[str] = set()
    for file_path in files:
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            enriched = dict(item)
            if not enriched.get("court"):
                enriched["court"] = default_court
            document = normalize_archive_document(enriched)
            if document is None:
                continue
            key = document.canonical_document_id.casefold()
            if key in seen:
                continue
            seen.add(key)
            yield document


def _manifest_files(batches_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    manifest_path = batches_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files: list[Path] = []
    missing: list[str] = []
    for entry in manifest.get("batches") or []:
        name = str(entry.get("file") or "").strip()
        if not name:
            continue
        path = batches_dir / name
        if path.exists():
            files.append(path)
        else:
            missing.append(name)
    if missing:
        raise FileNotFoundError(
            f"manifest files missing on disk: {missing[:10]}"
        )
    return files, manifest


def _resolve_ecli(raw: dict[str, Any]) -> str | None:
    for key in (
        "ecli",
        "canonical_document_id",
        "document_id",
    ):
        value = str(raw.get(key) or "").strip()
        if value and is_valid_ecli(value):
            return normalize_ecli(value)
    return None


def _resolve_court(raw: dict[str, Any], ecli: str) -> str | None:
    explicit = normalize_court_id(str(raw.get("court") or ""))
    if explicit:
        return explicit
    parts = ecli.split(":")
    if len(parts) >= 3:
        code = parts[2].upper()
        if code == "US":
            return COURT_CONSTITUTIONAL
        if code == "NS":
            return normalize_court_id("supreme_court")
        if code == "NSS":
            return normalize_court_id("supreme_administrative_court")
    return COURT_CONSTITUTIONAL


def _year_from_ecli(ecli: str) -> int | None:
    match = _ECLI_YEAR_RE.match(ecli)
    if not match:
        return None
    return int(match.group(1))


def _year_from_raw(raw: dict[str, Any]) -> int | None:
    for key in ("year",):
        try:
            value = int(raw.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if 1900 <= value <= 2100:
            return value
    return None


def _real_optional_text(*values: Any) -> str | None:
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        lowered = text.casefold()
        if lowered in {"none", "null", "n/a", "neuvedeno", "-"}:
            continue
        return text
    return None
