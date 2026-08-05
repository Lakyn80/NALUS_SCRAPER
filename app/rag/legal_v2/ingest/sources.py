from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from app.rag.legal_v2.adapters import LegalSourceDocument

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CZECH_DATE_RE = re.compile(r"^\s*(\d{1,2})\.\s*(\d{1,2})\.\s*(\d{4})\s*$")


@dataclass(frozen=True)
class DecisionDateRange:
    date_from: date | None = None
    date_to: date | None = None

    def as_summary(self) -> dict[str, str | None]:
        return {
            "decision_date_from": self.date_from.isoformat() if self.date_from else None,
            "decision_date_to": self.date_to.isoformat() if self.date_to else None,
        }


@dataclass(frozen=True)
class DecisionDateFilterResult:
    documents: list[LegalSourceDocument]
    summary: dict[str, Any]


def discover_source_documents(
    *,
    batches_dir: Path | None = None,
    nsoud_chunks_path: Path | None = None,
    limit: int | None = None,
) -> list[LegalSourceDocument]:
    documents: list[LegalSourceDocument] = []
    documents.extend(_load_nalus_batches(batches_dir or PROJECT_ROOT / "batches", limit=limit))
    if limit is not None and len(documents) >= limit:
        return documents[:limit]
    remaining = None if limit is None else max(0, limit - len(documents))
    documents.extend(
        _load_nsoud_chunks(
            nsoud_chunks_path
            or PROJECT_ROOT / "app/artifacts/nsoud/rag_ready/nsoud_chunks_2025_01_03.jsonl",
            limit=remaining,
        )
    )
    return documents[:limit] if limit is not None else documents


def discover_source_documents_by_ids(
    document_ids: list[str],
    *,
    batches_dir: Path | None = None,
    nsoud_chunks_path: Path | None = None,
) -> list[LegalSourceDocument]:
    requested = {document_id for document_id in document_ids if document_id}
    if not requested:
        return []
    documents: list[LegalSourceDocument] = []
    documents.extend(_load_nalus_batches_by_ids(batches_dir or PROJECT_ROOT / "batches", requested))
    found = {document.document_id for document in documents}
    remaining = requested - found
    if remaining:
        documents.extend(
            _load_nsoud_chunks_by_ids(
                nsoud_chunks_path
                or PROJECT_ROOT / "app/artifacts/nsoud/rag_ready/nsoud_chunks_2025_01_03.jsonl",
                remaining,
            )
        )
    by_id = {document.document_id: document for document in documents}
    return [by_id[document_id] for document_id in document_ids if document_id in by_id]


def parse_iso_decision_date(value: str, *, field_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must use YYYY-MM-DD format.") from exc


def parse_decision_date(value: Any) -> date | None:
    cleaned = str(value or "").strip()
    if not cleaned:
        return None
    try:
        return date.fromisoformat(cleaned)
    except ValueError:
        pass
    czech_match = _CZECH_DATE_RE.match(cleaned)
    if czech_match:
        day = int(czech_match.group(1))
        month = int(czech_match.group(2))
        year = int(czech_match.group(3))
        try:
            return date(year, month, day)
        except ValueError:
            return None
    return None


def filter_source_documents_by_decision_date(
    documents: list[LegalSourceDocument],
    date_range: DecisionDateRange,
) -> DecisionDateFilterResult:
    if date_range.date_from is not None and date_range.date_to is not None and date_range.date_from > date_range.date_to:
        raise ValueError("decision_date_from must be before or equal to decision_date_to.")

    parsed_dates: list[date] = []
    kept: list[LegalSourceDocument] = []
    missing_or_invalid = 0
    out_of_range = 0
    for document in documents:
        parsed = parse_decision_date(_decision_date_value(document.metadata))
        if parsed is None:
            missing_or_invalid += 1
            continue
        parsed_dates.append(parsed)
        if date_range.date_from is not None and parsed < date_range.date_from:
            out_of_range += 1
            continue
        if date_range.date_to is not None and parsed > date_range.date_to:
            out_of_range += 1
            continue
        kept.append(document)

    summary = {
        **date_range.as_summary(),
        "source_document_count_before_date_filter": len(documents),
        "date_filtered_document_count": len(kept),
        "date_missing_or_invalid_document_count": missing_or_invalid,
        "date_out_of_range_document_count": out_of_range,
        "date_min": min(parsed_dates).isoformat() if parsed_dates else None,
        "date_max": max(parsed_dates).isoformat() if parsed_dates else None,
    }
    return DecisionDateFilterResult(documents=kept, summary=summary)


def _load_nalus_batches(path: Path, *, limit: int | None) -> list[LegalSourceDocument]:
    if not path.exists():
        return []
    files = _manifest_files(path)
    documents: list[LegalSourceDocument] = []
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
            text = str(item.get("full_text") or "").strip()
            document_id = _document_identity(item)
            if not text or not document_id or document_id in seen:
                continue
            seen.add(document_id)
            metadata = dict(item)
            metadata["source"] = "constitutional"
            documents.append(
                LegalSourceDocument(
                    document_id=document_id,
                    source="constitutional",
                    text=text,
                    metadata=metadata,
                    origin_path=str(file_path),
                )
            )
            if limit is not None and len(documents) >= limit:
                return documents
    return documents


def _load_nalus_batches_by_ids(path: Path, document_ids: set[str]) -> list[LegalSourceDocument]:
    if not path.exists():
        return []
    documents: list[LegalSourceDocument] = []
    seen: set[str] = set()
    for file_path in _manifest_files(path):
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            document_id = _document_identity(item)
            text = str(item.get("full_text") or "").strip()
            if document_id not in document_ids or document_id in seen or not text:
                continue
            seen.add(document_id)
            metadata = dict(item)
            metadata["source"] = "constitutional"
            documents.append(
                LegalSourceDocument(
                    document_id=document_id,
                    source="constitutional",
                    text=text,
                    metadata=metadata,
                    origin_path=str(file_path),
                )
            )
            if seen == document_ids:
                return documents
    return documents


def _manifest_files(batches_dir: Path) -> list[Path]:
    manifest_path = batches_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        files = [
            batches_dir / str(entry.get("file"))
            for entry in manifest.get("batches", [])
            if isinstance(entry, dict) and entry.get("file")
        ]
        existing = [path for path in files if path.exists() and path.name != "manifest.json"]
        if existing:
            return existing
    return sorted(path for path in batches_dir.glob("*.json") if path.name != "manifest.json")


def _load_nsoud_chunks(path: Path, *, limit: int | None) -> list[LegalSourceDocument]:
    if limit == 0 or not path.exists():
        return []
    chunks_by_document: dict[str, list[dict[str, Any]]] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                document_id = _document_identity(item)
                text = str(item.get("text") or item.get("chunk_text") or "").strip()
                if not document_id or not text:
                    continue
                chunks_by_document.setdefault(document_id, []).append(item)
                if limit is not None and len(chunks_by_document) >= limit:
                    break
    except OSError:
        return []

    documents: list[LegalSourceDocument] = []
    for document_id, chunks in sorted(chunks_by_document.items()):
        ordered = sorted(chunks, key=lambda item: int(item.get("chunk_index") or 0))
        text = "\n\n".join(str(item.get("text") or item.get("chunk_text") or "") for item in ordered)
        metadata = _merge_metadata(ordered)
        metadata["source"] = "supreme"
        documents.append(
            LegalSourceDocument(
                document_id=document_id,
                source="supreme",
                text=text,
                metadata=metadata,
                origin_path=str(path),
            )
        )
    return documents[:limit] if limit is not None else documents


def _load_nsoud_chunks_by_ids(path: Path, document_ids: set[str]) -> list[LegalSourceDocument]:
    if not path.exists():
        return []
    chunks_by_document: dict[str, list[dict[str, Any]]] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                document_id = _document_identity(item)
                text = str(item.get("text") or item.get("chunk_text") or "").strip()
                if document_id in document_ids and text:
                    chunks_by_document.setdefault(document_id, []).append(item)
    except OSError:
        return []
    documents: list[LegalSourceDocument] = []
    for document_id, chunks in chunks_by_document.items():
        ordered = sorted(chunks, key=lambda item: int(item.get("chunk_index") or 0))
        text = "\n\n".join(str(item.get("text") or item.get("chunk_text") or "") for item in ordered)
        metadata = _merge_metadata(ordered)
        metadata["source"] = "supreme"
        documents.append(
            LegalSourceDocument(
                document_id=document_id,
                source="supreme",
                text=text,
                metadata=metadata,
                origin_path=str(path),
            )
        )
    by_id = {document.document_id: document for document in documents}
    return [by_id[document_id] for document_id in document_ids if document_id in by_id]


def _merge_metadata(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in items:
        for key, value in item.items():
            if key not in {"text", "chunk_text"} and value not in {None, ""}:
                merged.setdefault(key, value)
    return merged


def _decision_date_value(metadata: dict[str, Any]) -> Any:
    for key in ("decision_date", "date"):
        value = metadata.get(key)
        if value not in {None, ""}:
            return value
    return None


def _document_identity(item: dict[str, Any]) -> str:
    """Resolve production document identity. Prefer verified ECLI."""
    from app.rag.legal_v2.identity import is_valid_ecli, normalize_ecli

    ecli = str(item.get("ecli") or "").strip()
    if ecli and is_valid_ecli(ecli):
        return normalize_ecli(ecli)
    for key in ("canonical_document_id", "document_id"):
        value = str(item.get(key) or "").strip()
        if value and is_valid_ecli(value):
            return normalize_ecli(value)
    for key in ("source_document_id", "case_reference", "spisova_znacka", "result_id"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return ""
