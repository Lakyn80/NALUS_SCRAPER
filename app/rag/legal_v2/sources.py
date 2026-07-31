from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from app.rag.legal_v2.adapters import LegalSourceDocument

PROJECT_ROOT = Path(__file__).resolve().parents[3]


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


def _document_identity(item: dict[str, Any]) -> str:
    for key in ("ecli", "source_document_id", "document_id", "case_reference", "spisova_znacka", "result_id"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return ""
