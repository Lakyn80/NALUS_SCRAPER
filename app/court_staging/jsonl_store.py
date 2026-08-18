"""JSONL helpers for court staging with canonical_id upserts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from app.court_staging.identity import ChangeKind, classify_content_change, enrich_record_identity


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def load_canonical_index(paths: list[Path]) -> dict[str, str]:
    """canonical_id → content_hash (last wins)."""
    index: dict[str, str] = {}
    for path in paths:
        for record in iter_jsonl(path):
            enriched = enrich_record_identity(record)
            cid = str(enriched.get("canonical_id") or "")
            ch = str(enriched.get("content_hash") or "")
            if cid and ch:
                index[cid] = ch
    return index


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def rewrite_jsonl_upsert(
    path: Path,
    record: dict[str, Any],
    *,
    known: dict[str, str],
    source: str | None = None,
) -> ChangeKind:
    """Append NEW, or rewrite file replacing UPDATED canonical_id row.

    UNCHANGED returns without write. Updates ``known`` in place.
    """
    enriched = enrich_record_identity(record, source=source)
    canonical_id = str(enriched["canonical_id"])
    content_hash = str(enriched["content_hash"])
    kind = classify_content_change(
        canonical_id=canonical_id,
        content_hash=content_hash,
        known=known,
    )
    if kind is ChangeKind.UNCHANGED:
        return kind

    path.parent.mkdir(parents=True, exist_ok=True)

    if kind is ChangeKind.NEW or not path.exists():
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(enriched, ensure_ascii=False))
            handle.write("\n")
        known[canonical_id] = content_hash
        return kind

    # UPDATED: rewrite excluding old canonical_id, append new version.
    rows: list[dict[str, Any]] = []
    for existing in iter_jsonl(path):
        existing_enriched = enrich_record_identity(existing, source=source)
        if existing_enriched.get("canonical_id") == canonical_id:
            continue
        rows.append(existing_enriched)
    rows.append(enriched)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    temp.replace(path)
    known[canonical_id] = content_hash
    return kind
