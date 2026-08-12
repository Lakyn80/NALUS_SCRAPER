"""Persistent ColBERT id ↔ chunk/document mapping (JSONL beside the index)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from app.rag.legal_v2.retrieve.colbert.errors import ColbertMappingError


@dataclass(frozen=True)
class ColbertMappingRow:
    colbert_id: str
    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "colbert_id": self.colbert_id,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "metadata": dict(self.metadata),
        }


@dataclass
class ColbertChunkMapping:
    """In-memory mapping keyed by ColBERT passage id (chunk_id by default)."""

    rows: dict[str, ColbertMappingRow] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.rows)

    def get(self, colbert_id: str) -> ColbertMappingRow | None:
        return self.rows.get(str(colbert_id))

    def require(self, colbert_id: str) -> ColbertMappingRow:
        row = self.get(colbert_id)
        if row is None:
            raise ColbertMappingError(
                f"ColBERT hit id {colbert_id!r} is missing from chunk mapping"
            )
        return row

    def add(self, row: ColbertMappingRow) -> None:
        key = str(row.colbert_id)
        if key in self.rows:
            raise ColbertMappingError(f"duplicate colbert_id in mapping: {key!r}")
        self.rows[key] = row

    def integrity(self, *, expected_chunk_ids: set[str] | None = None) -> dict[str, int]:
        chunk_ids = [row.chunk_id for row in self.rows.values()]
        empty_texts = sum(1 for row in self.rows.values() if not str(row.text or "").strip())
        duplicates = len(chunk_ids) - len(set(chunk_ids))
        missing = 0
        if expected_chunk_ids is not None:
            missing = len(expected_chunk_ids - set(chunk_ids))
        return {
            "mapping_row_count": len(self.rows),
            "duplicate_chunk_ids": duplicates,
            "missing_chunk_ids": missing,
            "empty_texts": empty_texts,
        }


def write_mapping_jsonl(path: Path, rows: Iterable[ColbertMappingRow]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.as_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def load_mapping_jsonl(path: Path) -> ColbertChunkMapping:
    if not path.exists():
        raise ColbertMappingError(f"ColBERT mapping file missing: {path}")
    mapping = ColbertChunkMapping()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            cleaned = line.strip()
            if not cleaned:
                continue
            try:
                payload = json.loads(cleaned)
            except json.JSONDecodeError as exc:
                raise ColbertMappingError(
                    f"invalid JSONL at {path}:{line_no}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise ColbertMappingError(f"mapping row must be an object at line {line_no}")
            colbert_id = str(payload.get("colbert_id") or "").strip()
            chunk_id = str(payload.get("chunk_id") or "").strip()
            document_id = str(payload.get("document_id") or "").strip()
            text = str(payload.get("text") or "")
            metadata = payload.get("metadata") or {}
            if not colbert_id or not chunk_id or not document_id:
                raise ColbertMappingError(
                    f"mapping row missing ids at line {line_no}"
                )
            if not isinstance(metadata, dict):
                raise ColbertMappingError(
                    f"mapping metadata must be an object at line {line_no}"
                )
            mapping.add(
                ColbertMappingRow(
                    colbert_id=colbert_id,
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=text,
                    metadata=dict(metadata),
                )
            )
    return mapping
