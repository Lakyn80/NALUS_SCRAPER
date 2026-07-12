"""Deterministic same-document evidence windows for legal answer evaluation."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from app.rag.eval.legal_qa_benchmark import normalize_for_match
from app.rag.retrieval.errors import RetrievalConfigurationError


@dataclass(frozen=True)
class EvidenceWindowConfig:
    enabled: bool = False
    neighbor_chunks_before: int = 1
    neighbor_chunks_after: int = 1
    max_chunks: int = 3
    max_characters: int = 6000
    require_same_document: bool = True

    def validate(self) -> None:
        if self.neighbor_chunks_before < 0:
            raise RetrievalConfigurationError("Evidence window before-neighbor count must be >= 0.")
        if self.neighbor_chunks_after < 0:
            raise RetrievalConfigurationError("Evidence window after-neighbor count must be >= 0.")
        if self.max_chunks < 1:
            raise RetrievalConfigurationError("Evidence window max_chunks must be >= 1.")
        if self.max_characters < 1:
            raise RetrievalConfigurationError("Evidence window max_characters must be >= 1.")


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    chunk_index: int
    document_id: str
    source_document_id: str
    ecli: str
    text: str
    retrieval_rank: int | None = None
    is_anchor: bool = False


@dataclass(frozen=True)
class EvidenceWindow:
    anchor_chunk_id: str | None
    document_id: str | None
    ordered_chunk_ids: list[str] = field(default_factory=list)
    ordered_chunk_indexes: list[int] = field(default_factory=list)
    combined_text: str = ""
    truncated: bool = False
    missing_neighbors: list[int] = field(default_factory=list)
    provenance_valid: bool = False
    construction_reason: str = "disabled"
    source: str | None = None
    failure_reason: str | None = None

    @property
    def used(self) -> bool:
        return self.provenance_valid and bool(self.ordered_chunk_ids)

    @property
    def same_document_neighbor_count(self) -> int:
        if not self.used:
            return 0
        return max(0, len(self.ordered_chunk_ids) - 1)


class EvidenceSource(Protocol):
    source_name: str

    def load_chunks(
        self,
        *,
        document_id: str,
        chunk_indexes: set[int],
        anchor_chunk_id: str,
    ) -> dict[int, EvidenceChunk]:
        """Return chunks keyed by chunk_index."""


@dataclass(frozen=True)
class _HitProvenance:
    chunk_id: str
    chunk_index: int
    document_id: str
    source_document_id: str
    ecli: str
    text: str
    retrieval_rank: int | None


class RetrievalArtifactEvidenceSource:
    source_name = "retrieval_artifact"

    def __init__(self, hits: list[dict[str, Any]]) -> None:
        self._chunks_by_index: dict[tuple[str, int], EvidenceChunk] = {}
        for hit in hits:
            provenance = _parse_hit_provenance(hit, require_full_text=True)
            if provenance is None:
                continue
            key = (_normalized_document_id(provenance.source_document_id), provenance.chunk_index)
            candidate = EvidenceChunk(
                chunk_id=provenance.chunk_id,
                chunk_index=provenance.chunk_index,
                document_id=provenance.document_id,
                source_document_id=provenance.source_document_id,
                ecli=provenance.ecli,
                text=provenance.text,
                retrieval_rank=provenance.retrieval_rank,
                is_anchor=False,
            )
            existing = self._chunks_by_index.get(key)
            if existing is None or candidate.chunk_id < existing.chunk_id:
                self._chunks_by_index[key] = candidate

    def load_chunks(
        self,
        *,
        document_id: str,
        chunk_indexes: set[int],
        anchor_chunk_id: str,
    ) -> dict[int, EvidenceChunk]:
        normalized = _normalized_document_id(document_id)
        return {
            index: chunk
            for index in chunk_indexes
            if (chunk := self._chunks_by_index.get((normalized, index))) is not None
        }


class Bm25SidecarEvidenceSource:
    source_name = "bm25_sidecar"

    def __init__(self, path: Path) -> None:
        self._path = path
        if not path.exists():
            raise RetrievalConfigurationError(f"Evidence-window BM25 sidecar not found: {path}")

    def load_chunks(
        self,
        *,
        document_id: str,
        chunk_indexes: set[int],
        anchor_chunk_id: str,
    ) -> dict[int, EvidenceChunk]:
        if not chunk_indexes:
            return {}
        with _connect_read_only(self._path) as connection:
            table_name = _select_chunks_table(connection)
            placeholders = ", ".join("?" for _ in chunk_indexes)
            params: list[Any] = [document_id, document_id, document_id, *sorted(chunk_indexes)]
            rows = connection.execute(
                f"""
                SELECT chunk_id, text, document_id, source_document_id, ecli, chunk_index
                FROM {table_name}
                WHERE (
                    source_document_id = ?
                    OR document_id = ?
                    OR ecli = ?
                )
                AND chunk_index IN ({placeholders})
                """,
                params,
            ).fetchall()

        chunks: dict[int, EvidenceChunk] = {}
        for row in sorted(rows, key=lambda item: (int(item[5]), str(item[0]))):
            chunk_id, text, row_document_id, row_source_document_id, row_ecli, row_chunk_index = row
            row_payload = {
                "chunk_id": chunk_id,
                "text": text,
                "document_id": row_document_id,
                "source_document_id": row_source_document_id,
                "ecli": row_ecli,
                "chunk_index": row_chunk_index,
            }
            chunk = _chunk_from_row(row_payload, expected_document_id=document_id)
            if chunk is None:
                continue
            chunks.setdefault(chunk.chunk_index, chunk)
        return chunks


def build_evidence_window(
    *,
    anchor_hit: dict[str, Any] | None,
    hits: list[dict[str, Any]],
    config: EvidenceWindowConfig,
    sidecar_path: Path | None = None,
) -> EvidenceWindow:
    config.validate()
    if not config.enabled:
        return EvidenceWindow(anchor_chunk_id=None, document_id=None)
    if anchor_hit is None:
        return EvidenceWindow(
            anchor_chunk_id=None,
            document_id=None,
            construction_reason="no_anchor_hit",
            failure_reason="No gold anchor hit is available.",
        )

    anchor = _parse_hit_provenance(anchor_hit, require_full_text=False)
    if anchor is None:
        return EvidenceWindow(
            anchor_chunk_id=str(anchor_hit.get("chunk_id") or "").strip() or None,
            document_id=None,
            construction_reason="invalid_anchor_provenance",
            failure_reason="Anchor hit lacks valid document provenance or chunk_index.",
        )

    selected_indexes = _select_bounded_indexes(
        anchor_index=anchor.chunk_index,
        before=config.neighbor_chunks_before,
        after=config.neighbor_chunks_after,
        max_chunks=config.max_chunks,
    )
    sources: list[EvidenceSource] = [RetrievalArtifactEvidenceSource(hits)]
    if sidecar_path is not None:
        sources.append(Bm25SidecarEvidenceSource(sidecar_path))

    chunks_by_index: dict[int, EvidenceChunk] = {}
    source_name: str | None = None
    for source in sources:
        loaded = source.load_chunks(
            document_id=anchor.source_document_id,
            chunk_indexes=set(selected_indexes),
            anchor_chunk_id=anchor.chunk_id,
        )
        if set(selected_indexes).issubset(loaded.keys()):
            chunks_by_index = loaded
            source_name = source.source_name
            break
        if not chunks_by_index and loaded:
            chunks_by_index = loaded
            source_name = source.source_name

    if anchor.chunk_index not in chunks_by_index:
        anchor_text = anchor.text or str(anchor_hit.get("text_snippet") or "")
        chunks_by_index[anchor.chunk_index] = EvidenceChunk(
            chunk_id=anchor.chunk_id,
            chunk_index=anchor.chunk_index,
            document_id=anchor.document_id,
            source_document_id=anchor.source_document_id,
            ecli=anchor.ecli,
            text=anchor_text,
            retrieval_rank=anchor.retrieval_rank,
            is_anchor=True,
        )
        source_name = source_name or "anchor_snippet"

    ordered_chunks = [
        _mark_anchor(chunks_by_index[index], anchor.chunk_id)
        for index in sorted(chunks_by_index)
        if index in selected_indexes
    ]
    missing_neighbors = [
        index
        for index in selected_indexes
        if index != anchor.chunk_index and index not in chunks_by_index
    ]
    if config.require_same_document:
        invalid = [
            chunk.chunk_id
            for chunk in ordered_chunks
            if not _same_document(chunk.source_document_id, anchor.source_document_id)
            or (chunk.ecli and anchor.ecli and not _same_document(chunk.ecli, anchor.ecli))
        ]
        if invalid:
            return EvidenceWindow(
                anchor_chunk_id=anchor.chunk_id,
                document_id=anchor.source_document_id,
                construction_reason="provenance_mismatch",
                source=source_name,
                failure_reason=f"Evidence chunks crossed document boundary: {', '.join(invalid)}.",
            )

    combined_text = "\n\n".join(chunk.text.strip() for chunk in ordered_chunks if chunk.text.strip())
    truncated = False
    if len(combined_text) > config.max_characters:
        combined_text = combined_text[: config.max_characters]
        truncated = True

    return EvidenceWindow(
        anchor_chunk_id=anchor.chunk_id,
        document_id=anchor.source_document_id,
        ordered_chunk_ids=[chunk.chunk_id for chunk in ordered_chunks],
        ordered_chunk_indexes=[chunk.chunk_index for chunk in ordered_chunks],
        combined_text=combined_text,
        truncated=truncated,
        missing_neighbors=missing_neighbors,
        provenance_valid=True,
        construction_reason="constructed",
        source=source_name,
    )


def _parse_hit_provenance(
    hit: dict[str, Any],
    *,
    require_full_text: bool,
) -> _HitProvenance | None:
    metadata = dict(hit.get("metadata") or {})
    chunk_metadata = _parse_chunk_metadata(metadata)
    chunk_id = str(
        hit.get("chunk_id")
        or metadata.get("chunk_id")
        or chunk_metadata.get("chunk_id")
        or ""
    ).strip()
    chunk_index = _coerce_int(
        hit.get("chunk_index")
        or metadata.get("chunk_index")
        or chunk_metadata.get("chunk_index")
    )
    document_id = _first_text(hit, metadata, chunk_metadata, ("document_id",))
    source_document_id = _first_text(hit, metadata, chunk_metadata, ("source_document_id",))
    ecli = _first_text(hit, metadata, chunk_metadata, ("ecli",))
    stable_id = source_document_id or document_id or ecli
    text = _full_text_from_hit(hit, metadata)
    if require_full_text and not text:
        return None
    if not chunk_id or chunk_index is None or not stable_id:
        return None
    source_document_id = source_document_id or stable_id
    document_id = document_id or stable_id
    ecli = ecli or stable_id
    if not _consistent_identity(source_document_id, document_id, ecli):
        return None
    return _HitProvenance(
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        document_id=document_id,
        source_document_id=source_document_id,
        ecli=ecli,
        text=text,
        retrieval_rank=_coerce_int(hit.get("rank")),
    )


def _chunk_from_row(row: dict[str, Any], *, expected_document_id: str) -> EvidenceChunk | None:
    chunk_id = str(row.get("chunk_id") or "").strip()
    text = str(row.get("text") or "").strip()
    chunk_index = _coerce_int(row.get("chunk_index"))
    source_document_id = str(row.get("source_document_id") or "").strip()
    document_id = str(row.get("document_id") or "").strip()
    ecli = str(row.get("ecli") or "").strip()
    stable_id = source_document_id or document_id or ecli
    if not chunk_id or not text or chunk_index is None or not stable_id:
        return None
    source_document_id = source_document_id or stable_id
    document_id = document_id or stable_id
    ecli = ecli or stable_id
    if not _same_document(source_document_id, expected_document_id):
        return None
    if not _consistent_identity(source_document_id, document_id, ecli):
        return None
    return EvidenceChunk(
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        document_id=document_id,
        source_document_id=source_document_id,
        ecli=ecli,
        text=text,
    )


def _parse_chunk_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    raw = metadata.get("chunk_metadata")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return dict(decoded) if isinstance(decoded, dict) else {}
    return {}


def _first_text(
    hit: dict[str, Any],
    metadata: dict[str, Any],
    chunk_metadata: dict[str, Any],
    keys: tuple[str, ...],
) -> str:
    for key in keys:
        value = hit.get(key) or metadata.get(key) or chunk_metadata.get(key)
        if value not in {None, ""}:
            return str(value).strip()
    return ""


def _full_text_from_hit(hit: dict[str, Any], metadata: dict[str, Any]) -> str:
    for value in (hit.get("text"), metadata.get("text")):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _coerce_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _consistent_identity(*values: str) -> bool:
    present = [_normalized_document_id(value) for value in values if value]
    return len(set(present)) <= 1


def _same_document(left: str, right: str) -> bool:
    return _normalized_document_id(left) == _normalized_document_id(right)


def _normalized_document_id(value: str) -> str:
    return normalize_for_match(value)


def _select_bounded_indexes(
    *,
    anchor_index: int,
    before: int,
    after: int,
    max_chunks: int,
) -> list[int]:
    requested = list(range(anchor_index - before, anchor_index + after + 1))
    if len(requested) <= max_chunks:
        return requested
    closest = sorted(requested, key=lambda index: (abs(index - anchor_index), index))[:max_chunks]
    return sorted(closest)


def _mark_anchor(chunk: EvidenceChunk, anchor_chunk_id: str) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk.chunk_id,
        chunk_index=chunk.chunk_index,
        document_id=chunk.document_id,
        source_document_id=chunk.source_document_id,
        ecli=chunk.ecli,
        text=chunk.text,
        retrieval_rank=chunk.retrieval_rank,
        is_anchor=chunk.chunk_id == anchor_chunk_id,
    )


def _connect_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _select_chunks_table(connection: sqlite3.Connection) -> str:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    for table_name in ("bm25_chunks", "chunks", "rag_chunks"):
        if table_name in tables:
            return table_name
    raise RetrievalConfigurationError("Evidence sidecar does not contain a supported chunks table.")
