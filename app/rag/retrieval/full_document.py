"""Read-only full document reconstruction from indexed retrieval chunks.

This module deliberately does not perform ranking, reranking, embedding, query
rewrite, or document-level retrieval aggregation. It only reconstructs one
already identified document from same-document chunks stored in Qdrant.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from app.core.logging import get_logger
from app.rag.retrieval.production_profile import DEFAULT_QDRANT_COLLECTION

logger = get_logger(__name__)

FullTextAvailabilityStatus = Literal["available", "partial", "not_found"]

_MAX_DOCUMENT_ID_LENGTH = 256
_DEFAULT_SCROLL_PAGE_SIZE = 256
_DEFAULT_MAX_CHUNKS_PER_DOCUMENT = 2_000
_DOCUMENT_ID_KEYS = (
    "source_document_id",
    "document_id",
    "ecli",
    "case_reference",
    "case_number",
    "reference",
)
_TEXT_KEYS = ("text", "chunk_text", "content")
_CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x1f\x7f]")


class FullDocumentLookupError(RuntimeError):
    """Raised when the underlying document store cannot complete the lookup."""


@dataclass(frozen=True)
class FullDocumentChunk:
    chunk_id: str
    chunk_index: int | None
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FullDocumentDiagnostics:
    collection_name: str
    chunk_count: int
    missing_chunk_indexes: list[int]
    duplicate_chunk_indexes: list[int]
    all_chunks_have_index: bool
    reconstruction_method: str
    truncated: bool = False
    max_chunks: int = _DEFAULT_MAX_CHUNKS_PER_DOCUMENT


@dataclass(frozen=True)
class FullDocumentResult:
    document_id: str
    metadata: dict[str, Any]
    full_text: str
    chunks: list[FullDocumentChunk]
    source_url: str | None
    provenance_status: str
    full_text_availability_status: FullTextAvailabilityStatus
    diagnostics: FullDocumentDiagnostics


class FullDocumentStore(Protocol):
    def get(self, document_id: str) -> FullDocumentResult | None: ...


class _ScrollableQdrantClient(Protocol):
    def scroll(
        self,
        *,
        collection_name: str,
        scroll_filter: Any,
        limit: int,
        offset: Any = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> tuple[list[Any], Any]: ...


class QdrantFullDocumentStore:
    """Qdrant-backed read-only full-document store."""

    def __init__(
        self,
        *,
        qdrant_url: str | None = None,
        collection_name: str | None = None,
        max_chunks_per_document: int | None = None,
        scroll_page_size: int = _DEFAULT_SCROLL_PAGE_SIZE,
        client: _ScrollableQdrantClient | None = None,
    ) -> None:
        self._qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://qdrant:6333")
        self._collection_name = (
            collection_name
            or os.getenv("QDRANT_COLLECTION_NAME")
            or DEFAULT_QDRANT_COLLECTION
        )
        self._max_chunks_per_document = (
            max_chunks_per_document
            if max_chunks_per_document is not None
            else _max_chunks_from_env()
        )
        self._scroll_page_size = scroll_page_size
        self._client = client

    def get(self, document_id: str) -> FullDocumentResult | None:
        normalized_id = validate_document_id(document_id)
        payloads = self._scroll_payloads(normalized_id)
        result = build_full_document_from_payloads(
            document_id=normalized_id,
            payloads=payloads,
            collection_name=self._collection_name,
            max_chunks=self._max_chunks_per_document,
        )
        if result.full_text_availability_status == "not_found":
            return None
        return result

    def _scroll_payloads(self, document_id: str) -> list[dict[str, Any]]:
        try:
            client = self._client or _make_qdrant_client(self._qdrant_url)
            scroll_filter = _document_id_filter(document_id)
        except Exception as exc:  # noqa: BLE001
            raise FullDocumentLookupError("Full document store is not configured.") from exc

        payloads: list[dict[str, Any]] = []
        next_offset: Any = None
        limit = max(1, min(self._scroll_page_size, self._max_chunks_per_document))

        try:
            while len(payloads) < self._max_chunks_per_document:
                remaining = self._max_chunks_per_document - len(payloads)
                page_limit = min(limit, remaining)
                points, next_offset = client.scroll(
                    collection_name=self._collection_name,
                    scroll_filter=scroll_filter,
                    limit=page_limit,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False,
                )
                if not points:
                    break
                payloads.extend(_payload_from_point(point) for point in points)
                if next_offset is None:
                    break
        except Exception as exc:  # noqa: BLE001
            raise FullDocumentLookupError("Full document lookup failed.") from exc

        return payloads


def validate_document_id(document_id: str) -> str:
    value = str(document_id).strip()
    if not value:
        raise ValueError("Document id must not be empty.")
    if len(value) > _MAX_DOCUMENT_ID_LENGTH:
        raise ValueError("Document id is too long.")
    if _CONTROL_CHARACTER_PATTERN.search(value):
        raise ValueError("Document id contains control characters.")
    if "/" in value or "\\" in value or ".." in value:
        raise ValueError("Document id contains unsafe path characters.")
    return value


def build_full_document_from_payloads(
    *,
    document_id: str,
    payloads: list[dict[str, Any]],
    collection_name: str,
    max_chunks: int = _DEFAULT_MAX_CHUNKS_PER_DOCUMENT,
) -> FullDocumentResult:
    normalized_id = validate_document_id(document_id)
    bounded_payloads = payloads[:max(0, max_chunks)]
    matching_payloads = [
        payload
        for payload in bounded_payloads
        if _payload_matches_document_id(payload, normalized_id)
    ]

    chunks = _build_chunks(matching_payloads)
    ordered_chunks = _order_chunks(chunks)
    duplicate_indexes = _duplicate_indexes(ordered_chunks)
    missing_indexes = _missing_indexes(ordered_chunks)
    all_chunks_have_index = bool(ordered_chunks) and all(
        chunk.chunk_index is not None for chunk in ordered_chunks
    )
    full_text = "\n\n".join(chunk.text for chunk in ordered_chunks if chunk.text)
    truncated = len(payloads) > len(bounded_payloads)
    availability = _availability_status(
        ordered_chunks=ordered_chunks,
        full_text=full_text,
        all_chunks_have_index=all_chunks_have_index,
        duplicate_indexes=duplicate_indexes,
        missing_indexes=missing_indexes,
        truncated=truncated,
    )
    metadata = normalize_document_metadata(normalized_id, matching_payloads)

    return FullDocumentResult(
        document_id=normalized_id,
        metadata=metadata,
        full_text=full_text,
        chunks=ordered_chunks,
        source_url=_source_url(metadata),
        provenance_status="overeno" if ordered_chunks else "not_found",
        full_text_availability_status=availability,
        diagnostics=FullDocumentDiagnostics(
            collection_name=collection_name,
            chunk_count=len(ordered_chunks),
            missing_chunk_indexes=missing_indexes,
            duplicate_chunk_indexes=duplicate_indexes,
            all_chunks_have_index=all_chunks_have_index,
            reconstruction_method="qdrant_payload_chunk_index",
            truncated=truncated,
            max_chunks=max_chunks,
        ),
    )


def normalize_document_metadata(
    document_id: str,
    payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    base = _richest_payload(payloads)
    metadata: dict[str, Any] = dict(base)
    metadata["document_id"] = _first_text(base, ("document_id",)) or document_id
    metadata["source_document_id"] = (
        _first_text(base, ("source_document_id",)) or metadata["document_id"]
    )

    if document_id.startswith("ECLI:"):
        metadata["ecli"] = _first_text(base, ("ecli",)) or document_id
    elif _first_text(base, ("ecli",)):
        metadata["ecli"] = _first_text(base, ("ecli",))

    metadata["case_reference"] = (
        _first_text(base, ("case_reference", "case_number", "spisova_znacka", "reference"))
        or document_id
    )
    decision_date = _first_text(base, ("decision_date", "date"))
    if decision_date:
        metadata["decision_date"] = decision_date

    court_name = (
        _first_text(base, ("court_name", "court"))
        or _court_name_from_document_context(document_id, _first_text(base, ("source",)))
    )
    if court_name:
        metadata["court_name"] = court_name

    source = _first_text(base, ("source",))
    if source:
        metadata["source"] = source

    return metadata


def _make_qdrant_client(qdrant_url: str) -> _ScrollableQdrantClient:
    from qdrant_client import QdrantClient

    return QdrantClient(url=qdrant_url, timeout=10)


def _document_id_filter(document_id: str) -> Any:
    from qdrant_client import models

    return models.Filter(
        should=[
            models.FieldCondition(key=key, match=models.MatchValue(value=document_id))
            for key in _DOCUMENT_ID_KEYS
        ]
    )


def _max_chunks_from_env() -> int:
    raw_value = os.getenv("NALUS_FULL_DOCUMENT_MAX_CHUNKS", "").strip()
    if not raw_value:
        return _DEFAULT_MAX_CHUNKS_PER_DOCUMENT
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Invalid NALUS_FULL_DOCUMENT_MAX_CHUNKS value; using default.")
        return _DEFAULT_MAX_CHUNKS_PER_DOCUMENT
    return min(max(value, 1), 10_000)


def _payload_from_point(point: Any) -> dict[str, Any]:
    payload = dict(getattr(point, "payload", None) or {})
    if "original_id" not in payload and getattr(point, "id", None) is not None:
        payload["original_id"] = str(point.id)
    return payload


def _payload_matches_document_id(payload: dict[str, Any], document_id: str) -> bool:
    return any(_as_text(payload.get(key)) == document_id for key in _DOCUMENT_ID_KEYS)


def _build_chunks(payloads: list[dict[str, Any]]) -> list[FullDocumentChunk]:
    chunks: list[FullDocumentChunk] = []
    seen_chunk_ids: set[str] = set()
    for fallback_index, payload in enumerate(payloads):
        chunk_id = _chunk_id(payload, fallback_index)
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        text = _first_text(payload, _TEXT_KEYS) or ""
        chunks.append(
            FullDocumentChunk(
                chunk_id=chunk_id,
                chunk_index=_optional_int(payload.get("chunk_index")),
                text=text,
                metadata=dict(payload),
            )
        )
    return chunks


def _chunk_id(payload: dict[str, Any], fallback_index: int) -> str:
    return (
        _first_text(payload, ("original_id", "chunk_id", "id"))
        or f"chunk-{fallback_index}"
    )


def _order_chunks(chunks: list[FullDocumentChunk]) -> list[FullDocumentChunk]:
    return sorted(
        chunks,
        key=lambda chunk: (
            chunk.chunk_index is None,
            chunk.chunk_index if chunk.chunk_index is not None else 0,
            chunk.chunk_id,
        ),
    )


def _duplicate_indexes(chunks: list[FullDocumentChunk]) -> list[int]:
    seen: set[int] = set()
    duplicates: set[int] = set()
    for chunk in chunks:
        if chunk.chunk_index is None:
            continue
        if chunk.chunk_index in seen:
            duplicates.add(chunk.chunk_index)
        seen.add(chunk.chunk_index)
    return sorted(duplicates)


def _missing_indexes(chunks: list[FullDocumentChunk]) -> list[int]:
    indexes = sorted(
        chunk.chunk_index for chunk in chunks if chunk.chunk_index is not None
    )
    if not indexes:
        return []
    expected = range(indexes[0], indexes[-1] + 1)
    return [index for index in expected if index not in indexes]


def _availability_status(
    *,
    ordered_chunks: list[FullDocumentChunk],
    full_text: str,
    all_chunks_have_index: bool,
    duplicate_indexes: list[int],
    missing_indexes: list[int],
    truncated: bool,
) -> FullTextAvailabilityStatus:
    if not ordered_chunks or not full_text:
        return "not_found"
    if (
        all_chunks_have_index
        and not duplicate_indexes
        and not missing_indexes
        and not truncated
    ):
        return "available"
    return "partial"


def _richest_payload(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not payloads:
        return {}
    return max(payloads, key=lambda payload: sum(1 for value in payload.values() if value))


def _source_url(metadata: dict[str, Any]) -> str | None:
    return _first_text(metadata, ("url", "detail_url", "text_url", "source_url"))


def _court_name_from_document_context(document_id: str, source: str | None) -> str | None:
    normalized = f"{document_id} {source or ''}".lower()
    if "ecli:cz:us" in normalized or "nalus" in normalized or "constitutional" in normalized:
        return "Ústavní soud"
    if "ecli:cz:ns" in normalized or "supreme" in normalized:
        return "Nejvyšší soud"
    return None


def _first_text(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = _as_text(payload.get(key))
        if value:
            return value
    return None


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
