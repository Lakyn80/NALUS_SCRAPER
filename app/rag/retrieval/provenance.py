from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from app.rag.retrieval.errors import RetrievalConfigurationError
from app.rag.retrieval.production_profile import RetrievalProfile

_CHUNK_SUFFIX_RE = re.compile(r"__chunk_(\d+)$")


REQUIRED_PROVENANCE_FIELDS = (
    "embedding_provider",
    "embedding_model",
    "embedding_dimension",
    "retrieval_profile",
    "ingest_run_id",
    "qdrant_collection",
    "bm25_index_id",
    "source",
    "document_id",
    "chunk_index",
    "content_checksum",
)


def build_embedding_provenance(
    *,
    payload: dict[str, Any],
    profile: RetrievalProfile,
    ingest_run_id: str,
    qdrant_collection: str,
    bm25_index_id: str,
) -> dict[str, Any]:
    enriched = dict(payload)
    enriched.update(
        {
            "embedding_provider": profile.embedding_provider,
            "embedding_model": profile.embedding_model,
            "embedding_dimension": profile.embedding_dimension,
            "retrieval_profile": profile.name,
            "ingest_run_id": ingest_run_id,
            "qdrant_collection": qdrant_collection,
            "bm25_index_id": bm25_index_id,
        }
    )
    enriched.setdefault("content_checksum", content_checksum(enriched.get("text") or enriched.get("chunk_text") or ""))
    return enriched


def validate_embedding_provenance(
    payload: dict[str, Any],
    *,
    profile: RetrievalProfile,
    qdrant_collection: str,
    bm25_index_id: str,
) -> None:
    missing = [field for field in REQUIRED_PROVENANCE_FIELDS if payload.get(field) in {None, ""}]
    if missing:
        raise RetrievalConfigurationError(
            "Qdrant payload is missing required embedding provenance fields: "
            + ", ".join(missing)
        )

    expected = {
        "embedding_provider": profile.embedding_provider,
        "embedding_model": profile.embedding_model,
        "embedding_dimension": profile.embedding_dimension,
        "retrieval_profile": profile.name,
        "qdrant_collection": qdrant_collection,
        "bm25_index_id": bm25_index_id,
    }
    mismatches = [
        field
        for field, value in expected.items()
        if str(payload.get(field)) != str(value)
    ]
    if mismatches:
        raise RetrievalConfigurationError(
            "Qdrant payload embedding provenance does not match production profile: "
            + ", ".join(mismatches)
        )


def content_checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("chunk_metadata")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return dict(decoded) if isinstance(decoded, dict) else {}
    return {}


def derive_document_id(payload: dict[str, Any]) -> str:
    existing = str(payload.get("document_id") or "").strip()
    if existing:
        return existing

    metadata = _chunk_metadata(payload)
    for key in (
        "ecli",
        "case_reference",
        "spisova_znacka",
        "detail_url",
        "text_url",
        "source_document_id",
    ):
        value = str(payload.get(key) or metadata.get(key) or "").strip()
        if value:
            return value

    meta_chunk_id = str(metadata.get("chunk_id") or "").strip()
    if meta_chunk_id and "__chunk_" in meta_chunk_id:
        return meta_chunk_id.split("__chunk_", maxsplit=1)[0]

    raise RetrievalConfigurationError(
        "Cannot derive document_id from payload; expected document_id or one of "
        "ecli/case_reference/spisova_znacka/detail_url/text_url/source_document_id "
        "or chunk_metadata.source_document_id."
    )


def derive_chunk_index(payload: dict[str, Any]) -> int:
    chunk_index = payload.get("chunk_index")
    if chunk_index is not None and str(chunk_index).strip() != "":
        return int(chunk_index)

    metadata = _chunk_metadata(payload)
    meta_chunk_id = str(metadata.get("chunk_id") or "").strip()
    match = _CHUNK_SUFFIX_RE.search(meta_chunk_id)
    if match:
        return int(match.group(1))

    raise RetrievalConfigurationError(
        "Cannot derive chunk_index from payload; expected chunk_index or "
        "chunk_metadata.chunk_id with __chunk_NNNN suffix."
    )


def derive_source(payload: dict[str, Any]) -> str:
    existing = str(payload.get("source") or "").strip()
    if existing:
        return existing

    document_id = str(payload.get("document_id") or "").strip()
    if not document_id:
        try:
            document_id = derive_document_id(payload)
        except RetrievalConfigurationError:
            document_id = ""

    if document_id.startswith("ECLI:CZ:NS:"):
        return "nsoud"
    if document_id.startswith("ECLI:CZ:US:"):
        return "usoud / nalus"
    if _chunk_metadata(payload):
        return "nsoud"
    return "usoud / nalus"


def ensure_embedding_provenance(
    payload: dict[str, Any],
    *,
    profile: RetrievalProfile,
    qdrant_collection: str,
    bm25_index_id: str,
    ingest_run_id: str,
) -> dict[str, Any]:
    enriched = dict(payload)
    enriched["document_id"] = derive_document_id(enriched)
    enriched["chunk_index"] = derive_chunk_index(enriched)
    text = str(enriched.get("text") or enriched.get("chunk_text") or "")
    if not text.strip():
        raise RetrievalConfigurationError("Cannot compute content_checksum: payload text is empty.")
    enriched.setdefault("content_checksum", content_checksum(text))
    enriched.setdefault("source", derive_source(enriched))
    return build_embedding_provenance(
        payload=enriched,
        profile=profile,
        ingest_run_id=ingest_run_id,
        qdrant_collection=qdrant_collection,
        bm25_index_id=bm25_index_id,
    )
