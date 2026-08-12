"""Qdrant export helpers for ColBERT corpus rows (blocking; call via to_thread)."""

from __future__ import annotations

from typing import Any

from app.rag.legal_v2.retrieve.colbert.config import (
    COLBERT_PILOT_SOURCE_QDRANT_COLLECTION,
)
from app.rag.legal_v2.retrieve.colbert.errors import ColbertConfigurationError


def export_chunks_from_qdrant(
    *,
    qdrant_url: str,
    collection: str = COLBERT_PILOT_SOURCE_QDRANT_COLLECTION,
    batch_size: int = 256,
) -> list[dict[str, Any]]:
    """Scroll full collection into ColBERT source rows (one row per chunk)."""
    if not str(qdrant_url or "").strip():
        raise ColbertConfigurationError("qdrant_url must be set")
    if not str(collection or "").strip():
        raise ColbertConfigurationError("collection must be set")

    from qdrant_client import QdrantClient

    client = QdrantClient(url=qdrant_url, timeout=60)
    try:
        info = client.get_collection(collection)
    except Exception as exc:  # noqa: BLE001
        raise ColbertConfigurationError(
            f"Qdrant collection unavailable: {collection}"
        ) from exc
    points_count = int(getattr(info, "points_count", 0) or 0)
    if points_count <= 0:
        raise ColbertConfigurationError(f"Qdrant collection is empty: {collection}")

    rows: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=max(1, int(batch_size)),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = getattr(point, "payload", None) or {}
            chunk_id = str(payload.get("chunk_id") or "").strip()
            document_id = str(
                payload.get("document_id")
                or payload.get("ecli")
                or payload.get("canonical_document_id")
                or ""
            ).strip()
            text = str(payload.get("text") or "")
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "text": text,
                    "metadata": {
                        "ecli": payload.get("ecli"),
                        "section_type": payload.get("section_type"),
                        "chunk_index": payload.get("chunk_index"),
                        "source_document_id": payload.get("source_document_id"),
                        "court": payload.get("court"),
                        "decision_date": payload.get("decision_date"),
                        "qdrant_collection": collection,
                        "qdrant_point_id": str(getattr(point, "id", "")),
                    },
                }
            )
        if offset is None:
            break
    return rows
