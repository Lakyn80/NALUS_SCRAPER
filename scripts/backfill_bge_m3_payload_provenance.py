"""Backfill missing BGE-M3 embedding provenance fields in a candidate Qdrant collection."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.retrieval.errors import RetrievalConfigurationError
from app.rag.retrieval.production_profile import BGE_M3_DENSE_BM25_RRF
from app.rag.retrieval.provenance import ensure_embedding_provenance, validate_embedding_provenance

BACKFILL_VERSION = "usoud-bge-m3-provenance-backfill-v1"
DEFAULT_COLLECTION = "nalus_us_bge_m3_rag_combined_20260709"
DEFAULT_BM25_INDEX_ID = BGE_M3_DENSE_BM25_RRF.name
PRODUCTION_COLLECTION_DENYLIST = {
    "nalus",
    "nalus_live",
    "nalus_stable_20260326",
}
ALLOWED_COLLECTION_MARKERS = ("rag", "mvp", "combined", "tmp")
SCROLL_PAGE_SIZE = 256
UPSERT_BATCH_SIZE = 64


class SafetyError(ValueError):
    """Raised when a safety guard refuses the requested operation."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill missing embedding provenance fields in a BGE-M3 candidate collection.",
    )
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION)
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--bm25-index-id", default=DEFAULT_BM25_INDEX_ID)
    parser.add_argument(
        "--ingest-run-id",
        default=BACKFILL_VERSION,
        help="Value written to ingest_run_id for backfilled points.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply payload updates. Default is dry-run only.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/nalus_update/bge_m3_provenance_backfill",
    )
    return parser.parse_args(argv)


def validate_collection_name(collection_name: str) -> None:
    normalized = collection_name.strip()
    if not normalized:
        raise SafetyError("Collection name must not be empty.")
    if normalized in PRODUCTION_COLLECTION_DENYLIST or normalized.startswith("nalus_stable_"):
        raise SafetyError(f"Refusing protected collection: {normalized}")
    if not any(marker in normalized.lower() for marker in ALLOWED_COLLECTION_MARKERS):
        raise SafetyError(
            f"Collection name must include one of: {', '.join(ALLOWED_COLLECTION_MARKERS)}."
        )


def _qdrant_client(url: str | None) -> Any:
    from qdrant_client import QdrantClient

    return QdrantClient(url=url or "http://localhost:6333", timeout=60)


def _payload_needs_backfill(
    payload: dict[str, Any],
    *,
    collection_name: str,
    bm25_index_id: str,
) -> bool:
    try:
        validate_embedding_provenance(
            payload,
            profile=BGE_M3_DENSE_BM25_RRF,
            qdrant_collection=collection_name,
            bm25_index_id=bm25_index_id,
        )
        return False
    except RetrievalConfigurationError:
        return True


def backfill_collection(args: argparse.Namespace) -> dict[str, Any]:
    validate_collection_name(args.collection_name)
    client = _qdrant_client(args.qdrant_url)

    total_points = 0
    already_valid = 0
    needs_backfill = 0
    updated = 0
    sample_before: list[dict[str, Any]] = []
    sample_after: list[dict[str, Any]] = []
    upsert_buffer: list[Any] = []

    from qdrant_client.models import PointStruct

    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=args.collection_name,
            limit=SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=args.execute,
        )
        if not batch:
            break

        for point in batch:
            total_points += 1
            payload = dict(point.payload or {})
            if not _payload_needs_backfill(
                payload,
                collection_name=args.collection_name,
                bm25_index_id=args.bm25_index_id,
            ):
                already_valid += 1
                continue

            needs_backfill += 1
            enriched = ensure_embedding_provenance(
                payload,
                profile=BGE_M3_DENSE_BM25_RRF,
                qdrant_collection=args.collection_name,
                bm25_index_id=args.bm25_index_id,
                ingest_run_id=args.ingest_run_id,
            )
            validate_embedding_provenance(
                enriched,
                profile=BGE_M3_DENSE_BM25_RRF,
                qdrant_collection=args.collection_name,
                bm25_index_id=args.bm25_index_id,
            )

            if len(sample_before) < 3:
                sample_before.append(
                    {
                        "point_id": str(point.id),
                        "missing_document_id": not payload.get("document_id"),
                        "keys": sorted(payload.keys()),
                    }
                )
            if len(sample_after) < 3:
                sample_after.append(
                    {
                        "point_id": str(point.id),
                        "document_id": enriched.get("document_id"),
                        "embedding_model": enriched.get("embedding_model"),
                        "content_checksum": enriched.get("content_checksum"),
                    }
                )

            if args.execute:
                upsert_buffer.append(
                    PointStruct(id=point.id, vector=point.vector, payload=enriched)
                )
                if len(upsert_buffer) >= UPSERT_BATCH_SIZE:
                    client.upsert(collection_name=args.collection_name, points=upsert_buffer)
                    updated += len(upsert_buffer)
                    upsert_buffer.clear()

        if offset is None:
            break

    if upsert_buffer:
        client.upsert(collection_name=args.collection_name, points=upsert_buffer)
        updated += len(upsert_buffer)

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "backfill_version": BACKFILL_VERSION,
        "collection_name": args.collection_name,
        "bm25_index_id": args.bm25_index_id,
        "ingest_run_id": args.ingest_run_id,
        "execute": bool(args.execute),
        "total_points": total_points,
        "already_valid": already_valid,
        "needs_backfill": needs_backfill,
        "updated": updated,
        "sample_before": sample_before,
        "sample_after": sample_after,
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "execute" if args.execute else "dry_run"
    summary_path = output_dir / f"{suffix}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = backfill_collection(args)
    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(
        f"[{mode}] collection={summary['collection_name']} "
        f"total={summary['total_points']} valid={summary['already_valid']} "
        f"needs_backfill={summary['needs_backfill']} updated={summary['updated']}",
        file=sys.stderr,
    )
    if not args.execute and summary["needs_backfill"] > 0:
        print("Re-run with --execute to apply payload updates.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
