"""Merge multiple guarded BGE-M3 candidate Qdrant collections into one RAG target.

Copies existing 1024-dim vectors without re-embedding. Deduplicates by
(document_id, chunk_index), prefers later source collections on conflict,
remaps chunk_id/point_id for the target collection, and upserts points.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.retrieval.production_profile import BGE_M3_DENSE_BM25_RRF
from app.rag.retrieval.provenance import ensure_embedding_provenance

MERGE_VERSION = "usoud-bge-m3-combined-merge-v1"
BGE_M3_DIMENSION = 1024
BGE_M3_MODEL_NAME = "BAAI/bge-m3"
PRODUCTION_RETRIEVAL_PROFILE = "nalus_bge_m3_dense_bm25_rrf_v1"
DEFAULT_BM25_INDEX_ID = PRODUCTION_RETRIEVAL_PROFILE
DEFAULT_TARGET_COLLECTION = "nalus_us_bge_m3_rag_combined_20260709"
DEFAULT_SOURCE_COLLECTIONS = (
    "nalus_us_bge_m3_mvp_5y_20260708",
    "nalus_us_bge_m3_mvp_recent_3h_20260709",
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/nalus_update/usoud_bge_m3_rag_combined_20260709"
PRODUCTION_COLLECTION_DENYLIST = {
    "nalus",
    "nalus_live",
    "nalus_stable_20260326",
}
ALLOWED_TARGET_MARKERS = ("rag", "mvp", "combined", "tmp")
UPSERT_BATCH_SIZE = 64
SCROLL_PAGE_SIZE = 256


class SafetyError(ValueError):
    """Raised when a safety guard refuses the requested operation."""


@dataclass(frozen=True)
class SourcePoint:
    source_collection: str
    point_id: str
    vector: list[float]
    payload: dict[str, Any]
    source_priority: int


@dataclass(frozen=True)
class MergedPoint:
    point_id: str
    vector: list[float]
    payload: dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge guarded BGE-M3 candidate Qdrant collections.")
    parser.add_argument(
        "--source-collections",
        nargs="+",
        default=list(DEFAULT_SOURCE_COLLECTIONS),
        help="Source collections in ascending priority (later entries win dedupe conflicts).",
    )
    parser.add_argument("--target-collection", default=DEFAULT_TARGET_COLLECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--qdrant-url", default=None)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--recreate-target-collection",
        action="store_true",
        help="Recreate the target collection when it already exists.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    validate_collection_name(args.target_collection, is_target=True)
    if len(args.source_collections) < 1:
        raise SafetyError("At least one --source-collection is required.")
    for name in args.source_collections:
        validate_collection_name(name, is_target=False)
        if name == args.target_collection:
            raise SafetyError("Target collection must differ from source collections.")


def validate_collection_name(collection_name: str, *, is_target: bool) -> None:
    normalized = collection_name.strip()
    if not normalized:
        raise SafetyError("Collection name must not be empty.")
    if normalized in PRODUCTION_COLLECTION_DENYLIST:
        raise SafetyError(f"Refusing protected collection: {normalized}")
    if normalized.startswith("nalus_stable_"):
        raise SafetyError(f"Refusing stable production collection: {normalized}")
    if is_target and not any(marker in normalized.lower() for marker in ALLOWED_TARGET_MARKERS):
        raise SafetyError(
            f"Target collection name must include one of: {', '.join(ALLOWED_TARGET_MARKERS)}."
        )


def parse_decision_date(value: str | None) -> date | None:
    cleaned = str(value or "").strip()
    if not cleaned:
        return None
    import re

    iso_match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", cleaned)
    if iso_match:
        return date(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))
    czech_match = re.fullmatch(r"(\d{1,2})\.\s*(\d{1,2})\.\s*(\d{4})", cleaned)
    if czech_match:
        return date(int(czech_match.group(3)), int(czech_match.group(2)), int(czech_match.group(1)))
    return None


def dedupe_key(payload: dict[str, Any]) -> tuple[str, int]:
    document_id = str(payload.get("document_id") or payload.get("source_document_id") or "").strip()
    if not document_id:
        raise SafetyError("Point payload is missing document_id/source_document_id.")
    chunk_index = int(payload.get("chunk_index") if payload.get("chunk_index") is not None else -1)
    if chunk_index < 0:
        raise SafetyError(f"Point for {document_id!r} is missing chunk_index.")
    return document_id, chunk_index


def point_id_for(collection_name: str, identity: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{collection_name}:{identity}:{chunk_index}"))


def merge_source_points(
    points: list[SourcePoint],
    *,
    target_collection: str,
) -> tuple[list[MergedPoint], dict[str, int]]:
    deduped: dict[tuple[str, int], SourcePoint] = {}
    for point in points:
        key = dedupe_key(point.payload)
        current = deduped.get(key)
        if current is None or point.source_priority >= current.source_priority:
            deduped[key] = point

    ordered = sorted(
        deduped.values(),
        key=lambda item: (
            -(parse_decision_date(str(item.payload.get("decision_date") or "")) or date.min).toordinal(),
            str(item.payload.get("document_id") or ""),
            int(item.payload.get("chunk_index") or 0),
        ),
    )

    merged: list[MergedPoint] = []
    for seq_id, source in enumerate(ordered, start=1):
        payload = dict(source.payload)
        chunk_index = int(payload["chunk_index"])
        payload["chunk_id"] = seq_id
        payload["merge_source_collection"] = source.source_collection
        payload["merge_source_point_id"] = source.point_id
        payload = ensure_embedding_provenance(
            payload,
            profile=BGE_M3_DENSE_BM25_RRF,
            qdrant_collection=target_collection,
            bm25_index_id=DEFAULT_BM25_INDEX_ID,
            ingest_run_id=MERGE_VERSION,
        )
        document_id = str(payload["document_id"])
        merged.append(
            MergedPoint(
                point_id=point_id_for(target_collection, document_id, chunk_index),
                vector=list(source.vector),
                payload=payload,
            )
        )

    stats = {
        "input_point_count": len(points),
        "deduplicated_point_count": len(merged),
        "duplicate_dropped_count": len(points) - len(deduped),
    }
    return merged, stats


def validate_point_vector(vector: list[float]) -> None:
    if len(vector) != BGE_M3_DIMENSION:
        raise SafetyError(
            f"Vector dimension mismatch: expected {BGE_M3_DIMENSION}, got {len(vector)}."
        )


def validate_point_payload(payload: dict[str, Any], *, source_collection: str) -> None:
    model = str(payload.get("embedding_model") or "")
    if model and model != BGE_M3_MODEL_NAME:
        raise SafetyError(
            f"Collection {source_collection!r} contains embedding_model {model!r}; "
            f"expected {BGE_M3_MODEL_NAME}."
        )
    dimension = payload.get("embedding_dimension")
    if dimension is not None and int(dimension) != BGE_M3_DIMENSION:
        raise SafetyError(
            f"Collection {source_collection!r} contains embedding_dimension {dimension!r}; "
            f"expected {BGE_M3_DIMENSION}."
        )
    dedupe_key(payload)


def scroll_collection_points(client: Any, collection_name: str) -> list[SourcePoint]:
    points: list[SourcePoint] = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection_name,
            limit=SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not batch:
            break
        for point in batch:
            payload = dict(point.payload or {})
            vector = _to_float_vector(point.vector)
            validate_point_payload(payload, source_collection=collection_name)
            validate_point_vector(vector)
            points.append(
                SourcePoint(
                    source_collection=collection_name,
                    point_id=str(point.id),
                    vector=vector,
                    payload=payload,
                    source_priority=0,
                )
            )
        if offset is None:
            break
    return points


def _to_float_vector(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


def _qdrant_client(url: str | None) -> Any:
    import os

    from qdrant_client import QdrantClient

    return QdrantClient(url=url or os.getenv("QDRANT_URL", "http://qdrant:6333"))


def _count_collection(client: Any, collection_name: str) -> int | None:
    try:
        return int(client.count(collection_name=collection_name).count)
    except Exception:  # noqa: BLE001
        return None


def _prepare_target_collection(
    client: Any,
    *,
    collection_name: str,
    recreate: bool,
    existing_count: int | None,
) -> None:
    from qdrant_client.models import Distance, VectorParams

    exists = existing_count is not None
    if exists and recreate:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=BGE_M3_DIMENSION, distance=Distance.COSINE),
        )
        return
    if exists and existing_count:
        raise SafetyError(
            f"Target collection {collection_name!r} already has {existing_count} points. "
            "Pass --recreate-target-collection to replace it."
        )
    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=BGE_M3_DIMENSION, distance=Distance.COSINE),
        )


def _upsert_points(client: Any, *, collection_name: str, points: list[MergedPoint]) -> None:
    from qdrant_client.models import PointStruct

    qdrant_points = [
        PointStruct(id=point.point_id, vector=point.vector, payload=point.payload)
        for point in points
    ]
    for start in range(0, len(qdrant_points), UPSERT_BATCH_SIZE):
        client.upsert(collection_name=collection_name, points=qdrant_points[start : start + UPSERT_BATCH_SIZE])


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _production_safety_snapshot(client: Any) -> dict[str, Any]:
    aliases = client.get_aliases().aliases
    alias_rows = [
        {"alias_name": str(alias.alias_name), "collection_name": str(alias.collection_name)}
        for alias in aliases
    ]
    nalus_live_target = next(
        (row["collection_name"] for row in alias_rows if row["alias_name"] == "nalus_live"),
        None,
    )
    return {
        "generated_at": _utc_now(),
        "nalus_live_target": nalus_live_target,
        "nalus_live_points": _count_collection(client, "nalus_live"),
        "nalus_stable_20260326_points": _count_collection(client, "nalus_stable_20260326"),
        "aliases": sorted(alias_rows, key=lambda item: (item["alias_name"], item["collection_name"])),
    }


def run_merge(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    client = _qdrant_client(args.qdrant_url)

    loaded: list[SourcePoint] = []
    per_source_counts: dict[str, int] = {}
    for priority, collection_name in enumerate(args.source_collections):
        source_points = scroll_collection_points(client, collection_name)
        per_source_counts[collection_name] = len(source_points)
        loaded.extend(
            SourcePoint(
                source_collection=point.source_collection,
                point_id=point.point_id,
                vector=point.vector,
                payload=point.payload,
                source_priority=priority,
            )
            for point in source_points
        )

    merged, merge_stats = merge_source_points(
        loaded,
        target_collection=args.target_collection,
    )

    summary = {
        "generated_at": _utc_now(),
        "merge_version": MERGE_VERSION,
        "action": "dry-run" if args.dry_run else "execute",
        "source_collections": list(args.source_collections),
        "target_collection": args.target_collection,
        "per_source_point_counts": per_source_counts,
        "merge_stats": merge_stats,
        "merged_point_count": len(merged),
        "embedding_model": BGE_M3_MODEL_NAME,
        "embedding_dimension": BGE_M3_DIMENSION,
        "production_safety": _production_safety_snapshot(client),
        "final_status": "PASS",
    }

    if args.dry_run:
        return summary

    existing = _count_collection(client, args.target_collection)
    _prepare_target_collection(
        client,
        collection_name=args.target_collection,
        recreate=args.recreate_target_collection,
        existing_count=existing,
    )
    _upsert_points(client, collection_name=args.target_collection, points=merged)
    summary["target_point_count_after"] = _count_collection(client, args.target_collection)
    summary["production_safety_after"] = _production_safety_snapshot(client)
    return summary


def write_summary(args: argparse.Namespace, summary: dict[str, Any]) -> Path:
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ("merge_dry_run_summary.json" if args.dry_run else "merge_execute_summary.json")
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_merge(args)
    except SafetyError as exc:
        print(f"FAILURE: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"FAILURE: {exc}", file=sys.stderr)
        return 1

    path = write_summary(args, summary)
    print(
        f"[merge] {summary['action']} complete: "
        f"{summary['merged_point_count']} points planned for {summary['target_collection']}",
        file=sys.stderr,
    )
    print(f"Summary: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
