"""Build a SQLite BM25 sidecar from an existing Qdrant BGE-M3 collection."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_COLLECTION = "nalus_us_bge_m3_rag_combined_20260709"
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "storage/rag/bm25/nalus_us_bge_m3_rag_combined_20260709.sqlite"
SCROLL_PAGE_SIZE = 256
PRODUCTION_COLLECTION_DENYLIST = {
    "nalus",
    "nalus_live",
    "nalus_stable_20260326",
}


class SafetyError(ValueError):
    """Raised when a safety guard refuses the requested operation."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a Qdrant collection into a BM25 SQLite sidecar.")
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION)
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing SQLite sidecar file.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    name = args.collection_name.strip()
    if not name:
        raise SafetyError("Collection name must not be empty.")
    if name in PRODUCTION_COLLECTION_DENYLIST or name.startswith("nalus_stable_"):
        raise SafetyError(f"Refusing protected collection: {name}")
    if args.sqlite_path.exists() and not args.overwrite:
        raise SafetyError(
            f"SQLite sidecar already exists at {args.sqlite_path}. Pass --overwrite to replace it."
        )


def scroll_payload_rows(client: Any, collection_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection_name,
            limit=SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not batch:
            break
        for point in batch:
            payload = dict(point.payload or {})
            chunk_id = str(payload.get("chunk_id") or "").strip()
            text = str(payload.get("text") or "").strip()
            if not chunk_id or not text:
                raise SafetyError(
                    f"Point {point.id} in {collection_name!r} is missing chunk_id or text."
                )
            rows.append(payload)
        if offset is None:
            break
    return rows


def write_bm25_sqlite(rows: list[dict[str, Any]], sqlite_path: Path) -> int:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if sqlite_path.exists():
        sqlite_path.unlink()

    connection = sqlite3.connect(sqlite_path)
    try:
        connection.execute(
            """
            CREATE TABLE bm25_chunks (
                chunk_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                document_id TEXT,
                source_document_id TEXT,
                decision_date TEXT,
                chunk_index INTEGER,
                qdrant_collection TEXT,
                retrieval_profile TEXT,
                bm25_index_id TEXT
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO bm25_chunks (
                chunk_id,
                text,
                document_id,
                source_document_id,
                decision_date,
                chunk_index,
                qdrant_collection,
                retrieval_profile,
                bm25_index_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(row.get("chunk_id")),
                    str(row.get("text") or ""),
                    str(row.get("document_id") or ""),
                    str(row.get("source_document_id") or ""),
                    str(row.get("decision_date") or ""),
                    int(row.get("chunk_index") if row.get("chunk_index") is not None else -1),
                    str(row.get("qdrant_collection") or ""),
                    str(row.get("retrieval_profile") or ""),
                    str(row.get("bm25_index_id") or ""),
                )
                for row in rows
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return len(rows)


def _qdrant_client(url: str | None) -> Any:
    import os

    from qdrant_client import QdrantClient

    return QdrantClient(url=url or os.getenv("QDRANT_URL", "http://qdrant:6333"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_export(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    client = _qdrant_client(args.qdrant_url)
    rows = scroll_payload_rows(client, args.collection_name)
    if not rows:
        raise SafetyError(f"Collection {args.collection_name!r} returned no points.")

    sqlite_path = args.sqlite_path
    if not sqlite_path.is_absolute():
        sqlite_path = PROJECT_ROOT / sqlite_path

    row_count = write_bm25_sqlite(rows, sqlite_path)
    return {
        "generated_at": _utc_now(),
        "collection_name": args.collection_name,
        "sqlite_path": str(sqlite_path),
        "row_count": row_count,
        "table_name": "bm25_chunks",
        "final_status": "PASS",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_export(args)
    except SafetyError as exc:
        print(f"FAILURE: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"FAILURE: {exc}", file=sys.stderr)
        return 1

    print(
        f"[bm25-export] wrote {summary['row_count']} rows to {summary['sqlite_path']}",
        file=sys.stderr,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
