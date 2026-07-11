"""Repair NSoud BM25 sidecar provenance from read-only Qdrant payloads."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_bm25_sidecar_from_qdrant import (  # noqa: E402
    BM25_CHUNKS_TABLE,
    PRODUCTION_COLLECTION_DENYLIST,
    build_bm25_row,
    write_bm25_sqlite,
)

DEFAULT_COLLECTION = "nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1"
DEFAULT_SIDECAR_PATH = (
    PROJECT_ROOT
    / "storage/rag/bm25/nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1.sqlite"
)
DEFAULT_OUTPUT_SIDECAR_PATH = (
    PROJECT_ROOT
    / "storage/rag/bm25/nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1.provenance_repaired.sqlite"
)
SCROLL_PAGE_SIZE = 256
AUDIT_COLUMNS = (
    "document_id",
    "source_document_id",
    "ecli",
    "case_number",
    "chunk_index",
    "source",
)


class RepairError(ValueError):
    """Raised when the sidecar repair cannot be completed safely."""


@dataclass(frozen=True)
class SidecarAudit:
    path: str
    table_name: str
    row_count: int
    tables: list[str]
    columns: list[str]
    provenance_columns_present: list[str]
    blank_or_null_counts: dict[str, int]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair NSoud BM25 sidecar provenance from Qdrant.")
    parser.add_argument("--sidecar-path", type=Path, required=True)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION)
    parser.add_argument("--qdrant-url", default="http://qdrant:6333")
    parser.add_argument("--output-sidecar-path", type=Path, default=DEFAULT_OUTPUT_SIDECAR_PATH)
    parser.add_argument("--backup", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.sidecar_path.exists():
        raise RepairError(f"Sidecar not found: {args.sidecar_path}")
    collection_name = args.collection_name.strip()
    if not collection_name:
        raise RepairError("Collection name must not be empty.")
    if collection_name in PRODUCTION_COLLECTION_DENYLIST or collection_name.startswith("nalus_stable_"):
        raise RepairError(f"Refusing protected collection: {collection_name}")
    if args.execute and args.sidecar_path.resolve() == args.output_sidecar_path.resolve() and not args.backup:
        raise RepairError(
            "Refusing in-place execute without --backup. Write to a new output path or enable backup."
        )


def _qdrant_client(url: str) -> Any:
    from qdrant_client import QdrantClient

    return QdrantClient(url=url, timeout=30, check_compatibility=False)


def _select_chunks_table(connection: sqlite3.Connection) -> str:
    tables = [
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    ]
    for table_name in (BM25_CHUNKS_TABLE, "chunks", "rag_chunks"):
        if table_name in tables:
            return table_name
    raise RepairError("BM25 sidecar does not contain a supported chunks table.")


def inspect_sidecar(path: Path) -> SidecarAudit:
    with sqlite3.connect(path) as connection:
        tables = [
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        ]
        table_name = _select_chunks_table(connection)
        columns = [row[1] for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()]
        row_count = int(connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
        blank_counts: dict[str, int] = {}
        for column_name in AUDIT_COLUMNS:
            if column_name not in columns:
                blank_counts[column_name] = row_count
                continue
            if column_name == "chunk_index":
                blank_counts[column_name] = int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM {table_name} WHERE {column_name} IS NULL"
                    ).fetchone()[0]
                )
                continue
            blank_counts[column_name] = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM {table_name} WHERE {column_name} IS NULL OR TRIM(CAST({column_name} AS TEXT)) = ''"
                ).fetchone()[0]
            )
    return SidecarAudit(
        path=str(path),
        table_name=table_name,
        row_count=row_count,
        tables=tables,
        columns=columns,
        provenance_columns_present=[name for name in AUDIT_COLUMNS if name in columns],
        blank_or_null_counts=blank_counts,
    )


def load_sidecar_rows(path: Path) -> tuple[str, list[dict[str, Any]]]:
    with sqlite3.connect(path) as connection:
        table_name = _select_chunks_table(connection)
        columns = [row[1] for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()]
        rows = connection.execute(f"SELECT * FROM {table_name}").fetchall()
    return table_name, [dict(zip(columns, row, strict=True)) for row in rows]


def fetch_qdrant_payload_map(client: Any, collection_name: str) -> dict[str, dict[str, Any]]:
    offset = None
    payload_by_chunk_id: dict[str, dict[str, Any]] = {}
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
            if not chunk_id:
                raise RepairError(f"Qdrant point {point.id} is missing chunk_id.")
            if chunk_id in payload_by_chunk_id:
                raise RepairError(f"Ambiguous Qdrant mapping: duplicate chunk_id {chunk_id}.")
            payload_by_chunk_id[chunk_id] = payload
        if offset is None:
            break
    return payload_by_chunk_id


def enrich_rows(
    rows: list[dict[str, Any]],
    *,
    payload_by_chunk_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    enriched_rows: list[dict[str, Any]] = []
    stats = {
        "rows_processed": 0,
        "rows_enriched": 0,
        "rows_missing_qdrant_mapping": 0,
        "text_mismatch_count": 0,
    }
    for row in rows:
        chunk_id = str(row.get("chunk_id") or row.get("id") or row.get("original_id") or "").strip()
        text = str(row.get("text") or row.get("chunk_text") or "")
        if not chunk_id:
            raise RepairError("Sidecar row is missing deterministic chunk_id mapping.")
        payload = payload_by_chunk_id.get(chunk_id)
        if payload is None:
            stats["rows_missing_qdrant_mapping"] += 1
            raise RepairError(f"No Qdrant payload found for sidecar chunk_id {chunk_id}.")
        payload_text = str(payload.get("text") or "")
        if payload_text and payload_text != text:
            stats["text_mismatch_count"] += 1
        merged = dict(payload)
        merged["chunk_id"] = chunk_id
        merged["text"] = text
        enriched_rows.append(build_bm25_row(merged))
        stats["rows_processed"] += 1
        stats["rows_enriched"] += 1
    return enriched_rows, stats


def _backup_path_for(path: Path) -> Path:
    return path.with_name(f"{path.stem}.backup{path.suffix}")


def run_repair(args: argparse.Namespace, *, client: Any | None = None) -> dict[str, Any]:
    validate_args(args)
    sidecar_path = args.sidecar_path.resolve()
    output_path = args.output_sidecar_path.resolve()
    audit_before = inspect_sidecar(sidecar_path)
    _table_name, rows = load_sidecar_rows(sidecar_path)

    qdrant_client = client if client is not None else _qdrant_client(args.qdrant_url)
    payload_by_chunk_id = fetch_qdrant_payload_map(qdrant_client, args.collection_name)
    enriched_rows, enrichment_stats = enrich_rows(rows, payload_by_chunk_id=payload_by_chunk_id)

    summary: dict[str, Any] = {
        "mode": "dry_run" if args.dry_run else "execute",
        "sidecar_path": str(sidecar_path),
        "output_sidecar_path": str(output_path),
        "collection_name": args.collection_name,
        "qdrant_url": args.qdrant_url,
        "audit_before": audit_before.__dict__,
        "enrichment_stats": enrichment_stats,
        "qdrant_point_count": len(payload_by_chunk_id),
        "backup_created": None,
        "audit_after": None,
        "final_status": "PASS",
    }

    if args.dry_run:
        return summary

    if args.backup:
        backup_path = _backup_path_for(sidecar_path if sidecar_path == output_path else output_path)
        source_for_backup = sidecar_path if sidecar_path.exists() else output_path
        shutil.copy2(source_for_backup, backup_path)
        summary["backup_created"] = str(backup_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = write_bm25_sqlite(enriched_rows, output_path)
    audit_after = inspect_sidecar(output_path)
    if row_count != audit_before.row_count or audit_after.row_count != audit_before.row_count:
        raise RepairError(
            f"Row count changed during repair: before={audit_before.row_count}, written={row_count}, after={audit_after.row_count}"
        )
    summary["audit_after"] = audit_after.__dict__
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_repair(args)
    except RepairError as exc:
        print(f"FAILURE: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"FAILURE: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
