#!/usr/bin/env python3
"""Normalize mismatched bm25_index_id on pilot_600 Qdrant + BM25 sidecar.

Golden ECLI upsert accidentally stamped
`nalus_legal_paragraph_chunks_v2_pilot_600_bm25` on some payloads while the
rest of the pilot collection (and the evaluator) use
`nalus_legal_paragraph_bm25_v2_pilot_600`. Dense provenance checks then fail
whenever a mismatched chunk enters the candidate set.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_COLLECTION = "nalus_legal_paragraph_chunks_v2_pilot_600"
TARGET_BM25_ID = "nalus_legal_paragraph_bm25_v2_pilot_600"
WRONG_BM25_ID = "nalus_legal_paragraph_chunks_v2_pilot_600_bm25"
DEFAULT_BM25_PATH = Path(
    "/app/storage/rag/bm25/nalus_legal_paragraph_bm25_v2_pilot_600.sqlite"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", default="http://qdrant:6333")
    parser.add_argument("--qdrant-collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--target-bm25-index-id", default=TARGET_BM25_ID)
    parser.add_argument("--wrong-bm25-index-id", default=WRONG_BM25_ID)
    parser.add_argument("--bm25-sidecar-path", type=Path, default=DEFAULT_BM25_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from qdrant_client import QdrantClient  # type: ignore[import-not-found]
    from qdrant_client.http import models as qm  # type: ignore[import-not-found]

    client = QdrantClient(url=args.qdrant_url, timeout=120)
    before = _count_bm25_ids(client, args.qdrant_collection)
    print({"event": "qdrant_before", "bm25_index_id_counts": before})

    wrong_count = before.get(args.wrong_bm25_index_id, 0)
    if wrong_count and not args.dry_run:
        client.set_payload(
            collection_name=args.qdrant_collection,
            payload={"bm25_index_id": args.target_bm25_index_id},
            points=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="bm25_index_id",
                        match=qm.MatchValue(value=args.wrong_bm25_index_id),
                    )
                ]
            ),
        )
    after = _count_bm25_ids(client, args.qdrant_collection)
    print(
        {
            "event": "qdrant_after",
            "updated_or_would_update": wrong_count,
            "dry_run": args.dry_run,
            "bm25_index_id_counts": after,
        }
    )

    if args.bm25_sidecar_path.exists():
        conn = sqlite3.connect(str(args.bm25_sidecar_path), timeout=120)
        conn.execute("PRAGMA busy_timeout=120000")
        sql_before = conn.execute(
            "SELECT bm25_index_id, COUNT(*) FROM bm25_chunks GROUP BY bm25_index_id"
        ).fetchall()
        print({"event": "sqlite_before", "rows": sql_before})
        if not args.dry_run:
            conn.execute(
                "UPDATE bm25_chunks SET bm25_index_id = ? "
                "WHERE bm25_index_id IS NULL OR bm25_index_id != ?",
                (args.target_bm25_index_id, args.target_bm25_index_id),
            )
            conn.commit()
        sql_after = conn.execute(
            "SELECT bm25_index_id, COUNT(*) FROM bm25_chunks GROUP BY bm25_index_id"
        ).fetchall()
        print({"event": "sqlite_after", "rows": sql_after, "dry_run": args.dry_run})
        conn.close()
    else:
        print({"event": "sqlite_missing", "path": str(args.bm25_sidecar_path)})

    if after.get(args.wrong_bm25_index_id, 0) and not args.dry_run:
        raise SystemExit("Qdrant still has mismatched bm25_index_id rows.")
    print({"event": "repair_ok"})
    return 0


def _count_bm25_ids(client: object, collection: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    offset = None
    while True:
        points, offset = client.scroll(  # type: ignore[attr-defined]
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=["bm25_index_id"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            counts[str(payload.get("bm25_index_id"))] += 1
        if offset is None:
            break
    return dict(counts)


if __name__ == "__main__":
    raise SystemExit(main())
