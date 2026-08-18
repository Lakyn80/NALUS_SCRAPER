#!/usr/bin/env python3
"""US/NALUS incremental scrape into court_staging only (never batches/)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.court_staging.identity import ChangeKind, enrich_record_identity
from app.court_staging.jsonl_store import atomic_write_json, load_canonical_index, rewrite_jsonl_upsert
from app.court_staging.paths import assert_safe_staging_path, default_staging_root, ensure_staging_tree


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date-from", type=date.fromisoformat, required=True)
    p.add_argument("--date-to", type=date.fromisoformat, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--no-ingest", action="store_true", default=True)
    p.add_argument("--max-pages", type=int, default=5)
    p.add_argument("--limit", type=int, default=200)
    return p.parse_args()


def _as_record(item: object) -> dict:
    if hasattr(item, "__dict__"):
        raw = dict(item.__dict__)
    elif isinstance(item, dict):
        raw = dict(item)
    else:
        raise TypeError(f"Unsupported result type: {type(item)}")
    raw.setdefault("source", "nalus")
    return enrich_record_identity(raw, source="nalus")


def _in_window(record: dict, start: date, end: date) -> bool:
    for key in ("decision_date", "date", "publication_date"):
        value = str(record.get(key) or "")[:10]
        if not value:
            continue
        try:
            d = date.fromisoformat(value)
        except ValueError:
            continue
        return start <= d <= end
    # If no date fields, keep (caller still bounded by page/limit).
    return True


def main() -> int:
    args = parse_args()
    staging = ensure_staging_tree(default_staging_root())
    out_dir = assert_safe_staging_path(
        args.out_dir or (staging / "us" / "incremental"),
        staging_root=staging,
    )
    out_jsonl = out_dir / f"usoud_{args.date_from.isoformat()}_{args.date_to.isoformat()}.jsonl"
    meta_path = out_dir / f"usoud_{args.date_from.isoformat()}_{args.date_to.isoformat()}_meta.json"

    written = 0
    updated = 0
    unchanged = 0
    error = None
    try:
        from app.services.search_service import collect_results

        # NALUS search_service is page-based; date window applied locally when dates exist.
        results = collect_results(
            query="",
            page_start=1,
            page_end=max(1, args.max_pages),
            fetch_full_text=True,
            max_results=args.limit,
        )
        known = load_canonical_index([out_jsonl])
        for item in results:
            record = _as_record(item)
            if not _in_window(record, args.date_from, args.date_to):
                continue
            kind = rewrite_jsonl_upsert(out_jsonl, record, known=known, source="nalus")
            if kind is ChangeKind.NEW:
                written += 1
            elif kind is ChangeKind.UPDATED:
                updated += 1
            else:
                unchanged += 1
    except Exception as exc:
        error = str(exc)

    meta = {
        "court": "us",
        "date_from": args.date_from.isoformat(),
        "date_to": args.date_to.isoformat(),
        "out_jsonl": str(out_jsonl),
        "written": written,
        "updated": updated,
        "unchanged": unchanged,
        "batches_write": False,
        "ingest": False,
        "status": "ok" if error is None else "partial",
        "error": error,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(meta_path, meta)
    print(json.dumps(meta, ensure_ascii=False))
    # Non-zero only on hard failure with zero progress when browser required.
    if error and written + updated == 0 and "playwright" in error.lower():
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
