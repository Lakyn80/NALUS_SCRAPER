#!/usr/bin/env python3
"""US/NALUS incremental scrape into court_staging only (never batches/)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.court_staging.identity import ChangeKind, enrich_record_identity
from app.court_staging.jsonl_store import atomic_write_json, load_canonical_index, rewrite_jsonl_upsert
from app.court_staging.paths import assert_safe_staging_path, default_staging_root, ensure_staging_tree


@dataclass(frozen=True)
class CollectOutcome:
    items: list[object]
    pages_scanned: int
    total_pages: int
    document_failed: int
    incomplete_reason: str | None
    listing_complete: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date-from", type=date.fromisoformat, required=True)
    p.add_argument("--date-to", type=date.fromisoformat, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--no-ingest", action="store_true", default=True)
    p.add_argument(
        "--max-pages",
        type=int,
        default=50,
        help="Safety cap on NALUS result pages per run.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Safety cap on decisions collected per run.",
    )
    p.add_argument("--page-sleep", type=float, default=1.0)
    return p.parse_args()


def _as_record(item: object) -> dict:
    if isinstance(item, dict):
        raw = dict(item)
    else:
        try:
            from dataclasses import asdict, is_dataclass

            if is_dataclass(item) and not isinstance(item, type):
                raw = asdict(item)
            elif hasattr(item, "__dict__"):
                raw = dict(item.__dict__)
            else:
                raise TypeError(f"Unsupported result type: {type(item)}")
        except TypeError:
            raise
    raw.setdefault("source", "nalus")
    return enrich_record_identity(raw, source="nalus")


def _format_nalus_date(value: date) -> str:
    return f"{value.day}.{value.month}.{value.year}"


def _collect_date_scoped(
    *,
    date_from: date,
    date_to: date,
    max_pages: int,
    limit: int,
    page_sleep: float,
) -> CollectOutcome:
    from app.crawler.extractor import extract_search_page
    from app.crawler.playwright_crawler import fetch_page_html
    from app.services.decision_service import enrich_results_with_text

    decided_from = _format_nalus_date(date_from)
    decided_to = _format_nalus_date(date_to)
    collected: list[object] = []
    seen_ids: set[str] = set()
    document_failed = 0
    current_page = 1
    total_pages = 0
    incomplete_reason: str | None = None
    hit_result_limit = False

    while current_page <= max_pages:
        html = fetch_page_html(
            query="",
            page=current_page - 1,
            decided_from=decided_from,
            decided_to=decided_to,
        )
        search_page = extract_search_page(html, query="")
        total_pages = max(total_pages, int(search_page.total_pages or 0))

        if not search_page.results:
            break

        page_results = []
        for result in search_page.results:
            unique_id = result.ecli or result.case_reference
            if not unique_id:
                document_failed += 1
                continue
            if unique_id in seen_ids:
                continue
            seen_ids.add(unique_id)
            page_results.append(result)

        if page_results:
            page_results = enrich_results_with_text(page_results)
            collected.extend(page_results)

        if limit and len(collected) >= limit:
            collected = collected[:limit]
            if current_page < total_pages:
                hit_result_limit = True
            break

        if current_page >= total_pages:
            break

        current_page += 1
        if page_sleep > 0:
            time.sleep(page_sleep)

    pages_scanned = current_page
    if total_pages > max_pages and pages_scanned >= max_pages:
        incomplete_reason = "pagination_cap_reached"
    elif hit_result_limit:
        incomplete_reason = "result_limit_reached"

    listing_complete = incomplete_reason is None
    return CollectOutcome(
        items=collected,
        pages_scanned=pages_scanned,
        total_pages=total_pages,
        document_failed=document_failed,
        incomplete_reason=incomplete_reason,
        listing_complete=listing_complete,
    )


def _resolve_status(
    *,
    error: str | None,
    outcome: CollectOutcome | None,
    new_count: int,
    updated: int,
    document_failed: int,
) -> tuple[str, str | None]:
    if error:
        return ("partial" if new_count + updated > 0 else "failed"), "fetch_error"
    if outcome is None:
        return "failed", "no_outcome"
    if outcome.incomplete_reason:
        return "incomplete", outcome.incomplete_reason
    if document_failed > 0:
        return "partial", "document_parse_failures"
    return "ok", None


def main() -> int:
    args = parse_args()
    staging = ensure_staging_tree(default_staging_root())
    out_dir = assert_safe_staging_path(
        args.out_dir or (staging / "us" / "incremental"),
        staging_root=staging,
    )
    out_jsonl = out_dir / f"usoud_{args.date_from.isoformat()}_{args.date_to.isoformat()}.jsonl"
    meta_path = out_dir / f"usoud_{args.date_from.isoformat()}_{args.date_to.isoformat()}_meta.json"

    new_count = 0
    updated = 0
    unchanged = 0
    failed = 0
    error: str | None = None
    outcome: CollectOutcome | None = None

    try:
        outcome = _collect_date_scoped(
            date_from=args.date_from,
            date_to=args.date_to,
            max_pages=max(1, args.max_pages),
            limit=max(1, args.limit),
            page_sleep=max(0.0, args.page_sleep),
        )
        failed += outcome.document_failed
        known = load_canonical_index([out_jsonl])
        for item in outcome.items:
            try:
                record = _as_record(item)
            except Exception:
                failed += 1
                continue
            kind = rewrite_jsonl_upsert(out_jsonl, record, known=known, source="nalus")
            if kind is ChangeKind.NEW:
                new_count += 1
            elif kind is ChangeKind.UPDATED:
                updated += 1
            else:
                unchanged += 1
    except Exception as exc:
        error = str(exc)

    status, reason = _resolve_status(
        error=error,
        outcome=outcome,
        new_count=new_count,
        updated=updated,
        document_failed=failed,
    )
    watermark_advanced = status == "ok"

    meta = {
        "court": "us",
        "date_from": args.date_from.isoformat(),
        "date_to": args.date_to.isoformat(),
        "out_jsonl": str(out_jsonl),
        "new": new_count,
        "updated": updated,
        "unchanged": unchanged,
        "written": new_count,
        "failed": failed,
        "pages_scanned": outcome.pages_scanned if outcome else 0,
        "total_pages": outcome.total_pages if outcome else 0,
        "listing_complete": outcome.listing_complete if outcome else False,
        "max_pages": args.max_pages,
        "limit": args.limit,
        "batches_write": False,
        "ingest": False,
        "qdrant": False,
        "status": status,
        "reason": reason,
        "watermark_advanced": watermark_advanced,
        "error": error,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(meta_path, meta)
    print(json.dumps(meta, ensure_ascii=False))
    if status in {"failed", "incomplete", "partial"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
