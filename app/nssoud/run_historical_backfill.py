"""NSS historical backfill runner → artifacts/court_staging/nss/historical."""

from __future__ import annotations

import argparse
import calendar
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.court_staging.completeness import MonthCompleteness, finalize_month_status
from app.court_staging.jsonl_store import atomic_write_json
from app.court_staging.paths import assert_safe_staging_path, default_staging_root, ensure_staging_tree
from app.nssoud.scraper import ScrapeConfig, configure_logging, scrape

logger = logging.getLogger("nssoud_historical_backfill")


@dataclass
class MonthResult:
    year: int
    month: int
    date_from: str
    date_to: str
    output_path: str
    completeness: dict[str, Any]
    status: str
    error_message: str | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year-from", type=int, required=True)
    p.add_argument("--year-to", type=int, required=True)
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--max-pages", type=int, default=200)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--query", default="*")
    return p.parse_args()


def month_window(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def main() -> int:
    configure_logging()
    args = parse_args()
    if args.year_from > args.year_to:
        return 1

    staging = ensure_staging_tree(default_staging_root())
    out_dir = assert_safe_staging_path(
        args.out_dir or (staging / "nss" / "historical"),
        staging_root=staging,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "nssoud_historical_manifest.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {"months": {}}
    months = dict(manifest.get("months") or {})
    results: list[MonthResult] = []

    for year in range(args.year_from, args.year_to + 1):
        for month in range(1, 13):
            key = f"{year:04d}-{month:02d}"
            if args.resume and months.get(key, {}).get("status") == "ok":
                continue
            if date(year, month, 1) > date.today():
                continue
            date_from, date_to = month_window(year, month)
            output_path = out_dir / f"nssoud_{year:04d}_{month:02d}.jsonl"
            config = ScrapeConfig(
                limit=10_000_000,
                date_from=date_from,
                date_to=date_to,
                delay_seconds=args.delay,
                max_pages=args.max_pages,
                out_path=output_path,
                exhaust=True,
                query=args.query,
            )
            try:
                stats = scrape(config)
                completeness = MonthCompleteness(
                    site_total_results=stats.site_total_results,
                    discovered_entries=stats.records_discovered,
                    unique_source_ids=stats.unique_candidates,
                    fetched_ok=stats.records_written + stats.records_updated + stats.duplicates_skipped,
                    failed=stats.parse_failures,
                    duplicates=stats.duplicates_skipped,
                    failure_reasons=dict(stats.failure_reasons),
                    notes=list(stats.notes),
                )
                finalize_month_status(completeness)
                result = MonthResult(
                    year=year,
                    month=month,
                    date_from=date_from.isoformat(),
                    date_to=date_to.isoformat(),
                    output_path=str(output_path),
                    completeness=completeness.to_dict(),
                    status=completeness.status,
                )
            except Exception as exc:
                logger.exception("NSS month failed %s: %s", key, exc)
                completeness = MonthCompleteness(status="failed")
                completeness.bump_failure("month_exception")
                result = MonthResult(
                    year=year,
                    month=month,
                    date_from=date_from.isoformat(),
                    date_to=date_to.isoformat(),
                    output_path=str(output_path),
                    completeness=completeness.to_dict(),
                    status="failed",
                    error_message=str(exc),
                )
            results.append(result)
            months[key] = asdict(result)
            atomic_write_json(
                man_path,
                {
                    "version": 1,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "out_dir": str(out_dir),
                    "months": months,
                },
            )

    failed = sum(1 for r in results if r.status == "failed")
    print(f"months_run: {len(results)} failed: {failed} manifest: {man_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
