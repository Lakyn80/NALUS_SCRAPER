"""Multi-year NS historical backfill into court_staging (never batches/)."""

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
from app.court_staging.jsonl_store import atomic_write_json, load_canonical_index
from app.court_staging.paths import assert_safe_staging_path, default_staging_root, ensure_staging_tree
from app.nsoud.scraper import ScrapeConfig, configure_logging, scrape_sample

logger = logging.getLogger("nsoud_historical_backfill")


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
    parser = argparse.ArgumentParser(
        description="Exhaustive multi-year Nejvyšší soud backfill into artifacts/court_staging."
    )
    parser.add_argument("--year-from", type=int)
    parser.add_argument("--year-to", type=int)
    parser.add_argument(
        "--date-from",
        type=date.fromisoformat,
        default=None,
        help="Inclusive lower date bound in YYYY-MM-DD (month granularity).",
    )
    parser.add_argument(
        "--date-to",
        type=date.fromisoformat,
        default=None,
        help="Inclusive upper date bound in YYYY-MM-DD (month granularity).",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Process months from newest to oldest within the requested range.",
    )
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument(
        "--max-pages",
        type=int,
        default=500,
        help="Safety ceiling per month; month becomes partial if results remain.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to artifacts/court_staging/ns/historical",
    )
    parser.add_argument(
        "--seed-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Existing NS JSONL to seed canonical index (e.g. consolidated 150). Repeatable.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip months already marked status=ok in durable manifest.",
    )
    return parser.parse_args()


def month_window(year: int, month: int) -> tuple[date, date]:
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, 1), date(year, month, last_day)


def iter_months(
    *,
    year_from: int | None,
    year_to: int | None,
    date_from: date | None,
    date_to: date | None,
    reverse: bool,
) -> list[tuple[int, int]]:
    if (year_from is None) ^ (year_to is None):
        raise ValueError("--year-from and --year-to must be provided together.")
    if (date_from is None) ^ (date_to is None):
        raise ValueError("--date-from and --date-to must be provided together.")
    if date_from is None and year_from is None:
        raise ValueError("Provide either --year-from/--year-to or --date-from/--date-to.")
    if date_from is not None and year_from is not None:
        raise ValueError("Use either year bounds or date bounds, not both.")

    if date_from is not None and date_to is not None:
        start_year, start_month = date_from.year, date_from.month
        end_year, end_month = date_to.year, date_to.month
    else:
        assert year_from is not None and year_to is not None
        start_year, start_month = year_from, 1
        end_year, end_month = year_to, 12

    if (start_year, start_month) > (end_year, end_month):
        raise ValueError("Lower bound must be <= upper bound.")

    months: list[tuple[int, int]] = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        months.append((year, month))
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
    if reverse:
        months.reverse()
    return months


def manifest_path(out_dir: Path) -> Path:
    return out_dir / "nsoud_historical_manifest.json"


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "months": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def month_key(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def seed_known(out_dir: Path, seed_paths: list[Path]) -> dict[str, str]:
    paths = [p for p in seed_paths if p.exists()]
    # Also seed from any existing staging monthly files.
    paths.extend(sorted(out_dir.glob("nsoud_*.jsonl")))
    return load_canonical_index(paths)


def run_month(
    *,
    year: int,
    month: int,
    out_dir: Path,
    delay: float,
    max_pages: int,
) -> MonthResult:
    date_from, date_to = month_window(year, month)
    output_path = assert_safe_staging_path(out_dir / f"nsoud_{year:04d}_{month:02d}.jsonl")
    # Soft limit high — exhaust mode drives pagination; limit only bounds non-exhaust paths.
    config = ScrapeConfig(
        limit=10_000_000,
        date_from=date_from,
        date_to=date_to,
        delay_seconds=delay,
        max_pages=max_pages,
        out_path=output_path,
        debug_dir=None,
        exhaust=True,
    )
    try:
        stats = scrape_sample(config)
    except Exception as exc:
        logger.exception("Month %04d-%02d failed: %s", year, month, exc)
        completeness = MonthCompleteness(status="failed", notes=[str(exc)])
        completeness.bump_failure("month_exception")
        return MonthResult(
            year=year,
            month=month,
            date_from=date_from.isoformat(),
            date_to=date_to.isoformat(),
            output_path=str(output_path),
            completeness=completeness.to_dict(),
            status="failed",
            error_message=str(exc),
        )

    completeness = MonthCompleteness(
        site_total_results=stats.site_total_results,
        discovered_entries=stats.records_discovered,
        unique_source_ids=stats.unique_candidates,
        fetched_ok=stats.records_written + stats.records_updated + stats.duplicates_skipped,
        failed=stats.parse_failures,
        duplicates=stats.duplicates_skipped,
        skipped_classified=stats.locally_filtered_out,
        failure_reasons=dict(stats.failure_reasons),
    )
    # Locally filtered rows are classified skips (out of range), not failures.
    finalize_month_status(completeness)
    # If we hit max pages with remaining site results, force partial.
    if (
        stats.site_total_results
        and stats.unique_candidates < stats.site_total_results
        and stats.pages_visited >= max_pages
    ):
        completeness.status = "partial"
        completeness.notes.append("hit_max_pages_before_exhaust")

    return MonthResult(
        year=year,
        month=month,
        date_from=date_from.isoformat(),
        date_to=date_to.isoformat(),
        output_path=str(output_path),
        completeness=completeness.to_dict(),
        status=completeness.status,
        error_message=None,
    )


def main() -> int:
    configure_logging()
    args = parse_args()
    try:
        planned_months = iter_months(
            year_from=args.year_from,
            year_to=args.year_to,
            date_from=args.date_from,
            date_to=args.date_to,
            reverse=args.reverse,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    staging_root = ensure_staging_tree(default_staging_root())
    out_dir = assert_safe_staging_path(
        args.out_dir or (staging_root / "ns" / "historical"),
        staging_root=staging_root,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default seed: classic 150 consolidated sample if present.
    seeds = list(args.seed_jsonl)
    default_seed = PROJECT_ROOT / "app" / "artifacts" / "nsoud" / "nsoud_consolidated_2025_01_03.jsonl"
    if default_seed.exists() and default_seed not in seeds:
        seeds.append(default_seed)
    known = seed_known(out_dir, seeds)
    logger.info("Seeded %s canonical ids before backfill", len(known))

    man_path = manifest_path(out_dir)
    manifest = load_manifest(man_path)
    months: dict[str, Any] = dict(manifest.get("months") or {})

    results: list[MonthResult] = []
    for year, month in planned_months:
        key = month_key(year, month)
        if args.resume and months.get(key, {}).get("status") == "ok":
            logger.info("Skipping completed month %s", key)
            continue
        # Do not scrape future months.
        first_day = date(year, month, 1)
        if first_day > date.today():
            continue
        result = run_month(
            year=year,
            month=month,
            out_dir=out_dir,
            delay=args.delay,
            max_pages=args.max_pages,
        )
        results.append(result)
        months[key] = asdict(result)
        manifest = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "year_from": args.year_from,
            "year_to": args.year_to,
            "date_from": args.date_from.isoformat() if args.date_from else None,
            "date_to": args.date_to.isoformat() if args.date_to else None,
            "reverse": bool(args.reverse),
            "out_dir": str(out_dir),
            "seeded_canonical_ids": len(known),
            "months": months,
        }
        atomic_write_json(man_path, manifest)

    ok = sum(1 for r in results if r.status == "ok")
    partial = sum(1 for r in results if r.status == "partial")
    failed = sum(1 for r in results if r.status == "failed")
    print(f"months_run: {len(results)}")
    print(f"ok: {ok}")
    print(f"partial: {partial}")
    print(f"failed: {failed}")
    print(f"manifest: {man_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
