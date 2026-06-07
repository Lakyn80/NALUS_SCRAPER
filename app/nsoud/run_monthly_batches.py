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

from app.nsoud.scraper import ScrapeConfig, configure_logging, scrape_sample


logger = logging.getLogger("nsoud_monthly_runner")


@dataclass
class MonthlyBatchResult:
    year: int
    month: int
    date_from: str
    date_to: str
    output_path: str
    records_written: int
    duplicates_skipped: int
    pages_visited: int
    status: str
    error_message: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run safe monthly batches for Czech Supreme Court scraping.")
    parser.add_argument("--year", type=int, required=True, help="Target year, for example 2025.")
    parser.add_argument("--start-month", type=int, required=True, help="Starting month number, 1-12.")
    parser.add_argument("--end-month", type=int, required=True, help="Ending month number, 1-12.")
    parser.add_argument("--limit-per-month", type=int, required=True, help="Hard record cap per month.")
    parser.add_argument("--max-pages", type=int, required=True, help="Hard page cap per month.")
    parser.add_argument("--delay", type=float, required=True, help="Delay in seconds between page/detail requests.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for monthly JSONL outputs and manifest.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.year <= 0:
        raise ValueError("--year must be a positive integer.")
    if not 1 <= args.start_month <= 12:
        raise ValueError("--start-month must be between 1 and 12.")
    if not 1 <= args.end_month <= 12:
        raise ValueError("--end-month must be between 1 and 12.")
    if args.start_month > args.end_month:
        raise ValueError("--start-month must be less than or equal to --end-month.")
    if args.limit_per_month <= 0:
        raise ValueError("--limit-per-month must be a positive integer.")
    if args.max_pages <= 0:
        raise ValueError("--max-pages must be a positive integer.")
    if args.delay < 0:
        raise ValueError("--delay must be non-negative.")


def month_window(year: int, month: int) -> tuple[date, date]:
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, 1), date(year, month, last_day)


def month_output_path(out_dir: Path, year: int, month: int) -> Path:
    return out_dir / f"nsoud_{year}_{month:02d}.jsonl"


def manifest_path(out_dir: Path) -> Path:
    return out_dir / "nsoud_monthly_manifest.json"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def write_manifest(path: Path, args: argparse.Namespace, results: list[MonthlyBatchResult]) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runner_args": {
            "year": args.year,
            "start_month": args.start_month,
            "end_month": args.end_month,
            "limit_per_month": args.limit_per_month,
            "max_pages": args.max_pages,
            "delay": args.delay,
            "out_dir": str(args.out_dir),
        },
        "batches": [asdict(result) for result in results],
    }
    atomic_write_json(path, payload)


def run_month(
    *,
    year: int,
    month: int,
    args: argparse.Namespace,
) -> MonthlyBatchResult:
    date_from, date_to = month_window(year, month)
    output_path = month_output_path(args.out_dir, year, month)
    logger.info(
        "Starting monthly batch year=%s month=%s date_from=%s date_to=%s out=%s limit=%s max_pages=%s delay=%.3f",
        year,
        month,
        date_from.isoformat(),
        date_to.isoformat(),
        output_path,
        args.limit_per_month,
        args.max_pages,
        args.delay,
    )

    config = ScrapeConfig(
        limit=args.limit_per_month,
        date_from=date_from,
        date_to=date_to,
        delay_seconds=args.delay,
        max_pages=args.max_pages,
        out_path=output_path,
        debug_dir=None,
    )

    try:
        stats = scrape_sample(config)
    except Exception as exc:
        logger.exception("Monthly batch failed for %04d-%02d: %s", year, month, exc)
        return MonthlyBatchResult(
            year=year,
            month=month,
            date_from=date_from.isoformat(),
            date_to=date_to.isoformat(),
            output_path=str(output_path),
            records_written=0,
            duplicates_skipped=0,
            pages_visited=0,
            status="failed",
            error_message=str(exc),
        )

    logger.info(
        "Monthly batch completed year=%s month=%s records_written=%s duplicates_skipped=%s pages_visited=%s",
        year,
        month,
        stats.records_written,
        stats.duplicates_skipped,
        stats.pages_visited,
    )
    return MonthlyBatchResult(
        year=year,
        month=month,
        date_from=date_from.isoformat(),
        date_to=date_to.isoformat(),
        output_path=str(output_path),
        records_written=stats.records_written,
        duplicates_skipped=stats.duplicates_skipped,
        pages_visited=stats.pages_visited,
        status="success",
        error_message=None,
    )


def main() -> int:
    configure_logging()
    args = parse_args()

    try:
        validate_args(args)
    except Exception as exc:
        logger.error("Invalid arguments: %s", exc)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    results: list[MonthlyBatchResult] = []
    manifest_file = manifest_path(args.out_dir)

    for month in range(args.start_month, args.end_month + 1):
        result = run_month(year=args.year, month=month, args=args)
        results.append(result)
        write_manifest(manifest_file, args, results)

    total_months_attempted = len(results)
    successful_months = sum(1 for result in results if result.status == "success")
    failed_months = total_months_attempted - successful_months
    total_records_written = sum(result.records_written for result in results)

    print(f"total months attempted: {total_months_attempted}")
    print(f"successful months: {successful_months}")
    print(f"failed months: {failed_months}")
    print(f"total records written: {total_records_written}")
    print(f"manifest path: {manifest_file}")

    return 0 if failed_months == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
