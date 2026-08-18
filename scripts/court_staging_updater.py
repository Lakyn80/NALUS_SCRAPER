"""Unified daily court staging updater — US/NS/NSS → artifacts/court_staging only."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.court_staging.jsonl_store import atomic_write_json
from app.court_staging.paths import assert_safe_staging_path, default_staging_root, ensure_staging_tree

logger = logging.getLogger("court_staging_updater")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--courts", default="us,ns,nss", help="Comma-separated: us,ns,nss")
    p.add_argument("--mode", choices=("incremental", "dry-run-merge"), default="incremental")
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--overlap-days", type=int, default=7, help="Re-scrape window before watermark.")
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--max-pages", type=int, default=50)
    p.add_argument("--limit-per-court", type=int, default=500)
    return p.parse_args()


def _watermark_path(root: Path) -> Path:
    return root / "updater" / "watermarks.json"


def load_watermarks(root: Path) -> dict[str, Any]:
    path = _watermark_path(root)
    if not path.exists():
        return {"us": {}, "ns": {}, "nss": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_watermarks(root: Path, data: dict[str, Any]) -> None:
    atomic_write_json(_watermark_path(root), data)


def window_from_watermark(
    court_meta: dict[str, Any],
    *,
    overlap_days: int,
) -> tuple[date, date]:
    today = date.today()
    raw = court_meta.get("last_success_date") or court_meta.get("watermark_date")
    if raw:
        try:
            base = date.fromisoformat(str(raw)[:10])
        except ValueError:
            base = today - timedelta(days=overlap_days)
    else:
        base = today - timedelta(days=overlap_days)
    start = base - timedelta(days=overlap_days)
    return start, today


def run_ns_incremental(root: Path, *, start: date, end: date, delay: float, max_pages: int, limit: int) -> dict[str, Any]:
    from app.nsoud.scraper import ScrapeConfig, scrape_sample

    out = assert_safe_staging_path(
        root / "ns" / "incremental" / f"nsoud_incr_{start.isoformat()}_{end.isoformat()}.jsonl",
        staging_root=root,
    )
    stats = scrape_sample(
        ScrapeConfig(
            limit=limit,
            date_from=start,
            date_to=end,
            delay_seconds=delay,
            max_pages=max_pages,
            out_path=out,
            debug_dir=None,
            exhaust=False,
        )
    )
    return {
        "court": "ns",
        "out": str(out),
        "written": stats.records_written,
        "updated": stats.records_updated,
        "unchanged": stats.duplicates_skipped,
        "failed": stats.parse_failures,
        "status": "ok" if stats.parse_failures == 0 else "partial",
    }


def run_nss_incremental(root: Path, *, start: date, end: date, delay: float, max_pages: int, limit: int) -> dict[str, Any]:
    from app.nssoud.scraper import ScrapeConfig, scrape

    out = assert_safe_staging_path(
        root / "nss" / "incremental" / f"nssoud_incr_{start.isoformat()}_{end.isoformat()}.jsonl",
        staging_root=root,
    )
    stats = scrape(
        ScrapeConfig(
            limit=limit,
            date_from=start,
            date_to=end,
            delay_seconds=delay,
            max_pages=max_pages,
            out_path=out,
            exhaust=False,
        )
    )
    return {
        "court": "nss",
        "out": str(out),
        "written": stats.records_written,
        "updated": stats.records_updated,
        "unchanged": stats.duplicates_skipped,
        "failed": stats.parse_failures,
        "status": "ok" if stats.parse_failures == 0 else "partial",
        "notes": list(stats.notes),
    }


def run_us_incremental(root: Path, *, start: date, end: date) -> dict[str, Any]:
    """US delta into staging only — never writes batches/.

    Uses a lightweight marker + optional invoke of staging scrape helper.
    Full Playwright year crawl remains available via scrape_all_nalus with
    an explicit future staging adapter; daily path records the requested window
    and writes an incremental job stub that ops can execute when browser deps are ready.
    """
    out_dir = assert_safe_staging_path(root / "us" / "incremental", staging_root=root)
    report_path = out_dir / f"us_incr_{start.isoformat()}_{end.isoformat()}.json"
    # Prefer real helper if present.
    helper = PROJECT_ROOT / "scripts" / "scrape_us_staging_incremental.py"
    payload: dict[str, Any] = {
        "court": "us",
        "date_from": start.isoformat(),
        "date_to": end.isoformat(),
        "batches_write": False,
        "helper": str(helper),
        "status": "scheduled",
    }
    if helper.exists():
        # Import-run without touching batches.
        import runpy

        sys.argv = [
            str(helper),
            "--date-from",
            start.isoformat(),
            "--date-to",
            end.isoformat(),
            "--out-dir",
            str(out_dir),
            "--no-ingest",
        ]
        try:
            runpy.run_path(str(helper), run_name="__main__")
            payload["status"] = "ok"
        except SystemExit as exc:
            code = int(exc.code or 0) if not isinstance(exc.code, str) else 1
            payload["status"] = "ok" if code == 0 else "failed"
            payload["exit_code"] = code
        except Exception as exc:
            payload["status"] = "failed"
            payload["error"] = str(exc)
    else:
        payload["status"] = "pending_helper"
        payload["notes"] = [
            "Write US decisions only under court_staging/us/incremental.",
            "Do not run scrape_all_nalus into batches/ while Full B is active.",
        ]
    atomic_write_json(report_path, payload)
    payload["out"] = str(report_path)
    return payload


def dry_run_merge(root: Path) -> dict[str, Any]:
    """Report-only comparison stub — never merges into batches/."""
    batches = PROJECT_ROOT / "batches"
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batches_dir": str(batches),
        "staging_root": str(root),
        "policy": {
            "us_merge_to_batches": "future_explicit_only",
            "ns_merge_to_batches": "forbidden_until_canonical_ingestion",
            "nss_merge_to_batches": "forbidden_until_canonical_ingestion",
            "auto_merge": False,
        },
        "counts": {},
    }
    for court, pattern in (
        ("us", "us/**/*.jsonl"),
        ("ns", "ns/**/*.jsonl"),
        ("nss", "nss/**/*.jsonl"),
    ):
        files = list(root.glob(pattern))
        report["counts"][court] = {
            "jsonl_files": len(files),
            "paths_sample": [str(p) for p in files[:5]],
        }
    out = assert_safe_staging_path(root / "merge_dry_run" / "latest.json", staging_root=root)
    atomic_write_json(out, report)
    report["out"] = str(out)
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    root = ensure_staging_tree(args.out_root or default_staging_root())
    assert_safe_staging_path(root, staging_root=root)

    courts = [c.strip().lower() for c in args.courts.split(",") if c.strip()]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report: dict[str, Any] = {
        "run_id": run_id,
        "mode": args.mode,
        "courts": courts,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "results": [],
    }

    if args.mode == "dry-run-merge":
        report["results"].append(dry_run_merge(root))
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        out = root / "updater" / f"run_{run_id}.json"
        atomic_write_json(out, report)
        print(json.dumps({"out": str(out), "mode": args.mode}, ensure_ascii=False))
        return 0

    watermarks = load_watermarks(root)
    failed = 0
    for court in courts:
        start, end = window_from_watermark(watermarks.get(court) or {}, overlap_days=args.overlap_days)
        logger.info("Court=%s window=%s..%s", court, start, end)
        try:
            if court == "ns":
                result = run_ns_incremental(
                    root,
                    start=start,
                    end=end,
                    delay=args.delay,
                    max_pages=args.max_pages,
                    limit=args.limit_per_court,
                )
            elif court == "nss":
                result = run_nss_incremental(
                    root,
                    start=start,
                    end=end,
                    delay=args.delay,
                    max_pages=args.max_pages,
                    limit=args.limit_per_court,
                )
            elif court == "us":
                result = run_us_incremental(root, start=start, end=end)
            else:
                result = {"court": court, "status": "skipped", "error": "unknown_court"}
        except Exception as exc:
            logger.exception("Court %s failed: %s", court, exc)
            result = {"court": court, "status": "failed", "error": str(exc)}
            failed += 1

        report["results"].append(result)
        if result.get("status") in {"ok", "partial", "scheduled", "pending_helper"}:
            watermarks.setdefault(court, {})
            watermarks[court]["watermark_date"] = end.isoformat()
            watermarks[court]["last_success_date"] = end.isoformat()
            watermarks[court]["last_run_id"] = run_id
            watermarks[court]["last_status"] = result.get("status")
        else:
            failed += 1

    save_watermarks(root, watermarks)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["failed"] = failed
    out = root / "updater" / f"run_{run_id}.json"
    atomic_write_json(out, report)
    print(json.dumps({"out": str(out), "failed": failed, "results": len(report["results"])}, ensure_ascii=False))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
