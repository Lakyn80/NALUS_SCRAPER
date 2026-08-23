#!/usr/bin/env python3
"""Re-finalize partial months blocked only by empty_or_invalid_detail failures.

Moves historical `failed` counts from empty detail pages into
`skipped_unavailable` and re-runs finalize_month_status so months can close as ok.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.court_staging.completeness import MonthCompleteness, finalize_month_status  # noqa: E402


def reconcile_month(entry: dict) -> tuple[dict, bool]:
    if entry.get("status") != "partial":
        return entry, False

    completeness = dict(entry.get("completeness") or {})
    reasons = dict(completeness.get("failure_reasons") or {})
    empty_count = int(reasons.get("empty_or_invalid_detail") or 0)
    other_failed = int(completeness.get("failed") or 0) - empty_count
    if empty_count <= 0 or other_failed > 0:
        return entry, False

    stats = MonthCompleteness(
        site_total_results=completeness.get("site_total_results"),
        discovered_entries=int(completeness.get("discovered_entries") or 0),
        unique_source_ids=int(completeness.get("unique_source_ids") or 0),
        fetched_ok=int(completeness.get("fetched_ok") or 0),
        failed=0,
        duplicates=int(completeness.get("duplicates") or 0),
        skipped_classified=int(completeness.get("skipped_classified") or 0),
        skipped_unavailable=empty_count,
        failure_reasons={"empty_or_invalid_detail": empty_count},
        notes=[n for n in (completeness.get("notes") or []) if n != "has_explicit_failures"],
    )
    finalize_month_status(stats)
    entry = dict(entry)
    entry["completeness"] = stats.to_dict()
    entry["status"] = stats.status
    return entry, True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to nsoud_historical_manifest.json or nssoud_historical_manifest.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    months: dict = dict(manifest.get("months") or {})
    changed = 0
    promoted_ok = 0
    for key in sorted(months):
        updated, did_change = reconcile_month(months[key])
        if did_change:
            changed += 1
            if updated.get("status") == "ok":
                promoted_ok += 1
            months[key] = updated
            print(f"{key}: partial -> {updated.get('status')}")

    print(f"reconciled={changed} promoted_ok={promoted_ok}")
    if args.dry_run or changed == 0:
        return 0

    manifest["months"] = months
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
