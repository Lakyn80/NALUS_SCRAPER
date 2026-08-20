#!/usr/bin/env python3
"""Apply human-confirmed Golden v3 relevance grades into the review queue."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.benchmark.case_similarity_batch_review import (  # noqa: E402
    DEFAULT_QUEUE,
    apply_confirmed_to_queue,
    load_jsonl,
    write_jsonl,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument(
        "--confirmed",
        type=Path,
        required=True,
        help="Human-confirmed batch JSONL (final_grade set; no unresolved NEEDS_HUMAN_CHECK).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report updates without writing the queue.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    queue_rows = load_jsonl(args.queue)
    confirmed = load_jsonl(args.confirmed)
    updated_rows, updated = apply_confirmed_to_queue(queue_rows, confirmed)
    if not args.dry_run:
        write_jsonl(args.queue, updated_rows)
    report = {
        "confirmed_rows": len(confirmed),
        "updated_rows": updated,
        "queue_path": str(args.queue),
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
