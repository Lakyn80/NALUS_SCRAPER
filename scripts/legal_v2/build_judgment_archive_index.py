#!/usr/bin/env python3
"""Build the read-optimized document-level judgment archive SQLite index.

Reads Constitutional Court batch JSON (document records, not chunks). Does not
scan Qdrant and does not store full judgment text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.archive.builder import (  # noqa: E402
    build_archive_index_from_batches,
)
from app.rag.legal_v2.archive.courts import COURT_CONSTITUTIONAL  # noqa: E402
from app.rag.legal_v2.archive.store import default_archive_sqlite_path  # noqa: E402

DEFAULT_BATCHES = PROJECT_ROOT.parent / "nalus-scraper" / "batches"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches-dir", type=Path, default=DEFAULT_BATCHES)
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=default_archive_sqlite_path(),
    )
    parser.add_argument(
        "--court",
        default=COURT_CONSTITUTIONAL,
        help="Court id to stamp on records missing an explicit court field.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write the build summary JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_archive_index_from_batches(
        batches_dir=args.batches_dir,
        sqlite_path=args.sqlite_path,
        court_id=args.court,
    )
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
