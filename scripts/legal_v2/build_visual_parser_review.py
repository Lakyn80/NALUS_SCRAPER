from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.legal_v2.parser_review.models import DEFAULT_REVIEW_DIR  # noqa: E402
from scripts.legal_v2.parser_review.snapshot import build_snapshot  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the shared Legal v2 visual parser review snapshot.")
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--force-derived-rebuild", action="store_true", help="Regenerate derived snapshot files without deleting manual decisions.")
    parser.add_argument("--document", help="Optional review number, review id, or source id for a bounded rebuild.")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs and parser mapping without writing artifacts.")
    args = parser.parse_args(argv)
    result = build_snapshot(review_dir=args.review_dir, document_filter=args.document, validate_only=args.validate_only)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
