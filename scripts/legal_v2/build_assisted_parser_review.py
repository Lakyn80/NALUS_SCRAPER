from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.legal_v2.parser_review.assisted import build_assisted_review  # noqa: E402
from scripts.legal_v2.parser_review.models import DEFAULT_REVIEW_DIR  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build assisted parser-review suggestions.")
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--preview", action="store_true", help="Build and print suggestions without applying decisions.")
    parser.add_argument("--court", choices=["constitutional_court", "high_court_prague", "high_court_olomouc"])
    args = parser.parse_args(argv)
    result = build_assisted_review(review_dir=args.review_dir, court=args.court, write_artifacts=True)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
