from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.legal_v2.parser_review.full_export import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    export_full_review,
)
from scripts.legal_v2.parser_review.models import DEFAULT_REVIEW_DIR  # noqa: E402
from scripts.legal_v2.parser_review.status import AUDIT_DIR, GOLDEN_DIR  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a complete parser v6 review package for the 17 non-golden review documents."
    )
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--golden-dir", type=Path, default=GOLDEN_DIR)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include-golden-details", action="store_true")
    parser.add_argument("--verify-determinism", action="store_true")
    parser.add_argument("--first-commit", default=None)
    parser.add_argument("--second-commit", default=None)
    args = parser.parse_args(argv)
    result = export_full_review(
        snapshot_dir=args.snapshot_dir,
        golden_dir=args.golden_dir,
        audit_dir=args.audit_dir,
        output_dir=args.output_dir,
        include_golden_details=args.include_golden_details,
        verify_determinism=args.verify_determinism,
        first_commit=args.first_commit,
        second_commit=args.second_commit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
