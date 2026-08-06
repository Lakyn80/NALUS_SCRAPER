#!/usr/bin/env python3
"""Compare two Case Similarity Golden evaluation runs (read-only rank-diff audit)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.benchmark.case_similarity_run_comparison import (  # noqa: E402
    CaseSimilarityRunComparisonError,
    compare_and_write,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True, help="Before-run artifact directory")
    parser.add_argument("--after", type=Path, required=True, help="After-run artifact directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for JSON/CSV/Markdown audit outputs",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = compare_and_write(
            before_dir=args.before,
            after_dir=args.after,
            output_dir=args.output_dir,
        )
    except CaseSimilarityRunComparisonError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"verdict={payload['verdict']}")
    print(
        "hit1_before={before} hit1_after={after}".format(
            before=payload["hit1_arithmetic"]["hit_at_1_before_fraction"],
            after=payload["hit1_arithmetic"]["hit_at_1_after_fraction"],
        )
    )
    print(f"gained_hit1={payload['hit1_transitions']['gained_hit1']}")
    print(f"lost_hit1={payload['hit1_transitions']['lost_hit1']}")
    print(f"output_dir={args.output_dir}")
    print(json.dumps(payload["hit1_arithmetic"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
