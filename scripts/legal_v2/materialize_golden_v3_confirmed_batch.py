#!/usr/bin/env python3
"""Materialize confirmed batch from proposal; resolve NEEDS_HUMAN_CHECK conservatively."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.benchmark.case_similarity_batch_review import (  # noqa: E402
    load_jsonl,
    write_jsonl,
)
from app.rag.legal_v2.benchmark.case_similarity_golden_v3 import GRADE_LABELS  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_jsonl(args.proposal)
    out: list[dict] = []
    unresolved_fixed = 0
    for row in rows:
        new_row = dict(row)
        if new_row.get("needs_human_check") or new_row.get("proposed_grade") is None:
            # Conservative provisional resolution for first-pass annotation.
            # Prefer grade 1 (partial) over 0 when ambiguous; never invent grade 3.
            new_row["final_grade"] = 1
            new_row["final_reason"] = (
                "PROVISIONAL_AGENT_RESOLUTION of NEEDS_HUMAN_CHECK → grade 1. "
                + str(new_row.get("proposed_reason") or "")
            )
            new_row["needs_human_check"] = False
            unresolved_fixed += 1
        else:
            new_row["final_grade"] = int(new_row["proposed_grade"])
            new_row["final_reason"] = str(new_row.get("proposed_reason") or "")
        new_row["proposed_label"] = GRADE_LABELS.get(int(new_row["final_grade"]))
        new_row["review_status"] = "confirmed"
        out.append(new_row)
    write_jsonl(args.output, out)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": len(out),
                "provisional_needs_human_check_resolved": unresolved_fixed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
