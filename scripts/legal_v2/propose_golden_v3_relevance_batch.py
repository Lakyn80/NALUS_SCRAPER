#!/usr/bin/env python3
"""Export a Golden v3 relevance proposal batch (does not mutate the live queue)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.benchmark.case_similarity_batch_review import (  # noqa: E402
    DEFAULT_ARTIFACTS,
    DEFAULT_BATCHES,
    DEFAULT_QUEUE,
    build_proposal_rows,
    load_jsonl,
    load_legacy_primary_map,
    render_batch_summary,
    select_batch_query_ids,
    write_jsonl,
)
from app.rag.legal_v2.benchmark.case_similarity_golden_v3 import (  # noqa: E402
    DEFAULT_V3_DATASET,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_V3_DATASET)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--batches-dir", type=Path, default=DEFAULT_BATCHES)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--batch-index", type=int, default=1)
    parser.add_argument(
        "--include-reviewed",
        action="store_true",
        help="Include already-reviewed queries when selecting the batch window.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    queue_rows = load_jsonl(args.queue)
    selection = select_batch_query_ids(
        queue_rows,
        split=args.split,
        batch_size=args.batch_size,
        batch_index=args.batch_index,
        pending_only=not args.include_reviewed,
    )
    if not selection.query_ids:
        print(json.dumps({"error": "no_query_ids_for_batch", "selection": selection.__dict__}))
        return 2
    legacy = load_legacy_primary_map(args.benchmark)
    proposals = build_proposal_rows(
        queue_rows,
        query_ids=selection.query_ids,
        legacy_by_query=legacy,
    )
    batch_name = f"batch_{args.split}_{args.batch_index:02d}"
    args.batches_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = args.batches_dir / f"{batch_name}_proposal.jsonl"
    summary_path = args.batches_dir / f"{batch_name}_SUMMARY.md"
    write_jsonl(proposal_path, proposals)
    summary_path.write_text(render_batch_summary(proposals, batch_name=batch_name), encoding="utf-8")
    needs = sum(1 for row in proposals if row.get("needs_human_check") or row.get("proposed_grade") is None)
    report = {
        "batch_name": batch_name,
        "split": args.split,
        "batch_index": args.batch_index,
        "query_ids": selection.query_ids,
        "proposal_rows": len(proposals),
        "needs_human_check": needs,
        "proposal_path": str(proposal_path),
        "summary_path": str(summary_path),
        "note": "Do not apply until human confirms into *_confirmed.jsonl",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
