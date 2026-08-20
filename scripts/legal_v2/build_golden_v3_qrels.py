#!/usr/bin/env python3
"""Build Golden v3 qrels from reviewed queue rows only."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.benchmark.case_similarity_batch_review import (  # noqa: E402
    DEFAULT_ARTIFACTS,
    DEFAULT_QUEUE,
    assert_freeze_allowed,
    load_jsonl,
    qrels_to_jsonl_rows,
    reviewed_qrel_entries,
    split_review_complete,
    write_jsonl,
)
from app.rag.legal_v2.benchmark.case_similarity_golden_v3 import (  # noqa: E402
    DEFAULT_V3_DATASET,
    GRADE_LABELS,
    RelevanceJudgment,
    load_case_similarity_golden_v3_jsonl,
    write_case_similarity_golden_v3_jsonl,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_V3_DATASET)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Write frozen full qrels and sync relevance_judgments into the v3 benchmark. "
        "Requires DEV and TEST review to be complete.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    queue_rows = load_jsonl(args.queue)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    if args.freeze:
        assert_freeze_allowed(queue_rows)
        entries = reviewed_qrel_entries(queue_rows, split=None)
        out_path = args.artifacts_dir / "qrels.jsonl"
        write_jsonl(out_path, qrels_to_jsonl_rows(entries))
        items = load_case_similarity_golden_v3_jsonl(args.benchmark)
        by_query: dict[str, list[RelevanceJudgment]] = {}
        for entry in entries:
            by_query.setdefault(entry.query_id, []).append(
                RelevanceJudgment(
                    document_id=entry.document_id,
                    ecli=entry.document_id,
                    grade=entry.grade,
                    label=GRADE_LABELS[entry.grade],
                    review_reason=entry.review_reason,
                    review_status="reviewed",
                )
            )
        updated = []
        for item in items:
            payload = item.model_dump()
            payload["relevance_judgments"] = [
                judgment.model_dump() for judgment in by_query.get(item.query_id, [])
            ]
            from app.rag.legal_v2.benchmark.case_similarity_golden_v3 import CaseSimilarityGoldenV3Item

            updated.append(CaseSimilarityGoldenV3Item.model_validate(payload))
        write_case_similarity_golden_v3_jsonl(args.benchmark, updated)
        meta_path = args.benchmark.with_suffix(".meta.json")
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["human_review_status"] = "QRELS_FROZEN"
        meta["qrels_frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
        meta["qrels_count"] = len(entries)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report = {
            "mode": "freeze",
            "qrels_path": str(out_path),
            "qrels_count": len(entries),
            "benchmark": str(args.benchmark),
            "dev_complete": True,
            "test_complete": True,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    split = None if args.split == "all" else args.split
    if split and not split_review_complete(queue_rows, split):
        raise SystemExit(
            f"split={split} review incomplete; refuse to emit reviewed qrels for incomplete split"
        )
    entries = reviewed_qrel_entries(queue_rows, split=split)
    if args.split == "all":
        out_path = args.artifacts_dir / "qrels_all_reviewed.jsonl"
    else:
        out_path = args.artifacts_dir / f"qrels_{args.split}_reviewed.jsonl"
    write_jsonl(out_path, qrels_to_jsonl_rows(entries))
    report = {
        "mode": "reviewed_export",
        "split": args.split,
        "qrels_path": str(out_path),
        "qrels_count": len(entries),
        "dev_complete": split_review_complete(queue_rows, "dev"),
        "test_complete": split_review_complete(queue_rows, "test"),
        "freeze": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
