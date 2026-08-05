#!/usr/bin/env python3
"""Validate the tracked retrieval-golden v1 pilot against the development corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.benchmark.corpus import load_development_corpus  # noqa: E402
from app.rag.legal_v2.benchmark.retrieval_golden import (  # noqa: E402
    DEFAULT_PILOT_DATASET,
    load_retrieval_golden_jsonl,
    validate_retrieval_golden_dataset,
)

DEFAULT_REPORT = (
    PROJECT_ROOT / "artifacts" / "legal_v2" / "retrieval_golden_v1_pilot" / "validation_report.json"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_PILOT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--skip-corpus", action="store_true")
    args = parser.parse_args(argv)

    items = load_retrieval_golden_jsonl(args.dataset)
    blocks_by_id = None
    if not args.skip_corpus:
        corpus = load_development_corpus()
        blocks_by_id = corpus.blocks_by_id
    report = validate_retrieval_golden_dataset(
        items,
        blocks_by_id=blocks_by_id,
        dataset_path=str(args.dataset),
    )
    payload = report.model_dump()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
