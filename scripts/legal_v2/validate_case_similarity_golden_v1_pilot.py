#!/usr/bin/env python3
"""Validate the tracked case-similarity golden v1 pilot against the reviewed pool corpus."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.benchmark.case_similarity_golden import (  # noqa: E402
    DEFAULT_PILOT_DATASET,
    load_case_similarity_golden_jsonl,
    validate_case_similarity_dataset,
)
from app.rag.legal_v2.benchmark.corpus import (  # noqa: E402
    load_case_similarity_corpus,
    load_case_similarity_primary_document_ids,
)

DEFAULT_REPORT = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "case_similarity_golden_v1_pilot"
    / "validation_report.json"
)
BUILDER = PROJECT_ROOT / "scripts" / "legal_v2" / "build_case_similarity_golden_v1_pilot.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_PILOT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--skip-rebuild-check", action="store_true")
    args = parser.parse_args(argv)

    corpus = load_case_similarity_corpus()
    items = load_case_similarity_golden_jsonl(args.dataset)
    tracked_bytes = args.dataset.read_bytes()

    rebuild_bytes = None
    if not args.skip_rebuild_check:
        with tempfile.TemporaryDirectory() as tmp:
            rebuild_path = Path(tmp) / "rebuild.jsonl"
            build_report = Path(tmp) / "build_report.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(BUILDER),
                    "--output",
                    str(rebuild_path),
                    "--report",
                    str(build_report),
                ],
                cwd=str(PROJECT_ROOT),
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                print(completed.stdout)
                print(completed.stderr)
                print("rebuild_failed=1")
                return 1
            rebuild_bytes = rebuild_path.read_bytes()

    report = validate_case_similarity_dataset(
        items,
        corpus_documents=corpus.documents,
        blocks_by_id=corpus.blocks_by_id,
        expected_document_ids=load_case_similarity_primary_document_ids(),
        dataset_path=str(args.dataset),
        rebuild_bytes=rebuild_bytes,
        tracked_bytes=tracked_bytes if rebuild_bytes is not None else None,
    )
    payload = report.model_dump()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
