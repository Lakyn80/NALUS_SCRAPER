from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.evaluation import (  # noqa: E402
    LegalV2EvaluationCase,
    LegalV2PipelineResult,
    run_offline_legal_v2_comparison,
    write_legal_v2_evaluation_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a Legal Retrieval v2 comparison benchmark artifact.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "tests/fixtures/legal_v2_hard_negatives.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/legal_v2/benchmark")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = _load_cases(args.dataset)
    blocked = [
        LegalV2PipelineResult(
            case_id=case.case_id,
            pipeline="blocked",
            status="blocked",
            error="Live v2 benchmark requires an already built v2 index and explicit runner wiring.",
        )
        for case in cases
    ]
    report = run_offline_legal_v2_comparison(
        cases=cases,
        current_results=blocked,
        paragraph_child_results=blocked,
        paragraph_parent_results=blocked,
        status="blocked",
    )
    write_legal_v2_evaluation_report(output_dir=args.output_dir, report=report, status="blocked")
    print(args.output_dir)
    return 2


def _load_cases(path: Path) -> list[LegalV2EvaluationCase]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        cases.append(
            LegalV2EvaluationCase(
                case_id=str(item["document_id"]),
                query=str(item["query"]),
                hard_negative_document_ids=[str(item["document_id"])],
            )
        )
    return cases


if __name__ == "__main__":
    raise SystemExit(main())
