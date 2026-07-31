from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.qa_gate import (  # noqa: E402
    evaluate_parser_qa_gate,
    load_json,
    write_gate_decision,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Legal Retrieval v2 initial parser QA gate.")
    parser.add_argument(
        "--parser-quality",
        type=Path,
        default=PROJECT_ROOT / "artifacts/legal_v2/parser_quality_gate_20260730/parser_quality_gate.json",
    )
    parser.add_argument(
        "--manual-review-summary",
        type=Path,
        default=PROJECT_ROOT / "artifacts/legal_v2/parser_quality_gate_20260730/manual_review_summary.json",
    )
    parser.add_argument(
        "--parse-audit",
        type=Path,
        default=PROJECT_ROOT / "artifacts/legal_v2/parse_audit_full_20260730/legal_v2_parse_audit.json",
    )
    parser.add_argument(
        "--source-inventory",
        type=Path,
        default=PROJECT_ROOT / "artifacts/legal_v2/source_inventory_20260730.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/legal_v2/parser_quality_gate_20260730",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    decision = evaluate_parser_qa_gate(
        parser_quality=load_json(args.parser_quality),
        manual_review_summary=load_json(args.manual_review_summary),
        parse_audit=load_json(args.parse_audit),
        source_inventory=load_json(args.source_inventory),
    )
    json_path, markdown_path = write_gate_decision(decision, args.output_dir)
    print(json_path)
    print(markdown_path)
    print(f"decision={decision.final_decision} smoke_index_permitted={decision.smoke_index_permitted}")
    return 0 if decision.final_decision == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
