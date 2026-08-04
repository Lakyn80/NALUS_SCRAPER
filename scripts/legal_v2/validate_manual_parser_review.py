from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.legal_v2.parser_review.models import DEFAULT_REVIEW_DIR  # noqa: E402
from scripts.legal_v2.parser_review.store import append_decision  # noqa: E402
from scripts.legal_v2.parser_review.validation import validate_review  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate or update manual parser-review decisions.")
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--strict-complete", action="store_true")
    parser.add_argument("--write-summary", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--record-decision-json", type=Path)
    args = parser.parse_args(argv)
    if args.record_decision_json:
        payload = json.loads(args.record_decision_json.read_text(encoding="utf-8"))
        result = append_decision(args.review_dir, payload)
    else:
        result = validate_review(args.review_dir, strict_complete=args.strict_complete, write_summary=args.write_summary)
    if not args.quiet:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        if not args.record_decision_json:
            progress = result.get("progress") or {}
            completion = result.get("manual_review_completion") or {}
            print(
                "MANUAL REVIEW COMPLETE: PASS"
                if completion.get("status") == "pass" and result.get("status") == "pass"
                else "MANUAL REVIEW INCOMPLETE: FAIL"
            )
            print(f"remaining lines: {progress.get('line_pending', 0)}")
            print(f"remaining boundaries: {progress.get('boundary_pending', 0)}")
            print(f"incomplete documents: {progress.get('incomplete_documents', 0)}")
            print(f"unresolved items: {progress.get('unresolved_items', 0)}")
            stale_count = sum(
                int(error.get("count", 0))
                for error in completion.get("errors", [])
                if str(error.get("code", "")).startswith("stale_")
            )
            print(f"stale items: {stale_count}")
    return 0 if result.get("status", "pass") == "pass" or args.record_decision_json else 1


if __name__ == "__main__":
    raise SystemExit(main())
