from __future__ import annotations

import argparse
from pathlib import Path

from app.project_validation.report import render_json_report, render_markdown_report
from app.project_validation.validator import validate_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate NALUS task hygiene and safety checks.")
    parser.add_argument("--task-name", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("audit", "implementation", "artifact_only", "runtime_change", "eval_change"),
    )
    parser.add_argument("--expected-branch")
    parser.add_argument("--allow-branch", action="store_true")
    parser.add_argument("--allow-no-test-change")
    parser.add_argument("--allow-risk", action="append", default=[])
    parser.add_argument("--allow-candidate-run", action="append", default=["nsoud_sidecar_provenance_repaired"])
    parser.add_argument("--write-report", type=Path)
    parser.add_argument("--write-json", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    result = validate_task(
        repo_root=args.repo_root.resolve(),
        task_name=args.task_name,
        mode=args.mode,
        expected_branch=args.expected_branch,
        allow_branch=args.allow_branch,
        allow_no_test_change=args.allow_no_test_change,
        allow_risks=set(args.allow_risk),
        candidate_runs=set(args.allow_candidate_run),
    )
    markdown = render_markdown_report(result)
    json_text = render_json_report(result)

    if not args.no_write:
        if args.write_report is not None:
            _write_text(args.write_report, markdown)
        if args.write_json is not None:
            _write_text(args.write_json, json_text)

    print(f"STATUS: {result.status}")
    print(f"BRANCH: {result.git_summary.branch}")
    print(f"FINDINGS: {len(result.findings)}")
    print(f"NEXT: {result.recommended_next_action}")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
