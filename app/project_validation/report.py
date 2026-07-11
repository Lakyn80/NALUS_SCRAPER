from __future__ import annotations

import json
from collections import Counter

from app.project_validation.schemas import ValidationResult


def render_markdown_report(result: ValidationResult) -> str:
    classified = Counter(item.classification for item in result.classified_files)
    findings_by_severity = Counter(finding.severity for finding in result.findings)
    lines = [
        "# NALUS Task Validation",
        "",
        f"- Task name: `{result.task_name}`",
        f"- Mode: `{result.mode}`",
        f"- Branch: `{result.git_summary.branch}`",
        f"- Expected branch: `{result.git_summary.expected_branch}`",
        f"- Final status: `{result.status}`",
        f"- Exit code: `{result.exit_code}`",
        "",
        "## Git Summary",
        "",
        f"- Branch check passed: `{str(result.git_summary.branch_ok).lower()}`",
        f"- Staged paths: `{len(result.git_summary.staged_paths)}`",
        f"- Unstaged paths: `{len(result.git_summary.unstaged_paths)}`",
        f"- Untracked paths: `{len(result.git_summary.untracked_paths)}`",
        "",
        "## Classified Files",
        "",
    ]
    for classification, count in sorted(classified.items()):
        lines.append(f"- `{classification}`: {count}")
    lines.extend(["", "## Risk Findings", ""])
    if not result.findings:
        lines.append("- None")
    else:
        lines.append(f"- Warnings: `{findings_by_severity.get('warning', 0)}`")
        lines.append(f"- Failures: `{findings_by_severity.get('fail', 0)}`")
        for finding in result.findings:
            path_suffix = f" [{finding.path}]" if finding.path else ""
            lines.append(f"- `{finding.severity}` `{finding.rule_id}`{path_suffix}: {finding.message}")
    lines.extend(
        [
            "",
            "## Safety Summary",
            "",
            f"- Retrieval logic changed: `{result.safety_summary.retrieval_logic_changed}`",
            f"- Embedding logic changed: `{result.safety_summary.embedding_logic_changed}`",
            f"- BM25 behavior changed: `{result.safety_summary.bm25_behavior_changed}`",
            f"- RRF behavior changed: `{result.safety_summary.rrf_behavior_changed}`",
            f"- Qdrant modified: `{result.safety_summary.qdrant_modified}`",
            f"- Redis behavior changed: `{result.safety_summary.redis_behavior_changed}`",
            f"- Model download introduced: `{result.safety_summary.model_download_introduced}`",
            f"- Fallback introduced: `{result.safety_summary.fallback_introduced}`",
            f"- LLM/DeepSeek called: `{result.safety_summary.llm_or_deepseek_called}`",
            "",
            "## Documentation Check",
            "",
            f"- Status: `{result.documentation_check.status}`",
            f"- Required: `{str(result.documentation_check.required).lower()}`",
            f"- Passed: `{str(result.documentation_check.passed).lower()}`",
            f"- Message: {result.documentation_check.message}",
            "",
            "## Test Expectation Check",
            "",
            f"- Status: `{result.test_expectation_check.status}`",
            f"- Required: `{str(result.test_expectation_check.required).lower()}`",
            f"- Passed: `{str(result.test_expectation_check.passed).lower()}`",
            f"- Message: {result.test_expectation_check.message}",
            "",
            "## Recommended Next Action",
            "",
            f"- {result.recommended_next_action}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_json_report(result: ValidationResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n"
