from __future__ import annotations

from pathlib import Path

from app.project_validation.diff_scanner import build_safety_summary, scan_diff_text
from app.project_validation.file_classifier import classify_entries
from app.project_validation.git_status import get_current_branch, get_diff_text, get_status_entries
from app.project_validation.schemas import (
    ClassifiedFile,
    DocumentationCheck,
    GitSummary,
    RiskFinding,
    TestExpectationCheck,
    ValidationResult,
)

_IMPLEMENTATION_MODES = {"implementation", "runtime_change", "eval_change"}
_SELF_VALIDATOR_SCAN_EXCLUDES = {
    "app/project_validation/__init__.py",
    "app/project_validation/schemas.py",
    "app/project_validation/git_status.py",
    "app/project_validation/file_classifier.py",
    "app/project_validation/diff_scanner.py",
    "app/project_validation/report.py",
    "app/project_validation/validator.py",
    "scripts/validate_nalus_task.py",
    "tests/test_nalus_task_validator.py",
}


def _make_issue(severity: str, rule_id: str, message: str, *, path: str | None = None) -> RiskFinding:
    return RiskFinding(severity=severity, rule_id=rule_id, message=message, path=path, source="policy")


def _documentation_check(classified_files: list[ClassifiedFile], mode: str) -> tuple[DocumentationCheck, list[RiskFinding]]:
    has_code = any(item.classification == "source_code" for item in classified_files)
    has_progress = any(item.classification == "project_progress" for item in classified_files)
    required = has_code
    if not required:
        return DocumentationCheck(required=False, passed=True, status="PASS", message="No source or script changes detected."), []
    if has_progress:
        return DocumentationCheck(required=True, passed=True, status="PASS", message="PROJECT_PROGRESS.md update detected."), []
    severity = "fail" if mode in _IMPLEMENTATION_MODES else "warning"
    status = "FAIL" if severity == "fail" else "WARN"
    finding = _make_issue(
        severity,
        "missing_project_progress_update",
        "Source or script changes were detected without a PROJECT_PROGRESS.md update.",
    )
    return DocumentationCheck(required=True, passed=False, status=status, message=finding.message), [finding]


def _test_expectation_check(
    classified_files: list[ClassifiedFile],
    mode: str,
    allow_no_test_change: str | None,
) -> tuple[TestExpectationCheck, list[RiskFinding]]:
    has_code = any(item.classification == "source_code" for item in classified_files)
    has_tests = any(item.classification == "tests" for item in classified_files)
    required = has_code
    if not required:
        return TestExpectationCheck(required=False, passed=True, status="PASS", message="No source or script changes detected."), []
    if has_tests or allow_no_test_change:
        message = "Test change detected." if has_tests else f"Missing test change allowed explicitly: {allow_no_test_change}"
        return TestExpectationCheck(
            required=True,
            passed=True,
            status="PASS",
            message=message,
            allow_reason=allow_no_test_change,
        ), []
    severity = "fail" if mode in _IMPLEMENTATION_MODES else "warning"
    status = "FAIL" if severity == "fail" else "WARN"
    finding = _make_issue(
        severity,
        "missing_test_change",
        "Source or script changes were detected without a matching test change.",
    )
    return TestExpectationCheck(required=True, passed=False, status=status, message=finding.message), [finding]


def _recommended_next_action(findings: list[RiskFinding]) -> str:
    if any(finding.severity == "fail" for finding in findings):
        return "Resolve the failing policy findings before staging or committing this task."
    if findings:
        return "Review the warning findings, confirm they are intentional, and rerun the validator."
    return "Validation passed; proceed with the intended commit or final report."


def _status_to_exit_code(status: str) -> int:
    if status == "PASS":
        return 0
    if status == "WARN":
        return 2
    return 1


def validate_task(
    *,
    repo_root: Path,
    task_name: str,
    mode: str,
    expected_branch: str | None,
    allow_branch: bool,
    allow_no_test_change: str | None,
    allow_risks: set[str] | None,
    candidate_runs: set[str] | None,
) -> ValidationResult:
    branch = get_current_branch(repo_root)
    entries = get_status_entries(repo_root)
    classified_files = classify_entries(entries, candidate_runs=candidate_runs)
    findings: list[RiskFinding] = []

    branch_ok = allow_branch or expected_branch is None or branch == expected_branch
    if not branch_ok:
        findings.append(
            _make_issue(
                "fail",
                "branch_mismatch",
                f"Current branch {branch!r} does not match expected branch {expected_branch!r}.",
            )
        )

    for item in classified_files:
        if item.classification == "generated_baseline_artifacts":
            findings.append(
                _make_issue(
                    "warning",
                    "generated_baseline_artifact_dirty",
                    "Dirty baseline-generated artifact detected; do not stage by accident.",
                    path=item.path,
                )
            )
        elif item.classification in {"local_noise", "model_cache"}:
            findings.append(
                _make_issue(
                    "fail",
                    f"{item.classification}_dirty",
                    f"Forbidden local noise or cache artifact detected: {item.classification}.",
                    path=item.path,
                )
            )
        elif item.classification == "unknown":
            findings.append(
                _make_issue(
                    "warning",
                    "unknown_dirty_file",
                    "Dirty file does not match a known safe classification.",
                    path=item.path,
                )
            )

    allowed_risk_terms = allow_risks or set()
    classification_by_path = {item.path: item.classification for item in classified_files}
    for entry in entries:
        normalized_path = entry.path.replace("\\", "/")
        classification = classification_by_path.get(normalized_path, "unknown")
        if classification not in {"source_code", "tests"}:
            continue
        if normalized_path in _SELF_VALIDATOR_SCAN_EXCLUDES:
            continue
        diff_text = get_diff_text(repo_root, entry)
        if not diff_text.strip():
            continue
        findings.extend(scan_diff_text(entry.path, diff_text, allow_risks=allowed_risk_terms))

    documentation_check, documentation_findings = _documentation_check(classified_files, mode)
    findings.extend(documentation_findings)
    test_check, test_findings = _test_expectation_check(classified_files, mode, allow_no_test_change)
    findings.extend(test_findings)

    safety_summary = build_safety_summary(classified_files, findings)

    if any(finding.severity == "fail" for finding in findings):
        status = "FAIL"
    elif findings:
        status = "WARN"
    else:
        status = "PASS"

    git_summary = GitSummary(
        branch=branch,
        expected_branch=expected_branch,
        branch_ok=branch_ok,
        entries=entries,
        staged_paths=[entry.path for entry in entries if entry.is_staged],
        unstaged_paths=[entry.path for entry in entries if entry.is_unstaged],
        untracked_paths=[entry.path for entry in entries if entry.is_untracked],
    )

    return ValidationResult(
        task_name=task_name,
        mode=mode,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        exit_code=_status_to_exit_code(status),
        git_summary=git_summary,
        classified_files=classified_files,
        findings=findings,
        safety_summary=safety_summary,
        documentation_check=documentation_check,
        test_expectation_check=test_check,
        recommended_next_action=_recommended_next_action(findings),
        allowed_risks=sorted(allowed_risk_terms),
        candidate_runs=sorted(candidate_runs or set()),
    )
