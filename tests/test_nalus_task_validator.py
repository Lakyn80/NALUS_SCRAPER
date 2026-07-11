from __future__ import annotations

from pathlib import Path

from app.project_validation.diff_scanner import build_safety_summary, scan_diff_text
from app.project_validation.file_classifier import classify_entries, classify_path
from app.project_validation.report import render_json_report, render_markdown_report
from app.project_validation.schemas import (
    DocumentationCheck,
    GitStatusEntry,
    GitSummary,
    SafetySummary,
    TestExpectationCheck as ExpectationCheckSchema,
    ValidationResult,
)
from app.project_validation.validator import validate_task


def _entry(path: str, status_code: str = " M") -> GitStatusEntry:
    return GitStatusEntry(
        path=path,
        status_code=status_code,
        staged_status=status_code[0],
        unstaged_status=status_code[1],
        tracked=status_code != "??",
    )


def _result(status: str = "PASS") -> ValidationResult:
    return ValidationResult(
        task_name="demo",
        mode="implementation",
        status=status,  # type: ignore[arg-type]
        exit_code=0 if status == "PASS" else 2 if status == "WARN" else 1,
        git_summary=GitSummary(
            branch="main",
            expected_branch="main",
            branch_ok=True,
            entries=[],
            staged_paths=[],
            unstaged_paths=[],
            untracked_paths=[],
        ),
        classified_files=[],
        findings=[],
        safety_summary=SafetySummary(
            retrieval_logic_changed="no",
            embedding_logic_changed="no",
            bm25_behavior_changed="no",
            rrf_behavior_changed="no",
            qdrant_modified="no",
            redis_behavior_changed="no",
            model_download_introduced="no",
            fallback_introduced="no",
            llm_or_deepseek_called="no",
        ),
        documentation_check=DocumentationCheck(True, True, "PASS", "ok"),
        test_expectation_check=ExpectationCheckSchema(True, True, "PASS", "ok"),
        recommended_next_action="none",
    )


def test_classify_candidate_and_baseline_artifacts() -> None:
    assert classify_path(
        "artifacts/rag_eval/legal_qa/runs/nsoud_sidecar_provenance_repaired/metrics.json",
        candidate_runs={"nsoud_sidecar_provenance_repaired"},
    ) == "candidate_eval_artifacts"
    assert (
        classify_path("artifacts/rag_eval/legal_qa/answer_eval/usoud_no_llm_baseline/summary.json")
        == "generated_baseline_artifacts"
    )


def test_classify_local_noise_and_unknown() -> None:
    assert classify_path(".pytest_cache/state") == "local_noise"
    assert classify_path("models/bge-m3/model.bin") == "model_cache"
    assert classify_path("random/file.txt") == "unknown"


def test_risky_diff_detection_for_qdrant_write() -> None:
    findings = scan_diff_text("app/x.py", "+ client.qdrant.upsert(points=payload)")
    assert any(finding.rule_id == "qdrant_upsert" and finding.severity == "fail" for finding in findings)


def test_risky_diff_detection_for_model_download() -> None:
    findings = scan_diff_text("app/x.py", "+ model = AutoModel.from_pretrained('abc')")
    rule_ids = {finding.rule_id for finding in findings}
    assert "from_pretrained" in rule_ids
    assert "automodel" in rule_ids


def test_risky_diff_detection_for_alias_change() -> None:
    findings = scan_diff_text("app/x.py", "+ client.update_alias(name='nalus_live')")
    rule_ids = {finding.rule_id for finding in findings}
    assert "qdrant_update_alias" in rule_ids
    assert "protected_alias_live" in rule_ids


def test_safe_docs_only_task() -> None:
    classified = classify_entries([_entry("docs/NALUS_TASK_VALIDATOR.md")])
    summary = build_safety_summary(classified, [])
    assert summary.retrieval_logic_changed == "no"
    assert classified[0].classification == "docs"


def test_source_code_task_without_tests_gives_fail(tmp_path: Path) -> None:
    repo = tmp_path
    (repo / ".git").mkdir()

    from unittest.mock import patch

    with patch("app.project_validation.validator.get_current_branch", return_value="main"), patch(
        "app.project_validation.validator.get_status_entries",
        return_value=[_entry("scripts/validate_nalus_task.py"), _entry("PROJECT_PROGRESS.md")],
    ), patch("app.project_validation.validator.get_diff_text", return_value=""):
        result = validate_task(
            repo_root=repo,
            task_name="validator",
            mode="implementation",
            expected_branch="main",
            allow_branch=False,
            allow_no_test_change=None,
            allow_risks=set(),
            candidate_runs=set(),
        )
    assert result.status == "FAIL"
    assert any(finding.rule_id == "missing_test_change" for finding in result.findings)


def test_json_report_schema_contains_expected_fields() -> None:
    payload = render_json_report(_result("WARN"))
    assert '"status": "WARN"' in payload
    assert '"task_name": "demo"' in payload
    assert '"recommended_next_action": "none"' in payload


def test_markdown_report_includes_final_status() -> None:
    report = render_markdown_report(_result("FAIL"))
    assert "Final status: `FAIL`" in report
    assert "## Recommended Next Action" in report
