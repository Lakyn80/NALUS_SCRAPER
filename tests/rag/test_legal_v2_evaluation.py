from __future__ import annotations

import json
from pathlib import Path

from app.rag.legal_v2.evaluation import (
    LegalV2EvaluationCase,
    LegalV2PipelineResult,
    run_offline_legal_v2_comparison,
    write_legal_v2_evaluation_report,
)


def test_offline_comparison_reports_metrics_and_hard_negatives(tmp_path: Path) -> None:
    cases = [
        LegalV2EvaluationCase(
            case_id="positive",
            query="únos dítěte matkou z Česka do Ruska",
            expected_document_ids=["DOC-GOLD"],
        ),
        LegalV2EvaluationCase(
            case_id="hard-negative",
            query="únos dítěte matkou z Česka do Ruska",
            hard_negative_document_ids=["II. ÚS 859/23"],
        ),
    ]
    current = [
        LegalV2PipelineResult(
            case_id="positive",
            pipeline="current",
            retrieved_document_ids=["DOC-GOLD", "OTHER"],
            verified_document_ids=["DOC-GOLD"],
            latency_ms=10.0,
            chunk_count=4,
            average_token_count=120.0,
            section_distribution={"facts": 2},
        ),
        LegalV2PipelineResult(
            case_id="hard-negative",
            pipeline="current",
            retrieved_document_ids=["II. ÚS 859/23"],
            verified_document_ids=["II. ÚS 859/23"],
            latency_ms=8.0,
            chunk_count=1,
            average_token_count=90.0,
            section_distribution={"court_reasoning": 1},
        ),
    ]
    child = [
        LegalV2PipelineResult(
            case_id="positive",
            pipeline="child",
            retrieved_document_ids=["DOC-GOLD"],
            verified_document_ids=[],
            latency_ms=12.0,
            chunk_count=3,
            average_token_count=260.0,
            section_distribution={"facts": 1},
        ),
        LegalV2PipelineResult(
            case_id="hard-negative",
            pipeline="child",
            retrieved_document_ids=["II. ÚS 859/23"],
            verified_document_ids=[],
            latency_ms=11.0,
            chunk_count=2,
            average_token_count=250.0,
            reconstruction_failures=1,
        ),
    ]
    parent = [
        LegalV2PipelineResult(
            case_id="positive",
            pipeline="parent",
            retrieved_document_ids=["DOC-GOLD"],
            verified_document_ids=["DOC-GOLD"],
            latency_ms=14.0,
            chunk_count=3,
            average_token_count=900.0,
            section_distribution={"facts": 1, "court_reasoning": 1},
        ),
        LegalV2PipelineResult(
            case_id="hard-negative",
            pipeline="parent",
            retrieved_document_ids=["II. ÚS 859/23"],
            verified_document_ids=[],
            latency_ms=13.0,
            chunk_count=2,
            average_token_count=850.0,
        ),
    ]

    report = run_offline_legal_v2_comparison(
        cases=cases,
        current_results=current,
        paragraph_child_results=child,
        paragraph_parent_results=parent,
    )
    json_path, markdown_path = write_legal_v2_evaluation_report(
        output_dir=tmp_path,
        report=report,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["production_readiness_claimed"] is False
    assert payload["metrics_by_pipeline"]["current_production_chunks"][
        "hard_negative_false_positives"
    ] == 1
    assert payload["metrics_by_pipeline"]["paragraph_child_parent_windows"][
        "verified_document_precision"
    ] == 1.0
    assert markdown_path.read_text(encoding="utf-8").startswith(
        "# Universal Verified Legal Retrieval v2 evaluation"
    )


def test_evaluation_writer_handles_failure_blocked_and_exception(tmp_path: Path) -> None:
    cases = [LegalV2EvaluationCase(case_id="case-1", query="q")]
    result = [
        LegalV2PipelineResult(
            case_id="case-1",
            pipeline="p",
            status="blocked",
            error="missing dataset",
        )
    ]
    report = run_offline_legal_v2_comparison(
        cases=cases,
        current_results=result,
        paragraph_child_results=result,
        paragraph_parent_results=result,
        status="blocked",
    )

    blocked_json, _ = write_legal_v2_evaluation_report(
        output_dir=tmp_path / "blocked",
        report=report,
        status="blocked",
    )
    failed_json, _ = write_legal_v2_evaluation_report(
        output_dir=tmp_path / "failure",
        report=report,
        status="failure",
    )
    exception_json, _ = write_legal_v2_evaluation_report(
        output_dir=tmp_path / "exception",
        report=report,
        exception=RuntimeError("seed failure"),
    )

    assert json.loads(blocked_json.read_text(encoding="utf-8"))["summary"][
        "status"
    ] == "blocked"
    assert json.loads(failed_json.read_text(encoding="utf-8"))["summary"][
        "status"
    ] == "failure"
    exception_summary = json.loads(exception_json.read_text(encoding="utf-8"))[
        "summary"
    ]
    assert exception_summary["status"] == "exception"
    assert exception_summary["exception_type"] == "RuntimeError"


def test_known_hard_negative_fixture_is_permanent() -> None:
    fixture_path = Path("tests/fixtures/legal_v2_hard_negatives.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert {row["document_id"] for row in rows} == {
        "II. ÚS 859/23",
        "IV. ÚS 851/26",
        "IV. ÚS 1078/26",
        "II. ÚS 531/26",
    }
    assert all(row["expected"] == "not_verified" for row in rows)
    assert all(row["query"] == "únos dítěte matkou z Česka do Ruska" for row in rows)
