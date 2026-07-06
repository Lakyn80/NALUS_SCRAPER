"""Tests for legal-quality retrieval evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

RAG_EVAL_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "rag_eval"
sys.path.insert(0, str(RAG_EVAL_DIR))

from legal_quality_eval import (
    assess_generic_question_risk,
    classify_hit,
    evaluate_case,
    evaluate_winner_export,
    render_legal_quality_report,
)

DATASET = RAG_EVAL_DIR / "nalus_eval.json"
WINNER_QA = RAG_EVAL_DIR / "out_combined" / "winner_bge_m3_qa.json"


def test_generic_question_risk_high_for_dovolaci_duvod() -> None:
    assert (
        assess_generic_question_risk("dovolací důvod podle § 265b odst. 1 písm. g)")
        == "high"
    )


def test_classify_exact_dataset_match() -> None:
    classification, topic, marker_found, _ = classify_hit(
        document_id="ECLI:CZ:NS:2024:26.CDO.439.2024.1",
        text="nárok na vydání bezdůvodného obohacení za užívání bytu",
        expected_ecli=["ECLI:CZ:NS:2024:26.CDO.439.2024.1"],
        question="bezdůvodné obohacení za užívání bytu",
        markers=[{"marker": "bezdůvodného obohacení za užívání bytu", "aliases": []}],
    )
    assert classification == "exact_dataset_match"
    assert topic == "high"
    assert marker_found is True


def test_classify_alternate_relevant_with_marker() -> None:
    classification, _, marker_found, _ = classify_hit(
        document_id="ECLI:CZ:NS:2024:11.TDO.765.2024.1",
        text="Dovolací důvod podle § 265b odst. 1 písm. g) tr. řádu",
        expected_ecli=[
            "ECLI:CZ:NS:2024:3.TDO.650.2024.1",
            "ECLI:CZ:NS:2024:3.TDO.980.2024.1",
        ],
        question="dovolací důvod podle § 265b odst. 1 písm. g)",
        markers=[{"marker": "Dovolací důvod podle § 265b odst. 1 písm. g)", "aliases": []}],
    )
    assert classification == "alternate_relevant"
    assert marker_found is True


def test_evaluate_winner_export_has_eight_cases() -> None:
    if not WINNER_QA.exists():
        return
    payload = evaluate_winner_export(winner_qa_path=WINNER_QA, dataset_path=DATASET)
    assert payload["case_count"] == 8
    assert len(payload["cases"]) == 8
    report = render_legal_quality_report(payload)
    assert "Finální verdikt" in report
    assert "benchmark_alignment" in report


def test_evaluate_case_schema_keys() -> None:
    payload = json.loads(WINNER_QA.read_text(encoding="utf-8"))
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    case = payload["cases"][0]
    meta = dataset["cases"][0]
    result = evaluate_case(
        case_id=case["case_id"],
        question=case["question"],
        expected_ecli=meta["source_scope"]["document_ids"],
        expected_markers=[m["marker"] for m in meta["required_evidence"]],
        marker_defs=meta["required_evidence"],
        retrieval_hits=case["retrieval_hits"],
    )
    required = {
        "case_id",
        "question",
        "expected_ecli",
        "expected_markers",
        "generic_question_risk",
        "top1_classification",
        "best_relevant_rank",
        "best_relevant_document_id",
        "production_usefulness",
        "benchmark_alignment",
        "summary_cs",
        "hits",
        "notes",
    }
    assert required.issubset(result.keys())
