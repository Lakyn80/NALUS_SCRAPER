from __future__ import annotations

import json
from pathlib import Path

from app.rag.legal_v2.qa_gate import GATE_POLICY_VERSION, evaluate_initial_index_qa_gate


def test_current_thirty_approved_state_passes(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path)

    decision = evaluate_initial_index_qa_gate(**paths)

    assert decision.final_decision == "pass"
    assert decision.smoke_index_permitted is True
    assert decision.gate_policy_version == GATE_POLICY_VERSION
    assert decision.sample_count == 30
    assert decision.reviewed_count == 30
    assert decision.approved_count == 30
    assert decision.rejected_count == 0
    assert decision.needs_review_count == 0
    assert decision.source_incomplete_count == 55
    assert decision.duplicate_source_identifier_count == 502


def test_needs_review_blocks(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path)
    _mutate_quality(paths["parser_quality_path"], 0, review_status="needs_review")

    decision = evaluate_initial_index_qa_gate(**paths)

    assert decision.final_decision == "blocked"
    assert "approval_rate_not_100_percent" in decision.blocking_reasons
    assert "needs_review_samples_present" in decision.blocking_reasons


def test_rejected_sample_blocks(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path)
    _mutate_quality(paths["parser_quality_path"], 0, review_status="rejected")

    decision = evaluate_initial_index_qa_gate(**paths)

    assert decision.final_decision == "blocked"
    assert "rejected_samples_present" in decision.blocking_reasons


def test_missing_review_blocks(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path)
    _mutate_quality(paths["parser_quality_path"], 0, review_reason=None)

    decision = evaluate_initial_index_qa_gate(**paths)

    assert decision.final_decision == "blocked"
    assert "manual_review_coverage_not_100_percent" in decision.blocking_reasons


def test_failed_parse_audit_blocks(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path, parse_summary={"status": "fail"})

    decision = evaluate_initial_index_qa_gate(**paths)

    assert decision.final_decision == "blocked"
    assert "full_parse_audit_not_pass" in decision.blocking_reasons


def test_reconstruction_failure_blocks(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path, parse_summary={"reconstruction_failures": 1})

    decision = evaluate_initial_index_qa_gate(**paths)

    assert decision.final_decision == "blocked"
    assert "reconstruction_failures_present" in decision.blocking_reasons


def test_duplicate_chunk_id_blocks(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path, parse_summary={"duplicate_ids": 1})

    decision = evaluate_initial_index_qa_gate(**paths)

    assert decision.final_decision == "blocked"
    assert "duplicate_paragraph_or_chunk_ids_present" in decision.blocking_reasons


def test_cross_document_mixing_blocks(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path)
    _mutate_quality(paths["parser_quality_path"], 0, no_cross_document_mixing=False)

    decision = evaluate_initial_index_qa_gate(**paths)

    assert decision.final_decision == "blocked"
    assert "cross_document_mixing_present" in decision.blocking_reasons


def test_malformed_artifact_returns_invalid(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path)
    paths["parser_quality_path"].write_text("{", encoding="utf-8")

    decision = evaluate_initial_index_qa_gate(**paths)

    assert decision.final_decision == "invalid"
    assert decision.invalid_reasons


def test_missing_policy_field_returns_invalid(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path)
    paths["manual_review_summary_path"].write_text(json.dumps({}), encoding="utf-8")

    decision = evaluate_initial_index_qa_gate(**paths)

    assert decision.final_decision == "invalid"
    assert "missing_or_unknown_gate_policy_version" in decision.invalid_reasons


def test_source_risk_counts_are_reported_without_being_hidden(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path)

    decision = evaluate_initial_index_qa_gate(**paths)

    assert decision.source_incomplete_count == 55
    assert decision.duplicate_source_identifier_count == 502
    assert decision.reviewed_source_incomplete_count == 0
    assert decision.reviewed_duplicate_source_identifier_count == 0
    assert decision.final_decision == "pass"


def _write_artifacts(
    tmp_path: Path,
    *,
    parse_summary: dict | None = None,
    manual_summary: dict | None = None,
) -> dict[str, Path]:
    documents = [_document(index) for index in range(30)]
    parser_quality = {"summary": {"reviewed_documents": 30}, "documents": documents}
    manual = {"gate_policy_version": GATE_POLICY_VERSION}
    if manual_summary:
        manual.update(manual_summary)
    summary = {
        "status": "pass",
        "reconstruction_failures": 0,
        "boundary_violations": 0,
        "duplicate_ids": 0,
    }
    if parse_summary:
        summary.update(parse_summary)
    parse_audit = {"summary": summary}
    inventory = {
        "documents_missing_complete_text": 55,
        "duplicate_source_document_identifiers": 502,
    }
    paths = {
        "parser_quality_path": tmp_path / "parser_quality.json",
        "manual_review_summary_path": tmp_path / "manual_review_summary.json",
        "parse_audit_path": tmp_path / "parse_audit.json",
        "source_inventory_path": tmp_path / "source_inventory.json",
    }
    paths["parser_quality_path"].write_text(json.dumps(parser_quality), encoding="utf-8")
    paths["manual_review_summary_path"].write_text(json.dumps(manual), encoding="utf-8")
    paths["parse_audit_path"].write_text(json.dumps(parse_audit), encoding="utf-8")
    paths["source_inventory_path"].write_text(json.dumps(inventory), encoding="utf-8")
    return paths


def _document(index: int) -> dict:
    return {
        "document_id": f"DOC-{index}",
        "review_status": "approved",
        "review_reason": "Manual review confirmed parser and chunk evidence.",
        "categories": ["constitutional"] if index else ["supreme", "long_judgment"],
        "beginning_preserved": True,
        "ending_preserved": True,
        "legal_reasoning_preserved": True,
        "operative_part_preserved": True,
        "no_cross_document_mixing": True,
        "source_completeness_status": "complete_from_available_source",
        "duplicate_source_identifier_status": "none",
        "identified_defects": [],
    }


def _mutate_quality(path: Path, index: int, **updates) -> None:  # noqa: ANN003
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, value in updates.items():
        if value is None:
            payload["documents"][index].pop(key, None)
        else:
            payload["documents"][index][key] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
