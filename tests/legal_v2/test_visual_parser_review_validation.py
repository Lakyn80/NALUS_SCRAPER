from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.legal_v2.parser_review.models import REVIEW_SCHEMA_VERSION, write_json, write_jsonl
from scripts.legal_v2.parser_review.validation import validate_review
from scripts.legal_v2.validate_manual_parser_review import main as validator_main

PARSER_PROFILE = "legal-decision-parser.cz-courts.v4"
PARSER_HEAD = "test-head"


def _patch_identity(monkeypatch: pytest.MonkeyPatch, docs: list[SimpleNamespace]) -> None:
    monkeypatch.setattr("scripts.legal_v2.parser_review.validation.load_design_documents", lambda: ({}, docs))
    monkeypatch.setattr(
        "scripts.legal_v2.parser_review.validation.parser_git_identity",
        lambda: {"head": PARSER_HEAD, "parser_profile": PARSER_PROFILE},
    )


def _make_review(
    tmp_path: Path,
    *,
    document_count: int = 2,
    line_total: int = 4,
    boundary_total: int = 2,
    complete: bool = False,
    unresolved_line: bool = False,
    unresolved_boundary: bool = False,
    stale_checksum: bool = False,
    stale_profile: bool = False,
    stale_identity: bool = False,
    duplicate_decision: bool = False,
    missing_last_boundary: bool = False,
    missing_last_line: bool = False,
) -> tuple[Path, list[SimpleNamespace]]:
    review_dir = tmp_path / "review"
    docs = [
        SimpleNamespace(review_id=f"doc-{index}", source_checksum=f"sha-{index}")
        for index in range(1, document_count + 1)
    ]
    write_json(
        review_dir / "review_manifest.json",
        {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "parser_profile": PARSER_PROFILE,
            "head": PARSER_HEAD,
        },
    )
    write_jsonl(
        review_dir / "review_documents.jsonl",
        [
            {
                "schema_version": REVIEW_SCHEMA_VERSION,
                "document_id": doc.review_id,
                "review_number": index,
                "source_id": f"source-{index}",
                "court": "constitutional_court",
            }
            for index, doc in enumerate(docs, start=1)
        ],
    )
    line_rows = []
    for index in range(1, line_total + 1):
        doc = docs[(index - 1) % document_count]
        line_rows.append(
            {
                "schema_version": REVIEW_SCHEMA_VERSION,
                "item_type": "line",
                "item_id": f"line-{index}",
                "document_id": doc.review_id,
                "source_checksum": doc.source_checksum,
                "raw_line_number": index,
                "parser_proposed_line_class": "prose_start",
            }
        )
    boundary_rows = []
    for index in range(1, boundary_total + 1):
        doc = docs[(index - 1) % document_count]
        boundary_rows.append(
            {
                "schema_version": REVIEW_SCHEMA_VERSION,
                "item_type": "boundary",
                "item_id": f"boundary-{index}",
                "document_id": doc.review_id,
                "source_checksum": doc.source_checksum,
                "previous_line_number": index,
                "next_line_number": index + 1,
                "parser_proposed_boundary": False,
            }
        )
    write_jsonl(review_dir / "review_lines.jsonl", line_rows)
    write_jsonl(review_dir / "review_boundaries.jsonl", boundary_rows)
    decisions: list[dict[str, object]] = []
    if complete:
        line_decision_rows = line_rows[:-1] if missing_last_line else line_rows
        boundary_decision_rows = boundary_rows[:-1] if missing_last_boundary else boundary_rows
        for index, row in enumerate(line_decision_rows, start=1):
            decisions.append(_decision(row, item_type="line", item_id=str(row["item_id"]), status="accepted" if not (unresolved_line and index == 1) else "unresolved", manual_class="prose_start"))
        for index, row in enumerate(boundary_decision_rows, start=1):
            decisions.append(
                _decision(
                    row,
                    item_type="boundary",
                    item_id=str(row["item_id"]),
                    status="accepted" if not (unresolved_boundary and index == 1) else "unresolved",
                    manual_boundary_decision="preserve_parser",
                )
            )
    if decisions:
        if stale_checksum:
            decisions[0]["source_checksum"] = "stale"
        if stale_profile:
            decisions[0]["parser_profile"] = "stale-profile"
        if stale_identity:
            decisions[0]["parser_git_identity"] = "stale-head"
        if duplicate_decision:
            decisions.append(dict(decisions[0]))
    write_jsonl(review_dir / "manual_review_decisions.jsonl", decisions)
    write_jsonl(review_dir / "manual_review_history.jsonl", decisions)
    return review_dir, docs


def _decision(
    source: dict[str, object],
    *,
    item_type: str,
    item_id: str,
    status: str,
    manual_class: str | None = None,
    manual_boundary_decision: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "document_id": source["document_id"],
        "source_checksum": source["source_checksum"],
        "parser_profile": PARSER_PROFILE,
        "parser_git_identity": PARSER_HEAD,
        "item_type": item_type,
        "item_id": item_id,
        "raw_line_number": source.get("raw_line_number") or source.get("previous_line_number"),
        "parser_proposal": {},
        "previous_automated_annotation": None,
        "manual_class": manual_class,
        "manual_boundary_decision": manual_boundary_decision,
        "decision_status": status,
        "reviewer_comment": "",
        "revision_number": 1,
        "timestamp": "2026-08-04T00:00:00Z",
        "interface": "validator",
        "review_tool_version": "visual-parser-review-tool.v1",
    }


def _codes(result: dict[str, object]) -> set[str]:
    completion = result["manual_review_completion"]
    assert isinstance(completion, dict)
    return {str(error["code"]) for error in completion["errors"]}  # type: ignore[index]


def test_empty_review_store_fails_with_snapshot_integrity_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    review_dir, docs = _make_review(tmp_path)
    _patch_identity(monkeypatch, docs)

    result = validate_review(review_dir)

    assert result["status"] == "fail"
    assert result["snapshot_integrity"]["status"] == "pass"
    assert result["manual_review_completion"]["status"] == "fail"
    assert {"manual_line_decisions_missing", "manual_boundary_decisions_missing", "manual_documents_incomplete"} <= _codes(result)


def test_current_shape_empty_review_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    review_dir, docs = _make_review(tmp_path, document_count=20, line_total=1407, boundary_total=1387)
    _patch_identity(monkeypatch, docs)

    result = validate_review(review_dir)

    assert result["progress"]["line_pending"] == 1407
    assert result["progress"]["boundary_pending"] == 1387
    assert result["progress"]["incomplete_documents"] == 20
    assert result["status"] == "fail"


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"complete": True, "missing_last_boundary": True}, "manual_boundary_decisions_missing"),
        ({"complete": True, "missing_last_line": True}, "manual_line_decisions_missing"),
        ({"complete": True, "unresolved_line": True}, "manual_line_decisions_unresolved"),
        ({"complete": True, "unresolved_boundary": True}, "manual_boundary_decisions_unresolved"),
        ({"complete": True, "stale_checksum": True}, "stale_source_checksum_decisions"),
        ({"complete": True, "stale_profile": True}, "stale_parser_profile_decisions"),
        ({"complete": True, "stale_identity": True}, "stale_parser_git_identity_decisions"),
        ({"complete": True, "duplicate_decision": True}, "duplicate_active_decisions"),
    ],
)
def test_completion_failure_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, bool], code: str) -> None:
    review_dir, docs = _make_review(tmp_path, **kwargs)
    _patch_identity(monkeypatch, docs)

    result = validate_review(review_dir)

    assert result["status"] == "fail"
    assert code in _codes(result)


def test_one_incomplete_document_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    review_dir, docs = _make_review(tmp_path, document_count=2, line_total=2, boundary_total=2, complete=True)
    _patch_identity(monkeypatch, docs)
    rows = [json.loads(line) for line in (review_dir / "manual_review_decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row["document_id"] != "doc-2"]
    write_jsonl(review_dir / "manual_review_decisions.jsonl", rows)

    result = validate_review(review_dir)

    assert result["progress"]["incomplete_documents"] == 1
    assert result["status"] == "fail"


def test_complete_review_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    review_dir, docs = _make_review(tmp_path, complete=True)
    _patch_identity(monkeypatch, docs)

    result = validate_review(review_dir)

    assert result["status"] == "pass"
    assert result["snapshot_integrity"]["status"] == "pass"
    assert result["manual_review_completion"]["status"] == "pass"


def test_write_summary_preserves_incomplete_failure_and_store_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    review_dir, docs = _make_review(tmp_path)
    _patch_identity(monkeypatch, docs)
    before_decisions = (review_dir / "manual_review_decisions.jsonl").read_text(encoding="utf-8")
    before_history = (review_dir / "manual_review_history.jsonl").read_text(encoding="utf-8")
    code = validator_main(["--review-dir", str(review_dir), "--write-summary"])
    output = capsys.readouterr().out

    assert code == 1
    assert "MANUAL REVIEW INCOMPLETE: FAIL" in output
    assert (review_dir / "manual_review_summary.md").exists()
    assert (review_dir / "manual_review_decisions.jsonl").read_text(encoding="utf-8") == before_decisions
    assert (review_dir / "manual_review_history.jsonl").read_text(encoding="utf-8") == before_history


def test_write_summary_complete_exits_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    review_dir, docs = _make_review(tmp_path, complete=True)
    _patch_identity(monkeypatch, docs)

    code = validator_main(["--review-dir", str(review_dir), "--write-summary"])
    output = capsys.readouterr().out

    assert code == 0
    assert "MANUAL REVIEW COMPLETE: PASS" in output
