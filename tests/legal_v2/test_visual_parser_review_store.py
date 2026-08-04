from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.legal_v2.parser_review.models import REVIEW_SCHEMA_VERSION, write_json, write_jsonl
from scripts.legal_v2.parser_review.store import append_decision


def _review_dir(tmp_path: Path) -> Path:
    review_dir = tmp_path / "review"
    write_json(
        review_dir / "review_manifest.json",
        {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "parser_profile": "legal-decision-parser.cz-courts.v4",
            "head": "test-head",
        },
    )
    write_jsonl(
        review_dir / "review_documents.jsonl",
        [
            {
                "schema_version": REVIEW_SCHEMA_VERSION,
                "document_id": "doc-1",
                "review_number": 1,
                "source_id": "source-1",
                "court": "constitutional_court",
            }
        ],
    )
    write_jsonl(
        review_dir / "review_lines.jsonl",
        [
            {
                "schema_version": REVIEW_SCHEMA_VERSION,
                "item_type": "line",
                "item_id": "line-1",
                "document_id": "doc-1",
                "source_checksum": "sha",
                "raw_line_number": 1,
                "parser_proposed_line_class": "heading",
            }
        ],
    )
    write_jsonl(review_dir / "review_boundaries.jsonl", [])
    (review_dir / "manual_review_decisions.jsonl").write_text("", encoding="utf-8")
    (review_dir / "manual_review_history.jsonl").write_text("", encoding="utf-8")
    return review_dir


def test_store_appends_latest_and_history_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    review_dir = _review_dir(tmp_path)
    monkeypatch.setattr("scripts.legal_v2.parser_review.store.parser_git_identity", lambda: {"head": "test-head", "parser_profile": "legal-decision-parser.cz-courts.v4"})
    payload = {
        "item_type": "line",
        "item_id": "line-1",
        "document_id": "doc-1",
        "source_checksum": "sha",
        "decision_status": "overridden",
        "manual_class": "metadata",
        "interface": "powershell",
    }

    first = append_decision(review_dir, payload)
    second = append_decision(review_dir, {**payload, "manual_class": "heading"})

    active = [json.loads(line) for line in (review_dir / "manual_review_decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    history = [json.loads(line) for line in (review_dir / "manual_review_history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert first["revision_number"] == 1
    assert second["revision_number"] == 2
    assert len(active) == 1
    assert active[0]["manual_class"] == "heading"
    assert len(history) == 2


def test_store_rejects_unknown_manual_class(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    review_dir = _review_dir(tmp_path)
    monkeypatch.setattr("scripts.legal_v2.parser_review.store.parser_git_identity", lambda: {"head": "test-head", "parser_profile": "legal-decision-parser.cz-courts.v4"})
    with pytest.raises(ValueError, match="Unsupported manual class"):
        append_decision(
            review_dir,
            {
                "item_type": "line",
                "item_id": "line-1",
                "document_id": "doc-1",
                "source_checksum": "sha",
                "decision_status": "overridden",
                "manual_class": "freeform",
                "interface": "powershell",
            },
        )
