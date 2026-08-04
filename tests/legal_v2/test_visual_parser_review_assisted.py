from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.legal_v2.parser_review.assisted import build_assisted_review, occurrences_for_rule
from scripts.legal_v2.parser_review.batches import apply_batch, revert_batch
from scripts.legal_v2.parser_review.models import REVIEW_SCHEMA_VERSION, write_json, write_jsonl
from scripts.legal_v2.parser_review.progress import compute_progress
from scripts.legal_v2.parser_review.rules import line_signature
from scripts.legal_v2.parser_review.web_api import ReviewApi


def _decision(row: dict[str, object], *, value: str) -> dict[str, object]:
    item_type = str(row["item_type"])
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "document_id": row["document_id"],
        "source_checksum": row["source_checksum"],
        "parser_profile": "legal-decision-parser.cz-courts.v4",
        "parser_git_identity": "test-head",
        "item_type": item_type,
        "item_id": row["item_id"],
        "raw_line_number": row.get("raw_line_number") or row.get("previous_line_number"),
        "parser_proposal": {},
        "previous_automated_annotation": row.get("previous_automated_annotation") or row.get("previous_automated_boundary_annotation"),
        "manual_class": value if item_type == "line" else None,
        "manual_boundary_decision": value if item_type == "boundary" else None,
        "decision_status": "overridden",
        "reviewer_comment": "",
        "revision_number": 1,
        "timestamp": "2026-08-04T00:00:00Z",
        "interface": "html",
        "review_tool_version": "visual-parser-review-tool.v1",
    }


def _fixture_review(tmp_path: Path, *, complete_high_court: bool = False, conflict: bool = False) -> Path:
    review_dir = tmp_path / "review"
    write_json(
        review_dir / "review_manifest.json",
        {"schema_version": REVIEW_SCHEMA_VERSION, "parser_profile": "legal-decision-parser.cz-courts.v4", "head": "test-head"},
    )
    docs = [
        {"schema_version": REVIEW_SCHEMA_VERSION, "document_id": "doc-us-1", "review_number": 1, "source_id": "us-1", "court": "constitutional_court", "source_checksum": "sha-us-1"},
        {"schema_version": REVIEW_SCHEMA_VERSION, "document_id": "doc-us-2", "review_number": 2, "source_id": "us-2", "court": "constitutional_court", "source_checksum": "sha-us-2"},
        {"schema_version": REVIEW_SCHEMA_VERSION, "document_id": "doc-vs-1", "review_number": 3, "source_id": "vs-1", "court": "high_court_prague", "source_checksum": "sha-vs-1"},
    ]
    lines = [
        _line("doc-us-1", "constitutional_court", "sha-us-1", 1, "NALUS - databáze rozhodnutí Ústavního soudu", "page_header"),
        _line("doc-us-1", "constitutional_court", "sha-us-1", 2, "I.ÚS 1/24 ze dne 1. 1. 2024", "case_identifier"),
        _line("doc-us-2", "constitutional_court", "sha-us-2", 1, "NALUS - databáze rozhodnutí Ústavního soudu", "page_header"),
        _line("doc-us-2", "constitutional_court", "sha-us-2", 2, "I.ÚS 2/24 ze dne 2. 1. 2024", "case_identifier"),
        _line("doc-vs-1", "high_court_prague", "sha-vs-1", 1, "NALUS - databáze rozhodnutí Ústavního soudu", "page_header"),
    ]
    boundaries = [
        _boundary("doc-us-1", "sha-us-1", 1, 2, True),
        _boundary("doc-us-2", "sha-us-2", 1, 2, True),
    ]
    write_jsonl(review_dir / "review_documents.jsonl", docs)
    write_jsonl(review_dir / "review_lines.jsonl", lines)
    write_jsonl(review_dir / "review_boundaries.jsonl", boundaries)
    decisions = [
        _decision(lines[0], value="layout_noise" if not conflict else "metadata"),
        _decision(lines[1], value="metadata"),
        _decision(boundaries[0], value="split"),
    ]
    if complete_high_court:
        decisions.append(_decision(lines[4], value="layout_noise"))
    write_jsonl(review_dir / "manual_review_decisions.jsonl", decisions)
    write_jsonl(review_dir / "manual_review_history.jsonl", decisions)
    return review_dir


def _line(doc: str, court: str, checksum: str, num: int, text: str, previous: str) -> dict[str, object]:
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "item_type": "line",
        "item_id": f"{doc}:line:{num}",
        "document_id": doc,
        "court": court,
        "source_checksum": checksum,
        "raw_line_number": num,
        "raw_text": text,
        "parser_proposed_line_class": "prose_start",
        "previous_automated_annotation": previous,
        "suspicious_reason_codes": [],
    }


def _boundary(doc: str, checksum: str, left: int, right: int, parser_boundary: bool) -> dict[str, object]:
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "item_type": "boundary",
        "item_id": f"{doc}:boundary:{left}-{right}",
        "document_id": doc,
        "source_checksum": checksum,
        "previous_line_number": left,
        "next_line_number": right,
        "parser_proposed_boundary": parser_boundary,
        "previous_automated_boundary_annotation": True,
        "suspicious_reason_codes": [],
    }


def test_only_completed_documents_generate_court_scoped_safe_rules(tmp_path: Path) -> None:
    review_dir = _fixture_review(tmp_path)

    result = build_assisted_review(review_dir=review_dir)

    assert result["summary"]["completed_evidence_documents"][0]["document_id"] == "doc-us-1"
    assert result["summary"]["safe_rules"] > 0
    assert all(rule["court"] == "constitutional_court" for rule in result["rules"])
    assert "high_court_prague" in result["summary"]["high_court_gated"]


def test_constitutional_rules_do_not_apply_to_high_court(tmp_path: Path) -> None:
    review_dir = _fixture_review(tmp_path)
    result = build_assisted_review(review_dir=review_dir)

    assert not any(item["court"] == "high_court_prague" and not item["excluded"] for item in result["suggestions"])


def test_exact_text_rules_match_pending_items_and_exclude_existing_decisions(tmp_path: Path) -> None:
    review_dir = _fixture_review(tmp_path)
    result = build_assisted_review(review_dir=review_dir)

    exact_rules = [rule for rule in result["rules"] if rule["rule_type"] == "exact_normalized_line"]
    assert exact_rules
    occurrences = occurrences_for_rule(review_dir, exact_rules[0]["rule_id"])
    assert any(item["document_id"] == "doc-us-2" and not item["excluded"] for item in occurrences)
    assert any(item["document_id"] == "doc-us-1" and item["excluded_reason"] == "existing_manual_decision" for item in occurrences)


def test_preview_does_not_modify_store(tmp_path: Path) -> None:
    review_dir = _fixture_review(tmp_path)
    before = (review_dir / "manual_review_decisions.jsonl").read_text(encoding="utf-8")

    build_assisted_review(review_dir=review_dir)

    assert (review_dir / "manual_review_decisions.jsonl").read_text(encoding="utf-8") == before


def test_boundary_context_rules_preserve_boolean_semantics_for_preserve_parser(tmp_path: Path) -> None:
    review_dir = _fixture_review(tmp_path)
    rows = [json.loads(line) for line in (review_dir / "manual_review_decisions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if row["item_type"] == "boundary":
            row["manual_boundary_decision"] = "preserve_parser"
    write_jsonl(review_dir / "manual_review_decisions.jsonl", rows)
    result = build_assisted_review(review_dir=review_dir)

    boundary_rules = [rule for rule in result["rules"] if rule["item_type"] == "boundary"]

    assert boundary_rules
    assert boundary_rules[0]["target_value"] == "split"


def test_conflicting_completed_manual_evidence_blocks_rule(tmp_path: Path) -> None:
    review_dir = _fixture_review(tmp_path)
    lines = [json.loads(line) for line in (review_dir / "review_lines.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    boundaries = [json.loads(line) for line in (review_dir / "review_boundaries.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    decisions = [json.loads(line) for line in (review_dir / "manual_review_decisions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    decisions.extend(
        [
            _decision(lines[2], value="metadata"),
            _decision(lines[3], value="metadata"),
            _decision(boundaries[1], value="split"),
        ]
    )
    write_jsonl(review_dir / "manual_review_decisions.jsonl", decisions)
    write_jsonl(review_dir / "manual_review_history.jsonl", decisions)

    result = build_assisted_review(review_dir=review_dir)

    blocked = [rule for rule in result["rules"] if rule["confidence"] == "BLOCKED"]
    assert blocked
    assert any(conflict["code"] == "conflicting_completed_manual_evidence" for rule in blocked for conflict in rule["conflicts"])
    assert not any(batch["apply_allowed"] for batch in result["batches"] if batch["rule_id"] in {rule["rule_id"] for rule in blocked})


def test_stale_parser_profile_blocks_completed_evidence(tmp_path: Path) -> None:
    review_dir = _fixture_review(tmp_path)
    decisions = [json.loads(line) for line in (review_dir / "manual_review_decisions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    decisions[0]["parser_profile"] = "stale-profile"
    write_jsonl(review_dir / "manual_review_decisions.jsonl", decisions)

    result = build_assisted_review(review_dir=review_dir)

    assert result["summary"]["completed_evidence_documents"] == []
    assert result["summary"]["safe_rules"] == 0


def test_unresolved_item_blocks_completed_evidence(tmp_path: Path) -> None:
    review_dir = _fixture_review(tmp_path)
    decisions = [json.loads(line) for line in (review_dir / "manual_review_decisions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    decisions[0]["decision_status"] = "unresolved"
    write_jsonl(review_dir / "manual_review_decisions.jsonl", decisions)

    result = build_assisted_review(review_dir=review_dir)

    assert result["summary"]["completed_evidence_documents"] == []
    assert result["summary"]["safe_rules"] == 0


def test_anchored_templates_reject_partial_substring_matches() -> None:
    assert line_signature({"raw_text": "prefix Odůvodnění:", "previous_automated_annotation": "section_heading"}) == ""
    assert line_signature({"raw_text": "Poučení: proti tomuto rozhodnutí", "previous_automated_annotation": "section_heading"}) == "template:pouceni_prefix"


def test_batch_apply_requires_exact_confirmation_and_writes_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    review_dir = _fixture_review(tmp_path)
    monkeypatch.setattr("scripts.legal_v2.parser_review.store.parser_git_identity", lambda: {"head": "test-head", "parser_profile": "legal-decision-parser.cz-courts.v4"})
    result = build_assisted_review(review_dir=review_dir)
    rule = next(rule for rule in result["rules"] if rule["item_type"] == "line")

    with pytest.raises(ValueError, match="Confirmation mismatch"):
        apply_batch(review_dir, rule_id=rule["rule_id"], confirmation="APPLY wrong 1")
    applied = apply_batch(review_dir, rule_id=rule["rule_id"], confirmation=f"APPLY {rule['rule_id']} 1")

    active = [json.loads(line) for line in (review_dir / "manual_review_decisions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert applied["applied_count"] == 1
    assert any(row.get("assisted_rule_id") == rule["rule_id"] and row.get("assisted_batch_id") == applied["batch_id"] for row in active)


def test_batch_apply_partial_failure_leaves_store_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    review_dir = _fixture_review(tmp_path)
    docs = [json.loads(line) for line in (review_dir / "review_documents.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = [json.loads(line) for line in (review_dir / "review_lines.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    docs.append({"schema_version": REVIEW_SCHEMA_VERSION, "document_id": "doc-us-3", "review_number": 4, "source_id": "us-3", "court": "constitutional_court", "source_checksum": "sha-us-3"})
    lines.append(_line("doc-us-3", "constitutional_court", "sha-us-3", 1, "NALUS - databáze rozhodnutí Ústavního soudu", "page_header"))
    write_jsonl(review_dir / "review_documents.jsonl", docs)
    write_jsonl(review_dir / "review_lines.jsonl", lines)
    monkeypatch.setattr("scripts.legal_v2.parser_review.store.parser_git_identity", lambda: {"head": "test-head", "parser_profile": "legal-decision-parser.cz-courts.v4"})
    result = build_assisted_review(review_dir=review_dir)
    rule = next(rule for rule in result["rules"] if rule["rule_type"] == "exact_normalized_line")
    before_active = (review_dir / "manual_review_decisions.jsonl").read_text(encoding="utf-8")
    before_history = (review_dir / "manual_review_history.jsonl").read_text(encoding="utf-8")
    import scripts.legal_v2.parser_review.batches as batches_module

    original_decision_record = batches_module._decision_record
    calls = 0

    def flaky_decision_record(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("synthetic write failure")
        return original_decision_record(*args, **kwargs)

    monkeypatch.setattr(batches_module, "_decision_record", flaky_decision_record)

    with pytest.raises(ValueError, match="synthetic write failure"):
        apply_batch(review_dir, rule_id=rule["rule_id"], confirmation=f"APPLY {rule['rule_id']} 2")

    assert (review_dir / "manual_review_decisions.jsonl").read_text(encoding="utf-8") == before_active
    assert (review_dir / "manual_review_history.jsonl").read_text(encoding="utf-8") == before_history


def test_batch_revert_restores_pending_without_deleting_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    review_dir = _fixture_review(tmp_path)
    monkeypatch.setattr("scripts.legal_v2.parser_review.store.parser_git_identity", lambda: {"head": "test-head", "parser_profile": "legal-decision-parser.cz-courts.v4"})
    result = build_assisted_review(review_dir=review_dir)
    rule = next(rule for rule in result["rules"] if rule["item_type"] == "line")
    applied = apply_batch(review_dir, rule_id=rule["rule_id"], confirmation=f"APPLY {rule['rule_id']} 1")
    history_before = len((review_dir / "manual_review_history.jsonl").read_text(encoding="utf-8").splitlines())

    reverted = revert_batch(review_dir, batch_id=applied["batch_id"], confirmation=f"REVERT {applied['batch_id']}")

    active = (review_dir / "manual_review_decisions.jsonl").read_text(encoding="utf-8")
    history_after = len((review_dir / "manual_review_history.jsonl").read_text(encoding="utf-8").splitlines())
    assert reverted["reverted_count"] == 1
    assert applied["batch_id"] not in active
    assert history_after > history_before


def test_batch_revert_refuses_later_independent_decision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    review_dir = _fixture_review(tmp_path)
    monkeypatch.setattr("scripts.legal_v2.parser_review.store.parser_git_identity", lambda: {"head": "test-head", "parser_profile": "legal-decision-parser.cz-courts.v4"})
    result = build_assisted_review(review_dir=review_dir)
    rule = next(rule for rule in result["rules"] if rule["item_type"] == "line")
    applied = apply_batch(review_dir, rule_id=rule["rule_id"], confirmation=f"APPLY {rule['rule_id']} 1")
    active = [json.loads(line) for line in (review_dir / "manual_review_decisions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    batch_record = next(row for row in active if row.get("assisted_batch_id") == applied["batch_id"])
    independent_record = dict(batch_record)
    independent_record["assisted_batch_id"] = None
    independent_record["assisted_rule_id"] = None
    independent_record["interface"] = "html"
    independent_record["revision_number"] = int(batch_record["revision_number"]) + 1
    with (review_dir / "manual_review_history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(independent_record, ensure_ascii=False, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="later independent decision"):
        revert_batch(review_dir, batch_id=applied["batch_id"], confirmation=f"REVERT {applied['batch_id']}")


def test_progress_distinguishes_assisted_batch_decisions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    review_dir = _fixture_review(tmp_path)
    monkeypatch.setattr("scripts.legal_v2.parser_review.store.parser_git_identity", lambda: {"head": "test-head", "parser_profile": "legal-decision-parser.cz-courts.v4"})
    result = build_assisted_review(review_dir=review_dir)
    rule = next(rule for rule in result["rules"] if rule["item_type"] == "line")
    apply_batch(review_dir, rule_id=rule["rule_id"], confirmation=f"APPLY {rule['rule_id']} 1")

    progress = compute_progress(review_dir)

    assert progress["assisted_batch_decision_records"] == 1
    assert progress["manual_individual_decision_records"] == 3


def test_assisted_api_rules_and_confirmation_rejection(tmp_path: Path) -> None:
    review_dir = _fixture_review(tmp_path)
    status, payload = ReviewApi(review_dir).get("/api/assisted/rules", {})
    assert status == 200
    assert payload["rules"]

    status, response = ReviewApi(review_dir).post("/api/assisted/apply", {"rule_id": payload["rules"][0]["rule_id"], "confirmation": "wrong"})
    assert status == 400
    assert "Confirmation mismatch" in response["error"]
