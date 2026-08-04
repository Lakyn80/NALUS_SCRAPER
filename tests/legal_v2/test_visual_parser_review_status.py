from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.legal_v2.parser_review.status import (
    ManualReviewStatus,
    ParserValidationStatus,
    ReviewStatusBuilder,
)
from scripts.legal_v2.parser_review.web_api import ReviewApi

REVIEW_DIR = Path("artifacts/legal_v2/visual_parser_review")


def test_parser_and_manual_statuses_are_separate_for_golden_line() -> None:
    status, payload = ReviewApi(REVIEW_DIR).get(
        "/api/lines",
        {"document_id": ["doc-cfa470876b0d5ed7"]},
    )

    assert status == 200
    line36 = next(row for row in payload["lines"] if row["raw_line_number"] == 36)
    assert line36["parser_validation_status"] == ParserValidationStatus.AUTO_VALIDATED_GOLDEN.value
    assert line36["parser_validation_label"] == "AUTO-VALIDATED · GOLDEN v6"
    assert line36["parser_proposed_line_class"] == "list_or_table"
    assert line36["manual_review_status"] == ManualReviewStatus.NOT_MANUALLY_REVIEWED.value
    assert line36["manual_review_label"] == "Manual review: not performed"
    assert line36["manual_decision_status"] == "pending"


def test_exact_golden_boundary_and_document_blocks_are_auto_validated() -> None:
    api = ReviewApi(REVIEW_DIR)
    status, boundaries = api.get(
        "/api/boundary-cards",
        {"document_id": ["doc-cfa470876b0d5ed7"]},
    )
    status_docs, docs = api.get("/api/documents", {})

    assert status == 200
    assert status_docs == 200
    l35_l36 = next(card for card in boundaries["boundaries"] if card["before"]["line_number"] == 35)
    l42_l43 = next(card for card in boundaries["boundaries"] if card["before"]["line_number"] == 42)
    doc11 = next(row for row in docs["documents"] if row["document_id"] == "doc-cfa470876b0d5ed7")
    assert l35_l36["parser_validation_status"] == ParserValidationStatus.AUTO_VALIDATED_GOLDEN.value
    assert l35_l36["parser_boundary"]["display"] == "MERGE"
    assert l42_l43["parser_validation_status"] == ParserValidationStatus.AUTO_VALIDATED_GOLDEN.value
    assert l42_l43["parser_boundary"]["display"] == "SPLIT"
    assert doc11["parser_validation_label"] == "GOLDEN PASS"
    assert doc11["parser_block_validated"] == doc11["parser_block_total"] == 39


def test_document_05_and_document_16_display_golden_pass() -> None:
    status, payload = ReviewApi(REVIEW_DIR).get("/api/documents", {})

    assert status == 200
    docs = {row["document_id"]: row for row in payload["documents"]}
    assert docs["doc-e5ac4b1fcd075062"]["parser_validation_label"] == "GOLDEN PASS"
    assert docs["doc-4f3c37d9c5a1afb7"]["parser_validation_label"] == "GOLDEN PASS"


def test_non_golden_changed_item_is_review_recommended_and_unchanged_is_validated() -> None:
    api = ReviewApi(REVIEW_DIR)
    status, changed = api.get("/api/parser-v6/changes", {"document_id": ["doc-abd57ac0aa5dfe5b"]})
    assert status == 200
    changed_line = changed["changed_classes"][0]
    assert changed_line["parser_validation_status"] == ParserValidationStatus.PARSER_CHANGED_NEEDS_REVIEW.value

    status, lines = api.get("/api/lines", {"document_id": ["doc-abd57ac0aa5dfe5b"]})
    assert status == 200
    changed_numbers = {row["line"] for row in changed["changed_classes"]}
    unchanged_line = next(row for row in lines["lines"] if row["raw_line_number"] not in changed_numbers)
    assert unchanged_line["parser_validation_status"] == ParserValidationStatus.PARSER_VALIDATED.value


def test_manual_statuses_and_document2_completion_are_preserved() -> None:
    status, progress = ReviewApi(REVIEW_DIR).get("/api/progress", {})

    assert status == 200
    doc2 = next(row for row in progress["documents"] if row["document_id"] == "doc-b73cac9b3dfc8a42")
    assert doc2["line_reviewed"] == doc2["line_total"] == 13
    assert doc2["boundary_reviewed"] == doc2["boundary_total"] == 12
    assert doc2["line_unresolved"] == 0
    assert doc2["boundary_unresolved"] == 0
    assert progress["manual_review"]["overridden"] >= 25
    assert progress["manual_review"]["stale"] == 3


def test_status_generation_does_not_modify_manual_stores() -> None:
    before = _manual_hashes()
    api = ReviewApi(REVIEW_DIR)
    api.get("/api/progress", {})
    api.get("/api/lines", {"document_id": ["doc-cfa470876b0d5ed7"]})
    api.get("/api/boundary-cards", {"document_id": ["doc-cfa470876b0d5ed7"]})
    api.get("/api/parser-v6/changes", {"document_id": ["doc-cfa470876b0d5ed7"]})

    assert _manual_hashes() == before


def test_golden_mismatch_and_parser_profile_mismatch_are_conflicts() -> None:
    builder = ReviewStatusBuilder(REVIEW_DIR)
    line = next(
        row
        for row in builder.lines
        if row["document_id"] == "doc-cfa470876b0d5ed7" and row["raw_line_number"] == 36
    )
    bad_line = {**line, "parser_proposed_line_class": "metadata"}
    assert builder.parser_status_for_line(bad_line)["parser_validation_status"] == ParserValidationStatus.PARSER_CONFLICT.value

    builder.manifest["parser_profile"] = "legal-decision-parser.cz-courts.v5"
    builder._parser_profile_valid = False
    assert builder.parser_status_for_line(line)["parser_validation_status"] == ParserValidationStatus.PARSER_CONFLICT.value


def test_missing_validation_is_unvalidated_and_checksum_mismatch_conflicts() -> None:
    builder = ReviewStatusBuilder(REVIEW_DIR)
    non_golden = next(row for row in builder.lines if row["document_id"] == "doc-b73cac9b3dfc8a42")
    builder._corpus_by_doc.pop("doc-b73cac9b3dfc8a42", None)
    assert builder.parser_status_for_line(non_golden)["parser_validation_status"] == ParserValidationStatus.PARSER_UNVALIDATED.value

    golden = next(row for row in builder.lines if row["document_id"] == "doc-cfa470876b0d5ed7")
    bad_source = {**golden, "source_checksum": "bad-source-checksum"}
    assert builder.parser_status_for_line(bad_source)["parser_validation_status"] == ParserValidationStatus.PARSER_CONFLICT.value

    builder._golden_inputs_valid = False
    assert builder.parser_status_for_line(golden)["parser_validation_status"] == ParserValidationStatus.PARSER_CONFLICT.value


def _manual_hashes() -> dict[str, tuple[int, str]]:
    result = {}
    for name in ("manual_review_decisions.jsonl", "manual_review_history.jsonl"):
        path = REVIEW_DIR / name
        body = path.read_bytes()
        result[name] = (len(body), hashlib.sha256(body).hexdigest().upper())
    return result
