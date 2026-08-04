from __future__ import annotations

from pathlib import Path

import pytest

from scripts.legal_v2.parser_review.security import assert_local_bind
from scripts.legal_v2.parser_review.web_api import ReviewApi


def test_web_security_allows_loopback_only() -> None:
    assert_local_bind("127.0.0.1")
    assert_local_bind("::1")
    with pytest.raises(ValueError):
        assert_local_bind("0.0.0.0")


def test_boundary_card_api_contains_line_text_context_and_explicit_split_merge() -> None:
    status, payload = ReviewApi(Path("artifacts/legal_v2/visual_parser_review")).get(
        "/api/boundary-cards",
        {"document_id": ["doc-b73cac9b3dfc8a42"]},
    )

    assert status == 200
    card = next(
        item
        for item in payload["boundaries"]
        if item["before"]["line_number"] == 1 and item["after"]["line_number"] == 2
    )
    assert card["before"]["raw_text"] == "NALUS - databáze rozhodnutí Ústavního soudu"
    assert card["after"]["raw_text"] == "I.ÚS 3299/24 ze dne 20. 12. 2024"
    assert card["before"]["parser_class"] == "layout_noise"
    assert card["after"]["parser_class"] == "metadata"
    assert card["before"]["previous_class"] == "page_header"
    assert card["after"]["previous_class"] == "case_identifier"
    assert card["before"]["parser_block_id"]
    assert card["before"]["parser_block_id"] != card["after"]["parser_block_id"]
    assert card["parser_boundary"]["display"] == "SPLIT"
    assert card["previous_boundary"]["display"] == "SPLIT"
    assert card["manual_decision"]["stored_value"] == "split"
    assert "Force SPLIT before line 2." == card["manual_decision"]["explanation"]
    assert "Parser and previous annotation both say SPLIT." == card["conflict"]["text"]
    assert "parser_boundary_disagrees_previous_annotation" not in card["suspicious_reasons"]
    assert isinstance(card["context_after"], list)


def test_boundary_card_api_reports_agreement_text_when_parser_and_previous_match() -> None:
    status, payload = ReviewApi(Path("artifacts/legal_v2/visual_parser_review")).get(
        "/api/boundary-cards",
        {"document_id": ["doc-b73cac9b3dfc8a42"]},
    )

    assert status == 200
    card = next(item for item in payload["boundaries"] if item["parser_boundary"]["display"] == item["previous_boundary"]["display"])
    assert card["conflict"]["has_conflict"] is False
    assert "both say" in card["conflict"]["text"]


def test_parser_v6_changed_queue_api_filters_by_document() -> None:
    status, payload = ReviewApi(Path("artifacts/legal_v2/visual_parser_review")).get(
        "/api/parser-v6/changes",
        {"document_id": ["doc-e5ac4b1fcd075062"]},
    )

    assert status == 200
    assert payload["boundary_count"] > 0
    assert payload["class_count"] > 0
    assert payload["block_count"] > 0
    assert all(row["document_id"] == "doc-e5ac4b1fcd075062" for row in payload["changed_boundaries"])
    assert all(row["document_id"] == "doc-e5ac4b1fcd075062" for row in payload["changed_classes"])
    assert all(row["document_id"] == "doc-e5ac4b1fcd075062" for row in payload["changed_blocks"])
