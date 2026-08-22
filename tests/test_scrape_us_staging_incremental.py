"""Tests for US incremental scrape status and incomplete detection."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from app.models.search_result import NalusResult
from scripts.scrape_us_staging_incremental import (
    CollectOutcome,
    _as_record,
    _collect_date_scoped,
    _resolve_status,
)


def test_as_record_supports_slotted_nalus_result() -> None:
    item = NalusResult(
        result_id=1,
        case_reference="I. ÚS 1/26",
        ecli="ECLI:CZ:US:2026:1.US.1.26.1",
        judge_rapporteur=None,
        petitioner=None,
        popular_name=None,
        decision_date="01. 01. 2026",
        announcement_date=None,
        filing_date=None,
        publication_date=None,
    )
    record = _as_record(item)
    assert record["result_id"] == 1
    assert record["ecli"] == "ECLI:CZ:US:2026:1.US.1.26.1"
    assert record["source"] == "nalus"


def test_resolve_status_incomplete_on_cap() -> None:
    outcome = CollectOutcome(
        items=[],
        pages_scanned=50,
        total_pages=67,
        document_failed=0,
        incomplete_reason="pagination_cap_reached",
        listing_complete=False,
    )
    status, reason = _resolve_status(
        error=None,
        outcome=outcome,
        new_count=0,
        updated=0,
        document_failed=0,
    )
    assert status == "incomplete"
    assert reason == "pagination_cap_reached"


def test_resolve_status_partial_on_document_failures() -> None:
    outcome = CollectOutcome(
        items=[],
        pages_scanned=3,
        total_pages=3,
        document_failed=2,
        incomplete_reason=None,
        listing_complete=True,
    )
    status, reason = _resolve_status(
        error=None,
        outcome=outcome,
        new_count=1,
        updated=0,
        document_failed=2,
    )
    assert status == "partial"
    assert reason == "document_parse_failures"


def test_resolve_status_ok_when_complete() -> None:
    outcome = CollectOutcome(
        items=[],
        pages_scanned=2,
        total_pages=2,
        document_failed=0,
        incomplete_reason=None,
        listing_complete=True,
    )
    status, reason = _resolve_status(
        error=None,
        outcome=outcome,
        new_count=0,
        updated=0,
        document_failed=0,
    )
    assert status == "ok"
    assert reason is None


@patch("app.services.decision_service.enrich_results_with_text")
@patch("app.crawler.extractor.extract_search_page")
@patch("app.crawler.playwright_crawler.fetch_page_html")
def test_collect_marks_pagination_cap(
    mock_fetch: MagicMock,
    mock_extract: MagicMock,
    mock_enrich: MagicMock,
) -> None:
    mock_fetch.return_value = "<html/>"
    mock_enrich.side_effect = lambda items: items

    page_counter = {"n": 0}

    def _extract(html: str, query: str = "") -> MagicMock:
        page_counter["n"] += 1
        search_page = MagicMock()
        search_page.total_pages = 10
        result = MagicMock()
        result.ecli = f"ECLI:CZ:US:2024:1.US.{page_counter['n']}.24.1"
        result.case_reference = f"I.ÚS {page_counter['n']}/24"
        search_page.results = [result]
        return search_page

    mock_extract.side_effect = _extract

    outcome = _collect_date_scoped(
        date_from=date(2026, 8, 1),
        date_to=date(2026, 8, 22),
        max_pages=3,
        limit=500,
        page_sleep=0,
    )
    assert outcome.incomplete_reason == "pagination_cap_reached"
    assert outcome.listing_complete is False
