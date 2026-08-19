from __future__ import annotations

from datetime import date

import pytest

from app.nsoud.scraper import BaseProvider, ResponseData, SearchPageData, run_discovery


class _FakeProvider(BaseProvider):
    name = "fake"
    transport_name = "fake"

    def __init__(self, *, total_results: int, links: list) -> None:
        super().__init__(debug_dir=None)
        self._total_results = total_results
        self._links = links

    def fetch_home(self) -> ResponseData:
        return ResponseData(
            url="https://example.invalid/home",
            html="<html><body><form action='/search'></form></body></html>",
            headers={},
        )

    def submit_search(self, start_date, end_date, page_size: int) -> SearchPageData:
        return SearchPageData(
            url="https://example.invalid/search?Query=test",
            html="<html><body><h3>Výsledky 0 - 0 z 0</h3></body></html>",
            total_results=self._total_results,
            links=self._links,
        )

    def fetch_detail(self, url: str) -> ResponseData:
        return ResponseData(url=url, html="<html></html>", headers={})


def test_run_discovery_allows_legitimate_empty_month() -> None:
    provider = _FakeProvider(total_results=0, links=[])
    discovery = run_discovery(
        provider,
        window_start=date(2019, 12, 1),
        window_end=date(2019, 12, 31),
        page_size=20,
    )
    assert discovery.total_results == 0
    assert discovery.detail_urls == []


def test_run_discovery_still_fails_when_results_reported_but_no_links() -> None:
    provider = _FakeProvider(total_results=12, links=[])
    with pytest.raises(RuntimeError, match="Discovery did not yield any detail URLs"):
        run_discovery(
            provider,
            window_start=date(2019, 12, 1),
            window_end=date(2019, 12, 31),
            page_size=20,
        )
