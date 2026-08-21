"""Tests for the document-level jurisprudence archive index and API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.judgments_archive_router import (
    clear_judgment_archive_store_cache,
    get_judgment_archive_store,
    router,
)
from app.rag.legal_v2.archive.builder import (
    build_archive_index_from_records,
    normalize_archive_document,
)
from app.rag.legal_v2.archive.courts import (
    COURT_CONSTITUTIONAL,
    COURT_SUPREME,
    COURT_SUPREME_ADMINISTRATIVE,
    list_archive_courts,
)
from app.rag.legal_v2.archive.store import JudgmentArchiveStore


def _records() -> list[dict]:
    # Two chunk-like duplicates of the same decision must collapse to one row.
    return [
        {
            "ecli": "ECLI:CZ:US:2024:1.US.100.24.1",
            "case_reference": "I.ÚS 100/24 #1",
            "decision_date": "15. 3. 2024",
            "decision_form": "Usnesení",
            "popular_name": None,
            "court": "constitutional_court",
            "chunk_index": 0,
        },
        {
            "ecli": "ECLI:CZ:US:2024:1.US.100.24.1",
            "case_reference": "I.ÚS 100/24 #1",
            "decision_date": "15. 3. 2024",
            "decision_form": "Usnesení",
            "popular_name": None,
            "court": "constitutional_court",
            "chunk_index": 1,
        },
        {
            "ecli": "ECLI:CZ:US:2024:2.US.200.24.1",
            "case_reference": "II.ÚS 200/24",
            "decision_date": "20. 3. 2024",
            "decision_form": "Nález",
            "popular_name": "Ochrana osobních údajů",
            "court": "constitutional_court",
        },
        {
            "ecli": "ECLI:CZ:US:2024:3.US.300.24.1",
            "case_reference": "III.ÚS 300/24",
            "decision_date": "10. 2. 2024",
            "decision_form": "Usnesení",
            "court": "constitutional_court",
        },
        {
            "ecli": "ECLI:CZ:US:2023:1.US.10.23.1",
            "case_reference": "I.ÚS 10/23",
            "decision_date": "5. 11. 2023",
            "decision_form": "Usnesení",
            "court": "constitutional_court",
        },
        {
            "ecli": "ECLI:CZ:US:2023:4.US.40.23.1",
            "case_reference": "IV.ÚS 40/23",
            "decision_date": "1. 12. 2023",
            "decision_form": "Nález",
            "title": "",
            "court": "constitutional_court",
        },
    ]


@pytest.fixture()
def archive_sqlite(tmp_path: Path) -> Path:
    path = tmp_path / "judgment_archive_v1.sqlite"
    count = build_archive_index_from_records(
        records=_records(),
        sqlite_path=path,
        source_kind="test_fixture",
    )
    assert count == 5
    return path


@pytest.fixture()
def archive_client(archive_sqlite: Path):
    clear_judgment_archive_store_cache()
    store = JudgmentArchiveStore(archive_sqlite)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_judgment_archive_store] = lambda: store
    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()
        clear_judgment_archive_store_cache()


def test_deduplicates_same_decision_despite_multiple_chunk_like_rows(
    archive_sqlite: Path,
) -> None:
    store = JudgmentArchiveStore(archive_sqlite)
    page = store.list_decisions(year=2024, month=3, limit=50)
    ids = [item.canonical_document_id for item in page.items]
    assert ids.count("ECLI:CZ:US:2024:1.US.100.24.1") == 1
    assert len(ids) == len(set(ids))


def test_year_and_month_counts(archive_sqlite: Path) -> None:
    store = JudgmentArchiveStore(archive_sqlite)
    courts = store.list_courts_with_years()
    constitutional = next(c for c in courts if c.court_id == COURT_CONSTITUTIONAL)
    years = {bucket.year: bucket.count for bucket in constitutional.years}
    assert years[2024] == 3
    assert years[2023] == 2

    months_2024 = {
        bucket.month: bucket.count
        for bucket in store.list_months(year=2024, court_id=COURT_CONSTITUTIONAL)
    }
    assert months_2024[3] == 2
    assert months_2024[2] == 1


def test_chronological_ordering(archive_sqlite: Path) -> None:
    store = JudgmentArchiveStore(archive_sqlite)
    page = store.list_decisions(year=2024, month=3, limit=50)
    dates = [item.decision_date for item in page.items]
    assert dates == sorted(dates, reverse=True)
    # Deterministic secondary key when dates differ / for stability.
    assert page.items[0].canonical_document_id == "ECLI:CZ:US:2024:2.US.200.24.1"
    assert page.items[1].canonical_document_id == "ECLI:CZ:US:2024:1.US.100.24.1"


def test_pagination_cursor(archive_sqlite: Path) -> None:
    store = JudgmentArchiveStore(archive_sqlite)
    first = store.list_decisions(year=2024, month=3, limit=1)
    assert len(first.items) == 1
    assert first.has_more is True
    assert first.next_cursor

    second = store.list_decisions(
        year=2024,
        month=3,
        limit=1,
        cursor=first.next_cursor,
    )
    assert len(second.items) == 1
    assert second.items[0].canonical_document_id != first.items[0].canonical_document_id
    assert second.has_more is False


def test_canonical_ecli_document_identity() -> None:
    document = normalize_archive_document(
        {
            "ecli": "ecli:cz:us:2024:1.us.100.24.1",
            "canonical_document_id": "ECLI:CZ:US:2024:1.US.100.24.1",
            "decision_date": "2024-03-15",
            "case_reference": "I.ÚS 100/24",
            "court": "usoud",
        }
    )
    assert document is not None
    assert document.canonical_document_id == document.ecli
    assert document.canonical_document_id.startswith("ECLI:CZ:US:")
    assert document.court == COURT_CONSTITUTIONAL


def test_missing_title_is_not_invented() -> None:
    without_title = normalize_archive_document(
        {
            "ecli": "ECLI:CZ:US:2024:1.US.100.24.1",
            "decision_date": "15. 3. 2024",
            "case_reference": "I.ÚS 100/24",
            "topics_and_keywords": ["ochrana soukromí"],
            "popular_name": None,
        }
    )
    assert without_title is not None
    assert without_title.title is None

    empty_title = normalize_archive_document(
        {
            "ecli": "ECLI:CZ:US:2024:1.US.101.24.1",
            "decision_date": "15. 3. 2024",
            "case_reference": "I.ÚS 101/24",
            "title": "   ",
            "popular_name": "",
        }
    )
    assert empty_title is not None
    assert empty_title.title is None

    real_title = normalize_archive_document(
        {
            "ecli": "ECLI:CZ:US:2024:1.US.102.24.1",
            "decision_date": "15. 3. 2024",
            "case_reference": "I.ÚS 102/24",
            "popular_name": "Ochrana osobních údajů",
        }
    )
    assert real_title is not None
    assert real_title.title == "Ochrana osobních údajů"


def test_future_court_filter_and_schema_support(archive_sqlite: Path) -> None:
    court_ids = {court.court_id for court in list_archive_courts()}
    assert COURT_CONSTITUTIONAL in court_ids
    assert COURT_SUPREME in court_ids
    assert COURT_SUPREME_ADMINISTRATIVE in court_ids

    store = JudgmentArchiveStore(archive_sqlite)
    all_courts = store.list_courts_with_years()
    assert {c.court_id for c in all_courts} == court_ids
    supreme = next(c for c in all_courts if c.court_id == COURT_SUPREME)
    assert supreme.document_count == 0
    assert supreme.years == []
    assert supreme.ingest_ready is False

    filtered = store.list_courts_with_years(court_id=COURT_SUPREME_ADMINISTRATIVE)
    assert len(filtered) == 1
    assert filtered[0].document_count == 0


def test_archive_api_endpoints(archive_client: TestClient) -> None:
    overview = archive_client.get("/api/judgments/archive")
    assert overview.status_code == 200
    payload = overview.json()
    assert payload["index_ready"] is True
    constitutional = next(
        c for c in payload["courts"] if c["court_id"] == COURT_CONSTITUTIONAL
    )
    assert constitutional["document_count"] == 5
    assert constitutional["years"][0]["year"] == 2024

    months = archive_client.get(
        "/api/judgments/archive/2024",
        params={"court": COURT_CONSTITUTIONAL},
    )
    assert months.status_code == 200
    assert months.json()["months"][0]["month"] == 3

    decisions = archive_client.get(
        "/api/judgments/archive/2024/3",
        params={"court": COURT_CONSTITUTIONAL, "limit": 1},
    )
    assert decisions.status_code == 200
    body = decisions.json()
    assert body["has_more"] is True
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert item["canonical_document_id"] == item["ecli"]
    assert "title" in item

    page2 = archive_client.get(
        "/api/judgments/archive/2024/3",
        params={
            "court": COURT_CONSTITUTIONAL,
            "limit": 1,
            "cursor": body["next_cursor"],
        },
    )
    assert page2.status_code == 200
    assert page2.json()["items"][0]["canonical_document_id"] != item[
        "canonical_document_id"
    ]
