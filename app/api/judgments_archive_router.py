"""Read-only jurisprudence archive API (document-level browse).

Endpoints:
  GET /api/judgments/archive
  GET /api/judgments/archive/{year}
  GET /api/judgments/archive/{year}/{month}
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.rag.legal_v2.archive.models import (
    ArchiveCourtSummary,
    ArchiveDecision,
    ArchiveDecisionsPage,
    ArchiveMonthBucket,
)
from app.rag.legal_v2.archive.store import (
    DEFAULT_PAGE_SIZE,
    JudgmentArchiveStore,
    resolve_archive_sqlite_path,
)
from app.rag.legal_v2.archive.courts import normalize_court_id

logger = get_logger(__name__)

router = APIRouter(prefix="/api/judgments", tags=["judgments-archive"])


class ArchiveYearOut(BaseModel):
    year: int
    count: int


class ArchiveMonthOut(BaseModel):
    month: int
    count: int


class ArchiveCourtOut(BaseModel):
    court_id: str
    court_name: str
    document_count: int
    years: list[ArchiveYearOut]
    ingest_ready: bool


class ArchiveOverviewResponse(BaseModel):
    courts: list[ArchiveCourtOut]
    index_ready: bool
    index_path: str


class ArchiveMonthsResponse(BaseModel):
    court_id: str
    year: int
    months: list[ArchiveMonthOut]


class ArchiveDecisionOut(BaseModel):
    canonical_document_id: str
    ecli: str
    case_number: str | None = None
    court: str
    decision_date: str | None = None
    year: int
    month: int
    document_type: str | None = None
    title: str | None = Field(
        default=None,
        description="Present only when a real source title exists; never invented.",
    )


class ArchiveDecisionsResponse(BaseModel):
    court_id: str
    year: int
    month: int
    items: list[ArchiveDecisionOut]
    next_cursor: str | None = None
    has_more: bool
    limit: int


@lru_cache(maxsize=1)
def _cached_store(path: str) -> JudgmentArchiveStore:
    return JudgmentArchiveStore(path)


def get_judgment_archive_store() -> JudgmentArchiveStore:
    return _cached_store(str(resolve_archive_sqlite_path()))


def clear_judgment_archive_store_cache() -> None:
    _cached_store.cache_clear()


@router.get("/archive", response_model=ArchiveOverviewResponse)
async def get_judgment_archive_overview(
    court: str | None = Query(
        default=None,
        description=(
            "Optional court filter. Supported ids include constitutional_court; "
            "supreme_court and supreme_administrative_court are schema-ready."
        ),
    ),
    store: JudgmentArchiveStore = Depends(get_judgment_archive_store),
) -> ArchiveOverviewResponse:
    try:
        courts = await asyncio.to_thread(
            store.list_courts_with_years,
            court_id=court,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ArchiveOverviewResponse(
        courts=[_court_out(item) for item in courts],
        index_ready=await asyncio.to_thread(store.is_ready),
        index_path=str(store.sqlite_path),
    )


@router.get("/archive/{year}", response_model=ArchiveMonthsResponse)
async def get_judgment_archive_year(
    year: int,
    court: str = Query(default="constitutional_court"),
    store: JudgmentArchiveStore = Depends(get_judgment_archive_store),
) -> ArchiveMonthsResponse:
    try:
        months = await asyncio.to_thread(
            store.list_months,
            year=year,
            court_id=court,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ArchiveMonthsResponse(
        court_id=normalize_court_id(court) or "constitutional_court",
        year=year,
        months=[_month_out(item) for item in months],
    )


@router.get(
    "/archive/{year}/{month}",
    response_model=ArchiveDecisionsResponse,
)
async def get_judgment_archive_month(
    year: int,
    month: int,
    court: str = Query(default="constitutional_court"),
    cursor: str | None = Query(default=None),
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=100),
    store: JudgmentArchiveStore = Depends(get_judgment_archive_store),
) -> ArchiveDecisionsResponse:
    try:
        page = await asyncio.to_thread(
            store.list_decisions,
            year=year,
            month=month,
            court_id=court,
            cursor=cursor,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _decisions_response(
        court_id=court,
        year=year,
        month=month,
        page=page,
    )


def _court_out(item: ArchiveCourtSummary) -> ArchiveCourtOut:
    return ArchiveCourtOut(
        court_id=item.court_id,
        court_name=item.court_name,
        document_count=item.document_count,
        years=[ArchiveYearOut(year=y.year, count=y.count) for y in item.years],
        ingest_ready=item.ingest_ready,
    )


def _month_out(item: ArchiveMonthBucket) -> ArchiveMonthOut:
    return ArchiveMonthOut(month=item.month, count=item.count)


def _decision_out(item: ArchiveDecision) -> ArchiveDecisionOut:
    return ArchiveDecisionOut(
        canonical_document_id=item.canonical_document_id,
        ecli=item.ecli,
        case_number=item.case_number,
        court=item.court,
        decision_date=item.decision_date,
        year=item.year,
        month=item.month,
        document_type=item.document_type,
        title=item.title,
    )


def _decisions_response(
    *,
    court_id: str,
    year: int,
    month: int,
    page: ArchiveDecisionsPage,
) -> ArchiveDecisionsResponse:
    return ArchiveDecisionsResponse(
        court_id=normalize_court_id(court_id) or "constitutional_court",
        year=year,
        month=month,
        items=[_decision_out(item) for item in page.items],
        next_cursor=page.next_cursor,
        has_more=page.has_more,
        limit=page.limit,
    )
