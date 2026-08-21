"""Typed archive response models (document-level metadata only)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchiveYearBucket:
    year: int
    count: int


@dataclass(frozen=True)
class ArchiveMonthBucket:
    month: int
    count: int


@dataclass(frozen=True)
class ArchiveCourtSummary:
    court_id: str
    court_name: str
    document_count: int
    years: list[ArchiveYearBucket]
    ingest_ready: bool


@dataclass(frozen=True)
class ArchiveDecision:
    """One unique judicial decision (never a chunk).

    ``title`` is set only when a real source title exists; never invented.
    """

    canonical_document_id: str
    ecli: str
    case_number: str | None
    court: str
    decision_date: str | None
    year: int
    month: int
    document_type: str | None
    title: str | None


@dataclass(frozen=True)
class ArchiveDecisionsPage:
    items: list[ArchiveDecision]
    next_cursor: str | None
    has_more: bool
    limit: int
