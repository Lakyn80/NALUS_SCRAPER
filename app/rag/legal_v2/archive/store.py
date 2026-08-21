"""SQLite-backed read-optimized document metadata index for the archive.

This deliberately stores one row per unique judicial decision. It does not
store judgment full text and is never derived by scanning Qdrant chunks on
request.
"""

from __future__ import annotations

import base64
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from app.rag.legal_v2.archive.courts import (
    get_archive_court,
    list_archive_courts,
    normalize_court_id,
)
from app.rag.legal_v2.archive.models import (
    ArchiveCourtSummary,
    ArchiveDecision,
    ArchiveDecisionsPage,
    ArchiveMonthBucket,
    ArchiveYearBucket,
)

SCHEMA_VERSION = 1
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 100

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS archive_meta (
    schema_version INTEGER NOT NULL,
    built_at TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    document_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    canonical_document_id TEXT PRIMARY KEY,
    ecli TEXT NOT NULL,
    case_number TEXT,
    court TEXT NOT NULL,
    decision_date TEXT,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    document_type TEXT,
    title TEXT
);

CREATE INDEX IF NOT EXISTS idx_documents_court_year
    ON documents (court, year DESC);

CREATE INDEX IF NOT EXISTS idx_documents_court_year_month
    ON documents (court, year, month DESC);

CREATE INDEX IF NOT EXISTS idx_documents_list
    ON documents (court, year, month, decision_date DESC, canonical_document_id DESC);
"""


def default_archive_sqlite_path() -> Path:
    return Path("storage") / "rag" / "archive" / "judgment_archive_v1.sqlite"


def resolve_archive_sqlite_path(path: Path | str | None = None) -> Path:
    if path is not None and str(path).strip():
        return Path(path)
    env = os.getenv("NALUS_JUDGMENT_ARCHIVE_SQLITE_PATH", "").strip()
    if env:
        return Path(env)
    return default_archive_sqlite_path()


class JudgmentArchiveStore:
    """Read API over the document-level archive SQLite index."""

    def __init__(self, sqlite_path: Path | str | None = None) -> None:
        self.sqlite_path = resolve_archive_sqlite_path(sqlite_path)

    def is_ready(self) -> bool:
        path = self.sqlite_path
        if not path.exists():
            return False
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT document_count FROM archive_meta LIMIT 1"
                ).fetchone()
                return row is not None and int(row[0]) >= 0
        except sqlite3.Error:
            return False

    def list_courts_with_years(
        self,
        *,
        court_id: str | None = None,
    ) -> list[ArchiveCourtSummary]:
        wanted = normalize_court_id(court_id) if court_id else None
        if court_id and not wanted:
            raise ValueError(f"unsupported court_id: {court_id!r}")

        counts_by_court: dict[str, list[ArchiveYearBucket]] = {}
        totals: dict[str, int] = {}
        if self.sqlite_path.exists():
            with self._connect() as connection:
                params: list[Any] = []
                sql = (
                    "SELECT court, year, COUNT(*) AS cnt "
                    "FROM documents"
                )
                if wanted:
                    sql += " WHERE court = ?"
                    params.append(wanted)
                sql += " GROUP BY court, year ORDER BY court ASC, year DESC"
                for court, year, count in connection.execute(sql, params):
                    court_key = str(court)
                    totals[court_key] = totals.get(court_key, 0) + int(count)
                    counts_by_court.setdefault(court_key, []).append(
                        ArchiveYearBucket(year=int(year), count=int(count))
                    )

        summaries: list[ArchiveCourtSummary] = []
        for court in list_archive_courts():
            if wanted and court.court_id != wanted:
                continue
            summaries.append(
                ArchiveCourtSummary(
                    court_id=court.court_id,
                    court_name=court.display_name_cs,
                    document_count=int(totals.get(court.court_id, 0)),
                    years=list(counts_by_court.get(court.court_id, [])),
                    ingest_ready=court.ingest_ready,
                )
            )
        return summaries

    def list_months(
        self,
        *,
        year: int,
        court_id: str = "constitutional_court",
    ) -> list[ArchiveMonthBucket]:
        court = _require_court(court_id)
        _require_year(year)
        if not self.sqlite_path.exists():
            return []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT month, COUNT(*) AS cnt FROM documents "
                "WHERE court = ? AND year = ? "
                "GROUP BY month ORDER BY month DESC",
                (court, year),
            ).fetchall()
        return [
            ArchiveMonthBucket(month=int(month), count=int(count))
            for month, count in rows
        ]

    def list_decisions(
        self,
        *,
        year: int,
        month: int,
        court_id: str = "constitutional_court",
        cursor: str | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
    ) -> ArchiveDecisionsPage:
        court = _require_court(court_id)
        _require_year(year)
        _require_month(month)
        page_size = _clamp_limit(limit)
        cursor_date, cursor_id = decode_archive_cursor(cursor)

        if not self.sqlite_path.exists():
            return ArchiveDecisionsPage(
                items=[],
                next_cursor=None,
                has_more=False,
                limit=page_size,
            )

        params: list[Any] = [court, year, month]
        sql = (
            "SELECT canonical_document_id, ecli, case_number, court, "
            "decision_date, year, month, document_type, title "
            "FROM documents "
            "WHERE court = ? AND year = ? AND month = ?"
        )
        if cursor_id is not None:
            # Keyset pagination on (decision_date DESC, canonical_document_id DESC).
            # NULL decision_date sorts last under DESC in SQLite.
            sql += (
                " AND ("
                " ifnull(decision_date, '') < ifnull(?, '')"
                " OR ("
                "   ifnull(decision_date, '') = ifnull(?, '')"
                "   AND canonical_document_id < ?"
                " )"
                ")"
            )
            params.extend([cursor_date, cursor_date, cursor_id])
        sql += (
            " ORDER BY decision_date DESC, canonical_document_id DESC "
            "LIMIT ?"
        )
        params.append(page_size + 1)

        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()

        has_more = len(rows) > page_size
        page_rows = rows[:page_size]
        items = [_row_to_decision(row) for row in page_rows]
        next_cursor = None
        if has_more and items:
            last = items[-1]
            next_cursor = encode_archive_cursor(
                last.decision_date,
                last.canonical_document_id,
            )
        return ArchiveDecisionsPage(
            items=items,
            next_cursor=next_cursor,
            has_more=has_more,
            limit=page_size,
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.sqlite_path),
            timeout=30,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection


def write_archive_index(
    *,
    sqlite_path: Path | str,
    documents: Iterable[ArchiveDecision | dict[str, Any]],
    source_kind: str,
    built_at: str,
) -> int:
    """Replace the archive SQLite with the provided document metadata rows."""
    path = Path(sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    connection = sqlite3.connect(str(path))
    try:
        connection.executescript(_CREATE_SQL)
        rows: list[tuple[Any, ...]] = []
        seen: set[str] = set()
        for raw in documents:
            row = _document_to_row(raw)
            if row is None:
                continue
            doc_id = str(row[0])
            if doc_id.casefold() in seen:
                continue
            seen.add(doc_id.casefold())
            rows.append(row)

        connection.executemany(
            "INSERT INTO documents ("
            "canonical_document_id, ecli, case_number, court, decision_date, "
            "year, month, document_type, title"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        connection.execute(
            "INSERT INTO archive_meta "
            "(schema_version, built_at, source_kind, document_count) "
            "VALUES (?, ?, ?, ?)",
            (SCHEMA_VERSION, built_at, source_kind, len(rows)),
        )
        connection.commit()
        return len(rows)
    finally:
        connection.close()


def encode_archive_cursor(
    decision_date: str | None,
    canonical_document_id: str,
) -> str:
    payload = {
        "d": decision_date,
        "i": canonical_document_id,
    }
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_archive_cursor(
    cursor: str | None,
) -> tuple[str | None, str | None]:
    text = str(cursor or "").strip()
    if not text:
        return None, None
    padding = "=" * (-len(text) % 4)
    try:
        raw = base64.urlsafe_b64decode(text + padding)
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("invalid archive cursor") from exc
    if not isinstance(payload, dict):
        raise ValueError("invalid archive cursor")
    doc_id = str(payload.get("i") or "").strip()
    if not doc_id:
        raise ValueError("invalid archive cursor")
    decision_date = payload.get("d")
    if decision_date is None:
        return None, doc_id
    return str(decision_date), doc_id


def _document_to_row(
    raw: ArchiveDecision | dict[str, Any],
) -> tuple[Any, ...] | None:
    if isinstance(raw, ArchiveDecision):
        document = raw
    else:
        # Already-normalized dicts from the builder / tests.
        doc_id = str(
            raw.get("canonical_document_id") or raw.get("ecli") or ""
        ).strip()
        ecli = str(raw.get("ecli") or doc_id).strip()
        if not doc_id or not ecli:
            return None
        try:
            year = int(raw["year"])
            month = int(raw["month"])
        except (KeyError, TypeError, ValueError):
            return None
        document = ArchiveDecision(
            canonical_document_id=doc_id,
            ecli=ecli,
            case_number=_optional_str(raw.get("case_number")),
            court=str(raw.get("court") or "").strip(),
            decision_date=_optional_str(raw.get("decision_date")),
            year=year,
            month=month,
            document_type=_optional_str(raw.get("document_type")),
            title=_optional_str(raw.get("title")),
        )
    if not document.court:
        return None
    return (
        document.canonical_document_id,
        document.ecli,
        document.case_number,
        document.court,
        document.decision_date,
        document.year,
        document.month,
        document.document_type,
        document.title,
    )


def _row_to_decision(row: sqlite3.Row | tuple[Any, ...]) -> ArchiveDecision:
    if isinstance(row, sqlite3.Row):
        values = dict(row)
        return ArchiveDecision(
            canonical_document_id=str(values["canonical_document_id"]),
            ecli=str(values["ecli"]),
            case_number=_optional_str(values.get("case_number")),
            court=str(values["court"]),
            decision_date=_optional_str(values.get("decision_date")),
            year=int(values["year"]),
            month=int(values["month"]),
            document_type=_optional_str(values.get("document_type")),
            title=_optional_str(values.get("title")),
        )
    return ArchiveDecision(
        canonical_document_id=str(row[0]),
        ecli=str(row[1]),
        case_number=_optional_str(row[2]),
        court=str(row[3]),
        decision_date=_optional_str(row[4]),
        year=int(row[5]),
        month=int(row[6]),
        document_type=_optional_str(row[7]),
        title=_optional_str(row[8]),
    )


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _require_court(court_id: str) -> str:
    normalized = normalize_court_id(court_id)
    if not normalized or get_archive_court(normalized) is None:
        raise ValueError(f"unsupported court_id: {court_id!r}")
    return normalized


def _require_year(year: int) -> None:
    if not isinstance(year, int) or year < 1900 or year > 2100:
        raise ValueError(f"invalid year: {year!r}")


def _require_month(month: int) -> None:
    if not isinstance(month, int) or month < 1 or month > 12:
        raise ValueError(f"invalid month: {month!r}")


def _clamp_limit(limit: int) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("limit must be an integer") from exc
    if value < 1:
        raise ValueError("limit must be >= 1")
    return min(value, MAX_PAGE_SIZE)
