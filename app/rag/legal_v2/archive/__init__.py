"""Read-only jurisprudence archive (document-level, not chunk-level)."""

from app.rag.legal_v2.archive.courts import (
    ARCHIVE_COURT_IDS,
    COURT_CONSTITUTIONAL,
    COURT_SUPREME,
    COURT_SUPREME_ADMINISTRATIVE,
    ArchiveCourt,
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
from app.rag.legal_v2.archive.store import (
    JudgmentArchiveStore,
    default_archive_sqlite_path,
    resolve_archive_sqlite_path,
)

__all__ = [
    "ARCHIVE_COURT_IDS",
    "COURT_CONSTITUTIONAL",
    "COURT_SUPREME",
    "COURT_SUPREME_ADMINISTRATIVE",
    "ArchiveCourt",
    "ArchiveCourtSummary",
    "ArchiveDecision",
    "ArchiveDecisionsPage",
    "ArchiveMonthBucket",
    "ArchiveYearBucket",
    "JudgmentArchiveStore",
    "default_archive_sqlite_path",
    "get_archive_court",
    "list_archive_courts",
    "normalize_court_id",
    "resolve_archive_sqlite_path",
]
