"""Rich month-level completeness — avoid false 'complete' from raw site totals."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MonthCompleteness:
    site_total_results: int | None = None
    discovered_entries: int = 0
    unique_source_ids: int = 0
    fetched_ok: int = 0
    failed: int = 0
    duplicates: int = 0
    skipped_classified: int = 0
    status: str = "pending"
    failure_reasons: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def bump_failure(self, reason: str) -> None:
        key = (reason or "unknown").strip() or "unknown"
        self.failure_reasons[key] = self.failure_reasons.get(key, 0) + 1
        self.failed += 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def finalize_month_status(stats: MonthCompleteness) -> MonthCompleteness:
    """Month is ok only when every unique discovered id is accounted for.

    Accounted = fetched_ok + failed + skipped_classified (+ duplicates already
    collapsed into unique_source_ids).
    """
    unique = max(0, int(stats.unique_source_ids))
    accounted = (
        int(stats.fetched_ok)
        + int(stats.failed)
        + int(stats.skipped_classified)
    )

    if unique == 0 and (stats.discovered_entries or 0) == 0:
        # Empty month can be ok when site also reports zero / unknown empty search.
        if stats.site_total_results in (None, 0):
            stats.status = "ok"
            stats.notes.append("empty_month")
            return stats
        stats.status = "partial"
        stats.notes.append("site_reported_results_but_none_discovered")
        return stats

    if accounted < unique:
        stats.status = "partial"
        stats.notes.append(
            f"unaccounted_unique={unique - accounted}"
        )
        return stats

    if stats.failed > 0:
        stats.status = "partial"
        stats.notes.append("has_explicit_failures")
        return stats

    # Soft signal: site total much higher than unique discovered → incomplete crawl.
    if (
        stats.site_total_results is not None
        and stats.site_total_results > 0
        and unique > 0
        and stats.site_total_results > unique * 1.05 + 2
    ):
        stats.status = "partial"
        stats.notes.append(
            "site_total_exceeds_unique_discovered"
        )
        return stats

    stats.status = "ok"
    return stats
