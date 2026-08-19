"""Generic Legal v2 source filters applied to dense and BM25 channels.

These filters are request-level constraints (court / year / document type).
They must not encode query-specific keywords, case numbers, or expected ECLIs.
Empty filters are a no-op so production ranking stays unchanged.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable

from app.rag.retrieval.models import RetrievedChunk

_YEAR_RE = re.compile(r"^(?P<year>1[0-9]{3}|20[0-9]{2})")


@dataclass(frozen=True)
class RetrievalSourceFilters:
    courts: tuple[str, ...] = ()
    document_types: tuple[str, ...] = ()
    years: tuple[int, ...] = ()

    def is_active(self) -> bool:
        return bool(self.courts or self.document_types or self.years)

    def as_dict(self) -> dict[str, Any]:
        return {
            "courts": list(self.courts),
            "document_types": list(self.document_types),
            "years": list(self.years),
        }


def parse_retrieval_source_filters(
    *,
    courts: list[str] | tuple[str, ...] | None = None,
    document_types: list[str] | tuple[str, ...] | None = None,
    years: list[int] | tuple[int, ...] | None = None,
) -> RetrievalSourceFilters:
    cleaned_courts = tuple(item.strip() for item in (courts or ()) if str(item).strip())
    cleaned_types = tuple(
        item.strip() for item in (document_types or ()) if str(item).strip()
    )
    cleaned_years = tuple(int(year) for year in (years or ()))
    return RetrievalSourceFilters(
        courts=cleaned_courts,
        document_types=cleaned_types,
        years=cleaned_years,
    )


def chunk_matches_source_filters(
    metadata: dict[str, Any] | None,
    source_filters: RetrievalSourceFilters | None,
) -> bool:
    if source_filters is None or not source_filters.is_active():
        return True
    payload = metadata or {}
    if source_filters.courts and not _matches_any_court(payload, source_filters.courts):
        return False
    if source_filters.document_types and not _matches_any_document_type(
        payload, source_filters.document_types
    ):
        return False
    if source_filters.years and not _matches_any_year(payload, source_filters.years):
        return False
    return True


def filter_chunks(
    chunks: list[RetrievedChunk],
    source_filters: RetrievalSourceFilters | None,
) -> list[RetrievedChunk]:
    if source_filters is None or not source_filters.is_active():
        return list(chunks)
    return [
        chunk
        for chunk in chunks
        if chunk_matches_source_filters(chunk.metadata, source_filters)
    ]


def metadata_predicate_for(
    source_filters: RetrievalSourceFilters | None,
) -> Callable[[dict[str, Any]], bool] | None:
    if source_filters is None or not source_filters.is_active():
        return None

    def _predicate(metadata: dict[str, Any]) -> bool:
        return chunk_matches_source_filters(metadata, source_filters)

    return _predicate


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    without_marks = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return without_marks.casefold()


def _metadata_haystack(metadata: dict[str, Any]) -> str:
    parts = [
        metadata.get("court"),
        metadata.get("court_name"),
        metadata.get("source"),
        metadata.get("ecli"),
        metadata.get("document_id"),
        metadata.get("canonical_document_id"),
    ]
    return _fold(" ".join(str(part or "") for part in parts))


def _classify_court(metadata: dict[str, Any]) -> str | None:
    haystack = _metadata_haystack(metadata)
    if not haystack.strip():
        return None
    if (
        "ecli:cz:nss" in haystack
        or "spravni soud" in haystack
        or "nssoud" in haystack
        or "administrative" in haystack
    ):
        return "nssoud"
    if (
        "ustav" in haystack
        or "nalus" in haystack
        or "constitutional" in haystack
        or "usoud" in haystack
        or "ecli:cz:us" in haystack
    ):
        return "usoud"
    if (
        "ecli:cz:ns:" in haystack
        or "nejvyssi soud" in haystack
        or "nsoud" in haystack
        or "supreme" in haystack
    ):
        return "nsoud"
    return None


def _requested_court_key(court: str) -> str:
    folded = _fold(court).strip()
    aliases = {
        "ustavni soud": "usoud",
        "usoud": "usoud",
        "nejvyssi soud": "nsoud",
        "nsoud": "nsoud",
        "nejvyssi spravni soud": "nssoud",
        "nssoud": "nssoud",
    }
    return aliases.get(folded, folded)


def _matches_any_court(metadata: dict[str, Any], courts: tuple[str, ...]) -> bool:
    classified = _classify_court(metadata)
    if classified is None:
        haystack = _metadata_haystack(metadata)
        return any(_fold(court).strip() and _fold(court) in haystack for court in courts)
    return any(_requested_court_key(court) == classified for court in courts)


def _matches_any_document_type(
    metadata: dict[str, Any], document_types: tuple[str, ...]
) -> bool:
    actual = _fold(
        str(metadata.get("document_type") or metadata.get("decision_form") or "")
    )
    if not actual:
        return False
    return any(_fold(item) == actual for item in document_types if item)


def _matches_any_year(metadata: dict[str, Any], years: tuple[int, ...]) -> bool:
    raw = str(
        metadata.get("decision_date")
        or metadata.get("date")
        or metadata.get("decision_year")
        or ""
    ).strip()
    match = _YEAR_RE.match(raw)
    if match is None:
        return False
    return int(match.group("year")) in set(years)
