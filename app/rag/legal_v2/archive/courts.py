"""Future-ready court identifiers for the jurisprudence archive."""

from __future__ import annotations

from dataclasses import dataclass


COURT_CONSTITUTIONAL = "constitutional_court"
COURT_SUPREME = "supreme_court"
COURT_SUPREME_ADMINISTRATIVE = "supreme_administrative_court"

ARCHIVE_COURT_IDS: tuple[str, ...] = (
    COURT_CONSTITUTIONAL,
    COURT_SUPREME,
    COURT_SUPREME_ADMINISTRATIVE,
)


@dataclass(frozen=True)
class ArchiveCourt:
    court_id: str
    display_name_cs: str
    ecli_court_code: str | None
    ingest_ready: bool


_COURTS: dict[str, ArchiveCourt] = {
    COURT_CONSTITUTIONAL: ArchiveCourt(
        court_id=COURT_CONSTITUTIONAL,
        display_name_cs="Ústavní soud",
        ecli_court_code="US",
        ingest_ready=True,
    ),
    COURT_SUPREME: ArchiveCourt(
        court_id=COURT_SUPREME,
        display_name_cs="Nejvyšší soud",
        ecli_court_code="NS",
        ingest_ready=False,
    ),
    COURT_SUPREME_ADMINISTRATIVE: ArchiveCourt(
        court_id=COURT_SUPREME_ADMINISTRATIVE,
        display_name_cs="Nejvyšší správní soud",
        ecli_court_code="NSS",
        ingest_ready=False,
    ),
}


def list_archive_courts() -> list[ArchiveCourt]:
    return [_COURTS[court_id] for court_id in ARCHIVE_COURT_IDS]


def get_archive_court(court_id: str) -> ArchiveCourt | None:
    return _COURTS.get(normalize_court_id(court_id) or "")


def normalize_court_id(value: str | None) -> str | None:
    text = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    if not text:
        return None
    aliases = {
        "constitutional_court": COURT_CONSTITUTIONAL,
        "usoud": COURT_CONSTITUTIONAL,
        "us": COURT_CONSTITUTIONAL,
        "ustavni_soud": COURT_CONSTITUTIONAL,
        "supreme_court": COURT_SUPREME,
        "nsoud": COURT_SUPREME,
        "ns": COURT_SUPREME,
        "nejvyssi_soud": COURT_SUPREME,
        "supreme_administrative_court": COURT_SUPREME_ADMINISTRATIVE,
        "nssoud": COURT_SUPREME_ADMINISTRATIVE,
        "nss": COURT_SUPREME_ADMINISTRATIVE,
        "nejvyssi_spravni_soud": COURT_SUPREME_ADMINISTRATIVE,
    }
    return aliases.get(text)
