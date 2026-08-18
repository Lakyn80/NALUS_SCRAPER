"""Canonical judgment identity vs content-hash change detection."""

from __future__ import annotations

import hashlib
import re
from enum import Enum
from typing import Any, Mapping

_ECLI_RE = re.compile(
    r"^ECLI:CZ:(?P<court>US|NS|NSS):(?P<year>\d{4}):(?P<rest>.+)$",
    re.IGNORECASE,
)
_WS_RE = re.compile(r"\s+")


class ChangeKind(str, Enum):
    NEW = "new"
    UPDATED = "updated"
    UNCHANGED = "unchanged"


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    text = value.replace("\xa0", " ").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_ecli(value: str | None) -> str | None:
    text = normalize_text(value).upper().replace(" ", "")
    if not text:
        return None
    if not text.startswith("ECLI:"):
        return None
    match = _ECLI_RE.match(text)
    if not match:
        # Accept other CZ court codes if present, still normalize casing.
        if text.startswith("ECLI:CZ:") and len(text) > len("ECLI:CZ:"):
            return text
        return None
    court = match.group("court").upper()
    year = match.group("year")
    rest = match.group("rest")
    return f"ECLI:CZ:{court}:{year}:{rest}"


def is_valid_ecli(value: str | None) -> bool:
    return normalize_ecli(value) is not None


def _stable_token(value: str | None) -> str:
    return _WS_RE.sub(" ", normalize_text(value)).upper()


def _deterministic_fallback(
    *,
    source: str,
    url: str | None,
    case_number: str | None,
    decision_date: str | None,
) -> str:
    """Stable non-content fallback. Must not include full_text."""
    parts = [
        _stable_token(source),
        _stable_token(case_number),
        _stable_token(decision_date),
        _stable_token(url),
    ]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]
    return f"fallback:{source}:{digest}"


def resolve_canonical_id(
    record: Mapping[str, Any],
    *,
    source: str | None = None,
) -> tuple[str, str]:
    """Return (canonical_id, identity_class).

    Priority:
      ECLI → official source document ID → spis + decision_date → deterministic fallback
    """
    src = normalize_text(source or record.get("source") or "unknown").lower()

    ecli = normalize_ecli(
        str(record.get("ecli") or record.get("ECLI") or "") or None
    )
    if ecli:
        return ecli, "ecli"

    for key in (
        "source_document_id",
        "official_document_id",
        "document_id",
        "result_id",
    ):
        value = normalize_text(str(record.get(key) or ""))
        if value:
            # Prefer ECLI-shaped document ids.
            maybe_ecli = normalize_ecli(value)
            if maybe_ecli:
                return maybe_ecli, "ecli"
            return f"{src}:{value}", "source_document_id"

    case_number = normalize_text(
        str(
            record.get("case_number")
            or record.get("spisova_znacka")
            or record.get("case_reference")
            or ""
        )
    )
    decision_date = normalize_text(
        str(record.get("decision_date") or record.get("publication_date") or "")
    )
    if case_number and decision_date:
        return f"{src}:{_stable_token(case_number)}:{decision_date}", "spis_date"

    if case_number:
        return f"{src}:{_stable_token(case_number)}", "spis_only"

    url = normalize_text(str(record.get("url") or record.get("source_url") or ""))
    return (
        _deterministic_fallback(
            source=src,
            url=url or None,
            case_number=case_number or None,
            decision_date=decision_date or None,
        ),
        "deterministic_fallback",
    )


def compute_content_hash(*, full_text: str, url: str = "") -> str:
    """Content fingerprint for change detection — not document identity."""
    payload = f"{normalize_text(url)}\n{normalize_text(full_text)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def classify_content_change(
    *,
    canonical_id: str,
    content_hash: str,
    known: Mapping[str, str],
) -> ChangeKind:
    """known maps canonical_id → last content_hash."""
    previous = known.get(canonical_id)
    if previous is None:
        return ChangeKind.NEW
    if previous == content_hash:
        return ChangeKind.UNCHANGED
    return ChangeKind.UPDATED


def enrich_record_identity(record: dict[str, Any], *, source: str | None = None) -> dict[str, Any]:
    """Mutate/return record with canonical_id, identity_class, content_hash."""
    out = dict(record)
    src = source or str(out.get("source") or "unknown")
    out["source"] = src
    canonical_id, identity_class = resolve_canonical_id(out, source=src)
    out["canonical_id"] = canonical_id
    out["identity_class"] = identity_class
    full_text = str(out.get("full_text") or "")
    url = str(out.get("url") or out.get("source_url") or "")
    out["content_hash"] = compute_content_hash(full_text=full_text, url=url)
    return out
