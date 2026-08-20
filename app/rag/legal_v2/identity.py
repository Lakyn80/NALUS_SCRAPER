"""Canonical judicial-decision identity for Legal v2.

Invariant for indexed judicial decisions:

    document_id == canonical_document_id == ecli

``source_document_id`` (including historical ``doc-*`` review IDs) is secondary
traceability metadata only and must never be treated as the production
canonical document identity.
"""

from __future__ import annotations

import re
from typing import Any

IDENTITY_STATUS_VERIFIED = "verified"
IDENTITY_STATUS_BLOCKED_MISSING_ECLI = "blocked_missing_verified_ecli"
ALLOWED_IDENTITY_STATUSES = frozenset(
    {
        IDENTITY_STATUS_VERIFIED,
        IDENTITY_STATUS_BLOCKED_MISSING_ECLI,
    }
)

# Czech ECLI examples:
#   ECLI:CZ:US:2024:3.US.3203.24.1
#   ECLI:CZ:VSPH:2026:2.Cmo.28.2026.1
#   ECLI:CZ:VSOL:2025:6.To.41.2024.1
_ECLI_RE = re.compile(
    r"^ECLI:CZ:[A-Z]{2,8}:\d{4}:[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)+$",
    re.IGNORECASE,
)


class DecisionIdentityError(ValueError):
    """Raised when a judicial decision identity invariant fails."""


def normalize_ecli(value: str | None) -> str:
    """Normalize ECLI while preserving the verified ordinal-segment casing.

    Structural parts (``ECLI``, country, court code) are uppercased. The final
    ordinal segment keeps source casing so Justice Open Data / NALUS spellings
    match indexed payloads.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    if not text.upper().startswith("ECLI:"):
        return text
    parts = text.split(":")
    if len(parts) < 5:
        return "ECLI:" + text[5:]
    country = parts[1].upper()
    court = parts[2].upper()
    year = parts[3]
    ordinal = ":".join(parts[4:])
    return f"ECLI:{country}:{court}:{year}:{ordinal}"


def ecli_key(value: str | None) -> str:
    """Case-folded key for ECLI equality / set membership."""
    return normalize_ecli(value).casefold()


def eclis_equal(left: str | None, right: str | None) -> bool:
    left_key = ecli_key(left)
    right_key = ecli_key(right)
    return bool(left_key) and left_key == right_key


_TRAILING_ORDINAL_SEGMENT_RE = re.compile(r"^(?P<stem>.+)\.(?P<ord>\d+)$", re.IGNORECASE)


def ecli_ordinal_stem_and_suffix(value: str | None) -> tuple[str, str | None]:
    """Split a normalized ECLI into stem + optional trailing numeric ordinal segment.

    Example:
      ``ECLI:CZ:US:1999:4.US.23.99.1`` → (``ECLI:CZ:US:1999:4.US.23.99``, ``1``)
      ``ECLI:CZ:US:1999:4.US.23.99``   → (``ECLI:CZ:US:1999:4.US.23``, ``99``)

    Note: bare year-ending forms still have a numeric tail; use
    ``eclis_are_representation_variants`` for bare↔``.1`` identity candidates.
    """
    text = normalize_ecli(value)
    if not text:
        return "", None
    # Only inspect the ordinal payload after year (preserve ECLI:CZ:COURT:YEAR:…).
    parts = text.split(":")
    if len(parts) < 5:
        return text, None
    prefix = ":".join(parts[:4])
    ordinal = ":".join(parts[4:])
    match = _TRAILING_ORDINAL_SEGMENT_RE.match(ordinal)
    if not match:
        return text, None
    stem = f"{prefix}:{match.group('stem')}"
    return stem, match.group("ord")


def eclis_are_representation_variants(left: str | None, right: str | None) -> bool:
    """True when one ID is the other plus a single trailing ``.<digits>`` segment.

    Catches bare↔``.1`` NALUS pairs such as ``…4.US.23.99`` vs ``…4.US.23.99.1``.

    Does **not** treat ``…19.1`` and ``…19.2`` as the same decision (neither is a
    prefix of the other). Callers must still require content/metadata evidence
    before merging judgments; this helper is only an identity-candidate signal
    for pool deduplication / alias layers.
    """
    left_n = normalize_ecli(left)
    right_n = normalize_ecli(right)
    if not left_n or not right_n or eclis_equal(left_n, right_n):
        return False
    shorter, longer = sorted((left_n, right_n), key=len)
    if not longer.startswith(f"{shorter}."):
        return False
    suffix = longer[len(shorter) + 1 :]
    return bool(suffix) and suffix.isdigit()


def prefer_canonical_ecli_form(left: str | None, right: str | None) -> str:
    """Prefer the longer / suffixed form when two IDs are representation variants."""
    left_n = normalize_ecli(left)
    right_n = normalize_ecli(right)
    if not left_n:
        return right_n
    if not right_n:
        return left_n
    if not eclis_are_representation_variants(left_n, right_n):
        # Stable preference for non-variants: keep left.
        return left_n
    return left_n if len(left_n) >= len(right_n) else right_n


def collapse_document_ids_for_pool(document_ids: list[str]) -> list[str]:
    """Deduplicate pool IDs that are bare↔suffixed representation variants.

    Preserves first-seen order of the surviving canonical form (preferring the
    longer / ``.N`` form when a variant pair is found). Does not collapse
    distinct ordinals such as ``.1`` vs ``.2``.
    """
    survivors: list[str] = []
    for raw in document_ids:
        doc = normalize_ecli(raw) if is_valid_ecli(str(raw)) else str(raw or "").strip()
        if not doc:
            continue
        merged = False
        for idx, existing in enumerate(survivors):
            if eclis_equal(existing, doc):
                merged = True
                break
            if eclis_are_representation_variants(existing, doc):
                survivors[idx] = prefer_canonical_ecli_form(existing, doc)
                merged = True
                break
        if not merged:
            survivors.append(doc)
    return survivors


def is_valid_ecli(value: str | None) -> bool:
    text = normalize_ecli(value)
    return bool(text) and _ECLI_RE.match(text) is not None


def validate_decision_identity(
    *,
    ecli: str | None,
    canonical_document_id: str | None,
    require_present: bool = True,
) -> str:
    """Validate and return the normalized ECLI.

    When ``require_present`` is True (production judicial decisions), both values
    must be present, valid, and equal.
    """
    normalized_ecli = normalize_ecli(ecli)
    normalized_canonical = normalize_ecli(canonical_document_id)
    if not require_present and not normalized_ecli and not normalized_canonical:
        return ""
    if not normalized_ecli:
        raise DecisionIdentityError("ecli is required for a judicial decision")
    if not is_valid_ecli(normalized_ecli):
        raise DecisionIdentityError(f"malformed ecli: {ecli!r}")
    if not normalized_canonical:
        raise DecisionIdentityError("canonical_document_id is required for a judicial decision")
    if not eclis_equal(normalized_canonical, normalized_ecli):
        raise DecisionIdentityError(
            "canonical_document_id must equal ecli "
            f"({canonical_document_id!r} != {ecli!r})"
        )
    return normalized_ecli


def resolve_production_document_id(payload: dict[str, Any] | None) -> str:
    """Prefer ECLI / canonical identity from a retrieval or index payload."""
    metadata = payload or {}
    for key in ("ecli", "canonical_document_id", "document_id"):
        value = str(metadata.get(key) or "").strip()
        if value and is_valid_ecli(value):
            return normalize_ecli(value)
    for key in ("ecli", "canonical_document_id", "document_id", "source_document_id", "case_reference"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return ""


def production_identity_fields(
    *,
    ecli: str,
    source_document_id: str | None = None,
) -> dict[str, str | None]:
    """Build the production identity payload fragment for a verified decision."""
    normalized = validate_decision_identity(ecli=ecli, canonical_document_id=ecli)
    return {
        "document_id": normalized,
        "canonical_document_id": normalized,
        "ecli": normalized,
        "source_document_id": (str(source_document_id).strip() or None) if source_document_id else None,
    }
