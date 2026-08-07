"""Conservative identifier suppression for SearchBrief text."""

from __future__ import annotations

import re

# Keep statute citations like "§ 75 odst. 1" untouched.
ECLI_RE = re.compile(r"\bECLI:CZ:[A-Z]{2}:\d{4}:[^\s,;]+", re.IGNORECASE)
CASE_NUMBER_RE = re.compile(
    r"(?i)\b(?:sp\.?\s*zn\.?|spisov[áa]\s+zna[cč]ka)\s*[:.]?\s*"
    r"(?:[IVX]+\.?\s*)?(?:ÚS|US|Pl\.?\s*ÚS)\s*\d+/\d+(?:\s*#\d+)?"
)
CASE_NUMBER_BARE_RE = re.compile(
    r"(?i)\b(?:[IVX]+\.|Pl\.)\s*ÚS\s+\d+/\d+(?:\s*#\d+)?\b"
)
NS_CASE_RE = re.compile(r"(?i)\b\d+\s+(?:Tdo|Cdo|Nd|Ncu|ICdo|Tcu|Tvo|Td|Pzo)\s+\d+/\d+\b")
CJ_RE = re.compile(r"(?i)\bč\.\s*j\.\s*[0-9A-Za-zÁ-ž./\- ]{3,80}")
DOC_ID_RE = re.compile(r"\bdoc-[0-9a-f]{8,}\b", re.IGNORECASE)
BENCHMARK_ID_RE = re.compile(r"\bnalus-cs-pilot-\d+\b", re.IGNORECASE)
QDRANT_POINT_RE = re.compile(r"\b(?:point|chunk)[_-]?id[=:]\s*[^\s,]+\b", re.IGNORECASE)

_SUPPRESSORS: tuple[re.Pattern[str], ...] = (
    ECLI_RE,
    CASE_NUMBER_RE,
    CASE_NUMBER_BARE_RE,
    NS_CASE_RE,
    CJ_RE,
    DOC_ID_RE,
    BENCHMARK_ID_RE,
    QDRANT_POINT_RE,
)


def count_identifiers(text: str) -> int:
    return sum(len(pattern.findall(text or "")) for pattern in _SUPPRESSORS)


def suppress_identifiers(text: str) -> tuple[str, int]:
    """Remove lookup identifiers from brief text. Returns (cleaned, suppressed_count)."""
    value = text or ""
    suppressed = 0
    for pattern in _SUPPRESSORS:
        matches = pattern.findall(value)
        if not matches:
            continue
        suppressed += len(matches)
        value = pattern.sub(" ", value)
    value = re.sub(r"\s{2,}", " ", value).strip(" ,;.-")
    return value, suppressed
