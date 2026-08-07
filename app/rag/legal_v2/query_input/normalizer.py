"""Deterministic normalization for pasted legal text."""

from __future__ import annotations

import re
import unicodedata

_PAGE_MARKER_RE = re.compile(
    r"(?im)^(?:[-–—\s]*)(?:strana|str\.?|page|s\.)\s*\d+(?:\s*/\s*\d+)?(?:[-–—\s]*)$"
)
_REPEATED_HEADER_RE = re.compile(
    r"(?im)^(?:NALUS\b.*|Česká republika\s*$|Ústavní soud\s*$|Nejvyšší soud\s*$)$"
)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[^\S\n]{2,}")


def normalize_legal_input(text: str) -> str:
    """Normalize whitespace/artifacts without rewriting legal meaning."""
    value = unicodedata.normalize("NFC", text or "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\u00a0", " ").replace("\u200b", "")
    lines: list[str] = []
    for raw_line in value.split("\n"):
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        if _PAGE_MARKER_RE.match(line):
            continue
        if _REPEATED_HEADER_RE.match(line) and len(line) < 80:
            continue
        lines.append(line)
    value = "\n".join(lines)
    value = _MULTI_BLANK_RE.sub("\n\n", value)
    value = "\n".join(_MULTI_SPACE_RE.sub(" ", line).strip() for line in value.split("\n"))
    return value.strip()
