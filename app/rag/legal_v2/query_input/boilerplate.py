"""Conservative boilerplate downweighting for legal paste."""

from __future__ import annotations

import re

_BOILERPLATE_LINE_RE = re.compile(
    r"(?i)^(?:"
    r"Jménem republiky|"
    r"NALUS\b|"
    r"www\.|"
    r"tel\.|"
    r"e-?mail|"
    r"podpis|"
    r"razítko|"
    r"vyhotoveno|"
    r"za správnost|"
    r"strana\s+\d+"
    r").*$"
)
_SIGNATURE_RE = re.compile(r"(?i)\b(?:JUDr\.|Mgr\.|soudce zpravodaj|předseda senátu)\b")


def is_boilerplate_sentence(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    if len(cleaned) < 25:
        return True
    if _BOILERPLATE_LINE_RE.match(cleaned):
        return True
    if _SIGNATURE_RE.search(cleaned) and len(cleaned) < 120:
        return True
    if cleaned.count("Ústavní soud rozhodl") and "stížnost" not in cleaned.lower():
        return True
    return False
