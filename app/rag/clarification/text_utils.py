from __future__ import annotations

import re
import unicodedata


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return str(value).strip()


def simplify_text(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", normalize_text(text)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).strip()


def contains_any(haystack: str, needles: tuple[str, ...]) -> bool:
    simplified = f" {simplify_text(haystack)} "
    return any(f" {needle} " in simplified or needle in simplified for needle in needles)


def count_matches(haystack: str, needles: tuple[str, ...]) -> int:
    simplified = f" {simplify_text(haystack)} "
    return sum(1 for needle in needles if f" {needle} " in simplified or needle in simplified)
