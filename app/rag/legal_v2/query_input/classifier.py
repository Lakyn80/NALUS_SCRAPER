"""Deterministic input classification for short vs long legal paste."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.legal_v2.query_input.config import LongInputConfig
from app.rag.legal_v2.query_input.models import InputClassification

_WORD_RE = re.compile(r"\S+")
_LEGAL_MARKER_RE = re.compile(
    r"(?i)\b(?:"
    r"ústavní\s+stížnost|odůvodnění|výrok|usnesení|nález|"
    r"žalob[ay]|odvolání|dovolání|nezletil|vazb[ayě]|exekuc|"
    r"stěžovatel|soud rozhodl|podle\s+§"
    r")\b"
)
_SEARCH_LIKE_RE = re.compile(
    r"(?i)^\s*(?:hledám|hledam|potřebuji|chci|jak\s+|kde\s+|co\s+)\b"
)


@dataclass(frozen=True)
class ClassificationResult:
    classification: InputClassification
    char_count: int
    word_count: int
    paragraph_count: int
    newline_count: int
    legal_marker_hits: int
    reasons: tuple[str, ...]


def classify_input(text: str, config: LongInputConfig) -> ClassificationResult:
    cleaned = (text or "").strip()
    if not cleaned:
        return ClassificationResult(
            classification=InputClassification.EMPTY,
            char_count=0,
            word_count=0,
            paragraph_count=0,
            newline_count=0,
            legal_marker_hits=0,
            reasons=("empty",),
        )

    char_count = len(cleaned)
    words = _WORD_RE.findall(cleaned)
    word_count = len(words)
    paragraphs = [p for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    paragraph_count = len(paragraphs) if paragraphs else 1
    newline_count = cleaned.count("\n")
    legal_marker_hits = len(_LEGAL_MARKER_RE.findall(cleaned))
    reasons: list[str] = []

    if char_count > config.raw_hard_char_limit:
        return ClassificationResult(
            classification=InputClassification.OVERSIZED_INPUT,
            char_count=char_count,
            word_count=word_count,
            paragraph_count=paragraph_count,
            newline_count=newline_count,
            legal_marker_hits=legal_marker_hits,
            reasons=("exceeds_raw_hard_char_limit",),
        )

    long_votes = 0
    if char_count >= config.char_threshold:
        long_votes += 1
        reasons.append("char_threshold")
    if word_count >= config.word_threshold:
        long_votes += 1
        reasons.append("word_threshold")
    if paragraph_count >= config.paragraph_threshold:
        long_votes += 1
        reasons.append("paragraph_threshold")
    if newline_count >= config.newline_threshold:
        long_votes += 1
        reasons.append("newline_threshold")
    if legal_marker_hits >= 3 and char_count >= max(400, config.char_threshold // 2):
        long_votes += 1
        reasons.append("legal_markers")

    # Single-line search-like phrases stay short even near char threshold.
    if (
        long_votes <= 1
        and newline_count == 0
        and paragraph_count == 1
        and _SEARCH_LIKE_RE.search(cleaned)
        and char_count < config.char_threshold * 1.5
    ):
        return ClassificationResult(
            classification=InputClassification.SHORT_QUERY,
            char_count=char_count,
            word_count=word_count,
            paragraph_count=paragraph_count,
            newline_count=newline_count,
            legal_marker_hits=legal_marker_hits,
            reasons=("search_like_short",),
        )

    if long_votes >= 2 or char_count >= config.char_threshold * 2:
        return ClassificationResult(
            classification=InputClassification.LONG_LEGAL_INPUT,
            char_count=char_count,
            word_count=word_count,
            paragraph_count=paragraph_count,
            newline_count=newline_count,
            legal_marker_hits=legal_marker_hits,
            reasons=tuple(reasons) or ("long_default",),
        )

    return ClassificationResult(
        classification=InputClassification.SHORT_QUERY,
        char_count=char_count,
        word_count=word_count,
        paragraph_count=paragraph_count,
        newline_count=newline_count,
        legal_marker_hits=legal_marker_hits,
        reasons=("below_long_thresholds",),
    )
