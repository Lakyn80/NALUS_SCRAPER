"""Bounded structural segmentation for long legal inputs."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.legal_v2.query_input.config import LongInputConfig

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+")
_HEADING_RE = re.compile(
    r"(?im)^(?:I{1,3}|IV|V|VI{0,3}|IX|X|\d+)\.?\s+[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ].{3,80}$"
)


@dataclass(frozen=True)
class TextSegment:
    index: int
    text: str


def split_sentences(text: str, *, max_sentences: int) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text or "") if p.strip()]
    return parts[: max(0, max_sentences)]


def segment_text(text: str, config: LongInputConfig) -> list[TextSegment]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    # Prefer paragraph / heading boundaries.
    blocks: list[str] = []
    current: list[str] = []
    for line in cleaned.split("\n"):
        stripped = line.strip()
        if not stripped:
            if current:
                blocks.append(" ".join(current).strip())
                current = []
            continue
        if _HEADING_RE.match(stripped) and current:
            blocks.append(" ".join(current).strip())
            current = [stripped]
            continue
        current.append(stripped)
    if current:
        blocks.append(" ".join(current).strip())

    if not blocks:
        blocks = [cleaned]

    # Merge tiny blocks, then window oversized ones.
    merged: list[str] = []
    buf = ""
    for block in blocks:
        if not buf:
            buf = block
        elif len(buf) + 1 + len(block) <= config.segment_window_chars:
            buf = f"{buf} {block}"
        else:
            merged.append(buf)
            buf = block
    if buf:
        merged.append(buf)

    windows: list[str] = []
    for block in merged:
        if len(block) <= config.segment_window_chars:
            windows.append(block)
            continue
        start = 0
        while start < len(block) and len(windows) < config.max_segments:
            end = min(len(block), start + config.segment_window_chars)
            # Prefer sentence boundary near the window end.
            chunk = block[start:end]
            if end < len(block):
                cut = max(chunk.rfind(". "), chunk.rfind("? "), chunk.rfind("! "))
                if cut >= config.segment_window_chars // 3:
                    end = start + cut + 1
                    chunk = block[start:end]
            windows.append(chunk.strip())
            start = end

    limited = windows[: config.max_segments]
    return [TextSegment(index=i, text=t) for i, t in enumerate(limited) if t]
