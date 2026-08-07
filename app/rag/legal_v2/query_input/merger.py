"""Merge scored sentences into a bounded SearchBrief text."""

from __future__ import annotations

import hashlib
import re

from app.rag.legal_v2.query_input.config import LongInputConfig
from app.rag.legal_v2.query_input.identifiers import suppress_identifiers
from app.rag.legal_v2.query_input.models import CondensationMethod, ScoredSentence, SearchBrief
from app.rag.legal_v2.query_input.scoring import extract_negative_focus, extract_requested_focus


def _normalize_key(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _brief_signature(*, normalized_input: str, method: str, policy_version: str) -> str:
    payload = f"{normalized_input}\n{method}\n{policy_version}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def merge_scored_sentences(
    scored: list[ScoredSentence],
    *,
    original_length: int,
    normalized_text: str,
    config: LongInputConfig,
    method: CondensationMethod,
    condensation_latency_ms: float,
    segments_examined: int,
    warnings: list[str] | None = None,
) -> SearchBrief:
    # Prefer high scores, but keep deterministic order by (-score, segment, sentence).
    ordered = sorted(
        scored,
        key=lambda item: (-item.score, item.segment_index, item.sentence_index, item.text),
    )

    selected: list[ScoredSentence] = []
    seen: set[str] = set()
    total_chars = 0
    for item in ordered:
        if item.score <= 0:
            continue
        key = _normalize_key(item.text)
        if not key or key in seen:
            continue
        # Near-duplicate check: skip if key is substring of an already selected key.
        if any(key in existing or existing in key for existing in seen if len(key) > 40):
            continue
        cleaned, _ = suppress_identifiers(item.text)
        if len(cleaned) < 20:
            continue
        next_len = total_chars + len(cleaned) + (2 if selected else 0)
        if selected and next_len > config.max_brief_chars:
            continue
        selected.append(
            ScoredSentence(
                text=cleaned,
                score=item.score,
                segment_index=item.segment_index,
                sentence_index=item.sentence_index,
                flags=item.flags,
            )
        )
        seen.add(key)
        total_chars = next_len
        if len(selected) >= config.max_brief_sentences:
            break

    # Restore reading order for the brief.
    selected.sort(key=lambda item: (item.segment_index, item.sentence_index))
    brief_parts = [item.text for item in selected]
    brief_text = " ".join(brief_parts).strip()
    brief_text, suppressed = suppress_identifiers(brief_text)

    if len(brief_text) < config.min_brief_chars and ordered:
        # Fallback: take top sentences until min size without exceeding max.
        fallback: list[str] = []
        chars = 0
        for item in ordered:
            cleaned, _ = suppress_identifiers(item.text)
            if len(cleaned) < 20:
                continue
            if chars and chars + len(cleaned) + 1 > config.max_brief_chars:
                break
            fallback.append(cleaned)
            chars += len(cleaned) + 1
            if chars >= config.min_brief_chars and len(fallback) >= 3:
                break
        brief_text = " ".join(fallback).strip()
        brief_text, suppressed2 = suppress_identifiers(brief_text)
        suppressed += suppressed2

    joined_for_signals = " ".join(item.text for item in selected) or brief_text
    negative = tuple(dict.fromkeys(extract_negative_focus(joined_for_signals)))
    requested = tuple(dict.fromkeys(extract_requested_focus(joined_for_signals)))

    facts = tuple(
        item.text for item in selected if "fact" in item.flags
    )[:5]
    issues = tuple(
        item.text for item in selected if "issue" in item.flags or "defect" in item.flags
    )[:5]
    procedural = tuple(
        item.text for item in selected if "procedural" in item.flags
    )[:5]

    # Ensure negation phrases remain in the brief text when present.
    for neg in negative:
        if neg and neg.lower() not in brief_text.lower():
            prefix = f"Nehledám {neg}."
            candidate = f"{prefix} {brief_text}".strip()
            if len(candidate) <= config.max_brief_chars:
                brief_text = candidate

    for req in requested[:1]:
        if req and req.lower() not in brief_text.lower():
            suffix = f"Hledám {req}."
            candidate = f"{brief_text} {suffix}".strip()
            if len(candidate) <= config.max_brief_chars:
                brief_text = candidate

    segments_used = tuple(
        sorted({f"segment:{item.segment_index}" for item in selected})
    )

    return SearchBrief(
        original_length=original_length,
        normalized_length=len(normalized_text),
        brief_text=brief_text,
        method=method,
        was_condensed=True,
        policy_version=config.policy_version,
        brief_signature=_brief_signature(
            normalized_input=normalized_text,
            method=method.value,
            policy_version=config.policy_version,
        ),
        facts=facts,
        legal_issues=issues,
        procedural_signals=procedural,
        requested_focus=requested,
        negative_focus=negative,
        source_segments_used=segments_used,
        warnings=tuple(warnings or ()),
        suppressed_identifier_count=suppressed,
        segments_examined=segments_examined,
        segments_selected=len(segments_used),
        condensation_latency_ms=condensation_latency_ms,
    )
