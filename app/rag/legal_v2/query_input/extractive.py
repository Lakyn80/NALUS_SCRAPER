"""Extractive SearchBrief builder (no LLM)."""

from __future__ import annotations

import time

from app.rag.legal_v2.query_input.config import LongInputConfig
from app.rag.legal_v2.query_input.errors import CondensationFailedError, NoUsefulContentError
from app.rag.legal_v2.query_input.merger import merge_scored_sentences
from app.rag.legal_v2.query_input.models import CondensationMethod, ScoredSentence, SearchBrief
from app.rag.legal_v2.query_input.normalizer import normalize_legal_input
from app.rag.legal_v2.query_input.scoring import score_sentence
from app.rag.legal_v2.query_input.segments import segment_text, split_sentences


def build_extractive_search_brief(
    raw_text: str,
    *,
    config: LongInputConfig,
) -> SearchBrief:
    started = time.perf_counter()
    original_length = len(raw_text or "")
    normalized = normalize_legal_input(raw_text)
    if not normalized:
        raise NoUsefulContentError("Normalized input is empty.")

    segments = segment_text(normalized, config)
    if not segments:
        raise NoUsefulContentError("No segments produced from input.")

    scored: list[ScoredSentence] = []
    sentence_budget = 0
    for segment in segments:
        sentences = split_sentences(
            segment.text, max_sentences=config.max_sentences_per_segment
        )
        for sentence_index, sentence in enumerate(sentences):
            if sentence_budget >= config.max_sentences:
                break
            score, flags = score_sentence(sentence)
            scored.append(
                ScoredSentence(
                    text=sentence,
                    score=score,
                    segment_index=segment.index,
                    sentence_index=sentence_index,
                    flags=flags,
                )
            )
            sentence_budget += 1
        if sentence_budget >= config.max_sentences:
            break

    if not scored:
        raise NoUsefulContentError("No sentences available for condensation.")

    try:
        brief = merge_scored_sentences(
            scored,
            original_length=original_length,
            normalized_text=normalized,
            config=config,
            method=CondensationMethod.EXTRACTIVE,
            condensation_latency_ms=(time.perf_counter() - started) * 1000.0,
            segments_examined=len(segments),
        )
    except Exception as exc:  # noqa: BLE001
        raise CondensationFailedError(str(exc)) from exc

    if not brief.brief_text.strip():
        raise NoUsefulContentError("Extractive condensation produced an empty brief.")
    return brief
