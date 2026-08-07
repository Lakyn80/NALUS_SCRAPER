"""Typed configuration for long-input / SearchBrief preprocessing."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class LongInputConfig:
    """Centralized long-input policy (no scattered magic numbers)."""

    enabled: bool = False
    method: str = "extractive"  # extractive | precise (precise unavailable)
    policy_version: str = "extractive_v1"

    # Classification thresholds
    char_threshold: int = 700
    word_threshold: int = 80
    paragraph_threshold: int = 3
    newline_threshold: int = 4

    # Hard resource bounds
    raw_hard_char_limit: int = 100_000
    stage1_retrieval_char_limit: int = 8_000
    max_paragraphs: int = 400
    max_sentences: int = 800
    max_segments: int = 40
    max_sentences_per_segment: int = 80

    # Brief sizing
    target_brief_chars: int = 1_000
    min_brief_chars: int = 200
    max_brief_chars: int = 1_500
    max_brief_sentences: int = 8

    # Segmentation
    segment_window_chars: int = 2_500

    def validate(self) -> None:
        if self.char_threshold < 100:
            raise ValueError("char_threshold must be >= 100")
        if self.raw_hard_char_limit < self.stage1_retrieval_char_limit:
            raise ValueError("raw_hard_char_limit must be >= stage1_retrieval_char_limit")
        if self.target_brief_chars > self.max_brief_chars:
            raise ValueError("target_brief_chars must be <= max_brief_chars")
        if self.max_brief_chars > self.stage1_retrieval_char_limit:
            raise ValueError("max_brief_chars must be <= stage1_retrieval_char_limit")
        if self.method not in {"extractive", "precise"}:
            raise ValueError("method must be extractive or precise")


def long_input_config_from_env() -> LongInputConfig:
    config = LongInputConfig(
        enabled=_env_flag("NALUS_LEGAL_V2_LONG_INPUT_ENABLED", False),
        method=(os.getenv("NALUS_LEGAL_V2_LONG_INPUT_METHOD", "extractive").strip().lower() or "extractive"),
        policy_version=(
            os.getenv("NALUS_LEGAL_V2_CONDENSATION_POLICY_VERSION", "extractive_v1").strip()
            or "extractive_v1"
        ),
        char_threshold=_int_env("NALUS_LEGAL_V2_LONG_INPUT_CHAR_THRESHOLD", 700),
        word_threshold=_int_env("NALUS_LEGAL_V2_LONG_INPUT_WORD_THRESHOLD", 80),
        paragraph_threshold=_int_env("NALUS_LEGAL_V2_LONG_INPUT_PARAGRAPH_THRESHOLD", 3),
        newline_threshold=_int_env("NALUS_LEGAL_V2_LONG_INPUT_NEWLINE_THRESHOLD", 4),
        raw_hard_char_limit=_int_env("NALUS_LEGAL_V2_LONG_INPUT_HARD_LIMIT", 100_000),
        stage1_retrieval_char_limit=_int_env(
            "NALUS_LEGAL_V2_STAGE1_RETRIEVAL_CHAR_LIMIT", 8_000
        ),
        max_paragraphs=_int_env("NALUS_LEGAL_V2_LONG_INPUT_MAX_PARAGRAPHS", 400),
        max_sentences=_int_env("NALUS_LEGAL_V2_LONG_INPUT_MAX_SENTENCES", 800),
        max_segments=_int_env("NALUS_LEGAL_V2_LONG_INPUT_MAX_SEGMENTS", 40),
        max_sentences_per_segment=_int_env(
            "NALUS_LEGAL_V2_LONG_INPUT_MAX_SENTENCES_PER_SEGMENT", 80
        ),
        target_brief_chars=_int_env("NALUS_LEGAL_V2_SEARCH_BRIEF_TARGET_CHARS", 1_000),
        min_brief_chars=_int_env("NALUS_LEGAL_V2_SEARCH_BRIEF_MIN_CHARS", 200),
        max_brief_chars=_int_env("NALUS_LEGAL_V2_SEARCH_BRIEF_MAX_CHARS", 1_500),
        max_brief_sentences=_int_env("NALUS_LEGAL_V2_SEARCH_BRIEF_MAX_SENTENCES", 8),
        segment_window_chars=_int_env("NALUS_LEGAL_V2_LONG_INPUT_SEGMENT_CHARS", 2_500),
    )
    config.validate()
    return config
