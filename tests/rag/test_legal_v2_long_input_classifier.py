"""Classifier tests for Legal v2 long-input preprocessing."""

from __future__ import annotations

import pytest

from app.rag.legal_v2.query_input.classifier import classify_input
from app.rag.legal_v2.query_input.config import LongInputConfig
from app.rag.legal_v2.query_input.models import InputClassification


@pytest.fixture
def config() -> LongInputConfig:
    return LongInputConfig(enabled=True)


def test_short_ordinary_query(config: LongInputConfig) -> None:
    result = classify_input(
        "Hledám rozhodnutí o úpravě styku rodiče s nezletilým dítětem.",
        config,
    )
    assert result.classification == InputClassification.SHORT_QUERY


def test_long_narrative_query(config: LongInputConfig) -> None:
    text = (
        "Stěžovatel popisuje dlouhou historii sporu. "
        * 20
        + "Ústavní stížností se domáhá zrušení rozhodnutí pro formální vady."
    )
    result = classify_input(text, config)
    assert result.classification == InputClassification.LONG_LEGAL_INPUT


def test_multi_paragraph_legal_document(config: LongInputConfig) -> None:
    text = "\n\n".join(
        [
            "Ústavní stížností se stěžovatel domáhá zrušení napadených rozhodnutí.",
            "Nebyl zastoupen advokátem a neodstranil vady podání.",
            "Nehledá meritorní spor o péči, ale odmítnutí stížnosti pro vady.",
            "Odůvodnění obecných soudů považuje za nedostatečné.",
        ]
    )
    result = classify_input(text, config)
    assert result.classification == InputClassification.LONG_LEGAL_INPUT


def test_near_and_exact_threshold(config: LongInputConfig) -> None:
    near = "x" * (config.char_threshold - 1)
    assert classify_input(near, config).classification == InputClassification.SHORT_QUERY
    exact = ("slovo " * 40) + ("\n\nodstavec " * 3)
    assert classify_input(exact, config).classification in {
        InputClassification.SHORT_QUERY,
        InputClassification.LONG_LEGAL_INPUT,
    }


def test_oversized_input(config: LongInputConfig) -> None:
    text = "a" * (config.raw_hard_char_limit + 1)
    assert classify_input(text, config).classification == InputClassification.OVERSIZED_INPUT


def test_whitespace_only(config: LongInputConfig) -> None:
    assert classify_input("   \n\t  ", config).classification == InputClassification.EMPTY


def test_unicode_czech_legal_text(config: LongInputConfig) -> None:
    text = (
        "Stěžovatelka se ústavní stížností domáhá zrušení usnesení.\n\n"
        "Tvrdí porušení práva na soudní ochranu a spravedlivý proces.\n\n"
        "Současně uvádí, že nebyla zastoupena advokátem."
    )
    assert classify_input(text, config).classification == InputClassification.LONG_LEGAL_INPUT


def test_many_short_lines(config: LongInputConfig) -> None:
    text = "\n".join([f"Bod {i}: ústavní stížnost a formální vady." for i in range(12)])
    assert classify_input(text, config).classification == InputClassification.LONG_LEGAL_INPUT


def test_one_huge_paragraph(config: LongInputConfig) -> None:
    text = ("Stěžovatel popisuje skutkový stav a právní hodnocení. " * 50).strip()
    assert classify_input(text, config).classification == InputClassification.LONG_LEGAL_INPUT
