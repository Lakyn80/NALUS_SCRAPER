"""QueryInputService orchestration tests."""

from __future__ import annotations

import pytest

from app.rag.legal_v2.query_input.config import LongInputConfig
from app.rag.legal_v2.query_input.errors import InputTooLargeError
from app.rag.legal_v2.query_input.models import CondensationMethod, InputClassification
from app.rag.legal_v2.query_input.service import QueryInputService


def test_feature_off_passthrough_and_8k_limit() -> None:
    service = QueryInputService(LongInputConfig(enabled=False))
    prepared = service.prepare("Hledám rozhodnutí o výživném.")
    assert prepared.was_condensed is False
    assert prepared.retrieval_query == "Hledám rozhodnutí o výživném."
    assert prepared.condensation_method == CondensationMethod.NONE
    with pytest.raises(InputTooLargeError):
        service.prepare("x" * 8001)


def test_feature_on_short_passthrough() -> None:
    service = QueryInputService(LongInputConfig(enabled=True))
    query = "Hledám rozhodnutí o úpravě styku rodiče s nezletilým dítětem."
    prepared = service.prepare(query)
    assert prepared.classification == InputClassification.SHORT_QUERY
    assert prepared.was_condensed is False
    assert prepared.retrieval_query == query


def test_feature_on_long_condenses() -> None:
    service = QueryInputService(LongInputConfig(enabled=True))
    raw = """
Ústavní stížností se stěžovatel domáhá zrušení napadených rozhodnutí.

Nehledám meritorní spor o péči, ale odmítnutí ústavní stížnosti pro formální vady.
Nebyl zastoupen advokátem a výzvu k odstranění vad nesplnil.
Odůvodnění napadeného usnesení je nedostatečné.
""".strip()
    raw = (raw + "\n\n") * 3
    prepared = service.prepare(raw)
    assert prepared.classification == InputClassification.LONG_LEGAL_INPUT
    assert prepared.was_condensed is True
    assert prepared.condensation_method == CondensationMethod.EXTRACTIVE
    assert prepared.retrieval_query == prepared.brief.brief_text
    assert len(prepared.retrieval_query) < len(raw)
    assert "ECLI:" not in prepared.retrieval_query


def test_oversized_rejected_when_enabled() -> None:
    service = QueryInputService(LongInputConfig(enabled=True, raw_hard_char_limit=1000))
    with pytest.raises(InputTooLargeError):
        service.prepare("a" * 1001)
