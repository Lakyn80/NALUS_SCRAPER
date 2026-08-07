"""Normalizer tests for Legal v2 long-input preprocessing."""

from __future__ import annotations

from app.rag.legal_v2.query_input.normalizer import normalize_legal_input


def test_repeated_whitespace_and_crlf() -> None:
    text = "Ahoj\r\n\r\n\r\nsvěte   test"
    out = normalize_legal_input(text)
    assert "\r" not in out
    assert "\n\n\n" not in out
    assert "  " not in out


def test_page_markers_and_headers_removed() -> None:
    text = "Strana 3\nÚstavní soud\nMeritum sporu o výživné.\nNALUS databáze"
    out = normalize_legal_input(text)
    assert "Strana 3" not in out
    assert "Meritum sporu o výživné" in out


def test_preserves_negation_statutes_dates_money() -> None:
    text = (
        "Nehledám meritorní spor o péči, ale odmítnutí ústavní stížnosti.\n"
        "Podle § 75 odst. 1 zákona o Ústavním soudu.\n"
        "Částka 12 500 Kč ze dne 15. 3. 2024."
    )
    out = normalize_legal_input(text)
    assert "Nehledám meritorní spor o péči" in out
    assert "§ 75 odst. 1" in out
    assert "12 500 Kč" in out
    assert "15. 3. 2024" in out
