"""Extractive SearchBrief tests."""

from __future__ import annotations

import pytest

from app.rag.legal_v2.query_input.config import LongInputConfig
from app.rag.legal_v2.query_input.extractive import build_extractive_search_brief
from app.rag.legal_v2.query_input.identifiers import suppress_identifiers
from app.rag.legal_v2.query_input.models import CondensationMethod


@pytest.fixture
def config() -> LongInputConfig:
    return LongInputConfig(enabled=True)


def _procedural_fixture() -> str:
    return """
Ústavní stížností se stěžovatel domáhá zrušení rozhodnutí obecných soudů.

Ve věci šlo o zásah orgánu sociálně-právní ochrany dětí a odebrání dítěte.
Stěžovatel výslovně uvádí: Nehledám meritorní spor o péči, ale odmítnutí ústavní stížnosti pro formální vady.

Nebyl zastoupen advokátem a výzvu k odstranění vad nesplnil.
Napadené usnesení postrádá dostatečné odůvodnění.
Ústavní soud stížnost odmítá pro neodstranění vad.
""".strip()


def test_identifier_suppression_classes() -> None:
    samples = [
        "ECLI:CZ:US:2025:1.US.3575.25.1",
        "sp. zn. I. ÚS 3575/25",
        "č. j. 7 Afs 103/2020-27",
        "doc-1234567890abcdef",
        "nalus-cs-pilot-004",
    ]
    for sample in samples:
        cleaned, count = suppress_identifiers(f"Hledám podobné případy {sample} kvůli vadám.")
        assert count >= 1
        assert sample not in cleaned
    statute = "Podle § 75 odst. 1 zákona o Ústavním soudu"
    cleaned, count = suppress_identifiers(statute)
    assert "§ 75" in cleaned
    assert count == 0


def test_procedural_rejection_preserves_negation(config: LongInputConfig) -> None:
    brief = build_extractive_search_brief(_procedural_fixture(), config=config)
    text = brief.brief_text.lower()
    assert brief.method == CondensationMethod.EXTRACTIVE
    assert brief.was_condensed is True
    assert "advokát" in text or "zastoupen" in text
    assert "vad" in text or "odmít" in text
    assert "ústavní stížnost" in text or "stížnost" in text
    assert "nehledám" in text or brief.negative_focus
    # Custody merits must not dominate alone without procedural cue.
    assert not (text.count("péč") > 2 and "vad" not in text)


def test_limitation_dominates_damages_narrative(config: LongInputConfig) -> None:
    text = (
        "Žalobce popisuje rozsáhlou škodu na majetku, ušlý zisk a nemajetkovou újmu. "
        "Fakta o výši škody jsou podrobná a zabírají mnoho odstavců. "
        "Nejde mi o výši náhrady škody, ale o promlčení nároku. "
        "Hledám rozhodnutí o promlčecí lhůtě a prekluzi."
    )
    brief = build_extractive_search_brief(text * 3, config=config)
    low = brief.brief_text.lower()
    assert "promlč" in low or "prekluz" in low
    assert "nehled" in low or "nejde" in low or brief.negative_focus


def test_costs_vs_contract(config: LongInputConfig) -> None:
    text = (
        "Mezi stranami byla uzavřena smlouva o dílo a později spor o platnost smlouvy. "
        "Neřeším platnost smlouvy, nýbrž náhradu nákladů řízení. "
        "Hledám judikaturu k nákladům řízení před dovolacím soudem."
    )
    brief = build_extractive_search_brief((text + "\n\n") * 4, config=config)
    low = brief.brief_text.lower()
    assert "náklad" in low
    assert "neřeš" in low or "nikoli" in low or "nýbrž" in low or brief.negative_focus


def test_criminal_guilt_vs_admissibility(config: LongInputConfig) -> None:
    text = (
        "Obviněný byl uznán vinným ze spáchání trestného činu. "
        "Nehledám meritorní posouzení viny, ale přípustnost dovolání a formální vady. "
        "Dovolání bylo odmítnuto pro nepřípustnost."
    )
    brief = build_extractive_search_brief((text + "\n") * 5, config=config)
    low = brief.brief_text.lower()
    assert "přípust" in low or "vad" in low or "dovol" in low
    assert "nehledám" in low or brief.negative_focus


def test_substantive_case_without_procedural_contrast(config: LongInputConfig) -> None:
    text = (
        "Rodiče se neshodli na péči o nezletilé dítě po rozchodu. "
        "Matka navrhuje svěření do péče a úpravu styku otce. "
        "Soud posuzuje nejlepší zájem dítěte a výživné."
    )
    brief = build_extractive_search_brief((text + "\n\n") * 4, config=config)
    low = brief.brief_text.lower()
    assert "péč" in low or "styk" in low or "výživ" in low or "nezletil" in low


def test_determinism(config: LongInputConfig) -> None:
    text = _procedural_fixture()
    a = build_extractive_search_brief(text, config=config)
    b = build_extractive_search_brief(text, config=config)
    assert a.brief_text == b.brief_text
    assert a.brief_signature == b.brief_signature


def test_segmentation_keeps_end_evidence(config: LongInputConfig) -> None:
    prefix = ("Obecný popis rodinné situace a historie řízení. " * 80).strip()
    middle = "Dále se uvádí průběh opatrovnického řízení před okresním soudem. "
    decisive = (
        "Závěrem stěžovatel uvádí: Nehledám meritorní spor o péči, "
        "ale odmítnutí ústavní stížnosti pro chybějícího advokáta."
    )
    text = f"{prefix}\n\n{middle * 20}\n\n{decisive}"
    brief = build_extractive_search_brief(text, config=config)
    low = brief.brief_text.lower()
    assert "advokát" in low or "zastoupen" in low or "vad" in low
    assert "nehledám" in low or brief.negative_focus
