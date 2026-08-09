"""Regression tests for SectionType sticky / anti-header traps (parser v8)."""

from __future__ import annotations

from app.rag.legal_v2.models import SectionType
from app.rag.legal_v2.parser import parse_legal_document


def test_numbered_ustavni_soud_reasoning_is_not_header() -> None:
    text = "\n".join(
        [
            "Česká republika",
            "USNESENÍ",
            "Ústavního soudu",
            "Ústavní soud rozhodl v senátu složeném z předsedkyně Daniely Zemanové takto:",
            "Ústavní stížnost se odmítá.",
            "Odůvodnění",
            "7. Ústavní soud předně posoudil splnění procesních předpokladů řízení a dospěl "
            "k závěru, že ústavní stížnost byla podána včas.",
            "18. Vzhledem ke shora uvedeným důvodům proto Ústavní soud ústavní stížnost "
            "odmítl podle § 43 odst. 2 písm. a) zákona o Ústavním soudu.",
            "Poučení: Proti usnesení Ústavního soudu není odvolání přípustné.",
            "V Brně dne 20. prosince 2024",
            "Daniela Zemanová v. r.",
            "předsedkyně senátu",
        ]
    )
    parsed = parse_legal_document(
        document_id="DOC-US-HEADER-TRAP",
        text=text,
        metadata={"court": "constitutional_court"},
    )
    by_preview = {p.normalized_text[:40]: p.section_type for p in parsed.paragraphs}
    reasoning = [
        p
        for p in parsed.paragraphs
        if p.normalized_text.startswith("7. Ústavní soud předně")
        or p.normalized_text.startswith("18. Vzhledem ke shora")
    ]
    assert reasoning
    assert all(p.section_type == SectionType.COURT_REASONING for p in reasoning)
    assert all(p.section_type != SectionType.HEADER for p in reasoning)
    signatures = [
        p
        for p in parsed.paragraphs
        if "V Brně dne" in p.normalized_text or p.normalized_text.endswith("v. r.")
    ]
    assert signatures
    assert all(p.section_type != SectionType.HEADER for p in signatures)
    assert by_preview  # keep used


def test_operative_roman_item_not_cited_case_after_vyrok() -> None:
    text = "\n".join(
        [
            "Vrchní soud v Olomouci",
            "Výrok",
            "I. Rozsudek soudu prvního stupně se potvrzuje.",
            "Odůvodnění",
            "1. Odvolací soud přezkoumal napadené rozhodnutí.",
        ]
    )
    parsed = parse_legal_document(
        document_id="DOC-VSOL-OPERATIVE",
        text=text,
        metadata={"court": "high_court_olomouc"},
    )
    operative = next(
        p for p in parsed.paragraphs if "se potvrzuje" in p.normalized_text
    )
    assert operative.section_type == SectionType.OPERATIVE_PART
    assert operative.section_type != SectionType.CITED_CASE


def test_procedural_heading_then_reasoning_upgrades() -> None:
    text = "\n".join(
        [
            "Odůvodnění",
            "III. Splnění podmínek řízení",
            "7. Ústavní soud předně posoudil splnění procesních předpokladů řízení "
            "a dospěl k závěru, že ústavní stížnost byla podána včas.",
        ]
    )
    parsed = parse_legal_document(
        document_id="DOC-PROC-UPGRADE",
        text=text,
        metadata={"court": "constitutional_court"},
    )
    body = next(p for p in parsed.paragraphs if p.normalized_text.startswith("7."))
    assert body.section_type == SectionType.COURT_REASONING


def test_true_header_banners_remain_header() -> None:
    text = "\n".join(
        [
            "NALUS - databáze rozhodnutí Ústavního soudu",
            "III.ÚS 3203/24 ze dne 20. 12. 2024",
            "Česká republika",
            "USNESENÍ",
            "Ústavního soudu",
        ]
    )
    parsed = parse_legal_document(
        document_id="DOC-TRUE-HEADER",
        text=text,
        metadata={"court": "constitutional_court"},
    )
    assert all(p.section_type == SectionType.HEADER for p in parsed.paragraphs)
