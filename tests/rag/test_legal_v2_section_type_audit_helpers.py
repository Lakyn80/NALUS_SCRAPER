"""Unit tests for SectionType audit heuristics (no network)."""

from __future__ import annotations

from scripts.legal_v2.run_section_type_audit_pilot_300 import _flag_paragraph, _materiality_verdict


def test_flag_header_numbered_reasoning() -> None:
    flags = _flag_paragraph(
        section="header",
        text=(
            "7. Ústavní soud předně posoudil splnění procesních předpokladů řízení "
            "a dospěl k závěru, že ústavní stížnost byla podána včas."
        ),
        is_heading=False,
        token_count=40,
        prev_section="procedural_history",
    )
    assert "header_numbered_reasoning_cues" in flags


def test_flag_tiny_structural_heading() -> None:
    flags = _flag_paragraph(
        section="operative_part",
        text="Výrok",
        is_heading=True,
        token_count=1,
        prev_section="other",
    )
    assert "tiny_structural_heading_chunk_candidate" in flags
    assert "operative_heading_only_tiny" in flags


def test_flag_cited_case_operative() -> None:
    flags = _flag_paragraph(
        section="cited_case",
        text="I. Rozsudek soudu prvního stupně se potvrzuje.",
        is_heading=False,
        token_count=8,
        prev_section="operative_part",
    )
    assert "cited_case_looks_like_operative" in flags


def test_materiality_blocks_on_header_signal() -> None:
    stats = {
        "documents": 300,
        "flag_counts": {
            "header_numbered_reasoning_cues": 50,
            "tiny_structural_heading_chunk_candidate": 20,
        },
        "documents_with_header_suspicion": 80,
        "documents_with_any_suspicion": 120,
        "section_paragraph_share": {"header": 0.30},
    }
    out = _materiality_verdict(stats)
    assert out["verdict"] == "SECTION_TYPE_MATERIAL_REGRESSION"
    assert out["block_slice4"] is True


def test_materiality_ok_when_header_cleared_even_with_tiny_headings() -> None:
    stats = {
        "documents": 300,
        "flag_counts": {
            "tiny_structural_heading_chunk_candidate": 282,
            "header_long_nonheader_prose": 1,
        },
        "documents_with_header_suspicion": 1,
        "documents_with_any_suspicion": 236,
        "section_paragraph_share": {"header": 0.025},
    }
    out = _materiality_verdict(stats)
    assert out["verdict"] == "SECTION_TYPE_OK_FOR_CHUNKING_AB"
    assert out["block_slice4"] is False
    assert out["warnings"]
