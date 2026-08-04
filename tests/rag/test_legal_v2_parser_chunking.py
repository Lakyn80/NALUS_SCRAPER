from __future__ import annotations

import re

from app.rag.legal_v2.chunking import (
    HierarchicalChunkConfig,
    build_hierarchical_chunks,
    expand_parent_window,
)
from app.rag.legal_v2.models import MetadataProvenance, SectionType
from app.rag.legal_v2.parser import parse_legal_document


def test_parser_preserves_multiline_numbered_paragraph_28_with_spisova_znacka() -> None:
    text = "\n".join(
        [
            "28. Ve věci řešené v řízení",
            "sp. zn. IV. ÚS 1038/25",
            "zrušil služební orgán napadené rozhodnutí.",
        ]
    )

    parsed = parse_legal_document(document_id="DOC-P28", text=text)

    assert parsed.diagnostics.paragraph_count == 1
    paragraph = parsed.paragraphs[0]
    assert paragraph.numbering == "28"
    assert paragraph.section_type != SectionType.HEADER
    assert "28. Ve věci řešené v řízení" in paragraph.original_text
    assert "sp. zn. IV. ÚS 1038/25" in paragraph.original_text
    assert "zrušil služební orgán napadené rozhodnutí." in paragraph.original_text
    assert parsed.diagnostics.heading_count == 0
    assert parsed.diagnostics.numbered_paragraph_count == 1


def test_parser_preserves_multiline_numbered_paragraph_43_with_cislo_jednaci() -> None:
    text = "\n".join(
        [
            "43. V navazujícím posouzení soud připomněl, že v řízení",
            "č. j. 12 A 34/2024-56",
            "Nejvyšší správní soud navázal na předchozí závěry.",
        ]
    )

    parsed = parse_legal_document(document_id="DOC-P43", text=text)

    assert parsed.diagnostics.paragraph_count == 1
    paragraph = parsed.paragraphs[0]
    assert paragraph.numbering == "43"
    assert paragraph.section_type != SectionType.HEADER
    assert "č. j. 12 A 34/2024-56" in paragraph.original_text
    assert "Nejvyšší správní soud navázal" in paragraph.original_text


def test_numbered_paragraph_keywords_do_not_override_numbered_structure() -> None:
    keywords = ("řízení", "nález", "odůvodnění", "posouzení")
    for index, keyword in enumerate(keywords, start=1):
        parsed = parse_legal_document(
            document_id=f"DOC-KEY-{index}",
            text=f"{index}. Krátký právní odstavec obsahuje slovo {keyword}\n"
            "sp. zn. IV. ÚS 1038/25\n"
            "a pokračuje v téže právní větě.",
        )

        assert parsed.diagnostics.paragraph_count == 1
        assert parsed.paragraphs[0].numbering == str(index)
        assert parsed.diagnostics.heading_count == 0


def test_short_prose_with_heading_keywords_is_not_heading() -> None:
    for keyword in ("řízení", "nález", "odůvodnění", "posouzení"):
        parsed = parse_legal_document(
            document_id=f"DOC-PROSE-{keyword}",
            text=f"Soud pokračoval v {keyword} a provedl další dokazování.",
        )

        assert parsed.diagnostics.paragraph_count == 1
        assert parsed.diagnostics.heading_count == 0
        assert parsed.paragraphs[0].section_type != SectionType.HEADER


def test_genuine_headings_and_numbered_boundaries_are_preserved() -> None:
    text = "\n".join(
        [
            "Výrok",
            "1. Návrh se odmítá.",
            "2. Náklady řízení se nepřiznávají.",
            "Odůvodnění",
            "3. Ústavní soud posoudil věc samostatně.",
            "Posouzení Ústavního soudu",
            "4. Navazující odstavec zůstává číslovaný.",
        ]
    )

    parsed = parse_legal_document(document_id="DOC-HEADINGS", text=text)
    paragraphs = parsed.paragraphs

    assert [paragraph.original_text for paragraph in paragraphs] == [
        "Výrok",
        "1. Návrh se odmítá.",
        "2. Náklady řízení se nepřiznávají.",
        "Odůvodnění",
        "3. Ústavní soud posoudil věc samostatně.",
        "Posouzení Ústavního soudu",
        "4. Navazující odstavec zůstává číslovaný.",
    ]
    assert [paragraph.numbering for paragraph in paragraphs] == [
        None,
        "1",
        "2",
        None,
        "3",
        None,
        "4",
    ]
    assert parsed.diagnostics.heading_count == 3
    assert parsed.diagnostics.numbered_paragraph_count == 4
    assert paragraphs[0].section_type == SectionType.OPERATIVE_PART
    assert paragraphs[3].section_type == SectionType.COURT_REASONING
    assert paragraphs[5].section_type == SectionType.COURT_REASONING


def test_numbered_paragraph_followed_by_heading_and_heading_followed_by_numbered_paragraph() -> None:
    text = "\n".join(
        [
            "1. První odstavec pokračuje přes",
            "sp. zn. I. ÚS 1/24",
            "Výrok",
            "2. Druhý odstavec navazuje po nadpisu.",
        ]
    )

    parsed = parse_legal_document(document_id="DOC-BOUNDARY", text=text)

    assert [paragraph.original_text for paragraph in parsed.paragraphs] == [
        "1. První odstavec pokračuje přes sp. zn. I. ÚS 1/24",
        "Výrok",
        "2. Druhý odstavec navazuje po nadpisu.",
    ]
    assert [paragraph.numbering for paragraph in parsed.paragraphs] == ["1", None, "2"]
    assert parsed.paragraphs[1].section_type == SectionType.OPERATIVE_PART


def test_parser_emits_no_empty_or_orphan_citation_candidate_for_confirmed_example() -> None:
    text = "\n".join(
        [
            "28. Ve věci řešené v řízení",
            "sp. zn. IV. ÚS 1038/25",
            "zrušil služební orgán napadené rozhodnutí.",
            "",
            "43. V dalším odůvodnění soud odkázal na",
            "č. j. 12 A 34/2024-56",
            "a zachoval původní právní názor.",
        ]
    )

    parsed = parse_legal_document(document_id="DOC-NO-ORPHAN", text=text)

    assert len(parsed.paragraphs) == 2
    assert all(paragraph.original_text.strip() for paragraph in parsed.paragraphs)
    assert not any(
        paragraph.original_text.lower().startswith(("sp. zn.", "č. j."))
        for paragraph in parsed.paragraphs
    )


def test_parser_conserves_text_order_and_non_whitespace_content() -> None:
    text = "\n".join(
        [
            "Výrok",
            "1. Ve věci řešené v řízení",
            "sp. zn. IV. ÚS 1038/25",
            "zrušil služební orgán napadené rozhodnutí.",
            "2. Napadený nález byl následně zrušen.",
            "Odůvodnění",
            "3. Soud pokračoval v řízení a provedl dokazování.",
        ]
    )

    first = parse_legal_document(document_id="DOC-CONSERVE", text=text)
    second = parse_legal_document(document_id="DOC-CONSERVE", text=text)

    assert _non_whitespace(first.reconstruct_text()) == _non_whitespace(text)
    assert [paragraph.original_text for paragraph in first.paragraphs] == [
        "Výrok",
        "1. Ve věci řešené v řízení sp. zn. IV. ÚS 1038/25 zrušil služební orgán napadené rozhodnutí.",
        "2. Napadený nález byl následně zrušen.",
        "Odůvodnění",
        "3. Soud pokračoval v řízení a provedl dokazování.",
    ]
    assert [paragraph.paragraph_id for paragraph in first.paragraphs] == [
        paragraph.paragraph_id for paragraph in second.paragraphs
    ]
    assert [paragraph.start_offset for paragraph in first.paragraphs] == sorted(
        paragraph.start_offset for paragraph in first.paragraphs
    )


def test_parser_preserves_stable_paragraph_ids_and_damaged_formatting() -> None:
    text = "\n".join(
        [
            "ÚSTAVNÍ SOUD",
            "Účastníci řízení",
            "[1] Stěžovatelka je matkou nezletilého dítěte.",
            "[2] Otec namítá, že dítě bylo přemístěno.",
            "I. SKUTKOVÝ STAV",
            "[3] Ze spisu vyplývá, že dítě žilo v Česku.",
            "II. ODŮVODNĚNÍ",
            "[4] Ústavní soud dospěl k závěru, že věc je nutné posoudit.",
            "Srov. nález sp. zn. II. ÚS 859/23 a rozsudek č. j. 1 As 1/2020.",
            "Poučení: Proti rozhodnutí není odvolání přípustné.",
        ]
    )

    first = parse_legal_document(document_id="DOC-1", text=text)
    second = parse_legal_document(document_id="DOC-1", text=text)

    assert [p.paragraph_id for p in first.paragraphs] == [
        p.paragraph_id for p in second.paragraphs
    ]
    assert first.diagnostics.damaged_formatting_detected is True
    assert first.diagnostics.numbered_paragraph_count == 4
    assert first.diagnostics.heading_count >= 3
    assert any(p.section_type == SectionType.PARTICIPANTS for p in first.paragraphs)
    assert any(p.section_type == SectionType.FACTS for p in first.paragraphs)
    assert any(p.is_boilerplate for p in first.paragraphs)
    assert any(p.is_citation_block for p in first.paragraphs)
    assert first.paragraphs[2].numbering == "1"
    assert first.reconstruct_text()


def test_parser_profile_version_identifies_corrected_paragraph_parser() -> None:
    from app.rag.legal_v2.audit import PARSER_VERSION

    assert PARSER_VERSION == "legal-paragraph-parser.v3"


def test_chunking_merges_splits_overlaps_and_preserves_reconstruction() -> None:
    sentence_a = "První dlouhá věta zachovává celý právní význam pro test."
    sentence_b = "Druhá dlouhá věta zůstane nedotčená při dělení odstavce."
    sentence_c = "Třetí věta doplňuje kontext bez rozbití větné hranice."
    text = "\n\n".join(
        [
            "I. SKUTKOVÝ STAV",
            "[1] Krátký odstavec o dítěti.",
            "[2] Další krátký odstavec o matce.",
            f"[3] {sentence_a} {sentence_b} {sentence_c}",
            "II. ODŮVODNĚNÍ",
            "[4] Soud posoudil právní otázku samostatně.",
            "[5] Odůvodnění nepřekračuje předchozí skutkovou sekci.",
        ]
    )
    document = parse_legal_document(document_id="DOC-CHUNK", text=text)
    config = HierarchicalChunkConfig(
        child_target_min_tokens=8,
        child_target_max_tokens=14,
        child_hard_max_tokens=18,
        parent_target_min_tokens=12,
        parent_target_max_tokens=40,
        parent_hard_max_tokens=50,
        min_short_paragraph_tokens=7,
    )

    first = build_hierarchical_chunks(document, config=config)
    second = build_hierarchical_chunks(document, config=config)

    assert first.diagnostics.merged_short_paragraph_count >= 1
    assert first.diagnostics.split_overlong_paragraph_count == 1
    assert first.reconstruct_text() == document.reconstruct_text()
    assert [chunk.chunk_id for chunk in first.child_chunks] == [
        chunk.chunk_id for chunk in second.child_chunks
    ]
    assert any(len(chunk.paragraph_ids) > 1 for chunk in first.child_chunks)
    assert any(sentence_a in chunk.text for chunk in first.child_chunks)
    assert any(sentence_b in chunk.text for chunk in first.child_chunks)
    assert all(
        len({span.paragraph_id for span in chunk.source_spans}) >= 1
        for chunk in first.child_chunks
    )
    for chunk in first.child_chunks:
        section_types = {
            document.paragraphs[span.paragraph_index].section_type
            for span in chunk.source_spans
        }
        assert section_types == {chunk.section_type}

    adjacent_pairs = zip(first.child_chunks, first.child_chunks[1:], strict=False)
    assert any(
        set(left.paragraph_ids).intersection(right.paragraph_ids)
        for left, right in adjacent_pairs
        if left.section_type == right.section_type
    )

    anchor = first.child_chunks[0]
    parent = expand_parent_window(document, anchor, config=config)
    assert set(anchor.paragraph_ids).issubset(parent.paragraph_ids)
    assert parent.token_count <= config.parent_hard_max_tokens
    assert parent.section_types == [anchor.section_type]


def test_overlong_punctuation_heavy_paragraph_respects_hard_token_limit() -> None:
    entries = " ".join(f"067 EX {number}/10-{number};" for number in range(60))
    parsed = parse_legal_document(
        document_id="DOC-PUNCT",
        text=f"[1] {entries}",
        metadata={"court": "test"},
        provenance=MetadataProvenance(source="test", extraction_method="unit"),
    )
    config = HierarchicalChunkConfig(
        child_target_min_tokens=20,
        child_target_max_tokens=30,
        child_hard_max_tokens=40,
        parent_target_min_tokens=20,
        parent_target_max_tokens=80,
        parent_hard_max_tokens=120,
    )

    result = build_hierarchical_chunks(parsed, config=config)

    assert result.diagnostics.split_overlong_paragraph_count == 1
    assert result.child_chunks
    assert all(chunk.token_count <= config.child_hard_max_tokens for chunk in result.child_chunks)


def test_parent_window_for_overlong_paragraph_respects_hard_token_limit() -> None:
    long_paragraph = " ".join(
        f"Dlouhá věta číslo {index} zachovává právní argumentaci."
        for index in range(120)
    )
    parsed = parse_legal_document(
        document_id="DOC-PARENT-HARD",
        text=f"ODŮVODNĚNÍ\n\n[1] {long_paragraph}",
        metadata={"court": "test"},
        provenance=MetadataProvenance(source="test", extraction_method="unit"),
    )
    config = HierarchicalChunkConfig(
        child_target_min_tokens=40,
        child_target_max_tokens=80,
        child_hard_max_tokens=120,
        parent_target_min_tokens=120,
        parent_target_max_tokens=220,
        parent_hard_max_tokens=240,
    )

    result = build_hierarchical_chunks(parsed, config=config)

    assert result.diagnostics.split_overlong_paragraph_count == 1
    assert result.parent_windows
    assert any(window.truncated for window in result.parent_windows)
    assert all(window.token_count <= config.parent_hard_max_tokens for window in result.parent_windows)
    assert all(
        child.text == window.text
        for child, window in zip(result.child_chunks, result.parent_windows, strict=True)
        if window.truncated
    )


def test_chunking_never_crosses_incompatible_sections() -> None:
    text = "\n\n".join(
        [
            "I. SKUTKOVÝ STAV",
            "[1] Skutkový odstavec popisuje cestu dítěte z Česka.",
            "[2] Skutkový odstavec popisuje pobyt dítěte v Rusku.",
            "II. PRÁVNÍ ÚPRAVA",
            "[3] Podle čl. 8 Úmluvy se hodnotí rodinný život.",
            "[4] Ustanovení zákona se použije samostatně.",
        ]
    )
    document = parse_legal_document(document_id="DOC-SECTIONS", text=text)
    config = HierarchicalChunkConfig(
        child_target_min_tokens=4,
        child_target_max_tokens=60,
        child_hard_max_tokens=80,
        parent_target_min_tokens=4,
        parent_target_max_tokens=80,
        parent_hard_max_tokens=100,
    )

    result = build_hierarchical_chunks(document, config=config)

    assert result.child_chunks
    for chunk in result.child_chunks:
        assert {
            document.paragraphs[span.paragraph_index].section_type
            for span in chunk.source_spans
        } == {chunk.section_type}


def _non_whitespace(value: str) -> str:
    return re.sub(r"\s+", "", value)
