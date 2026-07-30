from __future__ import annotations

from app.rag.legal_v2.chunking import (
    HierarchicalChunkConfig,
    build_hierarchical_chunks,
    expand_parent_window,
)
from app.rag.legal_v2.models import MetadataProvenance, SectionType
from app.rag.legal_v2.parser import parse_legal_document


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
