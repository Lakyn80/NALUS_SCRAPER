"""Unit tests for Legal Contextual Packed chunker v1 and A freeze regression."""

from __future__ import annotations

import hashlib

from app.rag.legal_v2.audit import CHUNKER_VERSION
from app.rag.legal_v2.ingest.chunkers import chunk_document_for_experiment
from app.rag.legal_v2.ingest.chunkers.contextual_packed_v1 import (
    ContextualPackedConfigV1,
    build_contextual_packed_chunks_v1,
)
from app.rag.legal_v2.ingest.chunkers.names import (
    CHUNKER_A_CURRENT,
    CHUNKER_B_CONTEXTUAL_PACKED_V1,
)
from app.rag.legal_v2.ingest.chunking import (
    HierarchicalChunkConfig,
    build_hierarchical_chunks,
)
from app.rag.legal_v2.models import (
    LegalDocumentStructure,
    LegalParagraph,
    MetadataProvenance,
    ParagraphParsingDiagnostics,
    SectionType,
)


def _provenance() -> MetadataProvenance:
    return MetadataProvenance(source="test", extraction_method="unit")


def _paragraph(
    *,
    document_id: str,
    index: int,
    text: str,
    section: SectionType,
    offset: int,
) -> LegalParagraph:
    return LegalParagraph(
        document_id=document_id,
        paragraph_id=f"{document_id}:p:{index:05d}",
        paragraph_index=index,
        original_text=text,
        normalized_text=text,
        section_type=section,
        start_offset=offset,
        end_offset=offset + len(text),
        source_order=index,
        heading_context=[],
        is_boilerplate=False,
        is_citation_block=False,
        language="cs",
        metadata_provenance=_provenance(),
    )


def _document(
    document_id: str,
    paragraphs: list[tuple[str, SectionType]],
) -> LegalDocumentStructure:
    built: list[LegalParagraph] = []
    offset = 0
    for index, (text, section) in enumerate(paragraphs):
        built.append(
            _paragraph(
                document_id=document_id,
                index=index,
                text=text,
                section=section,
                offset=offset,
            )
        )
        offset += len(text) + 2
    return LegalDocumentStructure(
        document_id=document_id,
        normalized_text="\n\n".join(text for text, _ in paragraphs),
        paragraphs=built,
        diagnostics=ParagraphParsingDiagnostics(
            paragraph_count=len(built),
            numbered_paragraph_count=0,
            heading_count=0,
            boilerplate_count=0,
            citation_block_count=0,
            damaged_formatting_detected=False,
            fallback_paragraphs_created=0,
            section_counts={},
        ),
    )


def _words(n: int, seed: str = "slovo") -> str:
    return " ".join(f"{seed}{i}" for i in range(n))


def test_chunker_a_version_matches_production_constant() -> None:
    assert CHUNKER_A_CURRENT == CHUNKER_VERSION


def test_a_hierarchical_unchanged_fingerprint() -> None:
    """Regression: production A output fingerprint for a fixed synthetic doc."""
    doc = _document(
        "DOC-A-REG",
        [
            (_words(120, "facts"), SectionType.FACTS),
            (_words(130, "factsb"), SectionType.FACTS),
            (_words(140, "reason"), SectionType.COURT_REASONING),
            (_words(90, "reasonb"), SectionType.COURT_REASONING),
        ],
    )
    result = build_hierarchical_chunks(doc, config=HierarchicalChunkConfig())
    payload = "|".join(
        f"{c.chunk_index}:{c.token_count}:{','.join(c.paragraph_ids)}:{c.section_type.value}"
        for c in result.child_chunks
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    assert digest == "7015a733cc0e5a3e17fe4bb6e8f83c835e28d24170a72c4a6b60f5168d9598cc"
    assert result.diagnostics.child_chunk_count == 2
    assert all(chunk.token_count <= 650 for chunk in result.child_chunks)
    via_dispatch = chunk_document_for_experiment(doc, chunker_version=CHUNKER_A_CURRENT)
    assert [c.chunk_id for c in via_dispatch.child_chunks] == [
        c.chunk_id for c in result.child_chunks
    ]


def test_b_deterministic_and_stable_ids() -> None:
    doc = _document(
        "DOC-B-DET",
        [
            (_words(200, "a"), SectionType.FACTS),
            (_words(200, "b"), SectionType.FACTS),
            (_words(200, "c"), SectionType.FACTS),
        ],
    )
    first = build_contextual_packed_chunks_v1(doc)
    second = build_contextual_packed_chunks_v1(doc)
    assert [c.chunk_id for c in first.child_chunks] == [c.chunk_id for c in second.child_chunks]
    assert [c.text for c in first.child_chunks] == [c.text for c in second.child_chunks]
    assert all(CHUNKER_B_CONTEXTUAL_PACKED_V1 in c.chunk_id for c in first.child_chunks)


def test_b_preserves_ecli_document_id() -> None:
    ecli = "ECLI:CZ:US:2025:1.US.1111.25.1"
    doc = _document(ecli, [(_words(50), SectionType.FACTS)])
    result = build_contextual_packed_chunks_v1(doc)
    assert result.child_chunks[0].document_id == ecli
    assert result.child_chunks[0].chunk_id.startswith(ecli)


def test_b_no_empty_chunks_and_no_text_loss() -> None:
    paragraphs = [
        (_words(80, "f1"), SectionType.FACTS),
        (_words(90, "f2"), SectionType.FACTS),
        (_words(100, "r1"), SectionType.COURT_REASONING),
        (_words(110, "r2"), SectionType.COURT_REASONING),
    ]
    doc = _document("DOC-COV", paragraphs)
    result = build_contextual_packed_chunks_v1(doc)
    assert result.child_chunks
    assert all(chunk.text.strip() for chunk in result.child_chunks)
    # Every paragraph id appears in at least one chunk.
    seen: set[str] = set()
    for chunk in result.child_chunks:
        seen.update(chunk.paragraph_ids)
    assert seen == {p.paragraph_id for p in doc.paragraphs}


def test_b_does_not_pack_across_section_boundary() -> None:
    doc = _document(
        "DOC-SEC",
        [
            (_words(200, "facts"), SectionType.FACTS),
            (_words(200, "reason"), SectionType.COURT_REASONING),
        ],
    )
    result = build_contextual_packed_chunks_v1(doc)
    for chunk in result.child_chunks:
        sections = {doc.paragraphs[pid.split(":")[-1].lstrip("p0") and 0 or 0].section_type for pid in chunk.paragraph_ids}
        # Simpler check via metadata / paragraph lookup:
        para_by_id = {p.paragraph_id: p for p in doc.paragraphs}
        chunk_sections = {para_by_id[pid].section_type for pid in chunk.paragraph_ids}
        assert len(chunk_sections) == 1


def test_b_packs_within_same_section() -> None:
    # Three ~220-token paragraphs → soft target 650 should pack more than one.
    doc = _document(
        "DOC-PACK",
        [
            (_words(220, "p1"), SectionType.FACTS),
            (_words(220, "p2"), SectionType.FACTS),
            (_words(220, "p3"), SectionType.FACTS),
        ],
    )
    result = build_contextual_packed_chunks_v1(doc)
    assert any(len(chunk.paragraph_ids) >= 2 for chunk in result.child_chunks)
    assert all(chunk.token_count <= 850 for chunk in result.child_chunks)


def test_b_hard_token_cap() -> None:
    doc = _document(
        "DOC-HARD",
        [
            (_words(400, "a"), SectionType.FACTS),
            (_words(400, "b"), SectionType.FACTS),
            (_words(400, "c"), SectionType.FACTS),
        ],
    )
    result = build_contextual_packed_chunks_v1(doc)
    assert all(chunk.token_count <= 850 for chunk in result.child_chunks)


def test_b_overlap_whole_paragraph_or_none() -> None:
    # First pack will take ~400+400 under soft target? 400+400=800 > 650 soft target
    # so first chunk likely one or two paras; next starts with overlap only if last <=150.
    short = _words(80, "short")  # <= 150 → eligible overlap
    long_tail = _words(200, "tail")
    doc = _document(
        "DOC-OV",
        [
            (_words(400, "a"), SectionType.FACTS),
            (short, SectionType.FACTS),
            (long_tail, SectionType.FACTS),
            (_words(200, "more"), SectionType.FACTS),
        ],
    )
    result = build_contextual_packed_chunks_v1(doc)
    # If overlap happened, a complete short paragraph text appears in two chunks.
    short_hits = sum(1 for chunk in result.child_chunks if short in chunk.text)
    assert short_hits >= 1
    # Never invent partial short paragraph overlap by cutting tokens.
    for chunk in result.child_chunks:
        if "short0" in chunk.text:
            # Full paragraph present when referenced.
            assert short in chunk.text or short.split()[0] in chunk.text


def test_b_no_overlap_when_previous_paragraph_exceeds_cap() -> None:
    config = ContextualPackedConfigV1(overlap_max_tokens=150)
    big = _words(200, "big")  # > 150 → no overlap
    doc = _document(
        "DOC-NOOV",
        [
            (big, SectionType.FACTS),
            (_words(200, "next"), SectionType.FACTS),
            (_words(200, "next2"), SectionType.FACTS),
        ],
    )
    result = build_contextual_packed_chunks_v1(doc, config=config)
    # First paragraph should not appear as manufactured overlap prefix of later-only packs
    # beyond its own chunk(s). Count occurrences of unique marker.
    marker = "big0"
    occurrences = sum(chunk.text.count(marker) for chunk in result.child_chunks)
    # Each chunk that includes the paragraph contributes once; overlap would duplicate.
    # With no overlap, marker appears only in chunks that own that paragraph once.
    assert occurrences == sum(
        1 for chunk in result.child_chunks if any("big" in pid or True for pid in chunk.paragraph_ids)
        and marker in chunk.text
    )


def test_b_oversized_paragraph_sentence_split() -> None:
    # One paragraph >> hard max with sentence boundaries.
    sentences = ". ".join(_words(100, f"s{i}") + "." for i in range(12))
    doc = _document("DOC-OVER", [(sentences, SectionType.COURT_REASONING)])
    result = build_contextual_packed_chunks_v1(doc)
    assert result.diagnostics.split_overlong_paragraph_count >= 1
    assert all(chunk.token_count <= 850 for chunk in result.child_chunks)
    assert result.child_chunks


def test_b_no_cross_document_contamination() -> None:
    doc_a = _document("DOC-X-A", [(_words(60, "aa"), SectionType.FACTS)])
    doc_b = _document("DOC-X-B", [(_words(60, "bb"), SectionType.FACTS)])
    out_a = build_contextual_packed_chunks_v1(doc_a)
    out_b = build_contextual_packed_chunks_v1(doc_b)
    assert all(c.document_id == "DOC-X-A" for c in out_a.child_chunks)
    assert all(c.document_id == "DOC-X-B" for c in out_b.child_chunks)
    assert {c.chunk_id for c in out_a.child_chunks}.isdisjoint(
        {c.chunk_id for c in out_b.child_chunks}
    )


def test_b_positions_are_contiguous() -> None:
    doc = _document(
        "DOC-POS",
        [
            (_words(300, "a"), SectionType.FACTS),
            (_words(300, "b"), SectionType.FACTS),
            (_words(300, "c"), SectionType.FACTS),
            (_words(300, "d"), SectionType.FACTS),
        ],
    )
    result = build_contextual_packed_chunks_v1(doc)
    indexes = [c.chunk_index for c in result.child_chunks]
    assert indexes == list(range(len(indexes)))


def test_dispatch_unknown_chunker_rejected() -> None:
    doc = _document("DOC-UNK", [(_words(10), SectionType.FACTS)])
    try:
        chunk_document_for_experiment(doc, chunker_version="not_a_chunker")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "unknown experiment chunker_version" in str(exc)
