"""Focused tests for Stage-1 FE best-passage selection from chunk_evidence."""

from __future__ import annotations

from app.rag.legal_v2.evidence.selection import CandidateEvidenceDocument
from app.rag.legal_v2.models import LegalParagraph, MetadataProvenance, SectionType
from app.rag.legal_v2.retrieve import case_similarity_search as stage1


def _paragraph(
    *,
    paragraph_id: str,
    index: int,
    text: str,
    section: SectionType = SectionType.COURT_REASONING,
) -> LegalParagraph:
    return LegalParagraph(
        document_id="ECLI:CZ:US:2025:1.US.1111.25.1",
        paragraph_id=paragraph_id,
        paragraph_index=index,
        original_text=text,
        normalized_text=text,
        section_type=section,
        start_offset=0,
        end_offset=len(text),
        source_order=index,
        heading_context=[],
        is_boilerplate=False,
        is_citation_block=False,
        language="cs",
        metadata_provenance=MetadataProvenance(
            source="unit_test",
            extraction_method="manual",
        ),
    )


def _document(
    *,
    paragraphs: list[LegalParagraph],
    chunk_evidence: list[dict],
) -> CandidateEvidenceDocument:
    return CandidateEvidenceDocument(
        document_id="ECLI:CZ:US:2025:1.US.1111.25.1",
        metadata={
            "ecli": "ECLI:CZ:US:2025:1.US.1111.25.1",
            "court_name": "Ústavní soud",
            "case_reference": "I. ÚS 1111/25",
            "decision_date": "2025-01-15",
            "document_type": "nález",
            "source_document_id": "doc-abc123",
        },
        paragraphs=paragraphs,
        score=0.42,
        dense_rank=1,
        bm25_rank=2,
        rrf_score=0.42,
        chunk_evidence=chunk_evidence,
    )


def test_prefer_chunk_evidence_skips_heading_only_top_hit() -> None:
    matching = (
        "Soud dospěl k závěru, že mezinárodní únos dítěte je třeba posoudit "
        "podle Haagské úmluvy o občanskoprávních aspektech mezinárodních únosů dětí."
    )
    doc = _document(
        paragraphs=[
            _paragraph(paragraph_id="p0", index=0, text="Odůvodnění"),
            _paragraph(paragraph_id="p1", index=1, text="První odstavec v pořadí dokumentu."),
        ],
        chunk_evidence=[
            {
                "chunk_id": "c-heading",
                "text": "Odůvodnění",
                "rrf_rank": 1,
                "rrf_score": 0.09,
                "section": "court_reasoning",
                "retrieval_channels": ["rrf"],
            },
            {
                "chunk_id": "c-match",
                "text": matching,
                "rrf_rank": 2,
                "rrf_score": 0.08,
                "section": "court_reasoning",
                "retrieval_channels": ["rrf", "dense"],
            },
        ],
    )

    result = stage1._to_stage1_document(  # noqa: SLF001
        doc,
        rank=1,
        prefer_chunk_evidence=True,
        evidence_limit=5,
    )

    assert result.relevant_passages
    assert result.relevant_passages[0].chunk_id == "c-match"
    assert result.relevant_passages[0].text == matching
    assert "Odůvodnění" not in {p.text for p in result.relevant_passages}


def test_prefer_chunk_evidence_preserves_rrf_order_among_body_chunks() -> None:
    first = "První matching chunk o mezinárodním únosu dítěte a obvyklém bydlišti."
    second = "Druhý matching chunk o návratu dítěte do státu obvyklého bydliště."
    doc = _document(
        paragraphs=[_paragraph(paragraph_id="p0", index=0, text="Odůvodnění")],
        chunk_evidence=[
            {
                "chunk_id": "c1",
                "text": first,
                "rrf_rank": 1,
                "rrf_score": 0.11,
                "retrieval_channels": ["rrf"],
            },
            {
                "chunk_id": "c2",
                "text": second,
                "rrf_rank": 2,
                "rrf_score": 0.10,
                "retrieval_channels": ["rrf"],
            },
        ],
    )

    result = stage1._to_stage1_document(  # noqa: SLF001
        doc,
        rank=3,
        prefer_chunk_evidence=True,
        evidence_limit=2,
    )

    assert [p.chunk_id for p in result.relevant_passages] == ["c1", "c2"]
    assert result.rank == 3


def test_falls_back_to_paragraphs_when_chunk_evidence_is_only_headings() -> None:
    body = (
        "Ústavní soud konstatuje, že obecné soudy dostatečně nezohlednily "
        "nejlepší zájem dítěte při rozhodování o návratu."
    )
    doc = _document(
        paragraphs=[
            _paragraph(paragraph_id="p0", index=0, text="Odůvodnění"),
            _paragraph(paragraph_id="p1", index=1, text=body),
        ],
        chunk_evidence=[
            {
                "chunk_id": "c-heading",
                "text": "Odůvodnění",
                "rrf_rank": 1,
                "rrf_score": 0.05,
                "retrieval_channels": ["rrf"],
            },
            {
                "chunk_id": "c-vyrok",
                "text": "Výrok",
                "rrf_rank": 2,
                "rrf_score": 0.04,
                "retrieval_channels": ["bm25"],
            },
        ],
    )

    result = stage1._to_stage1_document(  # noqa: SLF001
        doc,
        rank=1,
        prefer_chunk_evidence=True,
    )

    assert result.relevant_passages
    assert result.relevant_passages[0].chunk_id == "p1"
    assert result.relevant_passages[0].text == body


def test_is_heading_only_passage_detects_common_markers() -> None:
    assert stage1._is_heading_only_passage("Odůvodnění")  # noqa: SLF001
    assert stage1._is_heading_only_passage("Odůvodnění:")  # noqa: SLF001
    assert stage1._is_heading_only_passage("Výrok")  # noqa: SLF001
    assert stage1._is_heading_only_passage("II.")  # noqa: SLF001
    assert not stage1._is_heading_only_passage(  # noqa: SLF001
        "Odůvodnění napadeného rozhodnutí je nepřezkoumatelné."
    )
