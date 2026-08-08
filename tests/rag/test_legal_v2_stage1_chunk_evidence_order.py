"""Prove additive chunk_evidence provenance does not change Stage 1 document order."""

from __future__ import annotations

from app.rag.legal_v2.evidence.selection import CandidateEvidenceDocument
from app.rag.legal_v2.query_spec import build_query_spec_v2
from app.rag.legal_v2.retrieve import retriever as retriever_module
from app.rag.retrieval.models import RetrievedChunk


def _chunk(chunk_id: str, document_id: str, score: float, order: int) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        text=f"text for {chunk_id}",
        score=score,
        source="hybrid",
        metadata={
            "ecli": document_id,
            "document_id": document_id,
            "source_order": order,
            "chunk_index": order,
            "rrf_score": score,
            "section_type": "facts",
        },
    )


def test_chunk_evidence_does_not_change_document_order() -> None:
    dense = [
        _chunk("d1", "ECLI:CZ:US:2025:1.US.1111.25.1", 0.9, 2),
        _chunk("d2", "ECLI:CZ:US:2025:1.US.2222.25.1", 0.8, 1),
    ]
    bm25 = [
        _chunk("b1", "ECLI:CZ:US:2025:1.US.1111.25.1", 0.7, 3),
        _chunk("b2", "ECLI:CZ:US:2025:1.US.3333.25.1", 0.6, 1),
    ]
    fused = [
        _chunk("f1", "ECLI:CZ:US:2025:1.US.1111.25.1", 0.95, 2),
        _chunk("f2", "ECLI:CZ:US:2025:1.US.2222.25.1", 0.85, 1),
        _chunk("f3", "ECLI:CZ:US:2025:1.US.3333.25.1", 0.75, 1),
        _chunk("f4", "ECLI:CZ:US:2025:1.US.1111.25.1", 0.70, 5),
    ]
    query_spec = build_query_spec_v2("testovací dotaz bez zlatých štítků")
    docs = retriever_module._aggregate_documents(  # noqa: SLF001
        fused,
        dense=dense,
        bm25=bm25,
        query_spec=query_spec,
        limit=10,
    )
    assert [doc.document_id for doc in docs] == [
        "ECLI:CZ:US:2025:1.US.1111.25.1",
        "ECLI:CZ:US:2025:1.US.2222.25.1",
        "ECLI:CZ:US:2025:1.US.3333.25.1",
    ]
    first = docs[0]
    assert isinstance(first, CandidateEvidenceDocument)
    assert first.chunk_evidence
    assert {row["chunk_id"] for row in first.chunk_evidence} >= {"f1", "f4"}
    by_id = {row["chunk_id"]: row for row in first.chunk_evidence}
    assert by_id["f1"]["rrf_rank"] == 1
    assert by_id["f4"]["rrf_rank"] == 4
