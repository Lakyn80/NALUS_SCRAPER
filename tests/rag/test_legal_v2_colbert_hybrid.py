"""Unit tests for experimental BGE-M3 + BM25 + ColBERT RRF orchestration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

from app.rag.legal_v2.query_spec import build_query_spec_v2
from app.rag.legal_v2.retrieve.colbert.models import ColbertHit, ColbertRetrievalResult
from app.rag.legal_v2.retrieve.colbert_hybrid import (
    EXPERIMENT_PROFILE_ID,
    retrieve_hybrid_plus_colbert,
)
from app.rag.legal_v2.retrieve.retriever import (
    LegalV2RetrievalResult,
    LegalV2RetrieverConfig,
)
from app.rag.retrieval.models import RetrievedChunk
from app.rag.retrieval.rrf import rrf_fuse


def _chunk(chunk_id: str, *, score: float, source: str, document_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        text=f"text-{chunk_id}",
        score=score,
        source=source,
        metadata={"document_id": document_id, "ecli": document_id},
    )


def test_rrf_fuse_accepts_three_ranked_lists() -> None:
    dense = [_chunk("c1", score=0.9, source="dense", document_id="ECLI:CZ:NS:2020:1")]
    bm25 = [_chunk("c2", score=5.0, source="bm25", document_id="ECLI:CZ:NS:2020:2")]
    colbert = [_chunk("c3", score=20.0, source="colbert", document_id="ECLI:CZ:NS:2020:3")]
    fused = rrf_fuse([dense, bm25, colbert], top_k=10, rrf_k=60)
    assert len(fused) == 3
    assert {c.id for c in fused} == {"c1", "c2", "c3"}
    # Equal ranks → equal RRF contribution; stable sort by id among ties.
    assert all(abs(c.score - (1.0 / 61.0)) < 1e-9 for c in fused)


@dataclass
class _FakeHybrid:
    config: LegalV2RetrieverConfig
    result: LegalV2RetrievalResult

    def retrieve(self, query_spec: Any) -> LegalV2RetrievalResult:  # noqa: ARG002
        return self.result


def test_retrieve_hybrid_plus_colbert_fuses_three_sources() -> None:
    config = LegalV2RetrieverConfig(
        dense_candidate_chunks=2,
        bm25_candidate_chunks=2,
        fused_candidate_chunks=10,
        candidate_documents=5,
    )
    dense = [
        _chunk("d1", score=0.9, source="dense", document_id="ECLI:CZ:NS:2020:1"),
        _chunk("d2", score=0.8, source="dense", document_id="ECLI:CZ:NS:2020:2"),
    ]
    bm25 = [
        _chunk("b1", score=4.0, source="bm25", document_id="ECLI:CZ:NS:2020:3"),
        _chunk("d1", score=3.5, source="bm25", document_id="ECLI:CZ:NS:2020:1"),
    ]
    base = LegalV2RetrievalResult(
        documents=[],
        dense_results=dense,
        bm25_results=bm25,
        fused_results=[],
        diagnostics={"collection": "coll_b", "bm25_index_id": "bm25_b"},
    )
    hybrid = _FakeHybrid(config=config, result=base)
    colbert_hits = ColbertRetrievalResult(
        hits=(
            ColbertHit(
                document_id="ECLI:CZ:NS:2020:4",
                chunk_id="cb1",
                rank=1,
                score=12.0,
                text="colbert text",
                metadata={"document_id": "ECLI:CZ:NS:2020:4", "ecli": "ECLI:CZ:NS:2020:4"},
            ),
        ),
        diagnostics={"latency_ms": 1.0},
    )
    colbert_retriever = AsyncMock()
    colbert_retriever.retrieve = AsyncMock(return_value=colbert_hits)

    query_spec = build_query_spec_v2("test query")

    result = asyncio.run(
        retrieve_hybrid_plus_colbert(
            hybrid_retriever=hybrid,  # type: ignore[arg-type]
            colbert_retriever=colbert_retriever,
            query_spec=query_spec,
            colbert_candidate_chunks=80,
        )
    )

    assert result.diagnostics["experiment_profile"] == EXPERIMENT_PROFILE_ID
    assert result.diagnostics["cross_encoder"] is False
    assert result.diagnostics["rrf_k"] == 60
    assert result.diagnostics["dense_candidate_chunks"] == 2
    assert result.diagnostics["bm25_candidate_chunks"] == 2
    assert result.diagnostics["colbert_candidate_chunks"] == 1
    assert len(result.fused_results) >= 3
    assert {doc.document_id for doc in result.documents}
    colbert_retriever.retrieve.assert_awaited_once()
