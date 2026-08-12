"""Experimental async orchestration: BGE-M3 + BM25 + ColBERT → RRF.

Not a production/default profile. Does not change FAST/CE canonical pins.
Reuses ``LegalV2HybridRetriever`` dense/BM25 channels, ``ColbertRetriever``,
and the shared ``rrf_fuse`` N-list fusion helper.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from app.rag.legal_v2.evidence import CandidateEvidenceDocument
from app.rag.legal_v2.indexing import LEGAL_V2_PROFILE
from app.rag.legal_v2.query_spec import QuerySpecV2
from app.rag.legal_v2.retrieve.colbert.models import ColbertRetrievalResult
from app.rag.legal_v2.retrieve.colbert.retriever import ColbertRetriever
from app.rag.legal_v2.retrieve.retriever import (
    LegalV2HybridRetriever,
    LegalV2RetrievalResult,
    aggregate_legal_v2_documents,
)
from app.rag.retrieval.models import RetrievedChunk
from app.rag.retrieval.rrf import rrf_fuse

EXPERIMENT_PROFILE_ID = "hybrid_b_plus_colbert_rrf_v1"


@dataclass(frozen=True)
class ColbertHybridRetrievalResult:
    """Document-level result of experimental three-source RRF (no CE)."""

    documents: list[CandidateEvidenceDocument]
    dense_results: list[RetrievedChunk]
    bm25_results: list[RetrievedChunk]
    colbert_results: list[RetrievedChunk]
    fused_results: list[RetrievedChunk]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _retrieval_query(query_spec: QuerySpecV2) -> str:
    if query_spec.retrieval_queries:
        return " ".join(query_spec.retrieval_queries[:3])
    return query_spec.original_query


async def retrieve_hybrid_plus_colbert(
    *,
    hybrid_retriever: LegalV2HybridRetriever,
    colbert_retriever: ColbertRetriever,
    query_spec: QuerySpecV2,
    colbert_candidate_chunks: int,
    fused_candidate_chunks: int | None = None,
    candidate_documents: int | None = None,
    rrf_k: int | None = None,
) -> ColbertHybridRetrievalResult:
    """Run dense+BM25 (thread) and ColBERT (async) then fuse with RRF.

    Dense/BM25 candidate depths come from ``hybrid_retriever`` config (canonical
    FAST knobs). ColBERT is an additional ranked list only; no CE rerank.
    """
    if colbert_candidate_chunks < 1:
        raise ValueError("colbert_candidate_chunks must be >= 1")

    started = time.perf_counter()
    query = _retrieval_query(query_spec)
    resolved_rrf_k = int(LEGAL_V2_PROFILE.rrf_k if rrf_k is None else rrf_k)
    hybrid_cfg = hybrid_retriever.config
    resolved_fused = int(
        hybrid_cfg.fused_candidate_chunks
        if fused_candidate_chunks is None
        else fused_candidate_chunks
    )
    resolved_docs = int(
        hybrid_cfg.candidate_documents if candidate_documents is None else candidate_documents
    )

    hybrid_task = asyncio.to_thread(hybrid_retriever.retrieve, query_spec)
    colbert_task = colbert_retriever.retrieve(query, top_k=colbert_candidate_chunks)
    base: LegalV2RetrievalResult
    colbert: ColbertRetrievalResult
    base, colbert = await asyncio.gather(hybrid_task, colbert_task)

    colbert_chunks = colbert.as_retrieved_chunks()
    if not base.dense_results or not base.bm25_results:
        raise RuntimeError("Hybrid dense and BM25 indexes must both return candidates.")
    if not colbert_chunks:
        raise RuntimeError("ColBERT returned no candidates for hybrid fusion.")

    fused_started = time.perf_counter()
    fused = rrf_fuse(
        [base.dense_results, base.bm25_results, colbert_chunks],
        top_k=resolved_fused,
        rrf_k=resolved_rrf_k,
    )
    fused_ms = (time.perf_counter() - fused_started) * 1000.0

    documents = aggregate_legal_v2_documents(
        fused,
        dense=base.dense_results,
        bm25=base.bm25_results,
        query_spec=query_spec,
        limit=resolved_docs,
    )

    return ColbertHybridRetrievalResult(
        documents=documents,
        dense_results=list(base.dense_results),
        bm25_results=list(base.bm25_results),
        colbert_results=colbert_chunks,
        fused_results=fused,
        diagnostics={
            "experiment_profile": EXPERIMENT_PROFILE_ID,
            "cross_encoder": False,
            "rrf_k": resolved_rrf_k,
            "dense_candidate_chunks": len(base.dense_results),
            "bm25_candidate_chunks": len(base.bm25_results),
            "colbert_candidate_chunks": len(colbert_chunks),
            "fused_candidate_chunks": len(fused),
            "candidate_documents": len(documents),
            "requested_dense_candidate_chunks": hybrid_cfg.dense_candidate_chunks,
            "requested_bm25_candidate_chunks": hybrid_cfg.bm25_candidate_chunks,
            "requested_colbert_candidate_chunks": colbert_candidate_chunks,
            "requested_fused_candidate_chunks": resolved_fused,
            "collection": base.diagnostics.get("collection"),
            "bm25_index_id": base.diagnostics.get("bm25_index_id"),
            "dense_latency_ms": base.diagnostics.get("dense_latency_ms"),
            "bm25_latency_ms": base.diagnostics.get("bm25_latency_ms"),
            "colbert_latency_ms": colbert.diagnostics.get("latency_ms"),
            "rrf_latency_ms": fused_ms,
            "total_retrieval_latency_ms": (time.perf_counter() - started) * 1000.0,
            "document_dedupe": (
                "group_fused_chunks_by_document_id_best_chunk_rrf_plus_evidence_bonus"
            ),
        },
    )
