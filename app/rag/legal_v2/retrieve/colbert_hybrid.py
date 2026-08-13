"""Experimental async orchestration: BGE-M3 + BM25 + ColBERT → RRF (+ optional CE).

Not a production/default profile. Does not change FAST/CE canonical pins.
Reuses ``LegalV2HybridRetriever`` dense/BM25 channels, ``ColbertRetriever``,
shared ``rrf_fuse``, document aggregation, and the existing CE reranker service.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from app.rag.legal_v2.evidence import CandidateEvidenceDocument
from app.rag.legal_v2.identity import ecli_key, is_valid_ecli, normalize_ecli, resolve_production_document_id
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
EXPERIMENT_CE_PROFILE_ID = "hybrid_b_plus_colbert_rrf_ce7_v1"


@dataclass(frozen=True)
class ColbertHybridRetrievalResult:
    """Document-level result of experimental three-source RRF (no CE)."""

    documents: list[CandidateEvidenceDocument]
    dense_results: list[RetrievedChunk]
    bm25_results: list[RetrievedChunk]
    colbert_results: list[RetrievedChunk]
    fused_results: list[RetrievedChunk]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ColbertHybridCeRetrievalResult:
    """Hybrid Stage-1 shortlist followed by canonical CE-7 rerank."""

    documents: list[Any]
    stage1: ColbertHybridRetrievalResult
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _retrieval_query(query_spec: QuerySpecV2) -> str:
    if query_spec.retrieval_queries:
        return " ".join(query_spec.retrieval_queries[:3])
    return query_spec.original_query


def _chunk_document_id(chunk: RetrievedChunk) -> str:
    metadata = dict(chunk.metadata or {})
    resolved = resolve_production_document_id(metadata)
    if resolved:
        return resolved
    return str(metadata.get("document_id") or "").strip()


def enrich_documents_with_colbert_evidence(
    documents: Sequence[CandidateEvidenceDocument],
    colbert_chunks: Sequence[RetrievedChunk],
    *,
    evidence_pool_limit: int,
) -> list[CandidateEvidenceDocument]:
    """Merge ColBERT chunk hits into Stage-1 ``chunk_evidence`` for CE selection.

    Does not change document ranking. Canonical CE selector remains unchanged;
    ColBERT-only passages enter the evidence pool (diversity fallback can use them).
    """
    if evidence_pool_limit < 1:
        raise ValueError("evidence_pool_limit must be >= 1")

    by_doc: dict[str, list[tuple[int, RetrievedChunk]]] = {}
    for rank, chunk in enumerate(colbert_chunks, start=1):
        doc_id = _chunk_document_id(chunk)
        if not doc_id:
            continue
        key = ecli_key(doc_id) if is_valid_ecli(doc_id) else doc_id
        by_doc.setdefault(key, []).append((rank, chunk))

    enriched: list[CandidateEvidenceDocument] = []
    for doc in documents:
        doc_key = (
            ecli_key(doc.document_id)
            if is_valid_ecli(doc.document_id)
            else doc.document_id
        )
        evidence = [dict(item) for item in list(doc.chunk_evidence or [])]
        seen = {
            str(item.get("chunk_id") or "")
            for item in evidence
            if str(item.get("chunk_id") or "").strip()
        }
        # Tag existing fused evidence that also appears in ColBERT.
        colbert_for_doc = by_doc.get(doc_key) or []
        colbert_rank_by_id = {chunk.id: rank for rank, chunk in colbert_for_doc}
        for item in evidence:
            chunk_id = str(item.get("chunk_id") or "")
            c_rank = colbert_rank_by_id.get(chunk_id)
            if c_rank is None:
                continue
            item["colbert_rank"] = c_rank
            channels = list(item.get("retrieval_channels") or [])
            if "colbert" not in channels:
                channels.append("colbert")
            item["retrieval_channels"] = channels

        # Append ColBERT-only chunks so CE can see late-interaction evidence.
        for rank, chunk in colbert_for_doc:
            chunk_id = str(chunk.id or "")
            text = str(chunk.text or "").strip()
            if not chunk_id or not text or chunk_id in seen:
                continue
            seen.add(chunk_id)
            metadata = dict(chunk.metadata or {})
            evidence.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "dense_rank": None,
                    "bm25_rank": None,
                    "rrf_rank": None,
                    "colbert_rank": rank,
                    "dense_score": None,
                    "bm25_score": None,
                    "rrf_score": None,
                    "section": str(metadata.get("section_type") or "") or None,
                    "page": metadata.get("page"),
                    "chunk_position": int(
                        metadata.get("source_order")
                        or metadata.get("chunk_index")
                        or rank
                    ),
                    "retrieval_channels": ["colbert"],
                }
            )
            if len(evidence) >= evidence_pool_limit:
                break
        enriched.append(
            replace(
                doc,
                chunk_evidence=evidence[:evidence_pool_limit],
            )
        )
    return enriched


def stage1_docs_from_hybrid_documents(
    documents: Sequence[CandidateEvidenceDocument],
    *,
    limit: int,
    evidence_limit: int,
) -> list[Any]:
    """Adapt hybrid documents to the Stage-1 shape expected by CE ``rerank``."""
    from app.rag.legal_v2.retrieve.case_similarity_search import (
        Stage1DocumentResult,
        Stage1Passage,
    )

    out: list[Stage1DocumentResult] = []
    for index, doc in enumerate(list(documents)[: max(0, limit)], start=1):
        raw_id = str(doc.document_id or "")
        meta = dict(doc.metadata or {})
        ecli_raw = str(meta.get("ecli") or raw_id)
        if not ecli_raw or not is_valid_ecli(ecli_raw):
            continue
        ecli = normalize_ecli(ecli_raw)
        chunk_evidence = [
            dict(item)
            for item in list(doc.chunk_evidence or [])[: max(0, evidence_limit)]
            if isinstance(item, dict)
        ]
        passages: list[Stage1Passage] = []
        for item in chunk_evidence:
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            passages.append(
                Stage1Passage(
                    text=text,
                    chunk_id=str(item.get("chunk_id") or f"p-{len(passages)}"),
                    section=item.get("section"),
                    page=item.get("page"),
                    score=item.get("rrf_score"),
                    dense_rank=item.get("dense_rank"),
                    bm25_rank=item.get("bm25_rank"),
                    rrf_rank=item.get("rrf_rank"),
                    retrieval_channels=tuple(item.get("retrieval_channels") or ()),
                    chunk_position=item.get("chunk_position"),
                )
            )
        out.append(
            Stage1DocumentResult(
                rank=index,
                document_id=ecli,
                canonical_document_id=ecli,
                ecli=ecli,
                court=meta.get("court"),
                case_number=meta.get("case_number"),
                decision_date=meta.get("decision_date"),
                document_type=meta.get("document_type"),
                score=float(doc.score),
                relevant_passages=passages,
                dense_rank=doc.dense_rank,
                bm25_rank=doc.bm25_rank,
                rrf_score=doc.rrf_score,
                metadata=meta,
                stage1_rank=index,
                stage1_score=float(doc.score),
                chunk_evidence=chunk_evidence,
            )
        )
    return out


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
        colbert=colbert_chunks,
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


async def retrieve_hybrid_plus_colbert_ce(
    *,
    hybrid_retriever: LegalV2HybridRetriever,
    colbert_retriever: ColbertRetriever,
    ce_service: Any,
    query: str,
    query_spec: QuerySpecV2,
    colbert_candidate_chunks: int,
    fused_candidate_chunks: int | None = None,
    candidate_documents: int | None = None,
    rrf_k: int | None = None,
    ce_candidate_documents: int = 30,
    evidence_pool_limit: int = 40,
) -> ColbertHybridCeRetrievalResult:
    """Hybrid three-source RRF shortlist → enrich ColBERT evidence → CE-7."""
    started = time.perf_counter()
    stage1 = await retrieve_hybrid_plus_colbert(
        hybrid_retriever=hybrid_retriever,
        colbert_retriever=colbert_retriever,
        query_spec=query_spec,
        colbert_candidate_chunks=colbert_candidate_chunks,
        fused_candidate_chunks=fused_candidate_chunks,
        candidate_documents=max(
            int(candidate_documents or hybrid_retriever.config.candidate_documents),
            int(ce_candidate_documents),
        ),
        rrf_k=rrf_k,
    )
    enriched = enrich_documents_with_colbert_evidence(
        stage1.documents,
        stage1.colbert_results,
        evidence_pool_limit=int(evidence_pool_limit),
    )
    stage1_docs = stage1_docs_from_hybrid_documents(
        enriched,
        limit=int(ce_candidate_documents),
        evidence_limit=int(evidence_pool_limit),
    )
    ce_started = time.perf_counter()
    reranked = await asyncio.to_thread(
        ce_service.rerank,
        query,
        stage1_docs,
        require_success=True,
    )
    ce_ms = (time.perf_counter() - ce_started) * 1000.0
    diagnostics = {
        **dict(stage1.diagnostics),
        "experiment_profile": EXPERIMENT_CE_PROFILE_ID,
        "cross_encoder": True,
        "ce_candidate_documents": len(stage1_docs),
        "ce_latency_ms": ce_ms,
        "evidence_pool_limit": int(evidence_pool_limit),
        "total_pipeline_latency_ms": (time.perf_counter() - started) * 1000.0,
    }
    return ColbertHybridCeRetrievalResult(
        documents=list(getattr(reranked, "documents", []) or []),
        stage1=replace(
            stage1,
            documents=enriched,
            diagnostics={**dict(stage1.diagnostics), "colbert_evidence_enriched": True},
        ),
        diagnostics=diagnostics,
    )
