from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.rag.legal_v2.evidence import CandidateEvidenceDocument
from app.rag.legal_v2.indexing import (
    LEGAL_V2_BM25_INDEX_ID,
    LEGAL_V2_COLLECTION_NAME,
    LEGAL_V2_PROFILE,
)
from app.rag.legal_v2.models import LegalParagraph, MetadataProvenance, SectionType
from app.rag.legal_v2.query_spec import QuerySpecV2
from app.rag.retrieval.bm25_sidecar import Bm25Sidecar
from app.rag.retrieval.models import RetrievedChunk
from app.rag.retrieval.production_profile import ProductionRetrievalConfig
from app.rag.retrieval.qdrant_dense_store import QdrantDenseStore
from app.rag.retrieval.rrf import rrf_fuse

logger = get_logger(__name__)


@dataclass(frozen=True)
class LegalV2RetrieverConfig:
    qdrant_collection: str = LEGAL_V2_COLLECTION_NAME
    bm25_sidecar_path: Path = Path("storage/rag/bm25/nalus_legal_paragraph_bm25_v2.sqlite")
    bm25_index_id: str = LEGAL_V2_BM25_INDEX_ID
    model_path: str = "/app/models/BAAI/bge-m3"
    dense_candidate_chunks: int = 80
    bm25_candidate_chunks: int = 80
    fused_candidate_chunks: int = 120
    candidate_documents: int = 40
    returned_verified_documents: int = 10
    evidence_windows_per_constraint: int = 2

    def validate(self) -> None:
        for field_name in (
            "dense_candidate_chunks",
            "bm25_candidate_chunks",
            "fused_candidate_chunks",
            "candidate_documents",
            "returned_verified_documents",
            "evidence_windows_per_constraint",
        ):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive.")


@dataclass(frozen=True)
class LegalV2RetrievalResult:
    documents: list[CandidateEvidenceDocument]
    dense_results: list[RetrievedChunk]
    bm25_results: list[RetrievedChunk]
    fused_results: list[RetrievedChunk]
    diagnostics: dict[str, Any] = field(default_factory=dict)


class LegalV2HybridRetriever:
    def __init__(
        self,
        *,
        dense_store: Any,
        bm25_sidecar: Bm25Sidecar,
        config: LegalV2RetrieverConfig,
    ) -> None:
        config.validate()
        self._dense = dense_store
        self._bm25 = bm25_sidecar
        self._config = config

    def retrieve(self, query_spec: QuerySpecV2) -> LegalV2RetrievalResult:
        started = time.perf_counter()
        query = _retrieval_query(query_spec)
        dense_started = time.perf_counter()
        dense = self._dense.search(query, top_k=self._config.dense_candidate_chunks)
        dense_ms = _elapsed_ms(dense_started)
        trace_event(logger, "legal_v2.dense_retrieval.completed", result_count=len(dense))
        bm25_started = time.perf_counter()
        bm25 = self._bm25.search(query, top_k=self._config.bm25_candidate_chunks)
        bm25_ms = _elapsed_ms(bm25_started)
        trace_event(logger, "legal_v2.bm25_retrieval.completed", result_count=len(bm25))
        if not dense or not bm25:
            raise RuntimeError("Legal v2 dense and BM25 indexes must both return candidates.")
        fused_started = time.perf_counter()
        fused = rrf_fuse([dense, bm25], top_k=self._config.fused_candidate_chunks, rrf_k=LEGAL_V2_PROFILE.rrf_k)
        fused_ms = _elapsed_ms(fused_started)
        trace_event(logger, "legal_v2.rrf_fusion.completed", result_count=len(fused))
        documents = _aggregate_documents(
            fused,
            dense=dense,
            bm25=bm25,
            query_spec=query_spec,
            limit=self._config.candidate_documents,
        )
        trace_event(logger, "legal_v2.document_aggregation.completed", document_count=len(documents))
        return LegalV2RetrievalResult(
            documents=documents,
            dense_results=dense,
            bm25_results=bm25,
            fused_results=fused,
            diagnostics={
                "dense_candidate_chunks": len(dense),
                "bm25_candidate_chunks": len(bm25),
                "fused_candidate_chunks": len(fused),
                "candidate_documents": len(documents),
                "dense_latency_ms": dense_ms,
                "bm25_latency_ms": bm25_ms,
                "rrf_latency_ms": fused_ms,
                "total_retrieval_latency_ms": _elapsed_ms(started),
                "collection": self._config.qdrant_collection,
                "bm25_index_id": self._config.bm25_index_id,
            },
        )


def legal_v2_retriever_config_from_env() -> LegalV2RetrieverConfig:
    return LegalV2RetrieverConfig(
        qdrant_collection=os.getenv("NALUS_LEGAL_V2_QDRANT_COLLECTION", LEGAL_V2_COLLECTION_NAME),
        bm25_sidecar_path=Path(os.getenv("NALUS_LEGAL_V2_BM25_SIDECAR_PATH", "storage/rag/bm25/nalus_legal_paragraph_bm25_v2.sqlite")),
        bm25_index_id=os.getenv("NALUS_LEGAL_V2_BM25_INDEX_ID", LEGAL_V2_BM25_INDEX_ID),
        model_path=os.getenv("EMBEDDING_MODEL_NAME", "/app/models/BAAI/bge-m3"),
        dense_candidate_chunks=_int_env("NALUS_LEGAL_V2_DENSE_CANDIDATE_CHUNKS", 80),
        bm25_candidate_chunks=_int_env("NALUS_LEGAL_V2_BM25_CANDIDATE_CHUNKS", 80),
        fused_candidate_chunks=_int_env("NALUS_LEGAL_V2_FUSED_CANDIDATE_CHUNKS", 120),
        candidate_documents=_int_env("NALUS_LEGAL_V2_CANDIDATE_DOCUMENTS", 50),
        returned_verified_documents=_int_env("NALUS_LEGAL_V2_RETURNED_VERIFIED_DOCUMENTS", 10),
        evidence_windows_per_constraint=_int_env("NALUS_LEGAL_V2_EVIDENCE_WINDOWS_PER_CONSTRAINT", 2),
    )


def build_live_legal_v2_retriever(client: Any, embedder: Any, config: LegalV2RetrieverConfig) -> LegalV2HybridRetriever:
    prod_config = ProductionRetrievalConfig(
        profile=LEGAL_V2_PROFILE,
        qdrant_collection=config.qdrant_collection,
        bm25_sidecar_path=config.bm25_sidecar_path,
        bm25_index_id=config.bm25_index_id,
        model_path=config.model_path,
        local_files_only=True,
        trust_remote_code=False,
        device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        candidate_multiplier=1,
        min_candidate_count=1,
        max_candidate_count=max(config.dense_candidate_chunks, config.bm25_candidate_chunks),
        lexical_filter_enabled=False,
    )
    dense = QdrantDenseStore(client=client, embedder=embedder, config=prod_config)
    bm25 = Bm25Sidecar(
        config.bm25_sidecar_path,
        k1=LEGAL_V2_PROFILE.bm25_k1,
        b=LEGAL_V2_PROFILE.bm25_b,
        index_id=config.bm25_index_id,
    )
    return LegalV2HybridRetriever(dense_store=dense, bm25_sidecar=bm25, config=config)


def _retrieval_query(query_spec: QuerySpecV2) -> str:
    return " ".join(query_spec.retrieval_queries[:3]) if query_spec.retrieval_queries else query_spec.original_query


def _aggregate_documents(
    fused: list[RetrievedChunk],
    *,
    dense: list[RetrievedChunk],
    bm25: list[RetrievedChunk],
    query_spec: QuerySpecV2,
    limit: int,
) -> list[CandidateEvidenceDocument]:
    from app.rag.legal_v2.identity import ecli_key, is_valid_ecli

    dense_rank = {chunk.id: index for index, chunk in enumerate(dense, start=1)}
    bm25_rank = {chunk.id: index for index, chunk in enumerate(bm25, start=1)}
    rrf_rank = {chunk.id: index for index, chunk in enumerate(fused, start=1)}
    grouped: dict[str, list[RetrievedChunk]] = {}
    display_ids: dict[str, str] = {}
    for chunk in fused:
        document_id = _document_id(chunk)
        if not document_id:
            continue
        group_key = ecli_key(document_id) if is_valid_ecli(document_id) else document_id
        display_ids.setdefault(group_key, document_id)
        grouped.setdefault(group_key, []).append(chunk)
    documents: list[CandidateEvidenceDocument] = []
    for group_key, chunks in grouped.items():
        document_id = display_ids[group_key]
        ordered = sorted(chunks, key=lambda chunk: int((chunk.metadata or {}).get("source_order") or (chunk.metadata or {}).get("chunk_index") or 0))
        paragraphs = _paragraphs_from_chunks(document_id, ordered)
        best = max(chunks, key=lambda chunk: chunk.score)
        evidence_score = _document_evidence_score(chunks, query_spec)
        chunk_evidence = _chunk_evidence_records(
            ordered,
            dense_rank=dense_rank,
            bm25_rank=bm25_rank,
            rrf_rank=rrf_rank,
        )
        documents.append(
            CandidateEvidenceDocument(
                document_id=document_id,
                metadata=_safe_metadata(best.metadata),
                paragraphs=paragraphs,
                score=best.score + evidence_score,
                dense_rank=min((dense_rank.get(chunk.id, 999999) for chunk in chunks), default=None),
                bm25_rank=min((bm25_rank.get(chunk.id, 999999) for chunk in chunks), default=None),
                rrf_score=float(best.metadata.get("rrf_score") or best.score),
                chunk_ids=[chunk.id for chunk in ordered],
                chunk_evidence=chunk_evidence,
            )
        )
    documents.sort(key=lambda doc: (-doc.score, _rank_value(doc.dense_rank), _rank_value(doc.bm25_rank), doc.document_id))
    return documents[:limit]


def _document_evidence_score(chunks: list[RetrievedChunk], query_spec: QuerySpecV2) -> float:
    text = " ".join(chunk.text for chunk in chunks).lower()
    hard_coverage = _constraint_coverage(text, query_spec.hard_constraints)
    soft_coverage = _constraint_coverage(text, query_spec.soft_constraints)
    relation_bonus = 0.0
    for relation in query_spec.relations:
        relation_terms = [
            relation.action,
            *[value for value in relation.qualifiers.values() if value],
        ]
        if all(_any_token_matches(text, value) for value in relation_terms if value):
            relation_bonus += 0.02
    multi_passage_bonus = min(0.02, max(0, len(chunks) - 1) * 0.004)
    return (hard_coverage * 0.08) + (soft_coverage * 0.02) + relation_bonus + multi_passage_bonus


def _constraint_coverage(text: str, constraints: list[Any]) -> float:
    if not constraints:
        return 0.0
    covered = sum(1 for constraint in constraints if _any_token_matches(text, constraint.normalized_value))
    return covered / len(constraints)


def _any_token_matches(text: str, value: str) -> bool:
    return any(token in text for token in value.lower().split() if len(token) >= 4)


def _rank_value(value: int | None) -> int:
    return value if value is not None else 999999


def _chunk_evidence_records(
    ordered: list[RetrievedChunk],
    *,
    dense_rank: dict[str, int],
    bm25_rank: dict[str, int],
    rrf_rank: dict[str, int],
) -> list[dict[str, Any]]:
    """Bounded per-chunk Stage-1 provenance for CE passage selection."""
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, chunk in enumerate(ordered):
        chunk_id = str(chunk.id or "")
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        metadata = dict(chunk.metadata or {})
        text = str(chunk.text or "").strip()
        if not text:
            continue
        d_rank = dense_rank.get(chunk_id)
        b_rank = bm25_rank.get(chunk_id)
        r_rank = rrf_rank.get(chunk_id)
        channels: list[str] = []
        if r_rank is not None:
            channels.append("rrf")
        if d_rank is not None:
            channels.append("dense")
        if b_rank is not None:
            channels.append("bm25")
        records.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "dense_rank": d_rank,
                "bm25_rank": b_rank,
                "rrf_rank": r_rank,
                "dense_score": None,
                "bm25_score": None,
                "rrf_score": float(metadata.get("rrf_score") or chunk.score)
                if r_rank is not None
                else None,
                "section": str(metadata.get("section_type") or "") or None,
                "page": metadata.get("page"),
                "chunk_position": int(
                    metadata.get("source_order")
                    or metadata.get("chunk_index")
                    or position
                ),
                "retrieval_channels": channels,
            }
        )
    return records


def _paragraphs_from_chunks(document_id: str, chunks: list[RetrievedChunk]) -> list[LegalParagraph]:
    paragraphs: list[LegalParagraph] = []
    seen: set[str] = set()
    for fallback_index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata or {})
        raw_paragraph_texts = metadata.get("paragraph_texts")
        paragraph_texts: dict[str, Any] = (
            raw_paragraph_texts if isinstance(raw_paragraph_texts, dict) else {}
        )
        paragraph_ids = metadata.get("paragraph_ids") if isinstance(metadata.get("paragraph_ids"), list) else []
        if not paragraph_ids:
            paragraph_ids = [str(metadata.get("paragraph_id") or chunk.id)]
            paragraph_texts = {paragraph_ids[0]: chunk.text}
        for paragraph_id in paragraph_ids:
            paragraph_id = str(paragraph_id)
            if paragraph_id in seen:
                continue
            seen.add(paragraph_id)
            section_type = _section_type(metadata.get("section_type"))
            paragraphs.append(
                LegalParagraph(
                    document_id=document_id,
                    paragraph_id=paragraph_id,
                    paragraph_index=len(paragraphs),
                    original_text=str(paragraph_texts.get(paragraph_id) or chunk.text),
                    normalized_text=str(paragraph_texts.get(paragraph_id) or chunk.text),
                    section_type=section_type,
                    start_offset=int(metadata.get("start_offset") or 0),
                    end_offset=int(metadata.get("end_offset") or len(chunk.text)),
                    source_order=int(metadata.get("source_order") or fallback_index),
                    heading_context=[str(item) for item in metadata.get("heading_context") or []],
                    is_boilerplate=bool(metadata.get("is_boilerplate", False)),
                    is_citation_block=bool(metadata.get("is_citation_block", False)),
                    language=str(metadata.get("language") or "cs"),
                    metadata_provenance=MetadataProvenance(source=str(metadata.get("source") or "v2_index"), extraction_method="qdrant_payload"),
                )
            )
    return sorted(paragraphs, key=lambda paragraph: (paragraph.source_order, paragraph.paragraph_id))


def _document_id(chunk: RetrievedChunk) -> str:
    metadata = chunk.metadata or {}
    from app.rag.legal_v2.identity import ecli_key, resolve_production_document_id

    resolved = resolve_production_document_id(metadata)
    # Prefer stable casefolded ECLI key for aggregation grouping, but return
    # the normalized display form so downstream consumers keep the literal ECLI.
    if resolved:
        return resolved
    return ""


def _section_type(value: Any) -> SectionType:
    try:
        return SectionType(str(value))
    except ValueError:
        return SectionType.OTHER


def _safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    forbidden = {"paragraph_texts", "paragraph_original_texts", "text", "chunk_text"}
    return {str(key): value for key, value in metadata.items() if key not in forbidden}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
