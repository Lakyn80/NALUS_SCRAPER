"""Deterministic passage selection from Stage 1 evidence."""

from __future__ import annotations

import hashlib
from typing import Sequence

from app.rag.legal_v2.rerank.models import (
    EvidenceChunkRecord,
    RerankCandidate,
    RerankPassage,
)
from app.rag.legal_v2.rerank.selectors.base import EvidencePassageSelector
from app.rag.legal_v2.rerank.selectors.names import FIRST_N_STAGE1_ORDER_V1
from app.rag.legal_v2.rerank.selectors.policy import get_evidence_passage_selector


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _passage_key(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).casefold().encode("utf-8")).hexdigest()


def select_passages_for_document(
    *,
    ecli: str,
    stage1_rank: int,
    passage_texts: Sequence[tuple[str, str]],
    max_passages: int,
) -> tuple[RerankPassage, ...]:
    """Select up to max_passages unique non-empty passages in given order.

    passage_texts: sequence of (chunk_id, text) in Stage 1 evidence order.
    CE-3 reference behavior (first_n_stage1_order_v1 core).
    """
    selected: list[RerankPassage] = []
    seen: set[str] = set()
    for chunk_id, raw_text in passage_texts:
        cleaned = _normalize_text(raw_text)
        if not cleaned:
            continue
        key = _passage_key(cleaned)
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            RerankPassage(
                ecli=ecli,
                text=cleaned,
                chunk_id=str(chunk_id or f"passage-{len(selected)}"),
                stage1_document_rank=stage1_rank,
                passage_index=len(selected),
                selection_slot=len(selected) + 1,
                selection_reason="stage1_order_primary",
            )
        )
        if len(selected) >= max_passages:
            break
    return tuple(selected)


def evidence_records_from_stage1_doc(doc: object) -> tuple[EvidenceChunkRecord, ...]:
    """Extract Stage-1 chunk provenance without depending on golden labels."""
    records: list[EvidenceChunkRecord] = []
    seen: set[str] = set()

    raw_evidence = list(getattr(doc, "chunk_evidence", None) or [])
    if raw_evidence:
        for item in raw_evidence:
            if isinstance(item, EvidenceChunkRecord):
                record = item
            elif isinstance(item, dict):
                channels = item.get("retrieval_channels") or ()
                record = EvidenceChunkRecord(
                    chunk_id=str(item.get("chunk_id") or ""),
                    text=str(item.get("text") or ""),
                    dense_rank=item.get("dense_rank"),
                    bm25_rank=item.get("bm25_rank"),
                    rrf_rank=item.get("rrf_rank"),
                    dense_score=item.get("dense_score"),
                    bm25_score=item.get("bm25_score"),
                    rrf_score=item.get("rrf_score"),
                    retrieval_channels=tuple(channels),
                    chunk_position=item.get("chunk_position"),
                    section=item.get("section"),
                    page=item.get("page"),
                )
            else:
                channels = getattr(item, "retrieval_channels", ()) or ()
                record = EvidenceChunkRecord(
                    chunk_id=str(getattr(item, "chunk_id", "") or ""),
                    text=str(getattr(item, "text", "") or ""),
                    dense_rank=getattr(item, "dense_rank", None),
                    bm25_rank=getattr(item, "bm25_rank", None),
                    rrf_rank=getattr(item, "rrf_rank", None),
                    dense_score=getattr(item, "dense_score", None),
                    bm25_score=getattr(item, "bm25_score", None),
                    rrf_score=getattr(item, "rrf_score", None),
                    retrieval_channels=tuple(channels),
                    chunk_position=getattr(item, "chunk_position", None),
                    section=getattr(item, "section", None),
                    page=getattr(item, "page", None),
                )
            chunk_id = str(record.chunk_id or "").strip()
            text = _normalize_text(record.text)
            if not chunk_id or not text or chunk_id in seen:
                continue
            seen.add(chunk_id)
            records.append(
                EvidenceChunkRecord(
                    chunk_id=chunk_id,
                    text=text,
                    dense_rank=record.dense_rank,
                    bm25_rank=record.bm25_rank,
                    rrf_rank=record.rrf_rank,
                    dense_score=record.dense_score,
                    bm25_score=record.bm25_score,
                    rrf_score=record.rrf_score,
                    retrieval_channels=tuple(record.retrieval_channels or ()),
                    chunk_position=record.chunk_position,
                    section=record.section,
                    page=record.page,
                )
            )
        return tuple(records)

    # Fallback: relevant_passages / paragraphs without channel ranks.
    passages_src = list(getattr(doc, "relevant_passages", None) or [])
    for index, passage in enumerate(passages_src):
        chunk_id = str(getattr(passage, "chunk_id", "") or f"passage-{index}")
        text = _normalize_text(str(getattr(passage, "text", "") or ""))
        if not text or chunk_id in seen:
            continue
        seen.add(chunk_id)
        channels = tuple(getattr(passage, "retrieval_channels", ()) or ())
        records.append(
            EvidenceChunkRecord(
                chunk_id=chunk_id,
                text=text,
                dense_rank=getattr(passage, "dense_rank", None),
                bm25_rank=getattr(passage, "bm25_rank", None),
                rrf_rank=getattr(passage, "rrf_rank", None),
                retrieval_channels=channels,
                chunk_position=getattr(passage, "chunk_position", None),
                section=getattr(passage, "section", None),
                page=getattr(passage, "page", None),
            )
        )
    if records:
        return tuple(records)

    for index, paragraph in enumerate(list(getattr(doc, "paragraphs", None) or [])):
        chunk_id = str(getattr(paragraph, "paragraph_id", "") or f"p-{index}")
        text = _normalize_text(
            str(
                getattr(paragraph, "normalized_text", None)
                or getattr(paragraph, "original_text", None)
                or ""
            )
        )
        if not text or chunk_id in seen:
            continue
        seen.add(chunk_id)
        section = getattr(paragraph.section_type, "value", None) or str(
            getattr(paragraph, "section_type", "") or ""
        ) or None
        records.append(
            EvidenceChunkRecord(
                chunk_id=chunk_id,
                text=text,
                chunk_position=getattr(paragraph, "paragraph_index", index),
                section=section,
            )
        )
    return tuple(records)


def build_candidates_from_stage1_docs(
    documents: Sequence[object],
    *,
    max_documents: int,
    max_passages: int,
    selector: EvidencePassageSelector | None = None,
    passage_selector_name: str | None = None,
) -> tuple[tuple[RerankCandidate, ...], tuple[str, ...]]:
    """Build CE candidates from Stage1DocumentResult-like objects."""
    active_selector = selector or get_evidence_passage_selector(
        passage_selector_name or FIRST_N_STAGE1_ORDER_V1
    )
    warnings: list[str] = []
    candidates: list[RerankCandidate] = []
    for doc in list(documents)[: max(0, max_documents)]:
        ecli = str(getattr(doc, "ecli", "") or "")
        stage1_rank = int(getattr(doc, "rank", 0) or 0)
        stage1_score = float(getattr(doc, "score", 0.0) or 0.0)
        evidence_pool = evidence_records_from_stage1_doc(doc)
        skeleton = RerankCandidate(
            ecli=ecli,
            stage1_rank=stage1_rank,
            stage1_score=stage1_score,
            passages=(),
            dense_rank=getattr(doc, "dense_rank", None),
            bm25_rank=getattr(doc, "bm25_rank", None),
            rrf_score=getattr(doc, "rrf_score", None),
            metadata=dict(getattr(doc, "metadata", None) or {}),
            evidence_pool=evidence_pool,
        )
        passages = tuple(active_selector.select(skeleton, limit=max_passages))
        if not passages:
            warnings.append(f"no_passages:{ecli}")
        candidates.append(
            RerankCandidate(
                ecli=ecli,
                stage1_rank=stage1_rank,
                stage1_score=stage1_score,
                passages=passages,
                dense_rank=getattr(doc, "dense_rank", None),
                bm25_rank=getattr(doc, "bm25_rank", None),
                rrf_score=getattr(doc, "rrf_score", None),
                metadata=dict(getattr(doc, "metadata", None) or {}),
                evidence_pool=evidence_pool,
            )
        )
    return tuple(candidates), tuple(warnings)
