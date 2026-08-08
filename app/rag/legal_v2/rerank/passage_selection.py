"""Deterministic passage selection from Stage 1 evidence."""

from __future__ import annotations

import hashlib
from typing import Sequence

from app.rag.legal_v2.rerank.models import RerankCandidate, RerankPassage


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
            )
        )
        if len(selected) >= max_passages:
            break
    return tuple(selected)


def build_candidates_from_stage1_docs(
    documents: Sequence[object],
    *,
    max_documents: int,
    max_passages: int,
) -> tuple[tuple[RerankCandidate, ...], tuple[str, ...]]:
    """Build CE candidates from Stage1DocumentResult-like objects."""
    warnings: list[str] = []
    candidates: list[RerankCandidate] = []
    for doc in list(documents)[: max(0, max_documents)]:
        ecli = str(getattr(doc, "ecli", "") or "")
        stage1_rank = int(getattr(doc, "rank", 0) or 0)
        stage1_score = float(getattr(doc, "score", 0.0) or 0.0)
        passages_src = list(getattr(doc, "relevant_passages", None) or [])
        pairs = [
            (
                str(getattr(p, "chunk_id", "") or ""),
                str(getattr(p, "text", "") or ""),
            )
            for p in passages_src
        ]
        passages = select_passages_for_document(
            ecli=ecli,
            stage1_rank=stage1_rank,
            passage_texts=pairs,
            max_passages=max_passages,
        )
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
            )
        )
    return tuple(candidates), tuple(warnings)
