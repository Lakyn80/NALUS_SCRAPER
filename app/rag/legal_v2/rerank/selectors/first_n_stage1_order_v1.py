"""CE-3 reference selector: first N unique passages in Stage-1 evidence order."""

from __future__ import annotations

import hashlib
from typing import Sequence

from app.rag.legal_v2.rerank.models import RerankCandidate, RerankPassage
from app.rag.legal_v2.rerank.selectors.names import FIRST_N_STAGE1_ORDER_V1

__all__ = ["FIRST_N_STAGE1_ORDER_V1", "FirstNStage1OrderSelectorV1"]


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _passage_key(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).casefold().encode("utf-8")).hexdigest()


class FirstNStage1OrderSelectorV1:
    policy_id = FIRST_N_STAGE1_ORDER_V1

    def select(
        self,
        candidate: RerankCandidate,
        *,
        limit: int,
    ) -> Sequence[RerankPassage]:
        evidence = list(candidate.evidence_pool or ())
        if evidence:
            pairs = [(item.chunk_id, item.text, item) for item in evidence]
        else:
            pairs = [(p.chunk_id, p.text, p) for p in candidate.passages]

        selected: list[RerankPassage] = []
        seen: set[str] = set()
        for chunk_id, raw_text, source in pairs:
            cleaned = _normalize_text(raw_text)
            if not cleaned:
                continue
            key = _passage_key(cleaned)
            if key in seen:
                continue
            seen.add(key)
            selected.append(
                RerankPassage(
                    ecli=candidate.ecli,
                    text=cleaned,
                    chunk_id=str(chunk_id or f"passage-{len(selected)}"),
                    stage1_document_rank=candidate.stage1_rank,
                    passage_index=len(selected),
                    selection_slot=len(selected) + 1,
                    selection_reason="stage1_order_primary",
                    dense_rank=getattr(source, "dense_rank", None),
                    bm25_rank=getattr(source, "bm25_rank", None),
                    rrf_rank=getattr(source, "rrf_rank", None),
                    retrieval_channels=tuple(
                        getattr(source, "retrieval_channels", ()) or ()
                    ),
                    chunk_position=getattr(source, "chunk_position", None),
                    section=getattr(source, "section", None),
                    page=getattr(source, "page", None),
                    requested_passages=limit,
                )
            )
            if len(selected) >= limit:
                break
        return tuple(
            RerankPassage(
                ecli=p.ecli,
                text=p.text,
                chunk_id=p.chunk_id,
                stage1_document_rank=p.stage1_document_rank,
                passage_index=p.passage_index,
                selection_slot=p.selection_slot,
                selection_reason=p.selection_reason,
                dense_rank=p.dense_rank,
                bm25_rank=p.bm25_rank,
                rrf_rank=p.rrf_rank,
                retrieval_channels=p.retrieval_channels,
                chunk_position=p.chunk_position,
                section=p.section,
                page=p.page,
                requested_passages=limit,
                selected_passages=len(selected),
            )
            for p in selected
        )
