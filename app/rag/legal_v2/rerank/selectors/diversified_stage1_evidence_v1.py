"""Deterministic diversified Stage-1 evidence selector (CE-7 policy).

Policy ``diversified_stage1_evidence_v1`` fills up to seven slots:

1. rrf_primary
2. dense_primary
3. bm25_primary
4. rrf_secondary
5. dense_secondary
6. bm25_secondary
7. diversity_support

Missing slots are filled from a deterministic multi-channel fallback pool.
Near-duplicates are suppressed via token Jaccard / containment thresholds
declared centrally below. No randomness; no golden labels; no expected ECLI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from app.rag.legal_v2.rerank.models import (
    EvidenceChunkRecord,
    RerankCandidate,
    RerankPassage,
)
from app.rag.legal_v2.rerank.selectors.names import DIVERSIFIED_STAGE1_EVIDENCE_V1

__all__ = ["DIVERSIFIED_STAGE1_EVIDENCE_V1", "DiversifiedStage1EvidenceSelectorV1"]

# Central near-duplicate policy (do not scatter thresholds).
NEAR_DUPLICATE_JACCARD_THRESHOLD = 0.82
NEAR_DUPLICATE_CONTAINMENT_THRESHOLD = 0.90
NEAR_DUPLICATE_MIN_TOKENS = 8

# Structural diversity: treat positions within this distance as same region.
STRUCTURAL_NEAR_POSITION_DISTANCE = 2

# Each channel slot picks the strongest remaining unused non-duplicate chunk
# from that channel (primary then secondary naturally follow selection order).
_SLOT_PLAN: tuple[tuple[str, str], ...] = (
    ("rrf", "rrf_primary"),
    ("dense", "dense_primary"),
    ("bm25", "bm25_primary"),
    ("rrf", "rrf_secondary"),
    ("dense", "dense_secondary"),
    ("bm25", "bm25_secondary"),
    ("diversity", "diversity_support"),
)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_MISSING_RANK = 10**9


@dataclass(frozen=True)
class _ChunkView:
    chunk_id: str
    text: str
    dense_rank: int | None
    bm25_rank: int | None
    rrf_rank: int | None
    retrieval_channels: tuple[str, ...]
    chunk_position: int | None
    section: str | None
    page: int | None
    tokens: frozenset[str]


class DiversifiedStage1EvidenceSelectorV1:
    policy_id = DIVERSIFIED_STAGE1_EVIDENCE_V1

    def select(
        self,
        candidate: RerankCandidate,
        *,
        limit: int,
    ) -> Sequence[RerankPassage]:
        if limit < 1:
            return ()
        pool = _build_chunk_views(candidate)
        if not pool:
            return ()

        selected: list[_ChunkView] = []
        reasons: list[str] = []
        used_ids: set[str] = set()
        filtered_near_dupes = 0

        for channel, reason in _SLOT_PLAN:
            if len(selected) >= limit:
                break
            if channel == "diversity":
                pick, suppressed = _pick_diversity(pool, selected, used_ids)
            else:
                pick, suppressed = _pick_channel(
                    pool,
                    selected=selected,
                    used_ids=used_ids,
                    channel=channel,
                )
            filtered_near_dupes += suppressed
            slot_reason = reason
            if pick is None:
                pick, suppressed = _pick_fallback(pool, selected, used_ids)
                filtered_near_dupes += suppressed
                if pick is None:
                    continue
                slot_reason = f"fallback_after_{reason}"
            used_ids.add(pick.chunk_id)
            selected.append(pick)
            reasons.append(slot_reason)

        out: list[RerankPassage] = []
        for index, (chunk, reason) in enumerate(zip(selected, reasons)):
            out.append(
                RerankPassage(
                    ecli=candidate.ecli,
                    text=chunk.text,
                    chunk_id=chunk.chunk_id,
                    stage1_document_rank=candidate.stage1_rank,
                    passage_index=index,
                    selection_slot=index + 1,
                    selection_reason=reason,
                    dense_rank=chunk.dense_rank,
                    bm25_rank=chunk.bm25_rank,
                    rrf_rank=chunk.rrf_rank,
                    retrieval_channels=chunk.retrieval_channels,
                    chunk_position=chunk.chunk_position,
                    section=chunk.section,
                    page=chunk.page,
                    near_duplicate_filtered_count=filtered_near_dupes if index == 0 else 0,
                    requested_passages=limit,
                    selected_passages=len(selected),
                )
            )
        return tuple(out)


def _build_chunk_views(candidate: RerankCandidate) -> list[_ChunkView]:
    records: list[EvidenceChunkRecord] = list(candidate.evidence_pool or ())
    if not records and candidate.passages:
        records = [
            EvidenceChunkRecord(
                chunk_id=p.chunk_id,
                text=p.text,
                dense_rank=p.dense_rank,
                bm25_rank=p.bm25_rank,
                rrf_rank=p.rrf_rank,
                retrieval_channels=p.retrieval_channels,
                chunk_position=p.chunk_position,
                section=p.section,
                page=p.page,
            )
            for p in candidate.passages
        ]
    views: list[_ChunkView] = []
    seen: set[str] = set()
    for record in records:
        chunk_id = str(record.chunk_id or "").strip()
        text = " ".join(str(record.text or "").split()).strip()
        if not chunk_id or not text or chunk_id in seen:
            continue
        seen.add(chunk_id)
        channels = tuple(
            ch
            for ch in (record.retrieval_channels or ())
            if ch in {"rrf", "dense", "bm25"}
        )
        if not channels:
            inferred: list[str] = []
            if record.rrf_rank is not None:
                inferred.append("rrf")
            if record.dense_rank is not None:
                inferred.append("dense")
            if record.bm25_rank is not None:
                inferred.append("bm25")
            channels = tuple(inferred)
        views.append(
            _ChunkView(
                chunk_id=chunk_id,
                text=text,
                dense_rank=record.dense_rank,
                bm25_rank=record.bm25_rank,
                rrf_rank=record.rrf_rank,
                retrieval_channels=channels,
                chunk_position=record.chunk_position,
                section=record.section,
                page=record.page,
                tokens=_tokenize(text),
            )
        )
    return views


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(token.casefold() for token in _TOKEN_RE.findall(text))


def _rank_key(value: int | None) -> int:
    return value if value is not None else _MISSING_RANK


def _channel_rank(chunk: _ChunkView, channel: str) -> int | None:
    if channel == "rrf":
        return chunk.rrf_rank
    if channel == "dense":
        return chunk.dense_rank
    if channel == "bm25":
        return chunk.bm25_rank
    return None


def _is_near_duplicate(a: _ChunkView, b: _ChunkView) -> bool:
    if a.chunk_id == b.chunk_id:
        return True
    if not a.tokens or not b.tokens:
        return a.text.casefold() == b.text.casefold()
    if (
        len(a.tokens) < NEAR_DUPLICATE_MIN_TOKENS
        and len(b.tokens) < NEAR_DUPLICATE_MIN_TOKENS
    ):
        return a.text.casefold() == b.text.casefold()
    inter = len(a.tokens & b.tokens)
    union = len(a.tokens | b.tokens)
    jaccard = inter / union if union else 0.0
    if jaccard >= NEAR_DUPLICATE_JACCARD_THRESHOLD:
        return True
    smaller = min(len(a.tokens), len(b.tokens))
    containment = inter / smaller if smaller else 0.0
    if containment >= NEAR_DUPLICATE_CONTAINMENT_THRESHOLD and smaller >= NEAR_DUPLICATE_MIN_TOKENS:
        return True
    return False


def _conflicts(chunk: _ChunkView, selected: Sequence[_ChunkView]) -> bool:
    return any(_is_near_duplicate(chunk, item) for item in selected)


def _fallback_sort_key(chunk: _ChunkView) -> tuple:
    channel_count = sum(
        1
        for rank in (chunk.rrf_rank, chunk.dense_rank, chunk.bm25_rank)
        if rank is not None
    )
    position = chunk.chunk_position if chunk.chunk_position is not None else _MISSING_RANK
    return (
        -channel_count,
        _rank_key(chunk.rrf_rank),
        _rank_key(chunk.dense_rank),
        _rank_key(chunk.bm25_rank),
        position,
        chunk.chunk_id,
    )


def _pick_channel(
    pool: Sequence[_ChunkView],
    *,
    selected: Sequence[_ChunkView],
    used_ids: set[str],
    channel: str,
) -> tuple[_ChunkView | None, int]:
    ranked = sorted(
        (
            chunk
            for chunk in pool
            if chunk.chunk_id not in used_ids and _channel_rank(chunk, channel) is not None
        ),
        key=lambda chunk: (
            _rank_key(_channel_rank(chunk, channel)),
            _fallback_sort_key(chunk),
        ),
    )
    suppressed = 0
    for chunk in ranked:
        if _conflicts(chunk, selected):
            suppressed += 1
            continue
        return chunk, suppressed
    return None, suppressed


def _min_position_distance(chunk: _ChunkView, selected: Sequence[_ChunkView]) -> int:
    if chunk.chunk_position is None or not selected:
        return _MISSING_RANK
    distances = []
    for item in selected:
        if item.chunk_position is None:
            continue
        distances.append(abs(chunk.chunk_position - item.chunk_position))
    return min(distances) if distances else _MISSING_RANK


def _diversity_sort_key(chunk: _ChunkView, selected: Sequence[_ChunkView]) -> tuple:
    # Prefer structurally distant evidence among similarly strong remaining chunks.
    # Strength uses the same fallback ordering (retrieval evidence only).
    strength = _fallback_sort_key(chunk)
    distance = _min_position_distance(chunk, selected)
    # Prefer larger distance, but never let weak evidence beat strong solely on distance:
    # compare strength first (already good), then prefer distance when strength ties closely.
    return (
        strength[0],
        strength[1],
        strength[2],
        strength[3],
        -distance if distance != _MISSING_RANK else 0,
        strength[4],
        strength[5],
    )


def _pick_diversity(
    pool: Sequence[_ChunkView],
    selected: Sequence[_ChunkView],
    used_ids: set[str],
) -> tuple[_ChunkView | None, int]:
    candidates = [
        chunk for chunk in pool if chunk.chunk_id not in used_ids and not _conflicts(chunk, selected)
    ]
    suppressed = sum(
        1
        for chunk in pool
        if chunk.chunk_id not in used_ids and _conflicts(chunk, selected)
    )
    if not candidates:
        return None, suppressed
    # Prefer chunks outside the clustered region of already-selected positions when
    # their retrieval strength is not worse than the strongest remaining candidate.
    strongest = min(candidates, key=_fallback_sort_key)
    strong_band = [
        chunk
        for chunk in candidates
        if _fallback_sort_key(chunk)[:4] == _fallback_sort_key(strongest)[:4]
        or (
            _rank_key(chunk.rrf_rank) <= _rank_key(strongest.rrf_rank) + 5
            and _fallback_sort_key(chunk)[0] <= _fallback_sort_key(strongest)[0]
        )
    ]
    pool_for_pick = strong_band or candidates
    pick = min(pool_for_pick, key=lambda chunk: _diversity_sort_key(chunk, selected))
    # If selected cluster is tight and a distinct distant chunk exists in band, prefer it.
    if selected and pick.chunk_position is not None:
        clustered = all(
            item.chunk_position is not None
            and abs(item.chunk_position - selected[0].chunk_position)  # type: ignore[arg-type]
            <= STRUCTURAL_NEAR_POSITION_DISTANCE * len(selected)
            for item in selected
            if item.chunk_position is not None
        )
        if clustered:
            distant = [
                chunk
                for chunk in pool_for_pick
                if chunk.chunk_position is not None
                and _min_position_distance(chunk, selected) > STRUCTURAL_NEAR_POSITION_DISTANCE
            ]
            if distant:
                pick = min(distant, key=lambda chunk: _diversity_sort_key(chunk, selected))
    return pick, suppressed


def _pick_fallback(
    pool: Sequence[_ChunkView],
    selected: Sequence[_ChunkView],
    used_ids: set[str],
) -> tuple[_ChunkView | None, int]:
    ranked = sorted(
        (chunk for chunk in pool if chunk.chunk_id not in used_ids),
        key=_fallback_sort_key,
    )
    suppressed = 0
    for chunk in ranked:
        if _conflicts(chunk, selected):
            suppressed += 1
            continue
        return chunk, suppressed
    return None, suppressed
