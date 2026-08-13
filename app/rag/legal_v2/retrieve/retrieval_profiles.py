"""Request-level retrieval profiles for Legal v2 Stage 1 (+ ColBERT / CE).

Product tiers (pinned after golden quality + latency benchmarks):

```text
FAST      = A hybrid          (BGE-M3 + BM25 + RRF)
BALANCED  = B + ColBERT       (B dense/BM25 + ColBERT → RRF; no CE)
PRECISE   = B + CE-7          (B Stage-1 shortlist + diversified CE)
```

Master-allow policy:
- ``NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED=1`` makes PRECISE/ce7 *available*
- ``NALUS_LEGAL_V2_COLBERT_ENABLED=1`` makes BALANCED *available*
- request ``retrieval_profile`` selects the mode (default ``fast``)

``ce7`` remains an accepted alias for ``precise`` (API backward compatibility).

``retrieval_stage`` is provenance for the ranking that was actually returned —
derived from runtime outcome, not from configuration intent alone.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from app.rag.legal_v2.rerank.config import CrossEncoderConfig, cross_encoder_config_from_env
from app.rag.legal_v2.rerank.selectors.names import DIVERSIFIED_STAGE1_EVIDENCE_V1

RetrievalProfileId = Literal["fast", "balanced", "precise"]

DEFAULT_RETRIEVAL_PROFILE: RetrievalProfileId = "fast"
KNOWN_RETRIEVAL_PROFILES: tuple[RetrievalProfileId, ...] = ("fast", "balanced", "precise")
# Public aliases normalize to a canonical product profile id.
PROFILE_ALIASES: dict[str, RetrievalProfileId] = {
    "ce7": "precise",
}

# ---------------------------------------------------------------------------
# Canonical Slice-4 profile index bindings (pinned after A/B benchmarks).
# FAST = chunking A; BALANCED/PRECISE = chunking B contextual + ColBERT/CE.
# Optional env overrides (ops only):
#   NALUS_LEGAL_V2_FAST_QDRANT_COLLECTION / _BM25_INDEX_ID / _BM25_SIDECAR_PATH
#   NALUS_LEGAL_V2_CE_QDRANT_COLLECTION / _BM25_INDEX_ID / _BM25_SIDECAR_PATH
# ---------------------------------------------------------------------------
FAST_CANONICAL_QDRANT_COLLECTION = (
    "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_a_current_300"
)
FAST_CANONICAL_BM25_INDEX_ID = (
    "nalus_legal_paragraph_bm25_v2_chunk_ab_v8_a_current_300"
)
FAST_CANONICAL_BM25_SIDECAR_PATH = Path(
    "storage/rag/bm25/nalus_legal_paragraph_bm25_v2_chunk_ab_v8_a_current_300.sqlite"
)

CE_CANONICAL_QDRANT_COLLECTION = (
    "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_b_contextual_300"
)
CE_CANONICAL_BM25_INDEX_ID = (
    "nalus_legal_paragraph_bm25_v2_chunk_ab_v8_b_contextual_300"
)
CE_CANONICAL_BM25_SIDECAR_PATH = Path(
    "storage/rag/bm25/nalus_legal_paragraph_bm25_v2_chunk_ab_v8_b_contextual_300.sqlite"
)

# BALANCED uses the same B dense/BM25 indexes as PRECISE (+ ColBERT index).
BALANCED_CANONICAL_QDRANT_COLLECTION = CE_CANONICAL_QDRANT_COLLECTION
BALANCED_CANONICAL_BM25_INDEX_ID = CE_CANONICAL_BM25_INDEX_ID
BALANCED_CANONICAL_BM25_SIDECAR_PATH = CE_CANONICAL_BM25_SIDECAR_PATH

DEFAULT_COLBERT_CANDIDATE_CHUNKS = 80


class RetrievalStage(str, Enum):
    """Public provenance labels for the ranking pipeline that produced results."""

    HYBRID_RRF_STAGE_1 = "hybrid_rrf_stage_1"
    HYBRID_RRF_COLBERT = "hybrid_rrf_colbert"
    HYBRID_RRF_CE7 = "hybrid_rrf_ce7"
    HYBRID_RRF_CE = "hybrid_rrf_ce"


def build_retrieval_stage(
    *,
    rerank_applied: bool,
    passages_per_document: int | None = None,
    colbert_applied: bool = False,
) -> str:
    """Map executed ranking outcome to a stable public ``retrieval_stage`` label.

    Uses execution truth, never configuration-only intent.
    CE wins over ColBERT when both were applied (PRECISE path).
    """
    if rerank_applied:
        if passages_per_document == 7:
            return RetrievalStage.HYBRID_RRF_CE7.value
        return RetrievalStage.HYBRID_RRF_CE.value
    if colbert_applied:
        return RetrievalStage.HYBRID_RRF_COLBERT.value
    return RetrievalStage.HYBRID_RRF_STAGE_1.value


@dataclass(frozen=True)
class ProfileIndexBinding:
    """Qdrant + BM25 targets bound to a retrieval profile."""

    qdrant_collection: str
    bm25_index_id: str
    bm25_sidecar_path: Path


@dataclass(frozen=True)
class ResolvedRetrievalProfile:
    profile_id: RetrievalProfileId
    use_cross_encoder: bool
    use_colbert: bool = False
    cross_encoder_config: CrossEncoderConfig | None = None
    colbert_candidate_chunks: int = DEFAULT_COLBERT_CANDIDATE_CHUNKS
    label: str = "FAST"
    notes: str = ""
    index: ProfileIndexBinding | None = None


def normalize_retrieval_profile(value: str | None) -> RetrievalProfileId:
    cleaned = str(value or DEFAULT_RETRIEVAL_PROFILE).strip().lower()
    if not cleaned:
        return DEFAULT_RETRIEVAL_PROFILE
    if cleaned in PROFILE_ALIASES:
        return PROFILE_ALIASES[cleaned]
    if cleaned not in KNOWN_RETRIEVAL_PROFILES:
        known = ", ".join((*KNOWN_RETRIEVAL_PROFILES, *sorted(PROFILE_ALIASES)))
        raise ValueError(f"unknown retrieval_profile={cleaned!r}; known: {known}")
    return cleaned  # type: ignore[return-value]


def _env_or_default(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip()


def _env_flag_enabled(name: str) -> bool:
    return str(os.getenv(name, "0")).strip().lower() in {"1", "true", "yes", "on"}


def colbert_master_allow_enabled() -> bool:
    """Ops master-allow for BALANCED (ColBERT) profile."""
    return _env_flag_enabled("NALUS_LEGAL_V2_COLBERT_ENABLED")


def fast_index_binding() -> ProfileIndexBinding:
    """FAST canonical indexes = Slice 4 variant A (A won FAST A/B)."""
    collection = _env_or_default(
        "NALUS_LEGAL_V2_FAST_QDRANT_COLLECTION",
        FAST_CANONICAL_QDRANT_COLLECTION,
    )
    bm25_id = _env_or_default(
        "NALUS_LEGAL_V2_FAST_BM25_INDEX_ID",
        FAST_CANONICAL_BM25_INDEX_ID,
    )
    path_raw = _env_or_default(
        "NALUS_LEGAL_V2_FAST_BM25_SIDECAR_PATH",
        str(FAST_CANONICAL_BM25_SIDECAR_PATH),
    )
    return ProfileIndexBinding(
        qdrant_collection=collection,
        bm25_index_id=bm25_id,
        bm25_sidecar_path=Path(path_raw),
    )


def ce_index_binding() -> ProfileIndexBinding:
    """PRECISE/CE canonical indexes = Slice 4 variant B contextual."""
    collection = _env_or_default(
        "NALUS_LEGAL_V2_CE_QDRANT_COLLECTION",
        CE_CANONICAL_QDRANT_COLLECTION,
    )
    bm25_id = _env_or_default(
        "NALUS_LEGAL_V2_CE_BM25_INDEX_ID",
        CE_CANONICAL_BM25_INDEX_ID,
    )
    path_raw = _env_or_default(
        "NALUS_LEGAL_V2_CE_BM25_SIDECAR_PATH",
        str(CE_CANONICAL_BM25_SIDECAR_PATH),
    )
    return ProfileIndexBinding(
        qdrant_collection=collection,
        bm25_index_id=bm25_id,
        bm25_sidecar_path=Path(path_raw),
    )


def balanced_index_binding() -> ProfileIndexBinding:
    """BALANCED dense/BM25 indexes = same B contextual pin as PRECISE."""
    # Optional dedicated overrides; fall back to CE/B bindings.
    collection = _env_or_default(
        "NALUS_LEGAL_V2_BALANCED_QDRANT_COLLECTION",
        _env_or_default(
            "NALUS_LEGAL_V2_CE_QDRANT_COLLECTION",
            BALANCED_CANONICAL_QDRANT_COLLECTION,
        ),
    )
    bm25_id = _env_or_default(
        "NALUS_LEGAL_V2_BALANCED_BM25_INDEX_ID",
        _env_or_default(
            "NALUS_LEGAL_V2_CE_BM25_INDEX_ID",
            BALANCED_CANONICAL_BM25_INDEX_ID,
        ),
    )
    path_raw = _env_or_default(
        "NALUS_LEGAL_V2_BALANCED_BM25_SIDECAR_PATH",
        _env_or_default(
            "NALUS_LEGAL_V2_CE_BM25_SIDECAR_PATH",
            str(BALANCED_CANONICAL_BM25_SIDECAR_PATH),
        ),
    )
    return ProfileIndexBinding(
        qdrant_collection=collection,
        bm25_index_id=bm25_id,
        bm25_sidecar_path=Path(path_raw),
    )


def _precise_cross_encoder_config() -> CrossEncoderConfig:
    base = cross_encoder_config_from_env()
    if not base.enabled:
        raise ValueError(
            "retrieval_profile='precise' (alias ce7) requires "
            "NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED=1 (CE master-allow). "
            "FAST remains the default request profile."
        )
    ce7 = CrossEncoderConfig(
        enabled=True,
        model_id=base.model_id,
        candidate_documents=base.candidate_documents,
        passages_per_document=7,
        batch_size=base.batch_size,
        device=base.device,
        max_length=base.max_length,
        allow_download=base.allow_download,
        local_files_only=base.local_files_only,
        aggregation="max",
        experiment_mode="ce_bge_v2m3_p7_diverse_v1",
        passage_selector=DIVERSIFIED_STAGE1_EVIDENCE_V1,
        evidence_pool_limit=max(base.evidence_pool_limit, 40),
    )
    ce7.validate()
    return ce7


def resolve_retrieval_profile(value: str | None = None) -> ResolvedRetrievalProfile:
    """Resolve a request profile against ColBERT/CE master-allow env flags."""
    profile_id = normalize_retrieval_profile(value)
    if profile_id == "fast":
        return ResolvedRetrievalProfile(
            profile_id="fast",
            use_cross_encoder=False,
            use_colbert=False,
            cross_encoder_config=None,
            label="FAST",
            notes=(
                "Stage 1 only (BGE-M3 + BM25 + RRF). "
                "FAST uses Slice 4 variant A because A won the FAST A/B benchmark."
            ),
            index=fast_index_binding(),
        )
    if profile_id == "balanced":
        if not colbert_master_allow_enabled():
            raise ValueError(
                "retrieval_profile='balanced' requires NALUS_LEGAL_V2_COLBERT_ENABLED=1 "
                "(ColBERT master-allow). FAST remains the default request profile."
            )
        colbert_k = int(
            _env_or_default(
                "NALUS_LEGAL_V2_COLBERT_CANDIDATE_CHUNKS",
                str(DEFAULT_COLBERT_CANDIDATE_CHUNKS),
            )
        )
        if colbert_k < 1:
            raise ValueError("NALUS_LEGAL_V2_COLBERT_CANDIDATE_CHUNKS must be >= 1")
        return ResolvedRetrievalProfile(
            profile_id="balanced",
            use_cross_encoder=False,
            use_colbert=True,
            cross_encoder_config=None,
            colbert_candidate_chunks=colbert_k,
            label="BALANCED",
            notes=(
                "B contextual dense + BM25 + ColBERT → RRF (no CE). "
                "BALANCED middle tier after latency/quality golden benchmarks."
            ),
            index=balanced_index_binding(),
        )
    # precise (and alias ce7)
    ce7 = _precise_cross_encoder_config()
    return ResolvedRetrievalProfile(
        profile_id="precise",
        use_cross_encoder=True,
        use_colbert=False,
        cross_encoder_config=ce7,
        label="PRECISE",
        notes=(
            "Stage 1 B shortlist + diversified 7-passage Cross-Encoder (CE-7). "
            "PRECISE quality ceiling; CE uses Slice 4 B contextual."
        ),
        index=ce_index_binding(),
    )
