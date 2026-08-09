"""Request-level retrieval profiles for Legal v2 Stage 1 (+ optional CE).

Profiles are additive and modular so future phases (e.g. PRECISE/ColBERT) can
register without changing the Stage 1 contract.

Master-allow policy:
- ``NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED=1`` makes CE *available*
- request ``retrieval_profile`` selects the mode (default ``fast`` = no CE)

``retrieval_stage`` is provenance for the ranking that was actually returned —
derived from runtime rerank outcome, not from configuration intent alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from app.rag.legal_v2.rerank.config import CrossEncoderConfig, cross_encoder_config_from_env
from app.rag.legal_v2.rerank.selectors.names import DIVERSIFIED_STAGE1_EVIDENCE_V1

RetrievalProfileId = Literal["fast", "ce7", "precise"]

DEFAULT_RETRIEVAL_PROFILE: RetrievalProfileId = "fast"
KNOWN_RETRIEVAL_PROFILES: tuple[RetrievalProfileId, ...] = ("fast", "ce7", "precise")


class RetrievalStage(str, Enum):
    """Public provenance labels for the ranking pipeline that produced results."""

    HYBRID_RRF_STAGE_1 = "hybrid_rrf_stage_1"
    HYBRID_RRF_CE7 = "hybrid_rrf_ce7"
    HYBRID_RRF_CE = "hybrid_rrf_ce"


def build_retrieval_stage(
    *,
    rerank_applied: bool,
    passages_per_document: int | None = None,
) -> str:
    """Map executed rerank outcome to a stable public ``retrieval_stage`` label.

    Uses execution truth (``rerank_applied``), never configuration-only intent.
    """
    if not rerank_applied:
        return RetrievalStage.HYBRID_RRF_STAGE_1.value
    if passages_per_document == 7:
        return RetrievalStage.HYBRID_RRF_CE7.value
    return RetrievalStage.HYBRID_RRF_CE.value


@dataclass(frozen=True)
class ResolvedRetrievalProfile:
    profile_id: RetrievalProfileId
    use_cross_encoder: bool
    cross_encoder_config: CrossEncoderConfig | None = None
    label: str = "FAST"
    notes: str = ""


def normalize_retrieval_profile(value: str | None) -> RetrievalProfileId:
    cleaned = str(value or DEFAULT_RETRIEVAL_PROFILE).strip().lower()
    if not cleaned:
        return DEFAULT_RETRIEVAL_PROFILE
    if cleaned not in KNOWN_RETRIEVAL_PROFILES:
        known = ", ".join(KNOWN_RETRIEVAL_PROFILES)
        raise ValueError(f"unknown retrieval_profile={cleaned!r}; known: {known}")
    return cleaned  # type: ignore[return-value]


def resolve_retrieval_profile(value: str | None = None) -> ResolvedRetrievalProfile:
    """Resolve a request profile against the CE master-allow env flag."""
    profile_id = normalize_retrieval_profile(value)
    if profile_id == "fast":
        return ResolvedRetrievalProfile(
            profile_id="fast",
            use_cross_encoder=False,
            cross_encoder_config=None,
            label="FAST",
            notes="Stage 1 only (BGE-M3 + BM25 + RRF)",
        )
    if profile_id == "precise":
        raise ValueError(
            "retrieval_profile='precise' is not available yet "
            "(reserved for a future late-interaction / ColBERT phase)"
        )
    # ce7
    base = cross_encoder_config_from_env()
    if not base.enabled:
        raise ValueError(
            "retrieval_profile='ce7' requires NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED=1 "
            "(CE master-allow). FAST remains the default request profile."
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
        experiment_mode="fast_plus_ce_experiment",
        passage_selector=DIVERSIFIED_STAGE1_EVIDENCE_V1,
        evidence_pool_limit=max(base.evidence_pool_limit, 40),
    )
    ce7.validate()
    return ResolvedRetrievalProfile(
        profile_id="ce7",
        use_cross_encoder=True,
        cross_encoder_config=ce7,
        label="CE-7",
        notes="Stage 1 shortlist + diversified 7-passage Cross-Encoder",
    )
