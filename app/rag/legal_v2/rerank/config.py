"""Typed configuration for Legal v2 Cross-Encoder reranking."""

from __future__ import annotations

import os
from dataclasses import dataclass

from app.rag.legal_v2.rerank.selectors.names import FIRST_N_STAGE1_ORDER_V1
from app.rag.legal_v2.rerank.selectors.policy import resolve_passage_selector_name

_TRUTHY = {"1", "true", "yes", "on"}

DEFAULT_CE_MODEL = "BAAI/bge-reranker-v2-m3"
HARD_MAX_CANDIDATE_DOCUMENTS = 80
# Hard ceiling allows future CE-10 experiments; this task does not run CE-10.
HARD_MAX_PASSAGES_PER_DOCUMENT = 10
HARD_MAX_BATCH_SIZE = 64
HARD_MAX_LENGTH = 1024
HARD_MAX_EVIDENCE_POOL = 80
DEFAULT_EVIDENCE_POOL_LIMIT = 40


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return int(str(raw).strip())


@dataclass(frozen=True)
class CrossEncoderConfig:
    enabled: bool = False
    model_id: str = DEFAULT_CE_MODEL
    candidate_documents: int = 30
    passages_per_document: int = 3
    batch_size: int = 16
    device: str = "auto"
    max_length: int = 512
    allow_download: bool = False
    local_files_only: bool = True
    aggregation: str = "max"
    experiment_mode: str = "fast_plus_ce_experiment"
    passage_selector: str = FIRST_N_STAGE1_ORDER_V1
    evidence_pool_limit: int = DEFAULT_EVIDENCE_POOL_LIMIT

    def validate(self) -> None:
        if self.candidate_documents < 1 or self.candidate_documents > HARD_MAX_CANDIDATE_DOCUMENTS:
            raise ValueError(
                f"candidate_documents must be 1..{HARD_MAX_CANDIDATE_DOCUMENTS}"
            )
        if (
            self.passages_per_document < 1
            or self.passages_per_document > HARD_MAX_PASSAGES_PER_DOCUMENT
        ):
            raise ValueError(
                f"passages_per_document must be 1..{HARD_MAX_PASSAGES_PER_DOCUMENT}"
            )
        if self.batch_size < 1 or self.batch_size > HARD_MAX_BATCH_SIZE:
            raise ValueError(f"batch_size must be 1..{HARD_MAX_BATCH_SIZE}")
        if self.max_length < 32 or self.max_length > HARD_MAX_LENGTH:
            raise ValueError(f"max_length must be 32..{HARD_MAX_LENGTH}")
        if self.aggregation != "max":
            raise ValueError("only aggregation='max' is supported in this experiment")
        if self.evidence_pool_limit < 1 or self.evidence_pool_limit > HARD_MAX_EVIDENCE_POOL:
            raise ValueError(
                f"evidence_pool_limit must be 1..{HARD_MAX_EVIDENCE_POOL}"
            )
        resolve_passage_selector_name(self.passage_selector)


def cross_encoder_config_from_env() -> CrossEncoderConfig:
    cfg = CrossEncoderConfig(
        enabled=_env_flag("NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED", False),
        model_id=(
            os.getenv("NALUS_LEGAL_V2_CROSS_ENCODER_MODEL", DEFAULT_CE_MODEL).strip()
            or DEFAULT_CE_MODEL
        ),
        candidate_documents=_int_env("NALUS_LEGAL_V2_CE_CANDIDATE_DOCUMENTS", 30),
        passages_per_document=_int_env("NALUS_LEGAL_V2_CE_PASSAGES_PER_DOCUMENT", 3),
        batch_size=_int_env("NALUS_LEGAL_V2_CE_BATCH_SIZE", 16),
        device=(os.getenv("NALUS_LEGAL_V2_CE_DEVICE", "auto").strip() or "auto"),
        max_length=_int_env("NALUS_LEGAL_V2_CE_MAX_LENGTH", 512),
        allow_download=_env_flag("NALUS_LEGAL_V2_CE_ALLOW_DOWNLOAD", False),
        local_files_only=not _env_flag("NALUS_LEGAL_V2_CE_ALLOW_DOWNLOAD", False),
        aggregation="max",
        experiment_mode="fast_plus_ce_experiment",
        passage_selector=resolve_passage_selector_name(
            os.getenv("NALUS_LEGAL_V2_CE_PASSAGE_SELECTOR", FIRST_N_STAGE1_ORDER_V1)
        ),
        evidence_pool_limit=_int_env(
            "NALUS_LEGAL_V2_CE_EVIDENCE_POOL_LIMIT", DEFAULT_EVIDENCE_POOL_LIMIT
        ),
    )
    cfg.validate()
    return cfg


def cross_encoder_enabled() -> bool:
    return cross_encoder_config_from_env().enabled
