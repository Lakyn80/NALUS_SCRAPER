"""Configuration for additive constraint-aware document verification."""

from __future__ import annotations

import os
from dataclasses import dataclass

from app.rag.retrieval.errors import RetrievalConfigurationError

STRUCTURED_QUERY_DETERMINISTIC = "deterministic_v1"
VERIFICATION_DETERMINISTIC = "deterministic_v1"
RANKING_RETRIEVAL_PLUS_CONSTRAINTS = "retrieval_plus_constraint_coverage_v1"

_MAX_CANDIDATE_CHUNKS_UPPER_BOUND = 2_000
_MAX_CANDIDATE_DOCUMENTS_UPPER_BOUND = 250
_MAX_RETURNED_DOCUMENTS_UPPER_BOUND = 100
_MAX_CHUNKS_PER_DOCUMENT_UPPER_BOUND = 100
_MAX_DOCUMENT_CHARACTERS_UPPER_BOUND = 250_000
_MAX_TIMEOUT_MS = 300_000


@dataclass(frozen=True)
class ConstraintRetrievalConfig:
    enabled: bool = False
    strict_mode: bool = True
    max_candidate_chunks: int = 200
    max_candidate_documents: int = 50
    max_returned_documents: int = 20
    max_supporting_chunks: int = 3
    max_chunks_per_document_for_verification: int = 24
    max_document_characters_for_verification: int = 40_000
    total_latency_budget_ms: int | None = 10_000
    document_verification_timeout_ms: int | None = 1_500
    structured_query_strategy: str = STRUCTURED_QUERY_DETERMINISTIC
    verification_strategy: str = VERIFICATION_DETERMINISTIC
    ranking_strategy: str = RANKING_RETRIEVAL_PLUS_CONSTRAINTS
    include_rejected_documents: bool = False

    def validate(self) -> None:
        _validate_int_range(
            "max_candidate_chunks",
            self.max_candidate_chunks,
            minimum=1,
            maximum=_MAX_CANDIDATE_CHUNKS_UPPER_BOUND,
        )
        _validate_int_range(
            "max_candidate_documents",
            self.max_candidate_documents,
            minimum=1,
            maximum=_MAX_CANDIDATE_DOCUMENTS_UPPER_BOUND,
        )
        _validate_int_range(
            "max_returned_documents",
            self.max_returned_documents,
            minimum=1,
            maximum=_MAX_RETURNED_DOCUMENTS_UPPER_BOUND,
        )
        _validate_int_range(
            "max_supporting_chunks",
            self.max_supporting_chunks,
            minimum=1,
            maximum=20,
        )
        _validate_int_range(
            "max_chunks_per_document_for_verification",
            self.max_chunks_per_document_for_verification,
            minimum=1,
            maximum=_MAX_CHUNKS_PER_DOCUMENT_UPPER_BOUND,
        )
        _validate_int_range(
            "max_document_characters_for_verification",
            self.max_document_characters_for_verification,
            minimum=1_000,
            maximum=_MAX_DOCUMENT_CHARACTERS_UPPER_BOUND,
        )
        if self.total_latency_budget_ms is not None:
            _validate_int_range(
                "total_latency_budget_ms",
                self.total_latency_budget_ms,
                minimum=1,
                maximum=_MAX_TIMEOUT_MS,
            )
        if self.document_verification_timeout_ms is not None:
            _validate_int_range(
                "document_verification_timeout_ms",
                self.document_verification_timeout_ms,
                minimum=1,
                maximum=_MAX_TIMEOUT_MS,
            )
        if self.structured_query_strategy != STRUCTURED_QUERY_DETERMINISTIC:
            raise RetrievalConfigurationError(
                "Only deterministic structured-query interpretation is enabled in this rollout."
            )
        if self.verification_strategy != VERIFICATION_DETERMINISTIC:
            raise RetrievalConfigurationError(
                "Only deterministic constraint verification is enabled in this rollout."
            )
        if self.ranking_strategy != RANKING_RETRIEVAL_PLUS_CONSTRAINTS:
            raise RetrievalConfigurationError(
                "Unsupported constraint-aware ranking strategy."
            )


def constraint_retrieval_config_from_env() -> ConstraintRetrievalConfig:
    config = ConstraintRetrievalConfig(
        enabled=_read_bool_env("NALUS_CONSTRAINT_RETRIEVAL_ENABLED", default=False),
        strict_mode=_read_bool_env("NALUS_CONSTRAINT_RETRIEVAL_STRICT_MODE", default=True),
        max_candidate_chunks=_read_int_env(
            "NALUS_CONSTRAINT_MAX_CANDIDATE_CHUNKS",
            default=200,
        ),
        max_candidate_documents=_read_int_env(
            "NALUS_CONSTRAINT_MAX_CANDIDATE_DOCUMENTS",
            default=50,
        ),
        max_returned_documents=_read_int_env(
            "NALUS_CONSTRAINT_MAX_RETURNED_DOCUMENTS",
            default=20,
        ),
        max_supporting_chunks=_read_int_env(
            "NALUS_CONSTRAINT_MAX_SUPPORTING_CHUNKS",
            default=3,
        ),
        max_chunks_per_document_for_verification=_read_int_env(
            "NALUS_CONSTRAINT_MAX_CHUNKS_PER_DOCUMENT_FOR_VERIFICATION",
            default=24,
        ),
        max_document_characters_for_verification=_read_int_env(
            "NALUS_CONSTRAINT_MAX_DOCUMENT_CHARACTERS_FOR_VERIFICATION",
            default=40_000,
        ),
        total_latency_budget_ms=_read_optional_int_env(
            "NALUS_CONSTRAINT_TOTAL_LATENCY_BUDGET_MS",
            default=10_000,
        ),
        document_verification_timeout_ms=_read_optional_int_env(
            "NALUS_CONSTRAINT_DOCUMENT_VERIFICATION_TIMEOUT_MS",
            default=1_500,
        ),
        structured_query_strategy=os.getenv(
            "NALUS_CONSTRAINT_STRUCTURED_QUERY_STRATEGY",
            STRUCTURED_QUERY_DETERMINISTIC,
        ).strip(),
        verification_strategy=os.getenv(
            "NALUS_CONSTRAINT_VERIFICATION_STRATEGY",
            VERIFICATION_DETERMINISTIC,
        ).strip(),
        ranking_strategy=os.getenv(
            "NALUS_CONSTRAINT_RANKING_STRATEGY",
            RANKING_RETRIEVAL_PLUS_CONSTRAINTS,
        ).strip(),
        include_rejected_documents=_read_bool_env(
            "NALUS_CONSTRAINT_INCLUDE_REJECTED_DOCUMENTS",
            default=False,
        ),
    )
    config.validate()
    return config


def _read_bool_env(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RetrievalConfigurationError(f"{name} must be a boolean value.")


def _read_int_env(name: str, *, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RetrievalConfigurationError(f"{name} must be an integer.") from exc


def _read_optional_int_env(name: str, *, default: int | None = None) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    if not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise RetrievalConfigurationError(f"{name} must be an integer.") from exc


def _validate_int_range(name: str, value: int, *, minimum: int, maximum: int) -> None:
    if value < minimum or value > maximum:
        raise RetrievalConfigurationError(f"{name} must be between {minimum} and {maximum}.")
