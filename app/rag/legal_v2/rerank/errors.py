"""Typed errors for Legal v2 Cross-Encoder reranking."""

from __future__ import annotations


class RerankerError(Exception):
    """Base reranker error."""


class RerankerUnavailableError(RerankerError):
    """Reranker feature or provider is not available."""


class RerankerModelLoadError(RerankerError):
    """Model weights could not be loaded."""


class RerankerInferenceError(RerankerError):
    """Model inference failed."""


class RerankerInvalidCandidateError(RerankerError):
    """Candidate set is invalid for reranking."""
