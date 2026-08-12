"""Typed errors for Legal v2 ColBERT retrieval."""

from __future__ import annotations


class ColbertError(Exception):
    """Base ColBERT module error."""


class ColbertConfigurationError(ColbertError):
    """ColBERT configuration is invalid."""


class ColbertBackendUnavailableError(ColbertError):
    """No ColBERT backend is injected / installed / ready."""


class ColbertNotImplementedError(ColbertError):
    """Requested ColBERT capability is not wired yet."""


class ColbertIndexError(ColbertError):
    """ColBERT index is missing, corrupt, or failed integrity checks."""


class ColbertMappingError(ColbertError):
    """Chunk mapping is missing, corrupt, or inconsistent with the index."""
