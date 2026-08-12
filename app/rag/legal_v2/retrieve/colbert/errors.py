"""Typed errors for Legal v2 ColBERT retrieval foundation."""

from __future__ import annotations


class ColbertError(Exception):
    """Base ColBERT module error."""


class ColbertConfigurationError(ColbertError):
    """ColBERT configuration is invalid."""


class ColbertBackendUnavailableError(ColbertError):
    """No ColBERT backend is injected / installed / ready."""


class ColbertNotImplementedError(ColbertError):
    """Requested ColBERT capability is foundation-only and not wired yet."""
