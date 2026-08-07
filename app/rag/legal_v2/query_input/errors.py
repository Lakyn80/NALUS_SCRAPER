"""Typed errors for long-input preprocessing."""

from __future__ import annotations


class QueryInputError(ValueError):
    """Base error for query-input preprocessing."""


class InputTooLargeError(QueryInputError):
    """Raw input exceeds configured hard limit."""


class CondensationFailedError(QueryInputError):
    """Extractive/hybrid condensation failed unexpectedly."""


class NoUsefulContentError(QueryInputError):
    """Normalized input had no usable legal content after filtering."""


class UnsupportedCondensationModeError(QueryInputError):
    """Requested condensation mode is not available."""
