"""Search-time Qdrant scalar quantization policy.

This module never mutates Qdrant. Enabling INT8 storage is an offline ops
action. Search defaults to full precision by sending ``ignore=True``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from app.rag.retrieval.errors import RetrievalConfigurationError

ENV_QUANTIZATION_ENABLED = "NALUS_QDRANT_QUANTIZATION_ENABLED"
ENV_QUANTIZATION_RESCORE = "NALUS_QDRANT_QUANTIZATION_RESCORE"
ENV_QUANTIZATION_OVERSAMPLING = "NALUS_QDRANT_QUANTIZATION_OVERSAMPLING"

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class QdrantQuantizationSearchPolicy:
    enabled: bool
    rescore: bool
    oversampling: float

    @property
    def ignore(self) -> bool:
        return not self.enabled

    def to_search_params(self) -> Any:
        from qdrant_client import models

        if self.enabled:
            quantization = models.QuantizationSearchParams(
                ignore=False,
                rescore=self.rescore,
                oversampling=self.oversampling,
            )
        else:
            quantization = models.QuantizationSearchParams(ignore=True)
        return models.SearchParams(quantization=quantization)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "quantization_enabled": self.enabled,
            "quantization_ignore": self.ignore,
            "quantization_rescore": self.rescore if self.enabled else False,
            "quantization_oversampling": self.oversampling if self.enabled else 1.0,
        }


def qdrant_quantization_policy_from_env(
    environ: dict[str, str] | None = None,
) -> QdrantQuantizationSearchPolicy:
    source = os.environ if environ is None else environ
    enabled = _read_bool_env(source, ENV_QUANTIZATION_ENABLED, default=False)
    rescore = _read_bool_env(source, ENV_QUANTIZATION_RESCORE, default=False)
    oversampling = _read_oversampling_env(source, ENV_QUANTIZATION_OVERSAMPLING, default=1.0)
    return QdrantQuantizationSearchPolicy(
        enabled=enabled,
        rescore=rescore,
        oversampling=oversampling,
    )


def _read_bool_env(source: Any, name: str, *, default: bool) -> bool:
    raw_value = source.get(name)
    if raw_value is None:
        return default
    normalized = str(raw_value).strip().lower()
    if normalized == "":
        return default
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise RetrievalConfigurationError(
        f"{name} must be a boolean value (0, 1, true, false, yes, no, on, off)."
    )


def _read_oversampling_env(source: Any, name: str, *, default: float) -> float:
    raw_value = source.get(name)
    if raw_value is None or str(raw_value).strip() == "":
        return default
    try:
        value = float(str(raw_value).strip())
    except ValueError as exc:
        raise RetrievalConfigurationError(f"{name} must be a floating-point number.") from exc
    if value < 1.0:
        raise RetrievalConfigurationError(f"{name} must be greater than or equal to 1.0.")
    return value
