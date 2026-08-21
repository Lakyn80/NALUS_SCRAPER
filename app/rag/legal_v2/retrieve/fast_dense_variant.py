"""FAST Dense implementation selector (current vs classic Legal v2).

Only applies when ``NALUS_FAST_RETRIEVAL_PROFILE=dense`` (or dense channel of
hybrid). BM25/hybrid ranking paths ignore this selector.

```text
NALUS_FAST_RETRIEVAL_PROFILE=dense
NALUS_FAST_DENSE_VARIANT=current|v2
```

- ``current`` — live dense path (includes env-controlled Qdrant INT8 search policy)
- ``v2`` — previous known-good Legal v2 FAST dense Qdrant search from before
  commit ``e9fa438`` (plain ``query_points`` without quantization search_params)
"""

from __future__ import annotations

import os
from typing import Literal

FastDenseVariantId = Literal["current", "v2"]

DEFAULT_FAST_DENSE_VARIANT: FastDenseVariantId = "current"
KNOWN_FAST_DENSE_VARIANTS: tuple[FastDenseVariantId, ...] = ("current", "v2")

# Source pin for the classic dense Qdrant search behavior restored as ``v2``.
V2_DENSE_SOURCE_COMMIT = "e9fa438^"  # parent of INT8 search-policy integration
V2_DENSE_SOURCE_NOTE = (
    "Legal v2 FAST dense QdrantDenseStore.search before env-controlled "
    "QuantizationSearchParams (plain query_points)"
)


def normalize_fast_dense_variant(value: str | None) -> FastDenseVariantId:
    cleaned = str(value or "").strip().lower()
    if not cleaned:
        raise ValueError("fast dense variant is empty")
    if cleaned not in KNOWN_FAST_DENSE_VARIANTS:
        known = ", ".join(KNOWN_FAST_DENSE_VARIANTS)
        raise ValueError(
            f"unknown NALUS_FAST_DENSE_VARIANT={cleaned!r}; known: {known}"
        )
    return cleaned  # type: ignore[return-value]


def resolve_fast_dense_variant() -> tuple[FastDenseVariantId, str]:
    """Resolve dense implementation variant and a short source label."""
    raw = os.getenv("NALUS_FAST_DENSE_VARIANT")
    if raw is not None and str(raw).strip():
        return normalize_fast_dense_variant(raw), "NALUS_FAST_DENSE_VARIANT"
    return DEFAULT_FAST_DENSE_VARIANT, "default"


def dense_variant_uses_legacy_qdrant_search(variant: FastDenseVariantId) -> bool:
    """True when dense search must omit quantization search_params (classic v2)."""
    return variant == "v2"
