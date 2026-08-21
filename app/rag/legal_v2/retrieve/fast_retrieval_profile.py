"""FAST-only retrieval channel selector (dense / bm25 / hybrid).

Product architecture:

```text
FAST
├── dense   ← default (WEDOS production)
├── bm25    ← implemented; enable via env on stronger hosts
└── hybrid  ← Dense + BM25 RRF; enable via env on stronger hosts

PODROBNÉ → ColBERT (separate request profile; not selected here)
PŘESNÉ   → Cross-Encoder (separate request profile; not selected here)
```

Ops switch (no frontend / API / source change):

    NALUS_FAST_RETRIEVAL_PROFILE=dense|bm25|hybrid

Legacy alias: ``NALUS_LEGAL_V2_STAGE1_FAST_DENSE_ONLY=1`` forces dense when the
new selector is unset.
"""

from __future__ import annotations

import os
from typing import Literal

FastRetrievalProfileId = Literal["dense", "bm25", "hybrid"]

DEFAULT_FAST_RETRIEVAL_PROFILE: FastRetrievalProfileId = "dense"
KNOWN_FAST_RETRIEVAL_PROFILES: tuple[FastRetrievalProfileId, ...] = (
    "dense",
    "bm25",
    "hybrid",
)

_TRUTHY = {"1", "true", "yes", "on"}


def normalize_fast_retrieval_profile(value: str | None) -> FastRetrievalProfileId:
    cleaned = str(value or "").strip().lower()
    if not cleaned:
        raise ValueError("fast retrieval profile is empty")
    if cleaned not in KNOWN_FAST_RETRIEVAL_PROFILES:
        known = ", ".join(KNOWN_FAST_RETRIEVAL_PROFILES)
        raise ValueError(
            f"unknown NALUS_FAST_RETRIEVAL_PROFILE={cleaned!r}; known: {known}"
        )
    return cleaned  # type: ignore[return-value]


def resolve_fast_retrieval_profile() -> tuple[FastRetrievalProfileId, str]:
    """Resolve effective FAST channel and a short source label for diagnostics.

    Priority:
    1. ``NALUS_FAST_RETRIEVAL_PROFILE`` when set (explicit; invalid → error)
    2. Legacy ``NALUS_LEGAL_V2_STAGE1_FAST_DENSE_ONLY=1`` → dense
    3. Default ``dense`` (WEDOS-safe)
    """
    raw = os.getenv("NALUS_FAST_RETRIEVAL_PROFILE")
    if raw is not None and str(raw).strip():
        profile = normalize_fast_retrieval_profile(raw)
        return profile, "NALUS_FAST_RETRIEVAL_PROFILE"

    legacy = os.getenv("NALUS_LEGAL_V2_STAGE1_FAST_DENSE_ONLY", "0")
    if str(legacy).strip().lower() in _TRUTHY:
        return "dense", "NALUS_LEGAL_V2_STAGE1_FAST_DENSE_ONLY"

    return DEFAULT_FAST_RETRIEVAL_PROFILE, "default"


def fast_profile_uses_dense(profile: FastRetrievalProfileId) -> bool:
    return profile in {"dense", "hybrid"}


def fast_profile_uses_bm25(profile: FastRetrievalProfileId) -> bool:
    return profile in {"bm25", "hybrid"}
