"""ColBERT retriever foundation (backend-injected; no fake results)."""

from __future__ import annotations

import time
from typing import Any

from app.rag.legal_v2.retrieve.colbert.backend import ColbertBackend, require_backend
from app.rag.legal_v2.retrieve.colbert.config import ColbertConfig
from app.rag.legal_v2.retrieve.colbert.errors import ColbertConfigurationError
from app.rag.legal_v2.retrieve.colbert.models import ColbertHit, ColbertRetrievalResult


class ColbertRetriever:
    """Query → ColBERT backend → ranked ``ColbertRetrievalResult``.

    Without an injected ``ColbertBackend``, ``retrieve`` fails explicitly.
    Never returns mock/fake production hits.
    """

    def __init__(
        self,
        config: ColbertConfig,
        *,
        backend: ColbertBackend | None = None,
    ) -> None:
        config.validate()
        self._config = config
        self._backend = backend

    @property
    def config(self) -> ColbertConfig:
        return self._config

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> ColbertRetrievalResult:
        cleaned = str(query or "").strip()
        if not cleaned:
            raise ColbertConfigurationError("query must not be blank")
        resolved_top_k = int(self._config.top_k if top_k is None else top_k)
        if resolved_top_k < 1:
            raise ColbertConfigurationError("top_k must be >= 1")

        backend = require_backend(self._backend)
        started = time.perf_counter()
        raw_hits = backend.search(cleaned, top_k=resolved_top_k)
        hits = tuple(_normalize_hit(hit, rank=index) for index, hit in enumerate(raw_hits, start=1))
        return ColbertRetrievalResult(
            hits=hits[:resolved_top_k],
            diagnostics={
                "model_name": self._config.model_name,
                "index_name": self._config.index_name,
                "index_path": self._config.index_path.as_posix(),
                "device": self._config.device,
                "top_k": resolved_top_k,
                "hit_count": len(hits[:resolved_top_k]),
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            },
        )


def _normalize_hit(hit: ColbertHit | Any, *, rank: int) -> ColbertHit:
    if isinstance(hit, ColbertHit):
        if hit.rank == rank:
            return hit
        return ColbertHit(
            document_id=hit.document_id,
            chunk_id=hit.chunk_id,
            rank=rank,
            score=hit.score,
            text=hit.text,
            metadata=dict(hit.metadata),
        )
    raise ColbertConfigurationError(
        f"ColBERT backend must return ColbertHit instances (got {type(hit)!r})"
    )
