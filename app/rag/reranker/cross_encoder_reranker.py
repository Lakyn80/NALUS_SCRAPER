"""
CrossEncoder reranker backed by sentence-transformers.

Replaces SimpleReranker when a real neural scoring model is available.
The interface (BaseReranker.rerank) is identical — swap is one line at
the injection site.

Dependency:
    pip install sentence-transformers

The model is injectable for unit tests — pass a pre-instantiated (or mock)
model to avoid loading weights during testing:

    reranker = CrossEncoderReranker(model=mock_model)

Usage (production):
    reranker = CrossEncoderReranker()
    service = RetrievalService(dense, keyword, reranker=reranker)
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.rag.reranker.base import BaseReranker
from app.rag.retrieval.models import RetrievedChunk

logger = get_logger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker(BaseReranker):
    """Neural reranker using a sentence-transformers CrossEncoder model.

    Falls back to the original retrieval order if model.predict() raises.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        model: Any = None,
    ) -> None:
        if model is not None:
            self._model = model
        else:
            # Lazy import — keeps the module importable without sentence-transformers
            # installed, and avoids loading weights at import time.
            from sentence_transformers import CrossEncoder  # type: ignore[import]

            self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        trace_event(
            logger,
            "rerank.start",
            query=query,
            input_count=len(chunks),
            model=type(self._model).__name__,
        )

        if not chunks:
            trace_event(logger, "rerank.done", output_count=0)
            return []

        pairs = [(query, chunk.text) for chunk in chunks]

        try:
            raw_scores: list[float] = [float(s) for s in self._model.predict(pairs)]
        except Exception as exc:
            logger.warning(
                "CrossEncoder.predict failed (%s); falling back to original order.",
                exc,
            )
            result = chunks[:top_k]
            trace_event(
                logger,
                "rerank.done",
                output_count=len(result),
                fallback=True,
            )
            return result

        rescored = [
            replace(chunk, score=score)
            for chunk, score in zip(chunks, raw_scores)
        ]
        rescored.sort(key=lambda c: c.score, reverse=True)
        result = rescored[:top_k]

        trace_event(logger, "rerank.done", output_count=len(result))
        return result
