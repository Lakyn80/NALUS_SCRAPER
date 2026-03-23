"""
Keyword-overlap reranker.

Boosts the score of every chunk by 0.05 for each query word found in
its text, then re-ranks.  No external models required.

Replace with a CrossEncoder implementation when ready — the interface
(BaseReranker.rerank) stays the same.

Usage:
    from app.rag.reranker.simple_reranker import SimpleReranker

    reranker = SimpleReranker()
    reranked = reranker.rerank("únos dítěte", chunks, top_k=5)
"""

from dataclasses import replace

from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.rag.reranker.base import BaseReranker
from app.rag.retrieval.models import RetrievedChunk

logger = get_logger(__name__)

_BOOST_PER_MATCH = 0.05


class SimpleReranker(BaseReranker):
    """Lexical-overlap reranker: score += 0.05 × matched_query_words."""

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
        )

        query_words = _tokenize(query)
        rescored = [_boost(chunk, query_words) for chunk in chunks]
        rescored.sort(key=lambda c: c.score, reverse=True)
        result = rescored[:top_k]

        trace_event(logger, "rerank.done", output_count=len(result))
        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Lowercase and split; keep words ≥ 3 chars."""
    return [w for w in text.lower().split() if len(w) >= 3]


def _boost(chunk: RetrievedChunk, query_words: list[str]) -> RetrievedChunk:
    """Return a new RetrievedChunk with score boosted by keyword overlap."""
    text_lower = chunk.text.lower()
    match_count = sum(1 for w in query_words if w in text_lower)
    if match_count == 0:
        return chunk
    return replace(chunk, score=chunk.score + match_count * _BOOST_PER_MATCH)
