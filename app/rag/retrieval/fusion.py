"""
Result fusion for hybrid retrieval.

Merges dense and keyword results:
  - deduplicates by chunk id (keeps the higher score)
  - sorts by score descending
  - returns top_k

Usage:
    final = fuse_results(dense_results, keyword_results, top_k=10)
"""

from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.rag.retrieval.models import RetrievedChunk

logger = get_logger(__name__)


def fuse_results(
    dense: list[RetrievedChunk],
    keyword: list[RetrievedChunk],
    top_k: int = 10,
) -> list[RetrievedChunk]:
    """Merge dense and keyword results into a single ranked list.

    When the same chunk id appears in both lists the higher score is kept.
    """
    trace_event(
        logger,
        "fusion.start",
        dense_count=len(dense),
        keyword_count=len(keyword),
        top_k=top_k,
    )

    merged = _merge(dense, keyword)
    merged.sort(key=lambda r: r.score, reverse=True)
    final = merged[:top_k]

    trace_event(logger, "fusion.done", final_count=len(final))
    return final


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _merge(
    dense: list[RetrievedChunk],
    keyword: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Deduplicate by id, keeping the entry with the higher score."""
    best: dict[str, RetrievedChunk] = {}

    for chunk in dense + keyword:
        existing = best.get(chunk.id)
        if existing is None or chunk.score > existing.score:
            best[chunk.id] = chunk

    return list(best.values())
