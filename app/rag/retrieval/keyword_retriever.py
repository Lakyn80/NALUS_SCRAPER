"""
Keyword retriever stub.

Scores chunks by counting how many unique query words appear in the chunk text.
The in-memory corpus will be replaced by a real BM25 / Qdrant sparse index
in a later step.

Usage:
    corpus = [
        ("chunk-id-1", "Haagská úmluva a únos dítěte"),
        ("chunk-id-2", "Vyživovací povinnost rodiče"),
    ]
    retriever = KeywordRetriever(corpus)
    results = retriever.retrieve("únos dítěte", top_k=5)
"""

from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.rag.retrieval.base import BaseRetriever
from app.rag.retrieval.models import RetrievedChunk

logger = get_logger(__name__)

# Corpus entry: (chunk_id, text)
CorpusEntry = tuple[str, str]

_BASE_SCORE = 0.3
_PER_MATCH_BONUS = 0.1
_MAX_SCORE = 0.9


class KeywordRetriever(BaseRetriever):
    """Stub keyword retriever backed by a plain in-memory list."""

    def __init__(self, corpus: list[CorpusEntry] | None = None) -> None:
        self._corpus: list[CorpusEntry] = corpus or []

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        trace_event(
            logger,
            "keyword.start",
            query=query,
            top_k=top_k,
            corpus_size=len(self._corpus),
        )

        query_words = _tokenize(query)
        results = _score_corpus(self._corpus, query_words)
        results.sort(key=lambda r: r.score, reverse=True)
        top = results[:top_k]

        trace_event(
            logger,
            "keyword.done",
            num_results=len(top),
            query_words=query_words,
        )
        return top


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on whitespace; drop tokens shorter than 3 chars."""
    return [w for w in text.lower().split() if len(w) >= 3]


def _score_corpus(
    corpus: list[CorpusEntry],
    query_words: list[str],
) -> list[RetrievedChunk]:
    """Return a RetrievedChunk for every corpus entry that matches at least one query word."""
    if not query_words:
        return []

    results: list[RetrievedChunk] = []
    for chunk_id, text in corpus:
        text_lower = text.lower()
        match_count = sum(1 for w in query_words if w in text_lower)
        if match_count == 0:
            continue
        score = min(_BASE_SCORE + _PER_MATCH_BONUS * match_count, _MAX_SCORE)
        results.append(
            RetrievedChunk(id=chunk_id, text=text, score=score, source="keyword")
        )
    return results
