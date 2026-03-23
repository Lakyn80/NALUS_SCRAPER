"""
Answer service — structures retrieval results into a human-readable response.

Rule-based implementation: no LLM required.
Interface is designed for drop-in replacement with an LLM-backed version later.

Usage:
    from app.rag.answer.answer_service import AnswerService

    answer_service = AnswerService()
    result = pipeline.run(query)
    answer = answer_service.generate(query, result.results)
"""

from dataclasses import dataclass

from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.rag.retrieval.models import RetrievedChunk

logger = get_logger(__name__)

_EXCERPT_MAX_LEN = 300
_MAX_EXCERPTS = 3
_MAX_CASES = 5

_NO_RESULTS_SUMMARY = "Nenalezeny žádné relevantní případy."
_SUMMARY_TEMPLATE = (
    "Na základě judikatury Ústavního soudu byly nalezeny relevantní případy "
    "týkající se: {query}"
)


@dataclass
class AnswerResult:
    query: str
    summary: str
    top_cases: list[str]
    excerpts: list[str]


class AnswerService:
    """Converts retrieved chunks into a structured answer.

    Stateless — safe to share across requests.
    """

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> AnswerResult:
        trace_event(
            logger,
            "answer.start",
            query=query,
            num_chunks=len(chunks),
        )

        if not chunks:
            result = AnswerResult(
                query=query,
                summary=_NO_RESULTS_SUMMARY,
                top_cases=[],
                excerpts=[],
            )
            trace_event(logger, "answer.done", num_cases=0, num_excerpts=0)
            return result

        summary = _SUMMARY_TEMPLATE.format(query=query)
        top_cases = _extract_cases(chunks)
        excerpts = _extract_excerpts(chunks)

        trace_event(
            logger,
            "answer.done",
            num_cases=len(top_cases),
            num_excerpts=len(excerpts),
        )

        return AnswerResult(
            query=query,
            summary=summary,
            top_cases=top_cases,
            excerpts=excerpts,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_cases(chunks: list[RetrievedChunk]) -> list[str]:
    """Return up to _MAX_CASES unique case identifiers, in score order."""
    seen: dict[str, None] = {}
    for chunk in chunks:
        case_id = chunk.id
        if case_id not in seen:
            seen[case_id] = None
        if len(seen) >= _MAX_CASES:
            break
    return list(seen)


def _extract_excerpts(chunks: list[RetrievedChunk]) -> list[str]:
    """Return up to _MAX_EXCERPTS text excerpts, each at most _EXCERPT_MAX_LEN chars."""
    return [
        chunk.text[:_EXCERPT_MAX_LEN]
        for chunk in chunks[:_MAX_EXCERPTS]
        if chunk.text
    ]
