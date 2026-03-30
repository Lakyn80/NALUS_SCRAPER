"""
Chunking layer for NALUS decisions.

Splits full_text from a NalusResult into overlapping character-based chunks
suitable for RAG ingestion.  No external dependencies — stdlib only.

Usage:
    from app.rag.chunking.chunker import chunk_document

    chunks = chunk_document(result, chunk_size=1500, overlap=200)
"""

import re
from dataclasses import dataclass

from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.models.search_result import NalusResult

logger = get_logger(__name__)


@dataclass
class TextChunk:
    id: str
    text: str
    case_reference: str | None
    ecli: str | None
    decision_date: str | None
    judge: str | None
    text_url: str | None
    chunk_index: int
    document_id: int | None = None


def chunk_document(
    result: NalusResult,
    chunk_size: int = 1500,
    overlap: int = 200,
) -> list[TextChunk]:
    """Split result.full_text into overlapping TextChunks.

    Returns an empty list when full_text is absent or empty.
    Raises ValueError when chunk_size <= overlap.
    """
    if chunk_size <= overlap:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be greater than overlap ({overlap})"
        )

    doc_id = result.case_reference or result.ecli or "unknown"

    if not result.full_text:
        trace_event(logger, "chunking.skip", doc_id=doc_id, reason="no_full_text")
        return []

    text = _clean_text(result.full_text)

    if not text:
        trace_event(logger, "chunking.skip", doc_id=doc_id, reason="empty_after_clean")
        return []

    trace_event(logger, "chunking.start", doc_id=doc_id, text_length=len(text))

    base_id = _normalize_case_reference(result.case_reference)
    step = chunk_size - overlap
    chunks: list[TextChunk] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]

        chunks.append(
            TextChunk(
                id=f"{base_id}_{len(chunks)}",
                text=chunk_text,
                case_reference=result.case_reference,
                ecli=result.ecli,
                decision_date=result.decision_date,
                judge=result.judge_rapporteur,
                text_url=result.text_url,
                chunk_index=len(chunks),
                document_id=result.result_id,
            )
        )

        if end == len(text):
            break
        start += step

    avg_chunk_len = round(sum(len(c.text) for c in chunks) / len(chunks)) if chunks else 0

    trace_event(
        logger,
        "chunking.done",
        doc_id=doc_id,
        num_chunks=len(chunks),
        avg_chunk_len=avg_chunk_len,
    )

    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_case_reference(case_reference: str | None) -> str:
    """Produce a filesystem-safe ID prefix from a case reference.

    'III.ÚS 255/26' → 'III.ÚS_255_26'
    """
    if not case_reference:
        return "unknown"
    normalized = case_reference.replace(" ", "_").replace("/", "_")
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def _clean_text(text: str) -> str:
    """Collapse horizontal whitespace runs; preserve newlines; strip edges."""
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
