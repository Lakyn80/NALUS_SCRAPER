"""
JSONL exporter for RAG-ready chunk datasets.

Each TextChunk is serialised as one JSON line:
  {"id": "...", "text": "...", "metadata": {...}}

Usage:
    from app.rag.exporter.jsonl_exporter import export_chunks_to_jsonl

    export_chunks_to_jsonl(chunks, "data/chunks.jsonl")
"""

import json
import os
from typing import Any

from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.rag.chunking.chunker import TextChunk

logger = get_logger(__name__)


def export_chunks_to_jsonl(chunks: list[TextChunk], output_path: str) -> None:
    """Write chunks to a JSONL file, one chunk per line.

    Creates or overwrites the file at output_path.
    An empty chunks list produces an empty file.
    """
    trace_event(
        logger,
        "export.start",
        num_chunks=len(chunks),
        output_path=output_path,
    )

    with open(output_path, "w", encoding="utf-8") as fh:
        for chunk in chunks:
            record = _chunk_to_record(chunk)
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")

    file_size = os.path.getsize(output_path)

    trace_event(
        logger,
        "export.done",
        num_written=len(chunks),
        file_size_bytes=file_size,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _chunk_to_record(chunk: TextChunk) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "text": chunk.text,
        "metadata": {
            "case_reference": chunk.case_reference,
            "ecli": chunk.ecli,
            "decision_date": chunk.decision_date,
            "judge": chunk.judge,
            "text_url": chunk.text_url,
            "chunk_index": chunk.chunk_index,
            "source": "nalus",
        },
    }
