"""Prepare rag-embedding-benchmark input: SQLite chunk store + eval JSON."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

DEFAULT_CHUNKS_PATH = Path("app/artifacts/nsoud/rag_ready/nsoud_chunks_2025_01_03.parquet")
DEFAULT_RELEVANCE_DATASET = Path(
    "app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/relevance_eval_dataset.json"
)
DEFAULT_SQLITE_PATH = Path("artifacts/rag_eval/nalus_chunks.sqlite")
DEFAULT_EVAL_PATH = Path("artifacts/rag_eval/nalus_eval.json")
DEFAULT_SOURCE_ID = 1
MAX_CASES = 8


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def _marker_present(text: str, marker: str) -> bool:
    return _normalize_text(marker) in _normalize_text(text)


def _document_id_from_chunk_id(chunk_id: str) -> str:
    if "__chunk_" in chunk_id:
        return chunk_id.split("__chunk_", maxsplit=1)[0]
    return chunk_id


def _find_literal_marker(term: str, text: str) -> str | None:
    if _marker_present(text, term):
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return match.group(0)
        return term

    pattern = re.compile(re.escape(term), re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group(0)

    words = [word for word in term.split() if word]
    if not words:
        return None

    lower_text = text.lower()
    first_word = words[0].lower()
    first_prefix = first_word[: max(6, min(len(first_word), 12))]
    search_from = 0
    while search_from < len(lower_text):
        start = lower_text.find(first_prefix, search_from)
        if start == -1:
            break

        end = start + len(first_prefix)
        while end < len(text) and text[end].isalnum():
            end += 1

        matched = True
        for word in words[1:]:
            next_pos = lower_text.find(word.lower(), end, end + 120)
            if next_pos == -1:
                matched = False
                break
            end = next_pos + len(word)

        if matched:
            return text[start:end]

        search_from = start + 1

    return None


def _pick_marker(source_terms: list[str], scoped_text: str) -> tuple[str, list[str]]:
    discovered: list[str] = []
    for term in sorted(source_terms, key=len, reverse=True):
        literal = _find_literal_marker(term, scoped_text)
        if literal is not None:
            discovered.append(literal)

    if not discovered:
        raise ValueError(f"No source term found in scoped chunk text: {source_terms!r}")

    marker = discovered[0]
    aliases = [item for item in discovered[1:] if item != marker]
    return marker, aliases


def _load_chunks(chunks_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(chunks_path)
    required = {"chunk_id", "document_id", "chunk_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Chunks parquet missing columns: {sorted(missing)}")
    return df


def _write_sqlite(*, df: pd.DataFrame, sqlite_path: Path, source_id: int) -> int:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if sqlite_path.exists():
        sqlite_path.unlink()

    connection = sqlite3.connect(sqlite_path)
    try:
        connection.execute(
            """
            CREATE TABLE rag_chunks (
                id INTEGER PRIMARY KEY,
                source_id INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_metadata TEXT NOT NULL,
                validation_status TEXT NOT NULL DEFAULT 'valid'
            )
            """
        )
        rows: list[tuple[int, int, str, str, str]] = []
        for index, row in enumerate(df.itertuples(index=False), start=1):
            metadata = {
                "source_document_id": str(row.document_id),
                "chunk_id": str(row.chunk_id),
                "case_number": str(getattr(row, "case_number", "") or ""),
                "section_type": str(getattr(row, "section_type", "") or ""),
                "legal_area": str(getattr(row, "legal_area", "") or ""),
            }
            rows.append(
                (
                    index,
                    source_id,
                    str(row.chunk_text or ""),
                    json.dumps(metadata, ensure_ascii=False),
                    "valid",
                )
            )
        connection.executemany(
            "INSERT INTO rag_chunks (id, source_id, chunk_text, chunk_metadata, validation_status) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        connection.commit()
    finally:
        connection.close()
    return len(rows)


def _build_eval_dataset(
    *,
    relevance_path: Path,
    chunks_df: pd.DataFrame,
    eval_path: Path,
    max_cases: int,
) -> dict[str, object]:
    payload = json.loads(relevance_path.read_text(encoding="utf-8"))
    positive_cases = list(payload.get("positive_answerable") or [])[:max_cases]
    if not positive_cases:
        raise ValueError("No positive_answerable cases found in relevance eval dataset.")

    chunk_lookup = {
        str(row.chunk_id): row
        for row in chunks_df.itertuples(index=False)
    }

    cases: list[dict[str, object]] = []
    source_documents: dict[str, dict[str, object]] = {}

    for index, item in enumerate(positive_cases, start=1):
        query = str(item["query"])
        source_chunk_ids = [str(chunk_id) for chunk_id in item.get("source_chunk_ids") or []]
        source_terms = [str(term) for term in item.get("source_terms") or []]
        if not source_chunk_ids or not source_terms:
            raise ValueError(f"Case {query!r} is missing source_chunk_ids or source_terms.")

        document_ids = sorted({_document_id_from_chunk_id(chunk_id) for chunk_id in source_chunk_ids})
        scoped_rows = [chunk_lookup[chunk_id] for chunk_id in source_chunk_ids if chunk_id in chunk_lookup]
        scoped_text = " ".join(str(row.chunk_text) for row in scoped_rows)

        try:
            marker, aliases = _pick_marker(source_terms, scoped_text)
        except ValueError:
            document_chunks = chunks_df[chunks_df["document_id"].isin(document_ids)]
            expanded_text = " ".join(str(text) for text in document_chunks["chunk_text"].tolist())
            marker, aliases = _pick_marker(source_terms, expanded_text)
            scoped_text = expanded_text
        scope_type = "multi_document" if len(document_ids) >= 2 else "document"
        if scope_type == "multi_document" and len(document_ids) < 2:
            scope_type = "document"

        case_id = f"nsoud-positive-{index:02d}"
        cases.append(
            {
                "id": case_id,
                "question": query,
                "expected_answer_type": "short_fact",
                "test_type": "short_fact" if scope_type == "document" else "multi_document",
                "source_scope": {
                    "scope_type": scope_type,
                    "document_ids": document_ids,
                },
                "required_evidence": [
                    {
                        "marker": marker,
                        "aliases": aliases,
                    }
                ],
                "minimum_coverage": 1.0,
                "allow_partial": False,
                "expected_citation_count_min": 1,
                "difficulty": "medium",
                "language": "cs",
                "expected_long_context": False,
                "minimum_context_chars": 0,
            }
        )

        for chunk_id in source_chunk_ids:
            if chunk_id not in chunk_lookup:
                continue
            row = chunk_lookup[chunk_id]
            document_id = str(row.document_id)
            excerpt = str(row.chunk_text or "")[:4000]
            if document_id not in source_documents:
                source_documents[document_id] = {
                    "document_id": document_id,
                    "content": excerpt,
                }
            elif _marker_present(excerpt, marker) and not _marker_present(
                str(source_documents[document_id]["content"]), marker
            ):
                source_documents[document_id]["content"] = excerpt

        for document_id in document_ids:
            if _marker_present(str(source_documents.get(document_id, {}).get("content", "")), marker):
                continue
            document_chunks = chunks_df[chunks_df["document_id"] == document_id]
            for text in document_chunks["chunk_text"].tolist():
                excerpt = str(text or "")[:4000]
                if _marker_present(excerpt, marker):
                    source_documents[document_id] = {
                        "document_id": document_id,
                        "content": excerpt,
                    }
                    break

    dataset = {
        "dataset_id": "nalus-nsoud-pilot-v1",
        "name": "NALUS NSOud Retrieval Pilot",
        "description": (
            "Pilot retrieval benchmark built from existing NSOud rag_ready artifacts "
            "and relevance_eval_dataset positive_answerable cases."
        ),
        "project_name": "NALUS Scraper",
        "metadata": {
            "court": "Nejvyšší soud ČR",
            "batch": "2025_01_03",
            "chunking_strategy": "document_section_aware",
            "external_dataset": True,
            "pilot": True,
        },
        "cases": cases,
        "source_documents": list(source_documents.values()),
    }

    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare rag-eval SQLite corpus and eval JSON.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--relevance-dataset", type=Path, default=DEFAULT_RELEVANCE_DATASET)
    parser.add_argument("--sqlite-out", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--eval-out", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--source-id", type=int, default=DEFAULT_SOURCE_ID)
    parser.add_argument("--max-cases", type=int, default=MAX_CASES)
    args = parser.parse_args()

    chunks_df = _load_chunks(args.chunks)
    row_count = _write_sqlite(df=chunks_df, sqlite_path=args.sqlite_out, source_id=args.source_id)
    dataset = _build_eval_dataset(
        relevance_path=args.relevance_dataset,
        chunks_df=chunks_df,
        eval_path=args.eval_out,
        max_cases=args.max_cases,
    )

    print(f"sqlite_path: {args.sqlite_out}")
    print(f"sqlite_rows: {row_count}")
    print(f"eval_path: {args.eval_out}")
    print(f"eval_cases: {len(dataset['cases'])}")
    print(f"source_documents: {len(dataset['source_documents'])}")


if __name__ == "__main__":
    main()
