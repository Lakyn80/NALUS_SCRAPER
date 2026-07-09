from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.rag.retrieval.errors import RetrievalConfigurationError
from app.rag.retrieval.models import RetrievedChunk


_TOKEN_RE = re.compile(r"[\wěščřžýáíéúůďťňó]+", re.IGNORECASE)


@dataclass(frozen=True)
class Bm25Record:
    id: str
    text: str
    metadata: dict[str, Any]


class Bm25Sidecar:
    """SQLite-backed BM25 sidecar, loaded lazily on first query."""

    def __init__(
        self,
        path: Path,
        *,
        k1: float,
        b: float,
        index_id: str,
    ) -> None:
        self._path = path
        self._k1 = k1
        self._b = b
        self._index_id = index_id
        self._index: _Bm25Index | None = None

    @property
    def index_id(self) -> str:
        return self._index_id

    def assert_ready(self) -> None:
        if not self._path.exists():
            raise RetrievalConfigurationError(
                f"BM25 sidecar is required for production retrieval but is missing: {self._path}"
            )

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self._index is None:
            self.assert_ready()
            self._index = _Bm25Index.from_records(_load_records(self._path), k1=self._k1, b=self._b)
        return self._index.search(query, top_k=top_k, index_id=self._index_id)

    @classmethod
    def from_records(
        cls,
        records: list[Bm25Record],
        *,
        k1: float,
        b: float,
        index_id: str,
    ) -> "Bm25Sidecar":
        sidecar = cls(Path("<memory>"), k1=k1, b=b, index_id=index_id)
        sidecar._index = _Bm25Index.from_records(records, k1=k1, b=b)
        return sidecar


class _Bm25Index:
    def __init__(self, records: list[Bm25Record], *, k1: float, b: float) -> None:
        self._records = records
        self._k1 = k1
        self._b = b
        self._tokens = [_tokenize(record.text) for record in records]
        self._term_counts = [Counter(tokens) for tokens in self._tokens]
        self._doc_lengths = [len(tokens) for tokens in self._tokens]
        self._avg_doc_length = (
            sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0
        )
        self._idf = self._build_idf()

    @classmethod
    def from_records(cls, records: list[Bm25Record], *, k1: float, b: float) -> "_Bm25Index":
        return cls(records, k1=k1, b=b)

    def search(self, query: str, *, top_k: int, index_id: str) -> list[RetrievedChunk]:
        query_terms = _tokenize(query)
        if not query_terms or not self._records:
            return []

        scored: list[RetrievedChunk] = []
        for index, record in enumerate(self._records):
            score = self._score_document(index, query_terms)
            if score <= 0:
                continue
            metadata = dict(record.metadata)
            metadata["bm25_score"] = score
            metadata["bm25_index_id"] = index_id
            scored.append(
                RetrievedChunk(
                    id=record.id,
                    text=record.text,
                    score=score,
                    source="bm25",
                    metadata=metadata,
                )
            )

        scored.sort(key=lambda chunk: (-chunk.score, chunk.id))
        return scored[:top_k]

    def _build_idf(self) -> dict[str, float]:
        doc_count = len(self._records)
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))

        return {
            term: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def _score_document(self, index: int, query_terms: list[str]) -> float:
        score = 0.0
        term_counts = self._term_counts[index]
        doc_length = self._doc_lengths[index]
        if doc_length == 0 or self._avg_doc_length == 0:
            return 0.0

        for term in query_terms:
            term_frequency = term_counts.get(term, 0)
            if term_frequency == 0:
                continue
            denominator = term_frequency + self._k1 * (
                1 - self._b + self._b * doc_length / self._avg_doc_length
            )
            score += self._idf.get(term, 0.0) * (term_frequency * (self._k1 + 1)) / denominator
        return score


def _load_records(path: Path) -> list[Bm25Record]:
    with sqlite3.connect(path) as connection:
        table_name = _select_chunks_table(connection)
        rows = connection.execute(f"SELECT * FROM {table_name}").fetchall()
        columns = [item[1] for item in connection.execute(f"PRAGMA table_info({table_name})")]

    records: list[Bm25Record] = []
    for row in rows:
        item = dict(zip(columns, row, strict=True))
        chunk_id = str(item.get("id") or item.get("chunk_id") or item.get("original_id") or "").strip()
        text = str(item.get("text") or item.get("chunk_text") or "").strip()
        if not chunk_id or not text:
            continue
        metadata = _parse_metadata(item.get("metadata") or item.get("payload") or item.get("payload_json"))
        metadata.setdefault("source", item.get("source"))
        metadata.setdefault("document_id", item.get("document_id"))
        metadata.setdefault("chunk_index", item.get("chunk_index"))
        records.append(Bm25Record(id=chunk_id, text=text, metadata=metadata))
    return records


def _select_chunks_table(connection: sqlite3.Connection) -> str:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    for table_name in ("bm25_chunks", "chunks", "rag_chunks"):
        if table_name in tables:
            return table_name
    raise RetrievalConfigurationError("BM25 sidecar does not contain a supported chunks table.")


def _parse_metadata(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return dict(decoded) if isinstance(decoded, dict) else {}


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]
