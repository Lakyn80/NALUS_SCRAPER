from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from collections.abc import Callable
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
    """SQLite-backed BM25 sidecar, loaded lazily on first query.

    Full-corpus sidecars are too large to materialize as Python strings. The
    in-memory structure is an inverted index (same BM25 k1/b/IDF formula as
    before); chunk text and full metadata are hydrated from SQLite for the
    returned top-k only.
    """

    def __init__(
        self,
        path: Path,
        *,
        k1: float,
        b: float,
        index_id: str,
    ) -> None:
        self._path = Path(path)
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

    def search(
        self,
        query: str,
        top_k: int,
        *,
        metadata_predicate: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[RetrievedChunk]:
        if self._index is None:
            self.assert_ready()
            self._index = _Bm25Index.from_sqlite(
                self._path, k1=self._k1, b=self._b
            )
        return self._index.search(
            query,
            top_k=top_k,
            index_id=self._index_id,
            metadata_predicate=metadata_predicate,
        )

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
    def __init__(
        self,
        *,
        ids: list[str],
        doc_lengths: list[int],
        metadata: list[dict[str, Any]],
        postings: dict[str, list[tuple[int, int]]],
        idf: dict[str, float],
        avg_doc_length: float,
        k1: float,
        b: float,
        texts: list[str] | None = None,
        sqlite_path: Path | None = None,
    ) -> None:
        self._ids = ids
        self._doc_lengths = doc_lengths
        self._metadata = metadata
        self._postings = postings
        self._idf = idf
        self._avg_doc_length = avg_doc_length
        self._k1 = k1
        self._b = b
        self._texts = texts
        self._sqlite_path = sqlite_path

    @classmethod
    def from_records(cls, records: list[Bm25Record], *, k1: float, b: float) -> "_Bm25Index":
        ids: list[str] = []
        texts: list[str] = []
        metadata: list[dict[str, Any]] = []
        token_lists: list[list[str]] = []
        for record in records:
            ids.append(record.id)
            texts.append(record.text)
            metadata.append(dict(record.metadata))
            token_lists.append(_tokenize(record.text))
        return cls._from_token_lists(
            ids=ids,
            token_lists=token_lists,
            metadata=metadata,
            k1=k1,
            b=b,
            texts=texts,
            sqlite_path=None,
        )

    @classmethod
    def from_sqlite(cls, path: Path, *, k1: float, b: float) -> "_Bm25Index":
        ids: list[str] = []
        metadata: list[dict[str, Any]] = []
        doc_lengths: list[int] = []
        document_frequency: Counter[str] = Counter()
        postings: dict[str, list[tuple[int, int]]] = {}
        with sqlite3.connect(path) as connection:
            table_name = _select_chunks_table(connection)
            columns = [
                item[1] for item in connection.execute(f"PRAGMA table_info({table_name})")
            ]
            rows = connection.execute(f"SELECT * FROM {table_name}")
            for row in rows:
                item = dict(zip(columns, row, strict=True))
                chunk_id = str(
                    item.get("id")
                    or item.get("chunk_id")
                    or item.get("original_id")
                    or ""
                ).strip()
                text = str(item.get("text") or item.get("chunk_text") or "").strip()
                if not chunk_id or not text:
                    continue
                payload = _parse_metadata(
                    item.get("metadata") or item.get("payload") or item.get("payload_json")
                )
                _set_metadata_defaults(payload, item)
                tokens = _tokenize(text)
                counts = Counter(tokens)
                doc_index = len(ids)
                ids.append(chunk_id)
                metadata.append(_lite_metadata(payload, item))
                doc_lengths.append(len(tokens))
                document_frequency.update(counts.keys())
                for term, term_frequency in counts.items():
                    postings.setdefault(term, []).append((doc_index, term_frequency))
        doc_count = len(ids)
        avg_doc_length = (sum(doc_lengths) / doc_count) if doc_count else 0.0
        idf = {
            term: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }
        return cls(
            ids=ids,
            doc_lengths=doc_lengths,
            metadata=metadata,
            postings=postings,
            idf=idf,
            avg_doc_length=avg_doc_length,
            k1=k1,
            b=b,
            texts=None,
            sqlite_path=Path(path),
        )

    @classmethod
    def _from_token_lists(
        cls,
        *,
        ids: list[str],
        token_lists: list[list[str]],
        metadata: list[dict[str, Any]],
        k1: float,
        b: float,
        texts: list[str] | None,
        sqlite_path: Path | None,
    ) -> "_Bm25Index":
        doc_count = len(ids)
        doc_lengths = [len(tokens) for tokens in token_lists]
        avg_doc_length = (sum(doc_lengths) / doc_count) if doc_count else 0.0
        document_frequency: Counter[str] = Counter()
        postings: dict[str, list[tuple[int, int]]] = {}
        for index, tokens in enumerate(token_lists):
            counts = Counter(tokens)
            document_frequency.update(counts.keys())
            for term, term_frequency in counts.items():
                postings.setdefault(term, []).append((index, term_frequency))
        idf = {
            term: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }
        return cls(
            ids=ids,
            doc_lengths=doc_lengths,
            metadata=metadata,
            postings=postings,
            idf=idf,
            avg_doc_length=avg_doc_length,
            k1=k1,
            b=b,
            texts=texts,
            sqlite_path=sqlite_path,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int,
        index_id: str,
        metadata_predicate: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[RetrievedChunk]:
        query_terms = _tokenize(query)
        if not query_terms or not self._ids:
            return []

        scores: dict[int, float] = {}
        for term in query_terms:
            idf = self._idf.get(term)
            if not idf:
                continue
            for doc_index, term_frequency in self._postings.get(term, ()):
                if metadata_predicate is not None and not metadata_predicate(
                    self._metadata[doc_index]
                ):
                    continue
                doc_length = self._doc_lengths[doc_index]
                if doc_length == 0 or self._avg_doc_length == 0:
                    continue
                denominator = term_frequency + self._k1 * (
                    1 - self._b + self._b * doc_length / self._avg_doc_length
                )
                scores[doc_index] = (
                    scores.get(doc_index, 0.0)
                    + idf * (term_frequency * (self._k1 + 1)) / denominator
                )

        ranked = sorted(
            (
                (score, self._ids[index], index)
                for index, score in scores.items()
                if score > 0
            ),
            key=lambda item: (-item[0], item[1]),
        )[:top_k]
        hydrated = self._hydrate([index for _, _, index in ranked])
        results: list[RetrievedChunk] = []
        for score, chunk_id, doc_index in ranked:
            text, metadata = hydrated[doc_index]
            payload = dict(metadata)
            payload["bm25_score"] = score
            payload["bm25_index_id"] = index_id
            results.append(
                RetrievedChunk(
                    id=chunk_id,
                    text=text,
                    score=score,
                    source="bm25",
                    metadata=payload,
                )
            )
        return results

    def _hydrate(self, indices: list[int]) -> dict[int, tuple[str, dict[str, Any]]]:
        if not indices:
            return {}
        if self._texts is not None:
            return {
                index: (self._texts[index], dict(self._metadata[index]))
                for index in indices
            }
        if self._sqlite_path is None:
            return {index: ("", dict(self._metadata[index])) for index in indices}
        wanted = {self._ids[index]: index for index in indices}
        hydrated: dict[int, tuple[str, dict[str, Any]]] = {
            index: ("", dict(self._metadata[index])) for index in indices
        }
        placeholders = ",".join("?" for _ in wanted)
        with sqlite3.connect(self._sqlite_path) as connection:
            table_name = _select_chunks_table(connection)
            columns = [
                item[1] for item in connection.execute(f"PRAGMA table_info({table_name})")
            ]
            id_column = "chunk_id" if "chunk_id" in columns else "id"
            rows = connection.execute(
                f"SELECT * FROM {table_name} WHERE {id_column} IN ({placeholders})",
                list(wanted.keys()),
            ).fetchall()
        for row in rows:
            item = dict(zip(columns, row, strict=True))
            chunk_id = str(
                item.get("id") or item.get("chunk_id") or item.get("original_id") or ""
            ).strip()
            doc_index = wanted.get(chunk_id)
            if doc_index is None:
                continue
            text = str(item.get("text") or item.get("chunk_text") or "").strip()
            payload = _parse_metadata(
                item.get("metadata") or item.get("payload") or item.get("payload_json")
            )
            _set_metadata_defaults(payload, item)
            hydrated[doc_index] = (text, payload)
        return hydrated


def _lite_metadata(metadata: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    lite: dict[str, Any] = {}
    for key in (
        "source",
        "document_id",
        "source_document_id",
        "ecli",
        "canonical_document_id",
        "case_number",
        "case_reference",
        "spisova_znacka",
        "court",
        "court_name",
        "decision_date",
        "document_type",
        "chunk_index",
    ):
        value = metadata.get(key, row.get(key))
        if value is not None and str(value).strip():
            lite[key] = value
    return lite


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


def _set_metadata_defaults(metadata: dict[str, Any], row: dict[str, Any]) -> None:
    for key in (
        "source",
        "document_id",
        "source_document_id",
        "ecli",
        "case_number",
        "spisova_znacka",
        "court",
        "decision_date",
        "chunk_index",
    ):
        metadata.setdefault(key, row.get(key))

    metadata.setdefault(
        "case_reference",
        row.get("case_reference") or row.get("case_number") or row.get("spisova_znacka"),
    )


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]
