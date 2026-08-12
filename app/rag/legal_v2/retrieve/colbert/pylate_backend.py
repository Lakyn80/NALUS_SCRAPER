"""PyLate-backed ColBERT implementation (lazy load; async public API)."""

from __future__ import annotations

import asyncio
import importlib.metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.rag.legal_v2.retrieve.colbert.backend import import_colbert_library
from app.rag.legal_v2.retrieve.colbert.config import ColbertConfig
from app.rag.legal_v2.retrieve.colbert.errors import (
    ColbertBackendUnavailableError,
    ColbertConfigurationError,
    ColbertIndexError,
    ColbertMappingError,
)
from app.rag.legal_v2.retrieve.colbert.mapping import (
    ColbertChunkMapping,
    ColbertMappingRow,
    load_mapping_jsonl,
    write_mapping_jsonl,
)
from app.rag.legal_v2.retrieve.colbert.models import (
    ColbertHit,
    ColbertIndexBuildResult,
)


def _resolve_device(preferred: str) -> str:
    pref = (preferred or "auto").strip().lower()
    if pref not in {"auto", "cpu", "cuda"}:
        pref = "auto"
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def _library_version() -> str:
    try:
        return importlib.metadata.version("pylate")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


class PyLateColbertBackend:
    """Concrete ColBERT backend using PyLate + PLAID.

    Blocking library calls always run via ``asyncio.to_thread``.
    Search/build share a semaphore so CUDA/model access stays bounded.
    """

    def __init__(self, config: ColbertConfig) -> None:
        config.validate()
        self._config = config
        self._device = _resolve_device(config.device)
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(int(config.concurrency_limit))
        self._model: Any | None = None
        self._index: Any | None = None
        self._retriever: Any | None = None
        self._mapping: ColbertChunkMapping | None = None
        self._initialized = False

    @property
    def config(self) -> ColbertConfig:
        return self._config

    @property
    def device(self) -> str:
        return self._device

    async def initialize(self) -> None:
        async with self._lock:
            if self._initialized:
                return
            await asyncio.to_thread(self._initialize_sync)

    async def close(self) -> None:
        async with self._lock:
            self._model = None
            self._index = None
            self._retriever = None
            self._mapping = None
            self._initialized = False

    async def __aenter__(self) -> "PyLateColbertBackend":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        await self.close()

    async def search(self, query: str, *, top_k: int) -> Sequence[ColbertHit]:
        cleaned = str(query or "").strip()
        if not cleaned:
            raise ColbertConfigurationError("query must not be blank")
        if int(top_k) < 1:
            raise ColbertConfigurationError("top_k must be >= 1")
        await self.initialize()
        async with self._semaphore:
            return await asyncio.to_thread(self._search_sync, cleaned, int(top_k))

    async def build_index(
        self,
        documents: Sequence[Mapping[str, Any]],
        *,
        source_collection: str | None = None,
    ) -> ColbertIndexBuildResult:
        rows = list(documents)
        collection = (
            source_collection
            or self._config.source_collection
            or "unknown"
        )
        async with self._semaphore:
            return await asyncio.to_thread(
                self._build_index_sync,
                rows,
                collection,
            )

    def _initialize_sync(self) -> None:
        mapping_path = self._config.resolved_mapping_path()
        index_root = Path(self._config.index_path)
        if not index_root.exists():
            raise ColbertIndexError(f"ColBERT index path missing: {index_root}")
        if not mapping_path.exists():
            raise ColbertMappingError(f"ColBERT mapping missing: {mapping_path}")

        import_colbert_library()
        mapping = load_mapping_jsonl(mapping_path)
        self._mapping = mapping
        local_files_only = not bool(self._config.allow_download)
        from pylate import indexes, models, retrieve

        self._model = models.ColBERT(
            model_name_or_path=self._config.model_name,
            device=self._device,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        self._index = indexes.PLAID(
            index_folder=str(index_root),
            index_name=self._config.index_name,
            override=False,
            use_fast=True,
            use_triton=False,
        )
        self._retriever = retrieve.ColBERT(index=self._index)
        self._initialized = True

    def _search_sync(self, query: str, top_k: int) -> list[ColbertHit]:
        if self._model is None or self._retriever is None or self._mapping is None:
            raise ColbertBackendUnavailableError("ColBERT backend is not initialized")
        query_embeddings = self._model.encode(
            [query],
            batch_size=1,
            is_query=True,
            show_progress_bar=False,
        )
        raw = self._retriever.retrieve(
            queries_embeddings=query_embeddings,
            k=top_k,
            device=self._device,
        )
        if not raw:
            return []
        first = raw[0]
        hits: list[ColbertHit] = []
        for rank, item in enumerate(first, start=1):
            if isinstance(item, Mapping):
                colbert_id = str(item.get("id") or item.get("document_id") or "")
                score = float(item.get("score") or 0.0)
            else:
                colbert_id = str(getattr(item, "id", "") or "")
                score = float(getattr(item, "score", 0.0) or 0.0)
            if not colbert_id:
                raise ColbertMappingError("ColBERT search returned a hit without id")
            row = self._mapping.require(colbert_id)
            hits.append(
                ColbertHit(
                    document_id=row.document_id,
                    chunk_id=row.chunk_id,
                    rank=rank,
                    score=score,
                    text=row.text,
                    metadata=dict(row.metadata),
                )
            )
        return hits

    def _build_index_sync(
        self,
        documents: list[Mapping[str, Any]],
        source_collection: str,
    ) -> ColbertIndexBuildResult:
        import_colbert_library()
        from pylate import indexes, models

        expected = int(self._config.expected_chunk_count)
        index_root = Path(self._config.index_path)
        mapping_path = self._config.resolved_mapping_path()
        index_root.mkdir(parents=True, exist_ok=True)
        mapping_path.parent.mkdir(parents=True, exist_ok=True)

        prepared: list[ColbertMappingRow] = []
        seen_chunk_ids: set[str] = set()
        empty_texts = 0
        for row in documents:
            chunk_id = str(row.get("chunk_id") or "").strip()
            document_id = str(row.get("document_id") or "").strip()
            text = str(row.get("text") or "")
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            if not chunk_id or not document_id:
                raise ColbertConfigurationError(
                    "each ColBERT source row requires chunk_id and document_id"
                )
            if chunk_id in seen_chunk_ids:
                raise ColbertConfigurationError(f"duplicate chunk_id in source: {chunk_id!r}")
            seen_chunk_ids.add(chunk_id)
            if not text.strip():
                empty_texts += 1
            prepared.append(
                ColbertMappingRow(
                    colbert_id=chunk_id,
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=text,
                    metadata=dict(metadata),
                )
            )

        missing = max(0, expected - len(prepared)) if expected else 0
        if empty_texts:
            return ColbertIndexBuildResult(
                status="failed",
                source_collection=source_collection,
                expected_chunk_count=expected,
                indexed_chunk_count=0,
                mapping_row_count=0,
                duplicate_chunk_ids=0,
                missing_chunk_ids=missing,
                empty_texts=empty_texts,
                index_path=str(index_root),
                mapping_path=str(mapping_path),
                model_name=self._config.model_name,
                library="pylate",
                library_version=_library_version(),
                device=self._device,
                diagnostics={"reason": "empty_texts"},
            )

        self._model = models.ColBERT(
            model_name_or_path=self._config.model_name,
            device=self._device,
            trust_remote_code=False,
            local_files_only=not bool(self._config.allow_download),
        )
        embeddings = self._model.encode(
            [row.text for row in prepared],
            batch_size=int(self._config.batch_size),
            is_query=False,
            show_progress_bar=True,
        )
        index = indexes.PLAID(
            index_folder=str(index_root),
            index_name=self._config.index_name,
            override=True,
            use_fast=True,
            use_triton=False,
        )
        index.add_documents(
            documents_ids=[row.colbert_id for row in prepared],
            documents_embeddings=embeddings,
        )
        mapping_count = write_mapping_jsonl(mapping_path, prepared)
        self._index = index
        self._retriever = None
        self._mapping = ColbertChunkMapping(
            rows={row.colbert_id: row for row in prepared}
        )
        self._initialized = False  # force search path to reopen cleanly

        indexed = len(prepared)
        integrity_ok = (
            indexed == expected
            and mapping_count == expected
            and empty_texts == 0
            and missing == 0
        )
        return ColbertIndexBuildResult(
            status="ok" if integrity_ok else "failed",
            source_collection=source_collection,
            expected_chunk_count=expected,
            indexed_chunk_count=indexed,
            mapping_row_count=mapping_count,
            duplicate_chunk_ids=0,
            missing_chunk_ids=missing,
            empty_texts=empty_texts,
            index_path=str(index_root),
            mapping_path=str(mapping_path),
            model_name=self._config.model_name,
            library="pylate",
            library_version=_library_version(),
            device=self._device,
            diagnostics={
                "batch_size": int(self._config.batch_size),
                "index_name": self._config.index_name,
                "use_fast": True,
                "use_triton": False,
                "index_backend": "pylate.indexes.PLAID/fast_plaid",
            },
        )
