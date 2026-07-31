from __future__ import annotations

import json
import sqlite3
import time
from importlib import import_module
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.rag.ingest.qdrant_ingest import point_id_from_original_id
from app.rag.legal_v2.adapters import LegalAdapterRegistry, LegalSourceDocument
from app.rag.legal_v2.audit import CHUNKER_VERSION, PARSER_VERSION, audit_documents
from app.rag.legal_v2.chunking import HierarchicalChunkConfig, RetrievalChildChunk, build_hierarchical_chunks
from app.rag.legal_v2.indexing import (
    LEGAL_V2_BM25_INDEX_ID,
    LEGAL_V2_COLLECTION_NAME,
    LEGAL_V2_PROFILE,
    payload_for_child_chunk,
)
from app.rag.retrieval.provenance import content_checksum

BUILDER_VERSION = "legal_v2_index_builder_v1"
PROTECTED_COLLECTIONS = {"nalus", "nalus_live", "nalus_bge_m3_chunks_v1"}


@dataclass(frozen=True)
class LegalV2BuildConfig:
    collection_name: str = LEGAL_V2_COLLECTION_NAME
    bm25_path: Path = Path("storage/rag/bm25/nalus_legal_paragraph_bm25_v2.sqlite")
    output_dir: Path = Path("artifacts/legal_v2/index_build")
    recreate_collection: bool = False
    overwrite_bm25: bool = False
    resume: bool = False
    batch_size: int = 64

    def validate(self) -> None:
        if self.collection_name in PROTECTED_COLLECTIONS or self.collection_name.startswith("nalus_stable_"):
            raise ValueError(f"Refusing protected collection: {self.collection_name}")
        if self.collection_name != LEGAL_V2_COLLECTION_NAME:
            raise ValueError(f"Legal v2 builder must write only {LEGAL_V2_COLLECTION_NAME}.")
        if self.bm25_path.exists() and not (self.overwrite_bm25 or self.resume):
            raise ValueError(f"BM25 sidecar already exists: {self.bm25_path}")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")


@dataclass(frozen=True)
class LegalV2BuildManifest:
    collection_name: str
    bm25_index_id: str
    bm25_path: str
    source_corpus: str
    source_document_count: int
    indexed_document_count: int
    excluded_document_count: int
    chunk_count: int
    embedding_model: str
    embedding_dimension: int
    parser_version: str
    chunker_version: str
    builder_version: str
    build_timestamp: str
    git_commit: str
    dirty: bool
    corpus_hash: str
    failed_documents: list[dict[str, Any]] = field(default_factory=list)
    resume_state: dict[str, Any] = field(default_factory=dict)
    validation_status: str = "pass"
    qdrant_write_status: str = "not_run"
    bm25_write_status: str = "not_run"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_legal_v2_index(
    *,
    documents: list[LegalSourceDocument],
    embedder: Any,
    qdrant_client: Any,
    config: LegalV2BuildConfig | None = None,
    chunk_config: HierarchicalChunkConfig | None = None,
    registry: LegalAdapterRegistry | None = None,
    git_commit: str = "unknown",
    dirty: bool = True,
) -> LegalV2BuildManifest:
    started = time.perf_counter()
    config = config or LegalV2BuildConfig()
    config.validate()
    chunk_config = chunk_config or HierarchicalChunkConfig()
    registry = registry or LegalAdapterRegistry()
    audit = audit_documents(documents, config=chunk_config, registry=registry)
    approved_ids = {
        item.document_id for item in audit.documents if item.status == "pass"
    }
    failed_documents = [
        {"document_id": item.document_id, "reasons": item.reasons}
        for item in audit.documents
        if item.status != "pass"
    ]
    chunks = _chunks_for_approved_documents(
        documents=documents,
        approved_ids=approved_ids,
        chunk_config=chunk_config,
        registry=registry,
    )
    if documents and not chunks:
        raise ValueError("Parser-approved documents produced no v2 chunks.")
    payloads = [payload_for_child_chunk(chunk) for chunk in chunks]
    _validate_payload_identity(payloads)
    _prepare_collection(qdrant_client, config)
    vectors = embedder.embed_texts([payload["text"] for payload in payloads])
    _validate_vectors(vectors, LEGAL_V2_PROFILE.embedding_dimension)
    _upsert_payloads(qdrant_client, collection_name=config.collection_name, payloads=payloads, vectors=vectors, batch_size=config.batch_size)
    _validate_qdrant_identity(qdrant_client, collection_name=config.collection_name, payloads=payloads)
    write_bm25_sidecar(payloads, config.bm25_path, overwrite=config.overwrite_bm25 or config.resume)
    _validate_bm25_identity(payloads, config.bm25_path)
    manifest = LegalV2BuildManifest(
        collection_name=config.collection_name,
        bm25_index_id=LEGAL_V2_BM25_INDEX_ID,
        bm25_path=str(config.bm25_path),
        source_corpus=_source_corpus(documents),
        source_document_count=len(documents),
        indexed_document_count=len(approved_ids),
        excluded_document_count=len(documents) - len(approved_ids),
        chunk_count=len(chunks),
        embedding_model=LEGAL_V2_PROFILE.embedding_model,
        embedding_dimension=LEGAL_V2_PROFILE.embedding_dimension,
        parser_version=PARSER_VERSION,
        chunker_version=CHUNKER_VERSION,
        builder_version=BUILDER_VERSION,
        build_timestamp=_utc_now(),
        git_commit=git_commit,
        dirty=dirty,
        corpus_hash=_corpus_hash(documents),
        failed_documents=failed_documents,
        resume_state={"resume": config.resume, "duration_ms": _elapsed_ms(started)},
        qdrant_write_status="pass",
        bm25_write_status="pass",
    )
    write_build_manifest(manifest, config.output_dir)
    return manifest


def write_bm25_sidecar(payloads: list[dict[str, Any]], path: Path, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not overwrite:
            raise ValueError(f"BM25 sidecar already exists: {path}")
        path.unlink()
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE bm25_chunks (
                chunk_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata TEXT NOT NULL,
                document_id TEXT,
                section_type TEXT,
                paragraph_ids TEXT,
                qdrant_collection TEXT,
                retrieval_profile TEXT,
                bm25_index_id TEXT
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO bm25_chunks (
                chunk_id, text, metadata, document_id, section_type, paragraph_ids,
                qdrant_collection, retrieval_profile, bm25_index_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    payload["chunk_id"],
                    payload["text"],
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    payload.get("document_id"),
                    payload.get("section_type"),
                    json.dumps(payload.get("paragraph_ids") or [], ensure_ascii=False),
                    payload.get("qdrant_collection"),
                    payload.get("retrieval_profile"),
                    payload.get("bm25_index_id"),
                )
                for payload in payloads
            ],
        )
        connection.commit()


def write_build_manifest(manifest: LegalV2BuildManifest, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "legal_v2_build_manifest.json"
    markdown_path = output_dir / "legal_v2_build_manifest.md"
    payload = manifest.to_dict()
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_manifest_markdown(payload), encoding="utf-8")
    return json_path, markdown_path


def _chunks_for_approved_documents(
    *,
    documents: list[LegalSourceDocument],
    approved_ids: set[str],
    chunk_config: HierarchicalChunkConfig,
    registry: LegalAdapterRegistry,
) -> list[RetrievalChildChunk]:
    chunks: list[RetrievalChildChunk] = []
    for source_document in documents:
        if source_document.document_id not in approved_ids:
            continue
        parsed = registry.adapter_for(source_document.source).parse(source_document)
        content_hash = content_checksum(parsed.normalized_text)
        result = build_hierarchical_chunks(parsed, config=chunk_config)
        parent_windows_by_child_id = {
            child_id: window
            for window in result.parent_windows
            for child_id in window.child_chunk_ids
        }
        for chunk in result.child_chunks:
            parent_window = parent_windows_by_child_id.get(chunk.chunk_id)
            metadata = dict(chunk.metadata)
            metadata.update(
                {
                    "language": "cs",
                    "source": source_document.source,
                    "court": parsed.metadata.get("court") or parsed.metadata.get("court_name"),
                    "case_reference": parsed.metadata.get("case_reference") or parsed.metadata.get("spisova_znacka"),
                    "decision_date": parsed.metadata.get("decision_date") or parsed.metadata.get("date"),
                    "document_type": parsed.metadata.get("document_type") or parsed.metadata.get("decision_form"),
                    "parser_version": PARSER_VERSION,
                    "chunker_version": CHUNKER_VERSION,
                    "document_content_hash": content_hash,
                    "parent_window_id": parent_window.window_id if parent_window else None,
                    "parent_window_paragraph_ids": parent_window.paragraph_ids if parent_window else [],
                    "parent_window_child_chunk_ids": parent_window.child_chunk_ids if parent_window else [],
                    "parent_window_text_checksum": content_checksum(parent_window.text) if parent_window else None,
                    "parent_window_token_count": parent_window.token_count if parent_window else None,
                    "parent_window_truncated": parent_window.truncated if parent_window else False,
                    "is_boilerplate": any(
                        paragraph.is_boilerplate for paragraph in parsed.paragraphs if paragraph.paragraph_id in chunk.paragraph_ids
                    ),
                    "is_citation_block": any(
                        paragraph.is_citation_block for paragraph in parsed.paragraphs if paragraph.paragraph_id in chunk.paragraph_ids
                    ),
                }
            )
            chunks.append(
                RetrievalChildChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    paragraph_ids=chunk.paragraph_ids,
                    paragraph_texts=chunk.paragraph_texts,
                    paragraph_original_texts=chunk.paragraph_original_texts,
                    source_spans=chunk.source_spans,
                    section_type=chunk.section_type,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    source_order=chunk.source_order,
                    heading_context=chunk.heading_context,
                    metadata=metadata,
                )
            )
    return chunks


def _prepare_collection(client: Any, config: LegalV2BuildConfig) -> None:
    try:
        qdrant_models = import_module("qdrant_client.models")
        Distance = getattr(qdrant_models, "Distance")
        VectorParams = getattr(qdrant_models, "VectorParams")
        vectors_config = VectorParams(size=LEGAL_V2_PROFILE.embedding_dimension, distance=Distance.COSINE)
    except ModuleNotFoundError:
        vectors_config = {
            "size": LEGAL_V2_PROFILE.embedding_dimension,
            "distance": "Cosine",
        }

    collections = {item.name for item in client.get_collections().collections}
    exists = config.collection_name in collections
    if exists and config.recreate_collection:
        client.delete_collection(collection_name=config.collection_name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=config.collection_name,
            vectors_config=vectors_config,
        )


def _upsert_payloads(
    client: Any,
    *,
    collection_name: str,
    payloads: list[dict[str, Any]],
    vectors: list[list[float]],
    batch_size: int,
) -> None:
    try:
        PointStruct = getattr(import_module("qdrant_client.models"), "PointStruct")
    except ModuleNotFoundError:
        PointStruct = None

    if PointStruct is None:
        points = [
            SimplePoint(
                id=point_id_from_original_id(payload["chunk_id"]),
                vector=vector,
                payload=payload,
            )
            for payload, vector in zip(payloads, vectors, strict=True)
        ]
    else:
        points = [
            PointStruct(
                id=point_id_from_original_id(payload["chunk_id"]),
                vector=vector,
                payload=payload,
            )
            for payload, vector in zip(payloads, vectors, strict=True)
        ]
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        try:
            client.upsert(collection_name=collection_name, points=batch, wait=True)
        except TypeError:
            client.upsert(collection_name=collection_name, points=batch)


@dataclass(frozen=True)
class SimplePoint:
    id: str
    vector: list[float]
    payload: dict[str, Any]


def _validate_qdrant_identity(
    client: Any,
    *,
    collection_name: str,
    payloads: list[dict[str, Any]],
) -> None:
    expected = {str(payload["chunk_id"]) for payload in payloads}
    actual = _qdrant_payload_chunk_ids(client, collection_name)
    if expected != actual:
        missing = sorted(expected - actual)[:10]
        unexpected = sorted(actual - expected)[:10]
        raise ValueError(
            "Qdrant v2 chunk identity mismatch after upsert: "
            f"missing={len(expected - actual)} sample_missing={missing}; "
            f"unexpected={len(actual - expected)} sample_unexpected={unexpected}"
        )


def _qdrant_payload_chunk_ids(client: Any, collection_name: str) -> set[str]:
    if hasattr(client, "scroll"):
        chunk_ids: set[str] = set()
        offset = None
        while True:
            batch, offset = client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            chunk_ids.update(
                str(point.payload.get("chunk_id"))
                for point in batch
                if getattr(point, "payload", None) and point.payload.get("chunk_id")
            )
            if offset is None:
                return chunk_ids
    if hasattr(client, "upserts"):
        return {
            str(point.payload.get("chunk_id"))
            for name, points in client.upserts
            if name == collection_name
            for point in points
            if getattr(point, "payload", None) and point.payload.get("chunk_id")
        }
    raise ValueError("Qdrant client does not support post-upsert identity validation.")


def _validate_payload_identity(payloads: list[dict[str, Any]]) -> None:
    chunk_ids = [str(payload.get("chunk_id") or "") for payload in payloads]
    if not all(chunk_ids):
        raise ValueError("Every v2 payload must have chunk_id.")
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate v2 chunk IDs detected.")


def _validate_vectors(vectors: list[list[float]], expected_dimension: int) -> None:
    for index, vector in enumerate(vectors):
        if len(vector) != expected_dimension:
            raise ValueError(f"Embedding vector {index} has dimension {len(vector)}, expected {expected_dimension}.")


def _validate_bm25_identity(payloads: list[dict[str, Any]], path: Path) -> None:
    expected = {str(payload["chunk_id"]) for payload in payloads}
    with sqlite3.connect(path) as connection:
        actual = {str(row[0]) for row in connection.execute("SELECT chunk_id FROM bm25_chunks")}
    if expected != actual:
        raise ValueError("Dense/BM25 v2 chunk identity mismatch.")


def _source_corpus(documents: list[LegalSourceDocument]) -> str:
    return ",".join(sorted({document.source for document in documents})) or "none"


def _corpus_hash(documents: list[LegalSourceDocument]) -> str:
    hasher = content_checksum
    return hasher(
        "\n".join(
            f"{document.document_id}:{content_checksum(document.text)}"
            for document in sorted(documents, key=lambda item: item.document_id)
        )
    )


def _manifest_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Legal Retrieval v2 build manifest",
            "",
            f"- Validation status: `{payload['validation_status']}`",
            f"- Collection: `{payload['collection_name']}`",
            f"- BM25 index: `{payload['bm25_index_id']}`",
            f"- BM25 path: `{payload['bm25_path']}`",
            f"- Source documents: {payload['source_document_count']}",
            f"- Indexed documents: {payload['indexed_document_count']}",
            f"- Excluded documents: {payload['excluded_document_count']}",
            f"- Chunks: {payload['chunk_count']}",
            f"- Embedding model: `{payload['embedding_model']}`",
            f"- Git commit: `{payload['git_commit']}`",
            f"- Dirty: `{payload['dirty']}`",
            "",
            "The v2 pipeline is disabled by default and is not yet production traffic.",
            "",
        ]
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
