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
CHECKPOINT_FILENAME = "legal_v2_execute_checkpoint.json"


def _is_legal_v2_collection_name(collection_name: str) -> bool:
    return collection_name == LEGAL_V2_COLLECTION_NAME or collection_name.startswith(
        f"{LEGAL_V2_COLLECTION_NAME}_"
    )


class LegalV2CheckpointStop(RuntimeError):
    """Raised after an intentional checkpointed stop requested by the CLI."""


@dataclass(frozen=True)
class LegalV2BuildConfig:
    collection_name: str = LEGAL_V2_COLLECTION_NAME
    bm25_index_id: str = LEGAL_V2_BM25_INDEX_ID
    bm25_path: Path = Path("storage/rag/bm25/nalus_legal_paragraph_bm25_v2.sqlite")
    output_dir: Path = Path("artifacts/legal_v2/index_build")
    recreate_collection: bool = False
    overwrite_bm25: bool = False
    resume: bool = False
    allow_existing_collection: bool = False
    batch_size: int = 64
    document_batch_size: int = 128
    checkpoint_path: Path | None = None
    stop_after_document_batches: int | None = None
    source_selection: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.collection_name in PROTECTED_COLLECTIONS or self.collection_name.startswith("nalus_stable_"):
            raise ValueError(f"Refusing protected collection: {self.collection_name}")
        if not _is_legal_v2_collection_name(self.collection_name):
            raise ValueError(
                "Legal v2 builder may write only the canonical collection or isolated "
                f"{LEGAL_V2_COLLECTION_NAME}_* collections."
            )
        if (
            self.collection_name != LEGAL_V2_COLLECTION_NAME
            and self.bm25_index_id == LEGAL_V2_BM25_INDEX_ID
        ):
            raise ValueError("Pilot Legal v2 builds must use a non-canonical BM25 index id.")
        if self.bm25_path.exists() and not (
            self.overwrite_bm25 or self.resume or self.allow_existing_collection
        ):
            raise ValueError(f"BM25 sidecar already exists: {self.bm25_path}")
        if self.resume and self.recreate_collection:
            raise ValueError("resume and recreate_collection cannot be used together.")
        if self.resume and self.overwrite_bm25:
            raise ValueError("resume and overwrite_bm25 cannot be used together.")
        if self.allow_existing_collection and self.recreate_collection:
            raise ValueError(
                "allow_existing_collection and recreate_collection cannot be used together."
            )
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.document_batch_size <= 0:
            raise ValueError("document_batch_size must be positive.")
        if self.stop_after_document_batches is not None and self.stop_after_document_batches <= 0:
            raise ValueError("stop_after_document_batches must be positive when set.")


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
    qdrant_upsert_batches: int = 0
    qdrant_upsert_points: int = 0
    batch_size: int = 64
    document_batch_size: int = 128
    source_selection: dict[str, Any] = field(default_factory=dict)
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
    _prepare_collection(qdrant_client, config)
    _prepare_bm25_sidecar(config.bm25_path, overwrite=config.overwrite_bm25, resume=config.resume)
    stream_summary = _stream_index_documents(
        documents=documents,
        approved_ids=approved_ids,
        chunk_config=chunk_config,
        registry=registry,
        embedder=embedder,
        qdrant_client=qdrant_client,
        config=config,
        corpus_hash=_corpus_hash(documents),
    )
    if approved_ids and stream_summary["chunk_count"] == 0:
        raise ValueError("Parser-approved documents produced no v2 chunks.")
    expected_chunk_ids = stream_summary["chunk_ids"]
    _validate_qdrant_identity_by_ids(
        qdrant_client,
        collection_name=config.collection_name,
        expected=expected_chunk_ids,
        require_exact_match=not config.allow_existing_collection,
    )
    _validate_bm25_identity_by_ids(
        expected_chunk_ids,
        config.bm25_path,
        require_exact_match=not config.allow_existing_collection,
    )
    manifest = LegalV2BuildManifest(
        collection_name=config.collection_name,
        bm25_index_id=config.bm25_index_id,
        bm25_path=str(config.bm25_path),
        source_corpus=_source_corpus(documents),
        source_document_count=len(documents),
        indexed_document_count=len(approved_ids),
        excluded_document_count=len(documents) - len(approved_ids),
        chunk_count=stream_summary["chunk_count"],
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
        qdrant_upsert_batches=stream_summary["qdrant_upsert_batches"],
        qdrant_upsert_points=stream_summary["qdrant_upsert_points"],
        batch_size=config.batch_size,
        document_batch_size=config.document_batch_size,
        source_selection=dict(config.source_selection),
        qdrant_write_status="pass",
        bm25_write_status="pass",
    )
    write_build_manifest(manifest, config.output_dir)
    _clear_checkpoint(config)
    return manifest


def write_bm25_sidecar(payloads: list[dict[str, Any]], path: Path, *, overwrite: bool) -> None:
    _prepare_bm25_sidecar(path, overwrite=overwrite, resume=False)
    _append_bm25_payloads(payloads, path)


def _prepare_bm25_sidecar(path: Path, *, overwrite: bool, resume: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if resume:
            return
        if not overwrite:
            raise ValueError(f"BM25 sidecar already exists: {path}")
        path.unlink()
    elif resume:
        raise ValueError(f"Cannot resume because BM25 sidecar does not exist: {path}")
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
        connection.commit()


def _append_bm25_payloads(payloads: list[dict[str, Any]], path: Path) -> None:
    with sqlite3.connect(path) as connection:
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


def _source_document_id_for_chunk(
    parsed_metadata: dict[str, Any],
    source_document: LegalSourceDocument,
) -> str | None:
    for candidate in (
        parsed_metadata.get("source_document_id"),
        source_document.metadata.get("source_document_id"),
        source_document.metadata.get("review_document_id"),
    ):
        text = str(candidate or "").strip()
        if text:
            return text
    # Keep legacy doc-* only when the production document_id is already ECLI.
    doc_id = str(source_document.document_id or "").strip()
    if doc_id.startswith("doc-"):
        return doc_id
    return None


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
                    "ecli": (
                        parsed.metadata.get("ecli")
                        or (chunk.document_id if str(chunk.document_id).upper().startswith("ECLI:") else None)
                    ),
                    "canonical_document_id": (
                        parsed.metadata.get("canonical_document_id")
                        or parsed.metadata.get("ecli")
                        or (chunk.document_id if str(chunk.document_id).upper().startswith("ECLI:") else None)
                    ),
                    "source_document_id": _source_document_id_for_chunk(
                        parsed.metadata, source_document
                    ),
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


def _stream_index_documents(
    *,
    documents: list[LegalSourceDocument],
    approved_ids: set[str],
    chunk_config: HierarchicalChunkConfig,
    registry: LegalAdapterRegistry,
    embedder: Any,
    qdrant_client: Any,
    config: LegalV2BuildConfig,
    corpus_hash: str,
) -> dict[str, Any]:
    approved_documents = [document for document in documents if document.document_id in approved_ids]
    total_documents = len(approved_documents)
    checkpoint = _resolve_checkpoint(
        config,
        corpus_hash=corpus_hash,
        source_document_count=len(documents),
        indexed_document_count=total_documents,
    )
    start_document_index = int(checkpoint.get("completed_document_count") or 0)
    chunk_ids = set()
    chunk_count = 0
    upsert_batches = int(checkpoint.get("qdrant_upsert_batches") or 0)
    upsert_points = int(checkpoint.get("qdrant_upsert_points") or 0)
    if start_document_index:
        chunk_ids = _expected_chunk_ids_for_documents(
            documents=approved_documents[:start_document_index],
            approved_ids=approved_ids,
            chunk_config=chunk_config,
            registry=registry,
        )
        _validate_resume_existing_identity(
            qdrant_client,
            collection_name=config.collection_name,
            bm25_path=config.bm25_path,
            expected=chunk_ids,
        )
        chunk_count = len(chunk_ids)
        upsert_points = max(upsert_points, chunk_count)

    completed_batches = 0
    for start in range(start_document_index, total_documents, config.document_batch_size):
        batch_documents = approved_documents[start : start + config.document_batch_size]
        chunks = _chunks_for_approved_documents(
            documents=batch_documents,
            approved_ids=approved_ids,
            chunk_config=chunk_config,
            registry=registry,
        )
        payloads = [
            payload_for_child_chunk(
                chunk,
                qdrant_collection=config.collection_name,
                bm25_index_id=config.bm25_index_id,
            )
            for chunk in chunks
        ]
        _validate_payload_identity(payloads, existing_chunk_ids=chunk_ids)
        chunk_ids.update(str(payload["chunk_id"]) for payload in payloads)
        upsert_summary = _embed_and_upsert_payloads(
            qdrant_client,
            collection_name=config.collection_name,
            payloads=payloads,
            embedder=embedder,
            batch_size=config.batch_size,
        )
        _append_bm25_payloads(payloads, config.bm25_path)
        chunk_count += len(payloads)
        upsert_batches += upsert_summary["batches"]
        upsert_points += upsert_summary["points"]
        next_document_index = min(start + config.document_batch_size, total_documents)
        checkpoint = _advance_checkpoint(
            checkpoint,
            completed_document_count=next_document_index,
            chunk_count=chunk_count,
            qdrant_upsert_batches=upsert_batches,
            qdrant_upsert_points=upsert_points,
        )
        _write_checkpoint(config, checkpoint)
        completed_batches += 1
        print(
            json.dumps(
                {
                    "event": "legal_v2_index_batch_complete",
                    "documents_processed": next_document_index,
                    "documents_total": total_documents,
                    "chunk_count": chunk_count,
                    "qdrant_upsert_points": upsert_points,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if (
            config.stop_after_document_batches is not None
            and completed_batches >= config.stop_after_document_batches
            and next_document_index < total_documents
        ):
            raise LegalV2CheckpointStop(
                "Intentional checkpointed stop after "
                f"{completed_batches} document batch(es); resume with --resume."
            )
    return {
        "chunk_ids": chunk_ids,
        "chunk_count": chunk_count,
        "qdrant_upsert_batches": upsert_batches,
        "qdrant_upsert_points": upsert_points,
    }


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
        if config.resume:
            raise ValueError(f"Cannot resume because collection does not exist: {config.collection_name}")
        client.create_collection(
            collection_name=config.collection_name,
            vectors_config=vectors_config,
        )
    elif not config.resume and not config.recreate_collection and not config.allow_existing_collection:
        existing = _qdrant_payload_chunk_ids(client, config.collection_name)
        if existing:
            raise ValueError(
                f"Legal v2 collection already contains {len(existing)} chunk IDs; "
                "use --recreate-v2-collection for a fresh build, --resume with a checkpoint, "
                "or allow_existing_collection for additive upsert."
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


def _embed_and_upsert_payloads(
    client: Any,
    *,
    collection_name: str,
    payloads: list[dict[str, Any]],
    embedder: Any,
    batch_size: int,
) -> dict[str, int]:
    batches = 0
    points = 0
    for start in range(0, len(payloads), batch_size):
        batch_payloads = payloads[start : start + batch_size]
        vectors = embedder.embed_texts([payload["text"] for payload in batch_payloads])
        _validate_vectors(vectors, LEGAL_V2_PROFILE.embedding_dimension)
        _upsert_payloads(
            client,
            collection_name=collection_name,
            payloads=batch_payloads,
            vectors=vectors,
            batch_size=batch_size,
        )
        batches += 1
        points += len(batch_payloads)
    return {"batches": batches, "points": points}


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
    _validate_qdrant_identity_by_ids(client, collection_name=collection_name, expected=expected)


def _validate_qdrant_identity_by_ids(
    client: Any,
    *,
    collection_name: str,
    expected: set[str],
    require_exact_match: bool = True,
) -> None:
    actual = _qdrant_payload_chunk_ids(client, collection_name)
    missing = expected - actual
    unexpected = actual - expected
    if missing or (require_exact_match and unexpected):
        raise ValueError(
            "Qdrant v2 chunk identity mismatch after upsert: "
            f"missing={len(missing)} sample_missing={sorted(missing)[:10]}; "
            f"unexpected={len(unexpected)} sample_unexpected={sorted(unexpected)[:10]}"
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


def _validate_payload_identity(
    payloads: list[dict[str, Any]],
    *,
    existing_chunk_ids: set[str] | None = None,
) -> None:
    chunk_ids = [str(payload.get("chunk_id") or "") for payload in payloads]
    if not all(chunk_ids):
        raise ValueError("Every v2 payload must have chunk_id.")
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate v2 chunk IDs detected.")
    existing = existing_chunk_ids or set()
    duplicate_existing = existing.intersection(chunk_ids)
    if duplicate_existing:
        raise ValueError(f"Duplicate v2 chunk IDs detected across batches: {sorted(duplicate_existing)[:10]}")


def _validate_vectors(vectors: list[list[float]], expected_dimension: int) -> None:
    for index, vector in enumerate(vectors):
        if len(vector) != expected_dimension:
            raise ValueError(f"Embedding vector {index} has dimension {len(vector)}, expected {expected_dimension}.")


def _validate_bm25_identity(payloads: list[dict[str, Any]], path: Path) -> None:
    expected = {str(payload["chunk_id"]) for payload in payloads}
    _validate_bm25_identity_by_ids(expected, path)


def _validate_bm25_identity_by_ids(
    expected: set[str],
    path: Path,
    *,
    require_exact_match: bool = True,
) -> None:
    with sqlite3.connect(path) as connection:
        actual = {str(row[0]) for row in connection.execute("SELECT chunk_id FROM bm25_chunks")}
    missing = expected - actual
    unexpected = actual - expected
    if missing or (require_exact_match and unexpected):
        raise ValueError(
            "Dense/BM25 v2 chunk identity mismatch: "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )


def _bm25_chunk_ids(path: Path) -> set[str]:
    with sqlite3.connect(path) as connection:
        return {str(row[0]) for row in connection.execute("SELECT chunk_id FROM bm25_chunks")}


def _expected_chunk_ids_for_documents(
    *,
    documents: list[LegalSourceDocument],
    approved_ids: set[str],
    chunk_config: HierarchicalChunkConfig,
    registry: LegalAdapterRegistry,
) -> set[str]:
    chunks = _chunks_for_approved_documents(
        documents=documents,
        approved_ids=approved_ids,
        chunk_config=chunk_config,
        registry=registry,
    )
    return {str(payload_for_child_chunk(chunk)["chunk_id"]) for chunk in chunks}


def _validate_resume_existing_identity(
    client: Any,
    *,
    collection_name: str,
    bm25_path: Path,
    expected: set[str],
) -> None:
    actual_qdrant = _qdrant_payload_chunk_ids(client, collection_name)
    actual_bm25 = _bm25_chunk_ids(bm25_path)
    if actual_qdrant != expected or actual_bm25 != expected:
        raise ValueError(
            "Cannot resume Legal v2 build because checkpointed Qdrant/BM25 identities "
            "do not match the completed source documents."
        )


def _checkpoint_path(config: LegalV2BuildConfig) -> Path:
    return config.checkpoint_path or config.output_dir / CHECKPOINT_FILENAME


def _resolve_checkpoint(
    config: LegalV2BuildConfig,
    *,
    corpus_hash: str,
    source_document_count: int,
    indexed_document_count: int,
) -> dict[str, Any]:
    path = _checkpoint_path(config)
    if config.resume:
        if not path.exists():
            raise ValueError(f"Cannot resume without checkpoint: {path}")
        try:
            checkpoint = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Checkpoint is invalid JSON: {path}") from exc
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Checkpoint must be a JSON object: {path}")
        _validate_checkpoint(
            checkpoint,
            config,
            corpus_hash=corpus_hash,
            source_document_count=source_document_count,
            indexed_document_count=indexed_document_count,
        )
        return checkpoint

    checkpoint = {
        "status": "in_progress",
        "builder_version": BUILDER_VERSION,
        "collection_name": config.collection_name,
        "bm25_path": str(config.bm25_path),
        "source_document_count": source_document_count,
        "indexed_document_count": indexed_document_count,
        "corpus_hash": corpus_hash,
        "batch_size": config.batch_size,
        "document_batch_size": config.document_batch_size,
        "source_selection": dict(config.source_selection),
        "completed_document_count": 0,
        "chunk_count": 0,
        "qdrant_upsert_batches": 0,
        "qdrant_upsert_points": 0,
        "updated_at": _utc_now(),
    }
    _write_checkpoint(config, checkpoint)
    return checkpoint


def _validate_checkpoint(
    checkpoint: dict[str, Any],
    config: LegalV2BuildConfig,
    *,
    corpus_hash: str,
    source_document_count: int,
    indexed_document_count: int,
) -> None:
    expected = {
        "status": "in_progress",
        "builder_version": BUILDER_VERSION,
        "collection_name": config.collection_name,
        "bm25_path": str(config.bm25_path),
        "source_document_count": source_document_count,
        "indexed_document_count": indexed_document_count,
        "corpus_hash": corpus_hash,
        "batch_size": config.batch_size,
        "document_batch_size": config.document_batch_size,
        "source_selection": dict(config.source_selection),
    }
    mismatches = [key for key, value in expected.items() if checkpoint.get(key) != value]
    if mismatches:
        raise ValueError(f"Checkpoint does not match current Legal v2 build request: {mismatches}")
    completed = int(checkpoint.get("completed_document_count") or 0)
    if completed < 0 or completed > indexed_document_count:
        raise ValueError("Checkpoint completed_document_count is out of range.")


def _advance_checkpoint(
    checkpoint: dict[str, Any],
    *,
    completed_document_count: int,
    chunk_count: int,
    qdrant_upsert_batches: int,
    qdrant_upsert_points: int,
) -> dict[str, Any]:
    updated = dict(checkpoint)
    updated.update(
        {
            "completed_document_count": completed_document_count,
            "chunk_count": chunk_count,
            "qdrant_upsert_batches": qdrant_upsert_batches,
            "qdrant_upsert_points": qdrant_upsert_points,
            "updated_at": _utc_now(),
        }
    )
    return updated


def _write_checkpoint(config: LegalV2BuildConfig, checkpoint: dict[str, Any]) -> None:
    path = _checkpoint_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _clear_checkpoint(config: LegalV2BuildConfig) -> None:
    path = _checkpoint_path(config)
    if path.exists():
        path.unlink()


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
            f"- Batch size: {payload.get('batch_size')}",
            f"- Document batch size: {payload.get('document_batch_size')}",
            f"- Qdrant upsert batches: {payload.get('qdrant_upsert_batches')}",
            f"- Qdrant upsert points: {payload.get('qdrant_upsert_points')}",
            f"- Source selection: `{json.dumps(payload.get('source_selection') or {}, ensure_ascii=False, sort_keys=True)}`",
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
