"""Guarded Stage 1 smoke builder for a parallel US/NALUS BGE-M3 candidate.

This script is intentionally limited to small smoke runs. It must not be used
for full-corpus ingest, production alias updates, or legacy mpnet ingestion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BUILDER_VERSION = "usoud-bge-m3-smoke-v1"
BGE_M3_MODEL_NAME = "BAAI/bge-m3"
BGE_M3_DIMENSION = 1024
SOURCE_ID = 1
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 1400
DEFAULT_CHUNK_OVERLAP_WORDS = 35
REPORT_PATH = PROJECT_ROOT / "artifacts/nalus_update/usoud_bge_m3_stage1_smoke_report.md"
PRODUCTION_COLLECTION_DENYLIST = {
    "nalus",
    "nalus_live",
    "nalus_stable_20260326",
}
ALLOWED_STAGE1_NAME_MARKERS = ("smoke", "tmp", "pilot")
RECREATE_ALLOWED_MARKERS = ("smoke", "tmp")
SMOKE_QUERIES = (
    "právo na spravedlivý proces",
    "opomenuté důkazy",
    "odůvodnění rozhodnutí",
    "porušení základních práv",
)


class SafetyError(ValueError):
    """Raised when a safety guard refuses the requested operation."""


@dataclass(frozen=True)
class SourceRecord:
    identity: str
    source_document_id: str
    case_reference: str | None
    ecli: str | None
    decision_date: str | None
    detail_url: str | None
    text_url: str | None
    full_text: str
    origin_file: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class SmokeChunk:
    seq_id: int
    point_id: str
    text: str
    payload: dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guarded Stage 1 smoke builder for a US/NALUS BGE-M3 candidate collection."
    )
    parser.add_argument("--mode", choices=["smoke"], required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--collection-name", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-batch", type=Path)
    source.add_argument("--source-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--recreate-smoke-collection", action="store_true")
    parser.add_argument("--no-alias-update", action="store_true", default=True)
    parser.add_argument("--top-k-smoke-test", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://qdrant:6333"))
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.mode != "smoke":
        raise SafetyError("Only --mode smoke is allowed. Full ingest is intentionally refused.")

    validate_collection_name(args.collection_name, execute=args.execute)
    validate_smoke_limit(args.limit)
    validate_top_k(args.top_k_smoke_test)

    if not args.no_alias_update:
        raise SafetyError("Alias updates are refused. --no-alias-update must remain enabled.")

    if args.recreate_smoke_collection and not _contains_marker(
        args.collection_name, RECREATE_ALLOWED_MARKERS
    ):
        raise SafetyError(
            "--recreate-smoke-collection is allowed only for collection names containing "
            "'smoke' or 'tmp'."
        )


def validate_collection_name(collection_name: str, *, execute: bool) -> None:
    normalized = collection_name.strip()
    if not normalized:
        raise SafetyError("--collection-name must be explicitly provided.")

    if normalized in PRODUCTION_COLLECTION_DENYLIST:
        raise SafetyError(f"Refusing to write to protected collection: {normalized}")

    if normalized.startswith("nalus_stable_"):
        raise SafetyError(f"Refusing to write to stable production collection: {normalized}")

    if execute and not _contains_marker(normalized, ALLOWED_STAGE1_NAME_MARKERS):
        raise SafetyError(
            "Stage 1 execution requires collection name to include one of: "
            "smoke, tmp, pilot."
        )


def validate_smoke_limit(limit: int) -> None:
    if limit <= 0:
        raise SafetyError("--limit must be greater than zero.")
    if limit > 100:
        raise SafetyError("Smoke mode refuses --limit above 100.")


def validate_top_k(top_k: int) -> None:
    if top_k <= 0:
        raise SafetyError("--top-k-smoke-test must be greater than zero.")
    if top_k > 20:
        raise SafetyError("--top-k-smoke-test above 20 is refused for smoke mode.")


def validate_vector_dimension(vectors: list[list[float]], expected_dim: int = BGE_M3_DIMENSION) -> None:
    for index, vector in enumerate(vectors):
        if len(vector) != expected_dim:
            raise SafetyError(
                f"BGE-M3 vector dimension validation failed at vector {index}: "
                f"expected {expected_dim}, got {len(vector)}."
            )


def _contains_marker(value: str, markers: tuple[str, ...]) -> bool:
    normalized = value.lower()
    return any(marker in normalized for marker in markers)


def load_source_records(args: argparse.Namespace) -> list[SourceRecord]:
    if args.source_batch:
        return load_batch_records(resolve_project_path(args.source_batch))
    return load_manifest_records(resolve_project_path(args.source_manifest))


def resolve_project_path(path: Path | None) -> Path:
    if path is None:
        raise ValueError("Path must not be None.")
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_batch_records(path: Path) -> list[SourceRecord]:
    raw = _read_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"Expected list of records in {path}.")
    return _records_from_items(raw, origin_file=path.name)


def load_manifest_records(path: Path) -> list[SourceRecord]:
    manifest = _read_json(path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected manifest object in {path}.")

    records: list[SourceRecord] = []
    for entry in manifest.get("batches", []):
        if not isinstance(entry, dict) or not entry.get("file"):
            continue
        batch_path = path.parent / str(entry["file"])
        if batch_path.name == "manifest.json" or not batch_path.exists():
            continue
        records.extend(load_batch_records(batch_path))
    return records


def select_records(records: list[SourceRecord], *, limit: int) -> list[SourceRecord]:
    with_text = [record for record in records if record.full_text.strip()]
    deduped = deduplicate_records(with_text)
    return deduped[:limit]


def deduplicate_records(records: list[SourceRecord]) -> list[SourceRecord]:
    best: dict[str, SourceRecord] = {}
    for record in records:
        current = best.get(record.identity)
        if current is None or _record_rank(record) > _record_rank(current):
            best[record.identity] = record
    return list(best.values())


def chunk_records(
    records: list[SourceRecord],
    *,
    collection_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> list[SmokeChunk]:
    chunks: list[SmokeChunk] = []
    for record in records:
        text_chunks = split_text_into_chunks(
            record.full_text,
            chunk_size=chunk_size,
            overlap_words=overlap_words,
        )
        chunk_count = len(text_chunks)
        for chunk_index, text in enumerate(text_chunks):
            seq_id = len(chunks) + 1
            point_id = _point_id(collection_name, record.identity, chunk_index)
            payload = {
                "chunk_id": seq_id,
                "source_id": SOURCE_ID,
                "text": text,
                "source": "usoud / nalus",
                "court": "Ústavní soud",
                "decision_date": record.decision_date,
                "source_document_id": record.source_document_id,
                "spisova_znacka": record.case_reference,
                "ecli": record.ecli,
                "detail_url": record.detail_url,
                "text_url": record.text_url,
                "origin_file": record.origin_file,
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "text_length": len(text),
                "builder_version": BUILDER_VERSION,
            }
            chunks.append(SmokeChunk(seq_id=seq_id, point_id=point_id, text=text, payload=payload))
    return chunks


def split_text_into_chunks(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n+", normalized) if paragraph.strip()]
    if not paragraphs:
        paragraphs = [normalized]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        parts = _split_long_paragraph(paragraph, chunk_size=chunk_size)
        for part in parts:
            if not current:
                current = part
                continue
            candidate = f"{current}\n\n{part}"
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            chunks.append(current)
            current = _with_overlap(current, part, overlap_words=overlap_words, chunk_size=chunk_size)

    if current:
        chunks.append(current)

    return chunks


def build_summary(args: argparse.Namespace, selected: list[SourceRecord], chunks: list[SmokeChunk]) -> dict[str, Any]:
    return {
        "generated_at": _utc_now(),
        "builder_version": BUILDER_VERSION,
        "script_path": "scripts/build_usoud_bge_m3_candidate.py",
        "mode": args.mode,
        "action": "execute" if args.execute else "dry-run",
        "command": _format_command(sys.argv),
        "input": str(args.source_batch or args.source_manifest),
        "collection_name": args.collection_name,
        "limit": args.limit,
        "selected_record_count": len(selected),
        "generated_chunk_count": len(chunks),
        "embedding_model": BGE_M3_MODEL_NAME,
        "expected_vector_dimension": BGE_M3_DIMENSION,
        "vector_dimension_validation": "not_run",
        "qdrant": {
            "collection_point_count_before": None,
            "collection_point_count_after": None,
            "nalus_live_before": None,
            "nalus_live_after": None,
            "nalus_stable_20260326_before": None,
            "nalus_stable_20260326_after": None,
            "aliases_changed": None,
        },
        "bm25_status": "not_run",
        "hybrid_status": "not_run",
        "smoke_queries": [],
        "warnings": [],
        "failures": [],
        "production_api_touched": False,
        "retrieval_logic_changed": False,
        "clarification_gate_changed": False,
        "aliases_touched": False,
        "stage2_recommendation": "not_ready",
    }


def run_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    records = load_source_records(args)
    selected = select_records(records, limit=args.limit)
    chunks = chunk_records(selected, collection_name=args.collection_name, chunk_size=args.chunk_size)
    summary = build_summary(args, selected, chunks)
    summary["stage2_recommendation"] = "safe_after_execute_smoke_passes"
    write_outputs(args, summary, dry_run=True)
    return summary


def run_execute(args: argparse.Namespace) -> dict[str, Any]:
    records = load_source_records(args)
    selected = select_records(records, limit=args.limit)
    chunks = chunk_records(selected, collection_name=args.collection_name, chunk_size=args.chunk_size)
    summary = build_summary(args, selected, chunks)
    dry_run_summary = _load_previous_dry_run(args.output_dir)
    if dry_run_summary:
        summary["dry_run_command"] = dry_run_summary.get("command")

    if not chunks:
        raise SafetyError("No chunks generated; refusing to create an empty smoke collection.")

    model = _load_bge_m3_model()
    vectors = _encode_chunks(model, [chunk.text for chunk in chunks], batch_size=args.embedding_batch_size)
    validate_vector_dimension(vectors)
    summary["vector_dimension_validation"] = f"PASS ({BGE_M3_DIMENSION})"

    client = _qdrant_client(args.qdrant_url)
    aliases_before = _aliases_snapshot(client)
    live_before = _count_collection(client, "nalus_live")
    stable_before = _count_collection(client, "nalus_stable_20260326")
    collection_before = _count_collection(client, args.collection_name)

    summary["qdrant"]["nalus_live_before"] = live_before
    summary["qdrant"]["nalus_stable_20260326_before"] = stable_before
    summary["qdrant"]["collection_point_count_before"] = collection_before

    _prepare_smoke_collection(
        client,
        collection_name=args.collection_name,
        recreate=args.recreate_smoke_collection,
        existing_count=collection_before,
    )
    _upsert_chunks(client, collection_name=args.collection_name, chunks=chunks, vectors=vectors)

    collection_after = _count_collection(client, args.collection_name)
    payload_check = _verify_payloads_present(client, args.collection_name)
    if not payload_check:
        raise SafetyError("Qdrant payload verification failed: no payloads returned from smoke collection.")

    query_results, bm25_status, hybrid_status = _run_smoke_queries(
        client=client,
        collection_name=args.collection_name,
        model=model,
        chunks=chunks,
        top_k=args.top_k_smoke_test,
    )

    aliases_after = _aliases_snapshot(client)
    live_after = _count_collection(client, "nalus_live")
    stable_after = _count_collection(client, "nalus_stable_20260326")

    summary["qdrant"]["collection_point_count_after"] = collection_after
    summary["qdrant"]["nalus_live_after"] = live_after
    summary["qdrant"]["nalus_stable_20260326_after"] = stable_after
    summary["qdrant"]["aliases_changed"] = aliases_before != aliases_after
    summary["bm25_status"] = bm25_status
    summary["hybrid_status"] = hybrid_status
    summary["smoke_queries"] = query_results

    if collection_after != len(chunks):
        summary["warnings"].append(
            f"Smoke collection point count is {collection_after}, expected {len(chunks)} after recreate."
        )
    if live_before != live_after:
        raise SafetyError(f"nalus_live changed during smoke run: {live_before} -> {live_after}")
    if stable_before != stable_after:
        raise SafetyError(
            "nalus_stable_20260326 changed during smoke run: "
            f"{stable_before} -> {stable_after}"
        )
    if aliases_before != aliases_after:
        raise SafetyError("Qdrant aliases changed during smoke run; this is forbidden.")

    summary["stage2_recommendation"] = "safe_after_review"
    write_outputs(args, summary, dry_run=False)
    return summary


def write_outputs(args: argparse.Namespace, summary: dict[str, Any], *, dry_run: bool) -> None:
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / ("dry_run_summary.json" if dry_run else "execute_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(summary), encoding="utf-8")


def render_report(summary: dict[str, Any]) -> str:
    qdrant = summary["qdrant"]
    dry_run_command = summary.get("dry_run_command")
    execute_command = summary["command"] if summary["action"] == "execute" else None
    if summary["action"] == "dry-run":
        dry_run_command = summary["command"]

    lines = [
        "# Ustavni soud / NALUS - BGE-M3 Stage 1 Smoke Report",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        f"- Script path: `{summary['script_path']}`",
        f"- Builder version: `{summary['builder_version']}`",
        f"- Action: `{summary['action']}`",
        f"- Dry-run command: `{dry_run_command or 'not recorded'}`",
        f"- Execute command: `{execute_command or 'not run'}`",
        f"- Input: `{summary['input']}`",
        f"- Selected records: `{summary['selected_record_count']}`",
        f"- Generated chunks: `{summary['generated_chunk_count']}`",
        f"- Embedding model: `{summary['embedding_model']}`",
        f"- Vector dimension validation: `{summary['vector_dimension_validation']}`",
        f"- Qdrant collection: `{summary['collection_name']}`",
        f"- Collection point count before: `{qdrant['collection_point_count_before']}`",
        f"- Collection point count after: `{qdrant['collection_point_count_after']}`",
        f"- `nalus_live` before/after: `{qdrant['nalus_live_before']}` / `{qdrant['nalus_live_after']}`",
        (
            "- `nalus_stable_20260326` before/after: "
            f"`{qdrant['nalus_stable_20260326_before']}` / `{qdrant['nalus_stable_20260326_after']}`"
        ),
        f"- BM25 status: `{summary['bm25_status']}`",
        f"- Hybrid/RRF status: `{summary['hybrid_status']}`",
        f"- Production API touched: `{summary['production_api_touched']}`",
        f"- Aliases touched: `{summary['aliases_touched']}`",
        f"- Aliases changed by verification: `{qdrant['aliases_changed']}`",
        f"- Retrieval logic changed: `{summary['retrieval_logic_changed']}`",
        f"- Clarification gate changed: `{summary['clarification_gate_changed']}`",
        f"- Stage 2 recommendation: `{summary['stage2_recommendation']}`",
        "",
        "## Smoke Query Results",
        "",
    ]

    if not summary["smoke_queries"]:
        lines.append("- Not run.")
    else:
        for item in summary["smoke_queries"]:
            lines.extend(
                [
                    f"### `{item['query']}`",
                    "",
                    f"- Dense results from smoke collection: `{item['dense_all_from_smoke_collection']}`",
                    f"- BM25 results: `{len(item.get('bm25_results') or [])}`",
                    f"- Hybrid results: `{len(item.get('hybrid_results') or [])}`",
                    "",
                ]
            )
            for result in item["dense_results"]:
                lines.append(
                    "- dense "
                    f"score=`{result['score']}` doc=`{result['source_document_id']}` "
                    f"date=`{result['decision_date']}` snippet=\"{result['snippet']}\""
                )
            if item.get("hybrid_results"):
                lines.append("")
                for result in item["hybrid_results"]:
                    lines.append(
                        "- hybrid "
                        f"score=`{result['score']}` doc=`{result['source_document_id']}` "
                        f"date=`{result['decision_date']}` snippet=\"{result['snippet']}\""
                    )
            lines.append("")

    lines.extend(["## Failures / Warnings", ""])
    warnings = summary.get("warnings") or []
    failures = summary.get("failures") or []
    if not warnings and not failures:
        lines.append("- None.")
    for warning in warnings:
        lines.append(f"- WARNING: {warning}")
    for failure in failures:
        lines.append(f"- FAILURE: {failure}")
    lines.append("")
    return "\n".join(lines)


def _load_bge_m3_model() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: sentence_transformers. Install project production "
            "requirements in Docker; no fallback embedding is allowed."
        ) from exc

    device = os.getenv("BGE_M3_DEVICE", "cpu")
    return SentenceTransformer(BGE_M3_MODEL_NAME, device=device)


def _encode_chunks(model: Any, texts: list[str], *, batch_size: int) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = model.encode(batch, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        vectors.extend(_to_float_vector(vector) for vector in encoded)
    return vectors


def _encode_query(model: Any, query: str) -> list[float]:
    encoded = model.encode([query], batch_size=1, normalize_embeddings=True, show_progress_bar=False)
    return _to_float_vector(encoded[0])


def _qdrant_client(qdrant_url: str) -> Any:
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: qdrant_client. Install project production requirements in Docker."
        ) from exc
    return QdrantClient(url=qdrant_url, timeout=120, check_compatibility=False)


def _prepare_smoke_collection(
    client: Any,
    *,
    collection_name: str,
    recreate: bool,
    existing_count: int | None,
) -> None:
    from qdrant_client.models import Distance, VectorParams

    exists = existing_count is not None
    if exists and recreate:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=BGE_M3_DIMENSION, distance=Distance.COSINE),
        )
        return

    if exists and existing_count:
        raise SafetyError(
            f"Smoke collection {collection_name!r} already has {existing_count} points. "
            "Pass --recreate-smoke-collection with a smoke/tmp collection name."
        )

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=BGE_M3_DIMENSION, distance=Distance.COSINE),
        )


def _upsert_chunks(
    client: Any,
    *,
    collection_name: str,
    chunks: list[SmokeChunk],
    vectors: list[list[float]],
) -> None:
    from qdrant_client.models import PointStruct

    points = [
        PointStruct(id=chunk.point_id, vector=vector, payload=chunk.payload)
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    for start in range(0, len(points), 64):
        client.upsert(collection_name=collection_name, points=points[start : start + 64])


def _run_smoke_queries(
    *,
    client: Any,
    collection_name: str,
    model: Any,
    chunks: list[SmokeChunk],
    top_k: int,
) -> tuple[list[dict[str, Any]], str, str]:
    bm25_index = None
    bm25_status = "not_available"
    hybrid_status = "not_available"
    rag_eval = _load_rag_eval_components()
    if rag_eval is not None:
        RagEvalChunk, BenchmarkRetrievalConfig, Bm25ChunkIndex, reciprocal_rank_fusion, RagEvalRetrievalResult = rag_eval
        bm25_chunks = [
            RagEvalChunk(
                chunk_id=chunk.seq_id,
                chunk_text=chunk.text,
                source_id=SOURCE_ID,
                chunk_metadata={k: v for k, v in chunk.payload.items() if k != "text"},
            )
            for chunk in chunks
        ]
        retrieval_config = BenchmarkRetrievalConfig(modes=["bm25"], bm25_k1=1.5, bm25_b=0.75)
        bm25_index = Bm25ChunkIndex(chunks=bm25_chunks, retrieval_config=retrieval_config)
        bm25_status = "available"
        hybrid_status = "available_rrf"
    else:
        reciprocal_rank_fusion = None
        RagEvalRetrievalResult = None

    results: list[dict[str, Any]] = []
    for query in SMOKE_QUERIES:
        dense_results = _dense_retrieve(
            client=client,
            collection_name=collection_name,
            model=model,
            query=query,
            top_k=top_k,
        )
        bm25_results = []
        hybrid_results = []
        if bm25_index is not None and reciprocal_rank_fusion is not None and RagEvalRetrievalResult is not None:
            bm25_response = bm25_index.retrieve(
                query=query,
                source_id=SOURCE_ID,
                top_k=top_k,
                collection_name=collection_name,
            )
            bm25_results = [_result_to_report(item, collection_name) for item in bm25_response.results]
            dense_for_rrf = [
                RagEvalRetrievalResult(
                    chunk_id=item["chunk_id"],
                    source_id=SOURCE_ID,
                    score=item["score"],
                    text=item["text"],
                    qdrant_collection=collection_name,
                    payload_metadata=item["payload_metadata"],
                )
                for item in dense_results
            ]
            hybrid_response = reciprocal_rank_fusion(
                [dense_for_rrf, bm25_response.results],
                top_k=top_k,
                rrf_k=60,
            )
            hybrid_results = [_result_to_report(item, collection_name) for item in hybrid_response.results]

        dense_report = [_dense_result_to_report(item, collection_name) for item in dense_results]
        results.append(
            {
                "query": query,
                "top_k": top_k,
                "collection": collection_name,
                "dense_results": dense_report,
                "bm25_results": bm25_results,
                "hybrid_results": hybrid_results,
                "dense_all_from_smoke_collection": all(
                    item["collection"] == collection_name for item in dense_report
                ),
            }
        )
    return results, bm25_status, hybrid_status


def _dense_retrieve(
    *,
    client: Any,
    collection_name: str,
    model: Any,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    vector = _encode_query(model, query)
    validate_vector_dimension([vector])
    response = client.query_points(collection_name=collection_name, query=vector, limit=top_k)
    dense_results: list[dict[str, Any]] = []
    for point in response.points:
        payload = dict(point.payload or {})
        dense_results.append(
            {
                "chunk_id": int(payload.get("chunk_id")),
                "score": float(point.score),
                "text": str(payload.get("text") or ""),
                "payload_metadata": {k: v for k, v in payload.items() if k != "text"},
                "collection": collection_name,
            }
        )
    return dense_results


def _load_rag_eval_components() -> tuple[Any, Any, Any, Any, Any] | None:
    try:
        from rag_eval.adapters.base import RagEvalChunk, RagEvalRetrievalResult
        from rag_eval.config import BenchmarkRetrievalConfig
        from rag_eval.retrieval.bm25 import Bm25ChunkIndex
        from rag_eval.retrieval.fusion import reciprocal_rank_fusion
    except ImportError:
        return None
    return (
        RagEvalChunk,
        BenchmarkRetrievalConfig,
        Bm25ChunkIndex,
        reciprocal_rank_fusion,
        RagEvalRetrievalResult,
    )


def _dense_result_to_report(item: dict[str, Any], collection_name: str) -> dict[str, Any]:
    metadata = item["payload_metadata"]
    return {
        "collection": collection_name,
        "source_document_id": metadata.get("source_document_id"),
        "decision_date": metadata.get("decision_date"),
        "score": round(float(item["score"]), 6),
        "snippet": _snippet(item["text"]),
        "chunk_id": item["chunk_id"],
    }


def _result_to_report(item: Any, collection_name: str) -> dict[str, Any]:
    metadata = dict(item.payload_metadata or {})
    return {
        "collection": item.qdrant_collection or collection_name,
        "source_document_id": metadata.get("source_document_id"),
        "decision_date": metadata.get("decision_date"),
        "score": round(float(item.score), 6),
        "snippet": _snippet(item.text),
        "chunk_id": item.chunk_id,
    }


def _verify_payloads_present(client: Any, collection_name: str) -> bool:
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=3,
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        return False
    return all(bool(point.payload and point.payload.get("source_document_id")) for point in points)


def _count_collection(client: Any, collection_name: str) -> int | None:
    try:
        return int(client.count(collection_name=collection_name).count)
    except Exception:  # noqa: BLE001 - missing collection must be represented as None.
        return None


def _aliases_snapshot(client: Any) -> list[dict[str, str]]:
    try:
        aliases = client.get_aliases().aliases
    except Exception:  # noqa: BLE001 - handled as unavailable snapshot.
        return []
    return sorted(
        (
            {"alias_name": str(alias.alias_name), "collection_name": str(alias.collection_name)}
            for alias in aliases
        ),
        key=lambda item: (item["alias_name"], item["collection_name"]),
    )


def _records_from_items(items: list[Any], *, origin_file: str) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        full_text = str(item.get("full_text") or "").strip()
        identity = _record_identity(item)
        if not identity:
            continue
        source_document_id = str(item.get("ecli") or item.get("case_reference") or item.get("result_id") or identity)
        records.append(
            SourceRecord(
                identity=identity,
                source_document_id=source_document_id,
                case_reference=_clean_optional(item.get("case_reference")),
                ecli=_clean_optional(item.get("ecli")),
                decision_date=_clean_optional(item.get("decision_date")),
                detail_url=_clean_optional(item.get("detail_url")),
                text_url=_clean_optional(item.get("text_url")),
                full_text=full_text,
                origin_file=origin_file,
                raw=item,
            )
        )
    return records


def _record_identity(item: dict[str, Any]) -> str:
    for key in ("ecli", "case_reference", "detail_url", "text_url"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    result_id = item.get("result_id")
    return str(result_id).strip() if result_id is not None else ""


def _record_rank(record: SourceRecord) -> tuple[int, int, int, str]:
    metadata_count = sum(
        1
        for value in (
            record.case_reference,
            record.ecli,
            record.decision_date,
            record.detail_url,
            record.text_url,
        )
        if value
    )
    return (int(bool(record.full_text.strip())), len(record.full_text), metadata_count, record.identity)


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    collapsed = "\n".join(lines)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
    return collapsed.strip()


def _split_long_paragraph(paragraph: str, *, chunk_size: int) -> list[str]:
    if len(paragraph) <= chunk_size:
        return [paragraph]

    words = paragraph.split()
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        additional = len(word) + (1 if current else 0)
        if current and current_len + additional > chunk_size:
            parts.append(" ".join(current))
            current = [word]
            current_len = len(word)
            continue
        current.append(word)
        current_len += additional
    if current:
        parts.append(" ".join(current))
    return parts


def _with_overlap(previous: str, next_part: str, *, overlap_words: int, chunk_size: int) -> str:
    previous_words = previous.split()
    overlap = " ".join(previous_words[-overlap_words:]) if previous_words and overlap_words > 0 else ""
    if not overlap:
        return next_part
    candidate = f"{overlap}\n\n{next_part}"
    if len(candidate) <= chunk_size:
        return candidate
    return next_part


def _point_id(collection_name: str, identity: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{collection_name}:{identity}:{chunk_index}"))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _clean_optional(value: Any) -> str | None:
    cleaned = str(value or "").strip()
    return cleaned or None


def _to_float_vector(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


def _snippet(text: str, limit: int = 220) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def _format_command(argv: list[str]) -> str:
    return "python " + " ".join(argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_previous_dry_run(output_dir: Path) -> dict[str, Any] | None:
    path = resolve_project_path(output_dir) / "dry_run_summary.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_args(args)
        summary = run_execute(args) if args.execute else run_dry_run(args)
    except Exception as exc:  # noqa: BLE001 - CLI must render a report on failure where possible.
        failure_summary = {
            "generated_at": _utc_now(),
            "builder_version": BUILDER_VERSION,
            "script_path": "scripts/build_usoud_bge_m3_candidate.py",
            "mode": getattr(args, "mode", "unknown"),
            "action": "execute" if getattr(args, "execute", False) else "dry-run",
            "command": _format_command(sys.argv),
            "input": str(getattr(args, "source_batch", None) or getattr(args, "source_manifest", None)),
            "collection_name": getattr(args, "collection_name", "unknown"),
            "limit": getattr(args, "limit", None),
            "selected_record_count": 0,
            "generated_chunk_count": 0,
            "embedding_model": BGE_M3_MODEL_NAME,
            "expected_vector_dimension": BGE_M3_DIMENSION,
            "vector_dimension_validation": "failed_or_not_run",
            "qdrant": {
                "collection_point_count_before": None,
                "collection_point_count_after": None,
                "nalus_live_before": None,
                "nalus_live_after": None,
                "nalus_stable_20260326_before": None,
                "nalus_stable_20260326_after": None,
                "aliases_changed": None,
            },
            "bm25_status": "not_run",
            "hybrid_status": "not_run",
            "smoke_queries": [],
            "warnings": [],
            "failures": [str(exc)],
            "production_api_touched": False,
            "retrieval_logic_changed": False,
            "clarification_gate_changed": False,
            "aliases_touched": False,
            "stage2_recommendation": "not_ready",
        }
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(render_report(failure_summary), encoding="utf-8")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
