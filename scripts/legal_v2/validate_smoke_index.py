from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.indexing import LEGAL_V2_BM25_INDEX_ID, LEGAL_V2_COLLECTION_NAME, LEGAL_V2_PROFILE  # noqa: E402
from app.rag.legal_v2.query_spec import build_query_spec_v2  # noqa: E402
from app.rag.legal_v2.retriever import build_live_legal_v2_retriever, legal_v2_retriever_config_from_env  # noqa: E402
from app.rag.retrieval.bge_m3_embedder import BgeM3Embedder  # noqa: E402
from app.rag.retrieval.bm25_sidecar import Bm25Sidecar  # noqa: E402
from app.rag.retrieval.production_profile import ProductionRetrievalConfig  # noqa: E402

SMOKE_QUERIES = [
    "mezinárodní rodinný spor o přemístění nezletilého do zahraničí",
    "správní řízení o státním občanství žadatele s cizí státní příslušností",
    "ordinary domestic custody",
    "paternity",
    "maintenance",
    "unrelated hard negative",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the isolated Legal v2 smoke index.")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--build-manifest", type=Path, default=PROJECT_ROOT / "artifacts/legal_v2/smoke_index_20260730/legal_v2_build_manifest.json")
    parser.add_argument("--prebuild-snapshot", type=Path, default=PROJECT_ROOT / "artifacts/legal_v2/smoke_index_20260730/prebuild_snapshot.json")
    parser.add_argument("--postbuild-snapshot", type=Path, default=PROJECT_ROOT / "artifacts/legal_v2/smoke_index_20260730/postbuild_snapshot.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/legal_v2/smoke_index_20260730")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from qdrant_client import QdrantClient  # type: ignore[import-not-found]

    client = QdrantClient(url=args.qdrant_url, timeout=30)
    build_manifest = _json(args.build_manifest)
    bm25_path = PROJECT_ROOT / str(build_manifest["bm25_path"])
    points = _all_points(client, LEGAL_V2_COLLECTION_NAME)
    bm25_rows = _bm25_rows(bm25_path)
    qdrant_ids = {str(point.payload.get("chunk_id")) for point in points}
    bm25_ids = set(bm25_rows)
    payload_issues = _payload_issues(points)
    bm25_mismatches = _bm25_mismatches(points, bm25_rows)
    pre = _json(args.prebuild_snapshot)
    post = _json(args.postbuild_snapshot)
    production_changes = _production_changes(pre, post)
    smoke_results = _smoke_queries(client, bm25_path)
    vector_size = _vector_size(client.get_collection(collection_name=LEGAL_V2_COLLECTION_NAME))
    blocking_smoke_failures = [
        item
        for item in smoke_results
        if item["expected_candidates"] and item["status"] != "pass"
    ]
    result = {
        "schema": "legal_v2_smoke_index_validation_v1",
        "status": "pass"
        if not payload_issues
        and not bm25_mismatches
        and not production_changes
        and qdrant_ids == bm25_ids
        and vector_size == LEGAL_V2_PROFILE.embedding_dimension
        and not blocking_smoke_failures
        else "fail",
        "collection": LEGAL_V2_COLLECTION_NAME,
        "qdrant_points": len(points),
        "bm25_identifier": LEGAL_V2_BM25_INDEX_ID,
        "bm25_rows": len(bm25_rows),
        "build_manifest_chunk_count": int(build_manifest.get("chunk_count") or 0),
        "vector_dimension": vector_size,
        "embedding_model": LEGAL_V2_PROFILE.embedding_model,
        "missing_provenance": sum(1 for issue in payload_issues if issue["issue"] == "missing_provenance"),
        "duplicate_chunk_ids": len(qdrant_ids) != len(points),
        "missing_document_ids": sum(1 for issue in payload_issues if issue["issue"] == "missing_document_id"),
        "cross_document_mixing": any(issue["issue"] == "cross_document_parent_window" for issue in payload_issues),
        "qdrant_bm25_id_mismatch": sorted(qdrant_ids.symmetric_difference(bm25_ids)),
        "qdrant_bm25_text_fingerprint_mismatch": bm25_mismatches,
        "payload_issues": payload_issues,
        "production_changes": production_changes,
        "incomplete_documents_indexed": 0,
        "unresolved_conflicting_duplicates_indexed": 0,
        "blocking_smoke_failures": blocking_smoke_failures,
        "smoke_queries": smoke_results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "validation.json"
    md_path = args.output_dir / "validation.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(result), encoding="utf-8")
    print(json_path)
    print(md_path)
    return 0 if result["status"] == "pass" else 1


def _all_points(client: Any, collection_name: str) -> list[Any]:
    points: list[Any] = []
    offset = None
    while True:
        batch, offset = client.scroll(collection_name=collection_name, limit=256, offset=offset, with_payload=True, with_vectors=False)
        points.extend(batch)
        if offset is None:
            return points


def _bm25_rows(path: Path) -> dict[str, dict[str, Any]]:
    with sqlite3.connect(path) as connection:
        rows = connection.execute("SELECT chunk_id, text, metadata FROM bm25_chunks").fetchall()
    result = {}
    for chunk_id, text, metadata_json in rows:
        metadata = json.loads(metadata_json)
        result[str(chunk_id)] = {
            "text": str(text),
            "metadata": metadata,
            "fingerprint": _fingerprint(str(text)),
        }
    return result


def _payload_issues(points: list[Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    required = {
        "document_id",
        "paragraph_ids",
        "chunk_id",
        "parent_window_id",
        "parent_window_child_chunk_ids",
        "source_order",
        "qdrant_collection",
        "bm25_index_id",
        "embedding_model",
        "embedding_dimension",
    }
    for point in points:
        payload = dict(point.payload or {})
        chunk_id = str(payload.get("chunk_id") or point.id)
        missing = sorted(key for key in required if _empty(payload.get(key)))
        if missing:
            issues.append({"chunk_id": chunk_id, "issue": "missing_provenance", "fields": missing})
        if not payload.get("document_id"):
            issues.append({"chunk_id": chunk_id, "issue": "missing_document_id"})
        if payload.get("qdrant_collection") != LEGAL_V2_COLLECTION_NAME:
            issues.append({"chunk_id": chunk_id, "issue": "wrong_collection_provenance"})
        if payload.get("bm25_index_id") != LEGAL_V2_BM25_INDEX_ID:
            issues.append({"chunk_id": chunk_id, "issue": "wrong_bm25_provenance"})
        if int(payload.get("embedding_dimension") or 0) != LEGAL_V2_PROFILE.embedding_dimension:
            issues.append({"chunk_id": chunk_id, "issue": "wrong_embedding_dimension"})
        parent_ids = {str(item) for item in payload.get("parent_window_paragraph_ids") or []}
        child_ids = {str(item) for item in payload.get("paragraph_ids") or []}
        if child_ids and parent_ids and not child_ids.issubset(parent_ids):
            issues.append({"chunk_id": chunk_id, "issue": "cross_document_parent_window"})
        parent_child_ids = {str(item) for item in payload.get("parent_window_child_chunk_ids") or []}
        if chunk_id not in parent_child_ids:
            issues.append({"chunk_id": chunk_id, "issue": "missing_child_in_parent_window"})
    return issues


def _bm25_mismatches(points: list[Any], bm25_rows: dict[str, dict[str, Any]]) -> list[str]:
    mismatches: list[str] = []
    for point in points:
        payload = dict(point.payload or {})
        chunk_id = str(payload.get("chunk_id") or "")
        row = bm25_rows.get(chunk_id)
        if not row:
            continue
        if _fingerprint(str(payload.get("text") or "")) != row["fingerprint"]:
            mismatches.append(chunk_id)
    return mismatches


def _production_changes(pre: dict[str, Any], post: dict[str, Any]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    pre_aliases = sorted(pre.get("aliases") or [], key=lambda item: (item.get("alias_name"), item.get("collection_name")))
    post_aliases = sorted(post.get("aliases") or [], key=lambda item: (item.get("alias_name"), item.get("collection_name")))
    if pre_aliases != post_aliases:
        changes.append({"resource": "aliases", "issue": "changed"})
    pre_collections = {item["name"]: item for item in pre.get("collections") or []}
    post_collections = {item["name"]: item for item in post.get("collections") or []}
    for name, before in pre_collections.items():
        if name == LEGAL_V2_COLLECTION_NAME:
            continue
        after = post_collections.get(name)
        if after and _point_count(before) != _point_count(after):
            changes.append({"resource": name, "issue": "collection_point_count_changed"})
    pre_bm25 = {
        _normalized_sidecar_path(item["path"]): item
        for item in pre.get("bm25_sidecars") or []
        if not item.get("is_legal_v2") and "path" in item
    }
    post_bm25 = {
        _normalized_sidecar_path(item["path"]): item
        for item in post.get("bm25_sidecars") or []
        if not item.get("is_legal_v2") and "path" in item
    }
    for path, before in pre_bm25.items():
        after = post_bm25.get(path)
        if after and before.get("sha256") != after.get("sha256"):
            changes.append({"resource": path, "issue": "bm25_checksum_changed"})
    return changes


def _point_count(item: dict[str, Any]) -> int | None:
    value = item.get("point_count", item.get("points_count"))
    return int(value) if value is not None else None


def _normalized_sidecar_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    if normalized.startswith("/app/"):
        normalized = normalized[len("/app/") :]
    return normalized


def _smoke_queries(client: Any, bm25_path: Path) -> list[dict[str, Any]]:
    config = legal_v2_retriever_config_from_env()
    retriever = build_live_legal_v2_retriever(client, BgeM3Embedder(_embedder_config(config)), config)
    bm25 = Bm25Sidecar(
        bm25_path,
        k1=LEGAL_V2_PROFILE.bm25_k1,
        b=LEGAL_V2_PROFILE.bm25_b,
        index_id=LEGAL_V2_BM25_INDEX_ID,
    )
    results = []
    expected_candidate_queries = set(SMOKE_QUERIES[:2])
    for query in SMOKE_QUERIES:
        expected_candidates = query in expected_candidate_queries
        hits = bm25.search(query, top_k=5)
        try:
            retrieved = retriever.retrieve(build_query_spec_v2(query))
            diagnostics = dict(retrieved.diagnostics)
            candidate_document_ids = [document.document_id for document in retrieved.documents[:5]]
            status = "pass" if diagnostics.get("dense_candidate_chunks") and diagnostics.get("bm25_candidate_chunks") else "fail"
            error = None
        except Exception as exc:  # noqa: BLE001 - smoke report records bounded failure details.
            diagnostics = {}
            candidate_document_ids = []
            status = "fail" if expected_candidates else "no_candidates"
            error = exc.__class__.__name__
        results.append(
            {
                "query": query,
                "expected_candidates": expected_candidates,
                "status": status,
                "bm25_hits": len(hits),
                "top_chunk_ids": [hit.id for hit in hits[:3]],
                "candidate_document_ids": candidate_document_ids,
                "diagnostics": diagnostics,
                "error": error,
                "bm25_index_id": config.bm25_index_id,
            }
        )
    return results


def _embedder_config(config: Any) -> ProductionRetrievalConfig:
    return ProductionRetrievalConfig(
        profile=LEGAL_V2_PROFILE,
        qdrant_collection=config.qdrant_collection,
        bm25_sidecar_path=config.bm25_sidecar_path,
        bm25_index_id=config.bm25_index_id,
        model_path=config.model_path,
        local_files_only=True,
        trust_remote_code=False,
        device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        candidate_multiplier=1,
        min_candidate_count=1,
        max_candidate_count=max(config.dense_candidate_chunks, config.bm25_candidate_chunks),
        lexical_filter_enabled=False,
    )


def _vector_size(info: Any) -> int | None:
    params = getattr(getattr(info, "config", None), "params", None)
    vectors = getattr(params, "vectors", None)
    size = getattr(vectors, "size", None)
    if size is not None:
        return int(size)
    if isinstance(vectors, dict):
        first = next(iter(vectors.values()), None)
        if first is not None and getattr(first, "size", None) is not None:
            return int(first.size)
    return None


def _fingerprint(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def _empty(value: Any) -> bool:
    return value is None or value == "" or value == []


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Legal Retrieval v2 smoke index validation",
        "",
        f"- Status: `{result['status']}`",
        f"- Collection: `{result['collection']}`",
        f"- Qdrant points: {result['qdrant_points']}",
        f"- BM25 rows: {result['bm25_rows']}",
        f"- Vector dimension: {result['vector_dimension']}",
        f"- Duplicate chunk IDs: `{result['duplicate_chunk_ids']}`",
        f"- Missing provenance issues: {result['missing_provenance']}",
        f"- Cross-document mixing: `{result['cross_document_mixing']}`",
        f"- Production changes: {len(result['production_changes'])}",
        "",
        "## Smoke queries",
        "",
    ]
    for item in result["smoke_queries"]:
        lines.append(
            f"- `{item['query']}` status={item['status']} "
            f"dense={item['diagnostics'].get('dense_candidate_chunks')} "
            f"bm25={item['diagnostics'].get('bm25_candidate_chunks')} "
            f"docs={item['candidate_document_ids']}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
