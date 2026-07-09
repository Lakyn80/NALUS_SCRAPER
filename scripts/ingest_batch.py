"""
Legacy standalone Qdrant ingest script.

Ingests one or more batch JSON files from the batches/ directory into Qdrant.
Safe to re-run — identical chunks are skipped (idempotent).

This is not the production BGE-M3 dense+BM25+RRF build path.

Usage:
    python scripts/ingest_batch.py                          # ingest all batches/
    python scripts/ingest_batch.py batches/foo.json         # ingest single file
    python scripts/ingest_batch.py --url http://host:6333   # custom Qdrant URL
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

BATCHES_DIR = PROJECT_ROOT / "batches"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest NALUS batch JSON files into Qdrant.")
    parser.add_argument(
        "files",
        nargs="*",
        help="Batch JSON files to ingest. Defaults to all *.json in batches/ (except manifest.json).",
    )
    parser.add_argument("--url", default="http://localhost:6333", help="Qdrant URL")
    parser.add_argument("--collection", default="nalus", help="Qdrant collection name")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument(
        "--allow-legacy-ingest",
        action="store_true",
        help="Explicitly allow this legacy dense-only ingest path.",
    )
    return parser.parse_args()


def _resolve_files(paths: list[str]) -> list[Path]:
    if paths:
        return [Path(p) for p in paths]
    return sorted(
        p for p in BATCHES_DIR.glob("*.json") if p.name != "manifest.json"
    )


def _load_manifest() -> dict:
    manifest_path = BATCHES_DIR / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"version": 1, "description": "Tracking ingestovaných batchů do Qdrant kolekce 'nalus'", "batches": []}


def _save_manifest(manifest: dict) -> None:
    manifest_path = BATCHES_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[INGEST] manifest updated -> {manifest_path}")


def main() -> int:
    args = _parse_args()
    if not args.allow_legacy_ingest:
        print(
            "[INGEST] refused: scripts/ingest_batch.py is a legacy dense-only path. "
            "Use the BGE-M3 candidate/build workflow for production retrieval, or pass "
            "--allow-legacy-ingest for an explicit legacy/test run."
        )
        return 2

    files = _resolve_files(args.files)

    if not files:
        print("[INGEST] no batch files found")
        return 1

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from app.data.runtime_corpus import build_runtime_corpus, load_results_from_json
    from app.rag.ingest.qdrant_ingest import QdrantIngestor, _MOCK_VECTOR_DIM
    from app.rag.retrieval.embedder import SentenceTransformersEmbedder

    class _Adapter:
        """Converts IngestPoint -> PointStruct before calling Qdrant."""
        def __init__(self, c: QdrantClient) -> None:
            self._c = c

        def upsert(self, collection_name: str, points: list) -> None:
            self._c.upsert(
                collection_name=collection_name,
                points=[PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points],
            )

        def retrieve(self, *, collection_name: str, ids: list, with_payload: bool = True, with_vectors: bool = False):
            return self._c.retrieve(collection_name=collection_name, ids=ids,
                                    with_payload=with_payload, with_vectors=with_vectors)

    client = QdrantClient(url=args.url, timeout=60)
    print(f"[INGEST] connected to Qdrant at {args.url}")

    existing = {c.name for c in client.get_collections().collections}
    if args.collection not in existing:
        client.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=_MOCK_VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"[INGEST] created collection '{args.collection}'")
    else:
        count = client.count(collection_name=args.collection).count
        print(f"[INGEST] collection '{args.collection}' exists ({count} points)")

    manifest = _load_manifest()
    already_ingested = {b["file"] for b in manifest.get("batches", [])}

    print("[INGEST] loading embedding model (sentence-transformers)...")
    embedder = SentenceTransformersEmbedder()
    print(f"[INGEST] embedder ready, dim={embedder.dim}")
    ingestor = QdrantIngestor(_Adapter(client), collection_name=args.collection, batch_size=args.batch_size, embedder=embedder)

    for batch_path in files:
        print(f"\n[INGEST] processing {batch_path.name}")

        results = load_results_from_json(batch_path)
        corpus = build_runtime_corpus(results, source_label=str(batch_path))
        print(f"[INGEST]   {len(results)} docs -> {len(corpus.chunks)} chunks")

        stats = ingestor.ingest_chunks(corpus.chunks)
        print(
            f"[INGEST]   inserted={stats.inserted_points} "
            f"updated={stats.updated_points} skipped={stats.skipped_points}"
        )

        entry = {
            "file": batch_path.name,
            "doc_count": len(results),
            "chunk_count": len(corpus.chunks),
            "ingested_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        }
        manifest["batches"] = [b for b in manifest["batches"] if b.get("file") != batch_path.name]
        manifest["batches"].append(entry)

    _save_manifest(manifest)
    print("\n[INGEST] all done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
