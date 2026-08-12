#!/usr/bin/env python3
"""Build the first Legal v2 ColBERT index over Slice 4 B contextual chunks.

Uses reusable ``app.rag.legal_v2.retrieve.colbert`` (PyLate backend).
Does not run golden benchmarks and does not change FAST/CE profiles.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.rag.legal_v2.retrieve.colbert import (  # noqa: E402
    COLBERT_PILOT_EXPECTED_CHUNK_COUNT,
    COLBERT_PILOT_SOURCE_QDRANT_COLLECTION,
    DEFAULT_COLBERT_MODEL,
    DEFAULT_INDEX_NAME,
    ColbertConfig,
    ColbertIndexer,
    ColbertRetriever,
    PyLateColbertBackend,
)
from app.rag.legal_v2.retrieve.colbert.corpus import (  # noqa: E402
    export_chunks_from_qdrant,
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "chunking_ab_pilot_300_v1"
    / "colbert_v1"
)


def _git_meta() -> tuple[str, str]:
    def _run(args: list[str]) -> str:
        try:
            return (
                subprocess.check_output(args, cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL)
                .decode("utf-8")
                .strip()
            )
        except Exception:  # noqa: BLE001
            return "unknown"

    return _run(["git", "rev-parse", "HEAD"]), _run(["git", "branch", "--show-current"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://nalus-scraper-qdrant-1:6333"),
    )
    p.add_argument(
        "--source-collection",
        default=COLBERT_PILOT_SOURCE_QDRANT_COLLECTION,
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model", default=DEFAULT_COLBERT_MODEL)
    p.add_argument("--device", default=os.getenv("COLBERT_DEVICE", "cuda"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    p.add_argument("--expected-chunks", type=int, default=COLBERT_PILOT_EXPECTED_CHUNK_COUNT)
    p.add_argument("--top-k-smoke", type=int, default=5)
    p.add_argument(
        "--smoke-queries",
        type=int,
        default=2,
        help="How many golden queries to run as smoke (0 disables).",
    )
    p.add_argument(
        "--golden-path",
        type=Path,
        default=PROJECT_ROOT / "benchmarks/legal_v2/case_similarity_golden_v1_pilot.jsonl",
    )
    p.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face model download if not cached.",
    )
    p.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Build + integrity only; skip retriever smoke queries.",
    )
    return p.parse_args(argv)


def _load_smoke_queries(path: Path, limit: int) -> list[str]:
    if limit <= 0 or not path.exists():
        return []
    queries: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(queries) >= limit:
                break
            payload = json.loads(line)
            q = str(payload.get("query") or "").strip()
            if q:
                queries.append(q)
    return queries


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir: Path = args.output_dir
    index_path = output_dir / "index"
    mapping_path = output_dir / "colbert_chunk_mapping.jsonl"
    manifest_path = output_dir / "colbert_index_manifest.json"
    report_path = output_dir / "colbert_build_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    git_head, git_branch = _git_meta()
    build_command = " ".join(
        [
            "python",
            "scripts/legal_v2/build_colbert_index_v1.py",
            f"--source-collection {args.source_collection}",
            f"--output-dir {output_dir.as_posix()}",
            f"--model {args.model}",
            f"--device {args.device}",
            f"--batch-size {args.batch_size}",
            f"--index-name {args.index_name}",
        ]
    )

    config = ColbertConfig(
        model_name=args.model,
        index_path=index_path,
        index_name=args.index_name,
        device=args.device,
        top_k=max(1, int(args.top_k_smoke)),
        batch_size=int(args.batch_size),
        concurrency_limit=1,
        mapping_path=mapping_path,
        allow_download=bool(args.allow_download),
        source_collection=args.source_collection,
        expected_chunk_count=int(args.expected_chunks),
    )
    backend = PyLateColbertBackend(config)
    indexer = ColbertIndexer(config, backend=backend)

    print(f"EXPORT_START collection={args.source_collection}")
    documents = await asyncio.to_thread(
        export_chunks_from_qdrant,
        qdrant_url=args.qdrant_url,
        collection=args.source_collection,
    )
    print(f"EXPORT_DONE rows={len(documents)}")

    print("INDEX_BUILD_START")
    build_result = await indexer.build(
        documents,
        source_collection=args.source_collection,
    )
    ready = bool(build_result.ready)
    print(
        "INDEX_BUILD_DONE "
        f"status={build_result.status} indexed={build_result.indexed_chunk_count} "
        f"mapping={build_result.mapping_row_count} ready={ready}"
    )

    smoke: dict[str, Any] = {"enabled": False, "queries": []}
    if ready and not args.skip_smoke:
        smoke["enabled"] = True
        retriever = ColbertRetriever(config, backend=backend)
        await backend.initialize()
        for query in _load_smoke_queries(args.golden_path, int(args.smoke_queries)):
            result = await retriever.retrieve(query, top_k=int(args.top_k_smoke))
            smoke["queries"].append(
                {
                    "query_preview": query[:160],
                    "hit_count": len(result.hits),
                    "hits": [
                        {
                            "rank": hit.rank,
                            "score": hit.score,
                            "document_id": hit.document_id,
                            "chunk_id": hit.chunk_id,
                            "text_preview": hit.text[:120].replace("\n", " "),
                        }
                        for hit in result.hits
                    ],
                    "diagnostics": result.diagnostics,
                }
            )
        await backend.close()

    timestamp = datetime.now(timezone.utc).isoformat()
    integrity = {
        "expected_chunks": build_result.expected_chunk_count,
        "indexed_chunks": build_result.indexed_chunk_count,
        "mapping_rows": build_result.mapping_row_count,
        "duplicate_chunk_ids": build_result.duplicate_chunk_ids,
        "missing_chunk_ids": build_result.missing_chunk_ids,
        "empty_texts": build_result.empty_texts,
    }
    manifest = {
        "timestamp": timestamp,
        "git_head": git_head,
        "branch": git_branch,
        "source_corpus": args.source_collection,
        "expected_chunk_count": build_result.expected_chunk_count,
        "actual_indexed_count": build_result.indexed_chunk_count,
        "model": build_result.model_name,
        "library": build_result.library,
        "library_version": build_result.library_version,
        "device": build_result.device,
        "index_configuration": {
            "index_name": args.index_name,
            "batch_size": args.batch_size,
            "backend": "pylate.indexes.PLAID",
        },
        "index_path": str(index_path),
        "mapping_path": str(mapping_path),
        "build_command": build_command,
        "status": "ok" if ready else "failed",
        "COLBERT_INDEX_READY": ready,
        "integrity": integrity,
    }
    report = {
        "manifest": manifest,
        "build_result": {
            "status": build_result.status,
            "ready": ready,
            "diagnostics": build_result.diagnostics,
            **integrity,
        },
        "smoke": smoke,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE {manifest_path}")
    print(f"WROTE {report_path}")
    print(f"COLBERT_INDEX_READY: {str(ready).lower()}")
    return 0 if ready else 2


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
