from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.index_builder import LegalV2BuildConfig, build_legal_v2_index  # noqa: E402
from app.rag.legal_v2.indexing import LEGAL_V2_PROFILE  # noqa: E402
from app.rag.legal_v2.retriever import legal_v2_retriever_config_from_env  # noqa: E402
from app.rag.legal_v2.sources import discover_source_documents  # noqa: E402
from app.rag.retrieval.bge_m3_embedder import BgeM3Embedder  # noqa: E402
from app.rag.retrieval.production_profile import ProductionRetrievalConfig  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the isolated Legal Retrieval v2 hybrid index.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/legal_v2/index_build")
    parser.add_argument("--batches-dir", type=Path, default=PROJECT_ROOT / "batches")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--overwrite-bm25", action="store_true")
    parser.add_argument("--recreate-v2-collection", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    retriever_config = legal_v2_retriever_config_from_env()
    documents = discover_source_documents(batches_dir=args.batches_dir, limit=args.limit)
    from qdrant_client import QdrantClient

    prod_config = ProductionRetrievalConfig(
        profile=LEGAL_V2_PROFILE,
        qdrant_collection=retriever_config.qdrant_collection,
        bm25_sidecar_path=retriever_config.bm25_sidecar_path,
        bm25_index_id=retriever_config.bm25_index_id,
        model_path=retriever_config.model_path,
        local_files_only=True,
        trust_remote_code=False,
        device="cpu",
        candidate_multiplier=1,
        min_candidate_count=1,
        max_candidate_count=1,
        lexical_filter_enabled=False,
    )
    manifest = build_legal_v2_index(
        documents=documents,
        embedder=BgeM3Embedder(prod_config),
        qdrant_client=QdrantClient(url=args.qdrant_url, timeout=60),
        config=LegalV2BuildConfig(
            bm25_path=retriever_config.bm25_sidecar_path,
            output_dir=args.output_dir,
            recreate_collection=args.recreate_v2_collection,
            overwrite_bm25=args.overwrite_bm25,
            resume=args.resume,
        ),
        git_commit=_git(["rev-parse", "HEAD"]),
        dirty=bool(_git(["status", "--short"])),
    )
    print(manifest.to_dict())
    return 0 if manifest.validation_status == "pass" else 1


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
