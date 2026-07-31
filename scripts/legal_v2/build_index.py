from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.index_builder import LegalV2BuildConfig, build_legal_v2_index  # noqa: E402
from app.rag.legal_v2.indexing import LEGAL_V2_PROFILE  # noqa: E402
from app.rag.legal_v2.retriever import legal_v2_retriever_config_from_env  # noqa: E402
from app.rag.legal_v2.sources import discover_source_documents, discover_source_documents_by_ids  # noqa: E402
from app.rag.retrieval.bge_m3_embedder import BgeM3Embedder  # noqa: E402
from app.rag.retrieval.production_profile import ProductionRetrievalConfig  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the isolated Legal Retrieval v2 hybrid index.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/legal_v2/index_build")
    parser.add_argument("--batches-dir", type=Path, default=PROJECT_ROOT / "batches")
    parser.add_argument("--document-ids-file", type=Path, default=None)
    parser.add_argument("--parser-quality-artifact", type=Path, default=None)
    parser.add_argument("--gate-decision", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--overwrite-bm25", action="store_true")
    parser.add_argument("--recreate-v2-collection", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.gate_decision is not None:
        _require_gate_pass(args.gate_decision)
    if args.document_ids_file is not None and args.parser_quality_artifact is not None:
        raise ValueError("--document-ids-file and --parser-quality-artifact are mutually exclusive")
    retriever_config = legal_v2_retriever_config_from_env()
    if args.parser_quality_artifact is not None:
        document_ids = _document_ids_from_parser_quality(args.parser_quality_artifact, limit=args.limit)
        documents = discover_source_documents_by_ids(document_ids, batches_dir=args.batches_dir)
        _require_all_documents_found(document_ids, documents)
    elif args.document_ids_file is not None:
        document_ids = [
            line.strip()
            for line in args.document_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if args.limit is not None:
            document_ids = document_ids[: args.limit]
        documents = discover_source_documents_by_ids(document_ids, batches_dir=args.batches_dir)
        _require_all_documents_found(document_ids, documents)
    else:
        documents = discover_source_documents(batches_dir=args.batches_dir, limit=args.limit)
    from qdrant_client import QdrantClient  # type: ignore[import-not-found]

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


def _require_gate_pass(path: Path) -> None:
    payload = _json_object(path)
    if payload.get("final_decision") != "pass" or payload.get("smoke_index_permitted") is not True:
        raise ValueError(f"Legal v2 parser QA gate does not permit indexing: {path}")


def _document_ids_from_parser_quality(path: Path, *, limit: int | None) -> list[str]:
    payload = _json_object(path)
    documents = payload.get("documents")
    if not isinstance(documents, list):
        raise ValueError(f"Parser quality artifact has no documents list: {path}")
    selected: list[str] = []
    for index, item in enumerate(documents):
        if not isinstance(item, dict):
            raise ValueError(f"Parser quality sample is not an object at index {index}: {path}")
        if item.get("review_status") != "approved":
            continue
        if item.get("identified_defects"):
            continue
        if str(item.get("source_completeness_status") or "") != "complete_from_available_source":
            continue
        duplicate_status = str(item.get("duplicate_source_identifier_status") or "")
        if duplicate_status not in {"none", "identical_duplicate_records", "metadata_only_duplicate_records"}:
            continue
        document_id = str(item.get("document_id") or "").strip()
        if document_id:
            selected.append(document_id)
        if limit is not None and len(selected) >= limit:
            break
    if not selected:
        raise ValueError(f"No approved complete parser QA documents selected from {path}")
    return selected


def _require_all_documents_found(document_ids: list[str], documents: list[Any]) -> None:
    found = {document.document_id for document in documents}
    missing = [document_id for document_id in document_ids if document_id not in found]
    if missing:
        raise ValueError(f"Requested Legal v2 source documents were not found: {missing}")


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
