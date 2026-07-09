"""Run retrieval-only legal Q&A benchmark over BGE-M3 hybrid RAG."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.eval.legal_qa_benchmark import (  # noqa: E402
    aggregate_metrics,
    build_hybrid_retriever,
    load_dataset,
    run_retrieval_benchmark,
    validate_collection_name,
    write_run_outputs,
)
from app.rag.retrieval.errors import RetrievalConfigurationError  # noqa: E402

DEFAULT_COLLECTION = "nalus_us_bge_m3_rag_combined_20260709"
DEFAULT_DATASET = PROJECT_ROOT / "artifacts/rag_eval/legal_qa/datasets/usoud_qa_v1.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/rag_eval/legal_qa/runs"
NSOUD_COLLECTION = "nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1"
DEFAULT_NSOUD_BM25_SIDECAR = (
    PROJECT_ROOT / "storage/rag/bm25/nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1.sqlite"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval-only legal Q&A benchmark.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--retrieval-only", action="store_true", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--use-redis-cache", action="store_true")
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument(
        "--bm25-sidecar-path",
        type=Path,
        default=None,
        help="Override BM25 SQLite sidecar for non-default benchmark collections.",
    )
    return parser.parse_args(argv)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / timestamp


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.top_k <= 0:
        raise RetrievalConfigurationError("--top-k must be positive.")
    if not args.retrieval_only:
        raise RetrievalConfigurationError("Only --retrieval-only mode is supported in v1.")

    dataset_path = args.dataset.resolve()
    if not dataset_path.exists():
        raise RetrievalConfigurationError(f"Dataset not found: {dataset_path}")

    validate_collection_name(args.collection_name)
    items = load_dataset(dataset_path, limit=args.limit)
    output_dir = resolve_output_dir(args).resolve()
    qdrant_url = args.qdrant_url or __import__("os").getenv("QDRANT_URL", "http://localhost:6333")

    search_fn = build_hybrid_retriever(
        collection_name=args.collection_name,
        qdrant_url=qdrant_url,
        use_redis_cache=args.use_redis_cache,
        bm25_sidecar_path=args.bm25_sidecar_path,
    )
    results = run_retrieval_benchmark(items=items, search_fn=search_fn, top_k=args.top_k)
    metrics = aggregate_metrics(results)
    write_run_outputs(
        output_dir=output_dir,
        dataset_path=dataset_path,
        collection_name=args.collection_name,
        top_k=args.top_k,
        retrieval_only=True,
        use_redis_cache=args.use_redis_cache,
        results=results,
        metrics=metrics,
    )

    print(
        f"[legal-qa-benchmark] questions={metrics.question_count} "
        f"hit@1={metrics.hit_at_1:.3f} hit@5={metrics.hit_at_5:.3f} "
        f"pass_rate={metrics.pass_rate:.3f} output={output_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RetrievalConfigurationError as exc:
        print(f"[legal-qa-benchmark] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
