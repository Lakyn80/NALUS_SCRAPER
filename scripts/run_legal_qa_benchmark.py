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
    MixedTwoPassConfig,
    aggregate_metrics,
    aggregate_mixed_metrics,
    build_hybrid_retriever,
    build_mixed_two_pass_search_fn,
    load_dataset,
    resolve_bm25_sidecar_path,
    run_mixed_retrieval_benchmark,
    run_retrieval_benchmark,
    validate_collection_name,
    write_mixed_run_outputs,
    write_run_outputs,
)
from app.rag.retrieval.errors import RetrievalConfigurationError  # noqa: E402

DEFAULT_COLLECTION = "nalus_us_bge_m3_rag_combined_20260709"
DEFAULT_DATASET = PROJECT_ROOT / "artifacts/rag_eval/legal_qa/datasets/usoud_qa_v1.jsonl"
DEFAULT_MIXED_DATASET = PROJECT_ROOT / "artifacts/rag_eval/legal_qa/datasets/mixed_qa_v1.jsonl"
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
    parser.add_argument(
        "--mixed-two-pass",
        action="store_true",
        help="Run two-pass mixed retrieval over ÚS and NSoud collections.",
    )
    parser.add_argument("--usoud-collection-name", default=DEFAULT_COLLECTION)
    parser.add_argument(
        "--usoud-bm25-sidecar-path",
        type=Path,
        default=None,
        help="ÚS BM25 sidecar override. Defaults to production config or collection-named sidecar.",
    )
    parser.add_argument("--nsoud-collection-name", default=NSOUD_COLLECTION)
    parser.add_argument(
        "--nsoud-bm25-sidecar-path",
        type=Path,
        default=DEFAULT_NSOUD_BM25_SIDECAR,
        help="NSoud BM25 sidecar path for mixed benchmark runs.",
    )
    return parser.parse_args(argv)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / timestamp


def _run_single_collection_benchmark(args: argparse.Namespace) -> int:
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


def _run_mixed_two_pass_benchmark(args: argparse.Namespace) -> int:
    dataset_path = args.dataset.resolve()
    if not dataset_path.exists():
        raise RetrievalConfigurationError(f"Dataset not found: {dataset_path}")

    validate_collection_name(args.usoud_collection_name)
    validate_collection_name(args.nsoud_collection_name)
    items = load_dataset(dataset_path, limit=args.limit)
    if any(item.corpus != "mixed" for item in items):
        raise RetrievalConfigurationError("Mixed two-pass mode requires corpus='mixed' dataset items.")

    usoud_bm25 = resolve_bm25_sidecar_path(
        collection_name=args.usoud_collection_name,
        explicit_path=args.usoud_bm25_sidecar_path,
    )
    nsoud_bm25 = resolve_bm25_sidecar_path(
        collection_name=args.nsoud_collection_name,
        explicit_path=args.nsoud_bm25_sidecar_path,
    )

    output_dir = resolve_output_dir(args).resolve()
    qdrant_url = args.qdrant_url or __import__("os").getenv("QDRANT_URL", "http://localhost:6333")
    mixed_config = MixedTwoPassConfig(
        usoud_collection_name=args.usoud_collection_name,
        nsoud_collection_name=args.nsoud_collection_name,
        usoud_bm25_sidecar_path=usoud_bm25,
        nsoud_bm25_sidecar_path=nsoud_bm25,
        qdrant_url=qdrant_url,
        use_redis_cache=args.use_redis_cache,
    )

    search_fn = build_mixed_two_pass_search_fn(mixed_config)
    results = run_mixed_retrieval_benchmark(items=items, search_fn=search_fn, top_k=args.top_k)
    metrics = aggregate_mixed_metrics(results)
    write_mixed_run_outputs(
        output_dir=output_dir,
        dataset_path=dataset_path,
        config=mixed_config,
        top_k=args.top_k,
        retrieval_only=True,
        results=results,
        metrics=metrics,
    )

    print(
        f"[legal-qa-benchmark] mode=mixed-two-pass questions={metrics.question_count} "
        f"corpus_hit@1={metrics.corpus_hit_at_1:.3f} retrieval_hit@1={metrics.retrieval_hit_at_1:.3f} "
        f"pass_rate={metrics.pass_rate:.3f} usoud_bm25={usoud_bm25} output={output_dir}",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.top_k <= 0:
        raise RetrievalConfigurationError("--top-k must be positive.")
    if not args.retrieval_only:
        raise RetrievalConfigurationError("Only --retrieval-only mode is supported in v1.")
    if args.use_redis_cache:
        raise RetrievalConfigurationError("Redis cache must remain disabled for baseline runs.")

    if args.mixed_two_pass:
        return _run_mixed_two_pass_benchmark(args)
    return _run_single_collection_benchmark(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RetrievalConfigurationError as exc:
        print(f"[legal-qa-benchmark] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
