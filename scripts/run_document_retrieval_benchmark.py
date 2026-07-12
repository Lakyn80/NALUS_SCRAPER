"""Run offline document-level legal retrieval benchmark."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.eval.document_retrieval_benchmark import (  # noqa: E402
    aggregate_document_benchmark_metrics,
    load_document_benchmark_dataset,
    run_document_retrieval_benchmark,
    write_document_benchmark_outputs,
)
from app.rag.eval.legal_qa_benchmark import (  # noqa: E402
    build_hybrid_retriever,
    validate_collection_name,
)
from app.rag.retrieval.document_retrieval import (  # noqa: E402
    DOCUMENT_SCORING_BEST_PLUS_AVERAGE,
    DocumentRetrievalConfig,
)
from app.rag.retrieval.errors import RetrievalConfigurationError  # noqa: E402

DEFAULT_COLLECTION = "nalus_us_bge_m3_rag_combined_20260709"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/rag_eval/legal_qa/document_retrieval_benchmark"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline document-level legal retrieval benchmark.",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION)
    parser.add_argument("--chunk-top-k", type=int, default=100)
    parser.add_argument("--candidate-pool-size", type=int, default=200)
    parser.add_argument("--max-returned-documents", type=int, default=100)
    parser.add_argument("--max-supporting-chunks", type=int, default=3)
    parser.add_argument("--document-threshold", type=float, default=0.0)
    parser.add_argument("--scoring-strategy", default=DOCUMENT_SCORING_BEST_PLUS_AVERAGE)
    parser.add_argument("--latency-budget-ms", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--bm25-sidecar-path", type=Path, default=None)
    parser.add_argument("--retrieval-only", action="store_true", required=True)
    parser.add_argument("--use-redis-cache", action="store_true")
    return parser.parse_args(argv)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / timestamp


def build_document_config(args: argparse.Namespace) -> DocumentRetrievalConfig:
    config = DocumentRetrievalConfig(
        enabled=True,
        max_candidate_chunks=args.candidate_pool_size,
        max_returned_documents=args.max_returned_documents,
        max_supporting_chunks_per_document=args.max_supporting_chunks,
        document_relevance_threshold=args.document_threshold,
        scoring_strategy=args.scoring_strategy,
        latency_budget_ms=args.latency_budget_ms,
    )
    config.validate()
    return config


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.retrieval_only:
        raise RetrievalConfigurationError("Only --retrieval-only mode is supported.")
    if args.use_redis_cache:
        raise RetrievalConfigurationError("Redis cache must remain disabled for benchmark runs.")
    if args.chunk_top_k <= 0:
        raise RetrievalConfigurationError("--chunk-top-k must be positive.")

    dataset_path = args.dataset.resolve()
    if not dataset_path.exists():
        raise RetrievalConfigurationError(f"Dataset not found: {dataset_path}")
    validate_collection_name(args.collection_name)

    document_config = build_document_config(args)
    items = load_document_benchmark_dataset(dataset_path, limit=args.limit)
    qdrant_url = args.qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
    search_fn = build_hybrid_retriever(
        collection_name=args.collection_name,
        qdrant_url=qdrant_url,
        use_redis_cache=False,
        bm25_sidecar_path=args.bm25_sidecar_path,
    )
    results = run_document_retrieval_benchmark(
        items=items,
        search_fn=search_fn,
        chunk_top_k=args.chunk_top_k,
        document_config=document_config,
    )
    metrics = aggregate_document_benchmark_metrics(results)
    output_dir = resolve_output_dir(args).resolve()
    write_document_benchmark_outputs(
        output_dir=output_dir,
        dataset_path=dataset_path,
        collection_name=args.collection_name,
        chunk_top_k=args.chunk_top_k,
        document_config=document_config,
        results=results,
        metrics=metrics,
    )

    print(
        f"[document-retrieval-benchmark] questions={metrics.question_count} "
        f"document_recall@10={metrics.document_recall_at_10:.3f} "
        f"candidate_pool_coverage={metrics.candidate_pool_coverage:.3f} "
        f"duplicate_rate={metrics.duplicate_rate:.3f} output={output_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RetrievalConfigurationError as exc:
        print(f"[document-retrieval-benchmark] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
