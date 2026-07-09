"""Run deterministic no-LLM legal answer evaluation over gold retrieval results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.eval.legal_answer_eval import (  # noqa: E402
    aggregate_answer_metrics,
    load_gold_registry_from_dataset,
    load_retrieval_results,
    resolve_retrieval_results_path,
    run_answer_eval,
    validate_gold_review_path,
    write_answer_eval_outputs,
)
from app.rag.eval.legal_qa_benchmark import load_dataset  # noqa: E402
from app.rag.retrieval.errors import RetrievalConfigurationError  # noqa: E402

DEFAULT_GOLD_REVIEW = PROJECT_ROOT / "artifacts/rag_eval/legal_qa/gold_source_review_20260709.md"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/rag_eval/legal_qa/answer_eval"

RETRIEVAL_CANDIDATES: dict[str, list[Path]] = {
    "usoud": [
        PROJECT_ROOT / "artifacts/rag_eval/legal_qa/runs/usoud_gold_eval/retrieval_results.jsonl",
        PROJECT_ROOT / "artifacts/rag_eval/legal_qa/runs/usoud_full_baseline/retrieval_results.jsonl",
    ],
    "nsoud": [
        PROJECT_ROOT / "artifacts/rag_eval/legal_qa/runs/nsoud_gold_eval/retrieval_results.jsonl",
        PROJECT_ROOT / "artifacts/rag_eval/legal_qa/runs/nsoud_full_baseline/retrieval_results.jsonl",
    ],
    "mixed": [
        PROJECT_ROOT / "artifacts/rag_eval/legal_qa/runs/mixed_gold_eval/retrieval_results.jsonl",
        PROJECT_ROOT / "artifacts/rag_eval/legal_qa/runs/mixed_two_pass_baseline/retrieval_results.jsonl",
    ],
}

DEFAULT_DATASETS: dict[str, Path] = {
    "usoud": PROJECT_ROOT / "artifacts/rag_eval/legal_qa/datasets/usoud_qa_v1.jsonl",
    "nsoud": PROJECT_ROOT / "artifacts/rag_eval/legal_qa/datasets/nsoud_qa_v1.jsonl",
    "mixed": PROJECT_ROOT / "artifacts/rag_eval/legal_qa/datasets/mixed_qa_v1.jsonl",
}

DEFAULT_OUTPUTS: dict[str, Path] = {
    "usoud": DEFAULT_OUTPUT_ROOT / "usoud_no_llm_baseline",
    "nsoud": DEFAULT_OUTPUT_ROOT / "nsoud_no_llm_baseline",
    "mixed": DEFAULT_OUTPUT_ROOT / "mixed_no_llm_baseline",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic no-LLM legal answer evaluation over gold retrieval results.",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--retrieval-results", type=Path, default=None)
    parser.add_argument("--gold-review", type=Path, default=DEFAULT_GOLD_REVIEW)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-llm", action="store_true", required=True)
    parser.add_argument("--require-citations", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def infer_corpus_key(dataset_path: Path) -> str:
    name = dataset_path.name.lower()
    if "usoud" in name:
        return "usoud"
    if "nsoud" in name:
        return "nsoud"
    if "mixed" in name:
        return "mixed"
    raise RetrievalConfigurationError(f"Cannot infer corpus from dataset name: {dataset_path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.no_llm:
        raise RetrievalConfigurationError("Only --no-llm mode is supported in v1.")

    dataset_path = args.dataset.resolve()
    if not dataset_path.exists():
        raise RetrievalConfigurationError(f"Dataset not found: {dataset_path}")

    gold_review_path = args.gold_review.resolve()
    validate_gold_review_path(gold_review_path)

    corpus_key = infer_corpus_key(dataset_path)
    if args.retrieval_results is not None:
        retrieval_path = args.retrieval_results.resolve()
        if not retrieval_path.exists():
            raise RetrievalConfigurationError(f"Retrieval results not found: {retrieval_path}")
    else:
        retrieval_path = resolve_retrieval_results_path(RETRIEVAL_CANDIDATES[corpus_key])

    items = load_dataset(dataset_path, limit=args.limit)
    registry = load_gold_registry_from_dataset(items)
    retrieval_by_id = load_retrieval_results(retrieval_path)
    results = run_answer_eval(
        items=items,
        registry=registry,
        retrieval_by_id=retrieval_by_id,
        citation_required=args.require_citations,
        limit=args.limit,
    )
    metrics = aggregate_answer_metrics(results)
    output_dir = args.output_dir.resolve()
    write_answer_eval_outputs(
        output_dir=output_dir,
        dataset_path=dataset_path,
        retrieval_results_path=retrieval_path,
        gold_review_path=gold_review_path,
        results=results,
        metrics=metrics,
        no_llm=True,
        citation_required=args.require_citations,
    )

    print(
        f"[legal-answer-eval] corpus={corpus_key} questions={metrics.total_questions} "
        f"gold={metrics.gold_available_count} pass_rate={metrics.answer_eval_pass_rate:.3f} "
        f"citation_rate={metrics.citation_available_rate:.3f} "
        f"unsupported_risk={metrics.unsupported_answer_risk_count} "
        f"retrieval={retrieval_path} output={output_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RetrievalConfigurationError as exc:
        print(f"[legal-answer-eval] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
