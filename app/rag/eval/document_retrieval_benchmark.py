"""Offline benchmark for document-level legal retrieval.

The benchmark is read-only: it does not modify retrieval, ranking, embeddings,
Qdrant, BM25, RRF, Redis, or LLM behavior. It evaluates already implemented
retrieval paths by calling an injected search function and aggregating the
returned candidate chunks into document-level results.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.rag.retrieval.document_retrieval import (
    DocumentRetrievalConfig,
    build_document_level_results,
    canonical_document_id,
)
from app.rag.retrieval.errors import RetrievalConfigurationError
from app.rag.retrieval.models import RetrievedChunk

logger = get_logger(__name__)

DOCUMENT_RECALL_K_VALUES = (10, 20, 50, 100)
FAILURE_RELEVANT_DOCUMENT_NEVER_RETRIEVED = "relevant_document_never_retrieved"
FAILURE_RELEVANT_DOCUMENT_REMOVED_BY_AGGREGATION = "relevant_document_removed_by_aggregation"
FAILURE_RELEVANT_DOCUMENT_REMOVED_BY_THRESHOLD = "relevant_document_removed_by_threshold"
FAILURE_RELEVANT_DOCUMENT_REMOVED_BY_RETURNED_LIMIT = "relevant_document_removed_by_returned_document_limit"
FAILURE_DUPLICATE_HANDLING_ISSUE = "duplicate_handling_issue"
FAILURE_METADATA_ISSUE = "metadata_issue"
FAILURE_UNKNOWN = "unknown"


class RetrievalSearchFn(Protocol):
    def __call__(self, query: str, top_k: int) -> list[RetrievedChunk]: ...


@dataclass(frozen=True)
class DocumentBenchmarkItem:
    id: str
    corpus: str
    question: str
    relevant_document_ids: list[str]
    legal_topic: str | None = None
    difficulty: str | None = None

    @classmethod
    def from_dict(cls, payload: dict) -> "DocumentBenchmarkItem":
        validate_document_benchmark_item(payload)
        return cls(
            id=str(payload["id"]).strip(),
            corpus=str(payload["corpus"]).strip(),
            question=str(payload["question"]).strip(),
            relevant_document_ids=_dedupe_document_ids(payload.get("relevant_document_ids") or []),
            legal_topic=_nullable_str(payload.get("legal_topic")),
            difficulty=_nullable_str(payload.get("difficulty")),
        )


@dataclass(frozen=True)
class PerQuestionDocumentMetrics:
    question_id: str
    corpus: str
    relevant_document_count: int
    chunk_recall_at_10: float
    chunk_recall_at_20: float
    chunk_recall_at_50: float
    chunk_recall_at_100: float
    document_recall_at_10: float
    document_recall_at_20: float
    document_recall_at_50: float
    document_recall_at_100: float
    precision_at_10: float
    precision_at_20: float
    precision_at_50: float
    precision_at_100: float
    candidate_recall: float
    final_recall: float
    candidate_chunk_count: int
    candidate_unique_document_count: int
    final_document_count: int
    duplicate_rate: float
    retrieval_latency_ms: float
    aggregation_latency_ms: float
    failure_category: str | None = None


@dataclass(frozen=True)
class DocumentBenchmarkQuestionResult:
    item: DocumentBenchmarkItem
    metrics: PerQuestionDocumentMetrics
    chunk_document_ids: list[str] = field(default_factory=list)
    candidate_document_ids: list[str] = field(default_factory=list)
    final_document_ids: list[str] = field(default_factory=list)
    failure_missing_document_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DocumentBenchmarkMetrics:
    question_count: int
    gold_document_count: int
    zero_relevant_question_count: int
    chunk_recall_at_10: float
    chunk_recall_at_20: float
    chunk_recall_at_50: float
    chunk_recall_at_100: float
    document_recall_at_10: float
    document_recall_at_20: float
    document_recall_at_50: float
    document_recall_at_100: float
    precision_at_10: float
    precision_at_20: float
    precision_at_50: float
    precision_at_100: float
    unique_document_coverage: float
    candidate_pool_coverage: float
    duplicate_rate: float
    zero_result_rate: float
    average_retrieved_documents: float
    average_candidate_chunks: float
    average_latency_ms: float
    document_aggregation_latency_ms: float
    failure_breakdown: dict[str, int] = field(default_factory=dict)


def validate_document_benchmark_item(payload: dict) -> None:
    if not isinstance(payload, dict):
        raise RetrievalConfigurationError("Document benchmark item must be a JSON object.")
    for field_name in ("id", "corpus", "question", "relevant_document_ids"):
        if field_name not in payload:
            raise RetrievalConfigurationError(
                f"Document benchmark item is missing required field: {field_name}."
            )
    if not str(payload.get("id") or "").strip():
        raise RetrievalConfigurationError("Document benchmark item id must not be empty.")
    if not str(payload.get("corpus") or "").strip():
        raise RetrievalConfigurationError(
            f"Document benchmark item {payload.get('id')!r} has empty corpus."
        )
    if not str(payload.get("question") or "").strip():
        raise RetrievalConfigurationError(
            f"Document benchmark item {payload.get('id')!r} has empty question."
        )
    relevant = payload.get("relevant_document_ids")
    if not isinstance(relevant, list):
        raise RetrievalConfigurationError(
            f"Document benchmark item {payload.get('id')!r} relevant_document_ids must be a list."
        )


def load_document_benchmark_dataset(
    path: Path,
    *,
    limit: int | None = None,
) -> list[DocumentBenchmarkItem]:
    items: list[DocumentBenchmarkItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RetrievalConfigurationError(
                    f"Invalid JSON in document benchmark dataset {path} line {line_number}: {exc}"
                ) from exc
            items.append(DocumentBenchmarkItem.from_dict(payload))
            if limit is not None and len(items) >= limit:
                break
    if not items:
        raise RetrievalConfigurationError(f"Document benchmark dataset {path} contains no items.")
    return items


def run_document_retrieval_benchmark(
    *,
    items: list[DocumentBenchmarkItem],
    search_fn: RetrievalSearchFn,
    chunk_top_k: int,
    document_config: DocumentRetrievalConfig,
) -> list[DocumentBenchmarkQuestionResult]:
    if chunk_top_k < 1:
        raise RetrievalConfigurationError("chunk_top_k must be >= 1.")
    document_config.validate()
    logger.info(
        "[document-benchmark] dataset_size=%d chunk_top_k=%d candidate_pool=%d",
        len(items),
        chunk_top_k,
        document_config.max_candidate_chunks,
    )
    results: list[DocumentBenchmarkQuestionResult] = []
    started = time.perf_counter()
    for item in items:
        results.append(
            evaluate_document_benchmark_item(
                item=item,
                search_fn=search_fn,
                chunk_top_k=chunk_top_k,
                document_config=document_config,
            )
        )
    metrics = aggregate_document_benchmark_metrics(results)
    trace_event(
        logger,
        "document_benchmark.done",
        questions=len(items),
        document_recall_at_10=metrics.document_recall_at_10,
        candidate_pool_coverage=metrics.candidate_pool_coverage,
        runtime_ms=round((time.perf_counter() - started) * 1000, 3),
    )
    logger.info(
        "[document-benchmark] done questions=%d doc_recall@10=%.3f candidate_coverage=%.3f",
        len(items),
        metrics.document_recall_at_10,
        metrics.candidate_pool_coverage,
    )
    return results


def evaluate_document_benchmark_item(
    *,
    item: DocumentBenchmarkItem,
    search_fn: RetrievalSearchFn,
    chunk_top_k: int,
    document_config: DocumentRetrievalConfig,
) -> DocumentBenchmarkQuestionResult:
    query = item.question
    retrieval_started = time.perf_counter()
    chunk_hits = search_fn(query, chunk_top_k)
    candidate_chunks = search_fn(query, document_config.max_candidate_chunks)
    retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000

    document_result = build_document_level_results(
        candidate_chunks=candidate_chunks,
        config=document_config,
        retrieval_latency_ms=retrieval_latency_ms,
    )
    diagnostic_config = replace(
        document_config,
        document_relevance_threshold=0.0,
        max_returned_documents=max(document_config.max_returned_documents, document_config.max_candidate_chunks),
    )
    unfiltered_document_result = build_document_level_results(
        candidate_chunks=candidate_chunks,
        config=diagnostic_config,
        retrieval_latency_ms=retrieval_latency_ms,
    )

    gold_ids = set(_dedupe_document_ids(item.relevant_document_ids))
    chunk_document_ids = _document_ids_from_chunks(chunk_hits)
    candidate_document_ids = _document_ids_from_chunks(candidate_chunks)
    final_document_ids = [normalize_document_id(document.document_id) for document in document_result.documents]
    unfiltered_document_ids = [
        normalize_document_id(document.document_id) for document in unfiltered_document_result.documents
    ]

    missing = sorted(gold_ids.difference(final_document_ids))
    failure_category = (
        classify_failure(
            gold_ids=gold_ids,
            candidate_document_ids=set(candidate_document_ids),
            unfiltered_document_ids=set(unfiltered_document_ids),
            final_document_ids=set(final_document_ids),
            final_ordered_document_ids=final_document_ids,
            document_config=document_config,
            chunks_missing_document_id=document_result.diagnostics.chunks_missing_document_id,
            duplicate_rate=_duplicate_rate(candidate_chunks),
        )
        if missing
        else None
    )
    metrics = PerQuestionDocumentMetrics(
        question_id=item.id,
        corpus=item.corpus,
        relevant_document_count=len(gold_ids),
        chunk_recall_at_10=_recall_at_k(gold_ids, chunk_document_ids, 10),
        chunk_recall_at_20=_recall_at_k(gold_ids, chunk_document_ids, 20),
        chunk_recall_at_50=_recall_at_k(gold_ids, chunk_document_ids, 50),
        chunk_recall_at_100=_recall_at_k(gold_ids, chunk_document_ids, 100),
        document_recall_at_10=_recall_at_k(gold_ids, final_document_ids, 10),
        document_recall_at_20=_recall_at_k(gold_ids, final_document_ids, 20),
        document_recall_at_50=_recall_at_k(gold_ids, final_document_ids, 50),
        document_recall_at_100=_recall_at_k(gold_ids, final_document_ids, 100),
        precision_at_10=_precision_at_k(gold_ids, final_document_ids, 10),
        precision_at_20=_precision_at_k(gold_ids, final_document_ids, 20),
        precision_at_50=_precision_at_k(gold_ids, final_document_ids, 50),
        precision_at_100=_precision_at_k(gold_ids, final_document_ids, 100),
        candidate_recall=_recall_at_k(gold_ids, candidate_document_ids, len(candidate_document_ids)),
        final_recall=_recall_at_k(gold_ids, final_document_ids, len(final_document_ids)),
        candidate_chunk_count=len(candidate_chunks),
        candidate_unique_document_count=len(set(candidate_document_ids)),
        final_document_count=len(final_document_ids),
        duplicate_rate=_duplicate_rate(candidate_chunks),
        retrieval_latency_ms=retrieval_latency_ms,
        aggregation_latency_ms=document_result.diagnostics.aggregation_latency_ms or 0.0,
        failure_category=failure_category,
    )
    return DocumentBenchmarkQuestionResult(
        item=item,
        metrics=metrics,
        chunk_document_ids=chunk_document_ids,
        candidate_document_ids=candidate_document_ids,
        final_document_ids=final_document_ids,
        failure_missing_document_ids=missing,
    )


def classify_failure(
    *,
    gold_ids: set[str],
    candidate_document_ids: set[str],
    unfiltered_document_ids: set[str],
    final_document_ids: set[str],
    final_ordered_document_ids: list[str],
    document_config: DocumentRetrievalConfig,
    chunks_missing_document_id: int,
    duplicate_rate: float,
) -> str:
    if not gold_ids:
        return FAILURE_UNKNOWN
    missing_from_final = gold_ids.difference(final_document_ids)
    if gold_ids.isdisjoint(candidate_document_ids):
        return FAILURE_RELEVANT_DOCUMENT_NEVER_RETRIEVED
    if chunks_missing_document_id > 0 and not gold_ids.issubset(candidate_document_ids):
        return FAILURE_METADATA_ISSUE
    if (
        missing_from_final
        and missing_from_final.issubset(unfiltered_document_ids)
        and document_config.document_relevance_threshold > 0
    ):
        return FAILURE_RELEVANT_DOCUMENT_REMOVED_BY_THRESHOLD
    if not gold_ids.isdisjoint(unfiltered_document_ids) and gold_ids.isdisjoint(final_document_ids):
        if document_config.document_relevance_threshold > 0:
            return FAILURE_RELEVANT_DOCUMENT_REMOVED_BY_THRESHOLD
        return FAILURE_RELEVANT_DOCUMENT_REMOVED_BY_RETURNED_LIMIT
    if not gold_ids.isdisjoint(unfiltered_document_ids):
        if missing_from_final and len(final_ordered_document_ids) >= document_config.max_returned_documents:
            return FAILURE_RELEVANT_DOCUMENT_REMOVED_BY_RETURNED_LIMIT
    if not gold_ids.issubset(unfiltered_document_ids):
        return FAILURE_RELEVANT_DOCUMENT_REMOVED_BY_AGGREGATION
    if duplicate_rate > 0:
        return FAILURE_DUPLICATE_HANDLING_ISSUE
    return FAILURE_UNKNOWN


def aggregate_document_benchmark_metrics(
    results: list[DocumentBenchmarkQuestionResult],
) -> DocumentBenchmarkMetrics:
    if not results:
        raise RetrievalConfigurationError("No document benchmark results to aggregate.")

    question_count = len(results)
    all_gold_ids: set[str] = set()
    all_candidate_gold_hits: set[str] = set()
    all_final_gold_hits: set[str] = set()
    failure_breakdown: dict[str, int] = {}
    for result in results:
        gold_ids = set(_dedupe_document_ids(result.item.relevant_document_ids))
        all_gold_ids.update(gold_ids)
        all_candidate_gold_hits.update(gold_ids.intersection(result.candidate_document_ids))
        all_final_gold_hits.update(gold_ids.intersection(result.final_document_ids))
        if result.metrics.failure_category is not None:
            failure_breakdown[result.metrics.failure_category] = (
                failure_breakdown.get(result.metrics.failure_category, 0) + 1
            )

    def mean(field_name: str) -> float:
        return sum(float(getattr(result.metrics, field_name)) for result in results) / question_count

    return DocumentBenchmarkMetrics(
        question_count=question_count,
        gold_document_count=sum(result.metrics.relevant_document_count for result in results),
        zero_relevant_question_count=sum(
            1 for result in results if result.metrics.relevant_document_count == 0
        ),
        chunk_recall_at_10=mean("chunk_recall_at_10"),
        chunk_recall_at_20=mean("chunk_recall_at_20"),
        chunk_recall_at_50=mean("chunk_recall_at_50"),
        chunk_recall_at_100=mean("chunk_recall_at_100"),
        document_recall_at_10=mean("document_recall_at_10"),
        document_recall_at_20=mean("document_recall_at_20"),
        document_recall_at_50=mean("document_recall_at_50"),
        document_recall_at_100=mean("document_recall_at_100"),
        precision_at_10=mean("precision_at_10"),
        precision_at_20=mean("precision_at_20"),
        precision_at_50=mean("precision_at_50"),
        precision_at_100=mean("precision_at_100"),
        unique_document_coverage=(
            len(all_final_gold_hits) / len(all_gold_ids) if all_gold_ids else 0.0
        ),
        candidate_pool_coverage=(
            len(all_candidate_gold_hits) / len(all_gold_ids) if all_gold_ids else 0.0
        ),
        duplicate_rate=mean("duplicate_rate"),
        zero_result_rate=sum(1 for result in results if result.metrics.final_document_count == 0)
        / question_count,
        average_retrieved_documents=mean("final_document_count"),
        average_candidate_chunks=mean("candidate_chunk_count"),
        average_latency_ms=mean("retrieval_latency_ms"),
        document_aggregation_latency_ms=mean("aggregation_latency_ms"),
        failure_breakdown=failure_breakdown,
    )


def write_document_benchmark_outputs(
    *,
    output_dir: Path,
    dataset_path: Path,
    collection_name: str,
    chunk_top_k: int,
    document_config: DocumentRetrievalConfig,
    results: list[DocumentBenchmarkQuestionResult],
    metrics: DocumentBenchmarkMetrics,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_name = output_dir.name
    summary = {
        "generated_at": generated_at,
        "run_name": run_name,
        "corpus": _infer_corpus(results),
        "dataset": str(dataset_path),
        "collection_name": collection_name,
        "chunk_top_k": chunk_top_k,
        "document_config": asdict(document_config),
        **asdict(metrics),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(output_dir / "per_question.jsonl", results)
    _write_csv(output_dir / "per_question.csv", results)
    _write_markdown_summary(
        output_dir / "summary.md",
        generated_at=generated_at,
        dataset_path=dataset_path,
        collection_name=collection_name,
        chunk_top_k=chunk_top_k,
        document_config=document_config,
        metrics=metrics,
    )


def normalize_document_id(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _write_jsonl(path: Path, results: list[DocumentBenchmarkQuestionResult]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            payload = {
                "question_id": result.item.id,
                "corpus": result.item.corpus,
                "legal_topic": result.item.legal_topic,
                "difficulty": result.item.difficulty,
                "relevant_document_count": result.metrics.relevant_document_count,
                "metrics": asdict(result.metrics),
                "failure_missing_document_ids": result.failure_missing_document_ids,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_csv(path: Path, results: list[DocumentBenchmarkQuestionResult]) -> None:
    fieldnames = [
        "question_id",
        "corpus",
        "relevant_document_count",
        "chunk_recall_at_10",
        "document_recall_at_10",
        "document_recall_at_20",
        "document_recall_at_50",
        "document_recall_at_100",
        "precision_at_10",
        "candidate_recall",
        "final_recall",
        "candidate_chunk_count",
        "candidate_unique_document_count",
        "final_document_count",
        "duplicate_rate",
        "retrieval_latency_ms",
        "aggregation_latency_ms",
        "failure_category",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = asdict(result.metrics)
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_markdown_summary(
    path: Path,
    *,
    generated_at: str,
    dataset_path: Path,
    collection_name: str,
    chunk_top_k: int,
    document_config: DocumentRetrievalConfig,
    metrics: DocumentBenchmarkMetrics,
) -> None:
    lines = [
        "# Document-level retrieval benchmark summary",
        "",
        f"- Generated: {generated_at}",
        f"- Dataset: `{dataset_path}`",
        f"- Collection: `{collection_name}`",
        f"- Questions: {metrics.question_count}",
        f"- Gold documents: {metrics.gold_document_count}",
        f"- Chunk top-k: {chunk_top_k}",
        f"- Candidate chunks: {document_config.max_candidate_chunks}",
        f"- Max returned documents: {document_config.max_returned_documents}",
        f"- Relevance threshold: {document_config.document_relevance_threshold}",
        f"- Scoring strategy: `{document_config.scoring_strategy}`",
        "",
        "## Metrics",
        "",
        f"- chunk_recall@10: {metrics.chunk_recall_at_10:.3f}",
        f"- document_recall@10: {metrics.document_recall_at_10:.3f}",
        f"- document_recall@20: {metrics.document_recall_at_20:.3f}",
        f"- document_recall@50: {metrics.document_recall_at_50:.3f}",
        f"- document_recall@100: {metrics.document_recall_at_100:.3f}",
        f"- precision@10: {metrics.precision_at_10:.3f}",
        f"- unique_document_coverage: {metrics.unique_document_coverage:.3f}",
        f"- candidate_pool_coverage: {metrics.candidate_pool_coverage:.3f}",
        f"- duplicate_rate: {metrics.duplicate_rate:.3f}",
        f"- zero_result_rate: {metrics.zero_result_rate:.3f}",
        f"- average_retrieved_documents: {metrics.average_retrieved_documents:.3f}",
        f"- average_candidate_chunks: {metrics.average_candidate_chunks:.3f}",
        f"- average_latency_ms: {metrics.average_latency_ms:.3f}",
        f"- document_aggregation_latency_ms: {metrics.document_aggregation_latency_ms:.3f}",
        "",
        "## Failure breakdown",
        "",
    ]
    if not metrics.failure_breakdown:
        lines.append("- None")
    else:
        for category, count in sorted(metrics.failure_breakdown.items()):
            lines.append(f"- {category}: {count}")
    lines.extend(
        [
            "",
            "## Comparison guidance",
            "",
            "- Chunk recall and document recall are reported side by side.",
            "- No winner is declared automatically.",
            "- Thresholds and limits are reported so runs remain reproducible.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _document_ids_from_chunks(chunks: list[RetrievedChunk]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for chunk in chunks:
        document_id = canonical_document_id(chunk)
        if document_id is None:
            continue
        normalized = normalize_document_id(document_id)
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _recall_at_k(gold_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    return len(gold_ids.intersection(retrieved_ids[:k])) / len(gold_ids)


def _precision_at_k(gold_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    if k < 1:
        return 0.0
    return len(gold_ids.intersection(retrieved_ids[:k])) / k


def _duplicate_rate(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    unique_documents = set(_document_ids_from_chunks(chunks))
    return max(0.0, (len(chunks) - len(unique_documents)) / len(chunks))


def _dedupe_document_ids(values: list) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = normalize_document_id(str(value or ""))
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _infer_corpus(results: list[DocumentBenchmarkQuestionResult]) -> str:
    corpora = {result.item.corpus for result in results}
    if len(corpora) == 1:
        return next(iter(corpora))
    return "mixed"


def _nullable_str(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None
