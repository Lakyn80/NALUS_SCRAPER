from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.rag.eval.document_retrieval_benchmark import (
    FAILURE_RELEVANT_DOCUMENT_NEVER_RETRIEVED,
    FAILURE_RELEVANT_DOCUMENT_REMOVED_BY_RETURNED_LIMIT,
    FAILURE_RELEVANT_DOCUMENT_REMOVED_BY_THRESHOLD,
    DocumentBenchmarkItem,
    aggregate_document_benchmark_metrics,
    load_document_benchmark_dataset,
    run_document_retrieval_benchmark,
    validate_document_benchmark_item,
    write_document_benchmark_outputs,
)
from app.rag.retrieval.document_retrieval import DocumentRetrievalConfig
from app.rag.retrieval.errors import RetrievalConfigurationError
from app.rag.retrieval.models import RetrievedChunk


def _item(
    *,
    item_id: str = "doc-qa-001",
    relevant_document_ids: list[str] | None = None,
) -> DocumentBenchmarkItem:
    return DocumentBenchmarkItem(
        id=item_id,
        corpus="usoud",
        question=f"Question {item_id}",
        relevant_document_ids=(
            ["DOC-A", "DOC-C"] if relevant_document_ids is None else relevant_document_ids
        ),
        legal_topic="topic",
        difficulty="medium",
    )


def _chunk(chunk_id: str, *, document_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        text=f"text {chunk_id}",
        score=score,
        source="hybrid",
        metadata={"document_id": document_id, "case_reference": document_id},
    )


def _config(**overrides) -> DocumentRetrievalConfig:
    data = {
        "enabled": True,
        "max_candidate_chunks": 5,
        "max_returned_documents": 5,
        "max_supporting_chunks_per_document": 2,
        "document_relevance_threshold": 0.0,
    }
    data.update(overrides)
    return DocumentRetrievalConfig(**data)


def test_dataset_loader_supports_multiple_and_duplicate_gold_documents(tmp_path: Path) -> None:
    path = tmp_path / "document_dataset.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "doc-qa-001",
                "corpus": "usoud",
                "question": "Which decisions are relevant?",
                "relevant_document_ids": ["DOC-A", "doc-a", "DOC-B"],
                "legal_topic": "topic",
                "difficulty": "medium",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    items = load_document_benchmark_dataset(path)

    assert items[0].relevant_document_ids == ["doc-a", "doc-b"]


def test_invalid_dataset_item_fails_clearly() -> None:
    with pytest.raises(RetrievalConfigurationError, match="missing required field"):
        validate_document_benchmark_item({"id": "x"})


def test_run_executes_chunk_and_document_candidate_paths_independently() -> None:
    calls: list[int] = []

    def search_fn(query: str, top_k: int) -> list[RetrievedChunk]:
        del query
        calls.append(top_k)
        return [
            _chunk("a-1", document_id="DOC-A", score=0.9),
            _chunk("b-1", document_id="DOC-B", score=0.8),
            _chunk("c-1", document_id="DOC-C", score=0.7),
        ][:top_k]

    results = run_document_retrieval_benchmark(
        items=[_item()],
        search_fn=search_fn,
        chunk_top_k=2,
        document_config=_config(max_candidate_chunks=3),
    )

    assert calls == [2, 3]
    assert results[0].metrics.chunk_recall_at_10 == 0.5
    assert results[0].metrics.candidate_recall == 1.0
    assert results[0].metrics.final_recall == 1.0


def test_aggregate_metrics_cover_candidate_final_precision_and_duplicates() -> None:
    def search_fn(query: str, top_k: int) -> list[RetrievedChunk]:
        del query, top_k
        return [
            _chunk("a-1", document_id="DOC-A", score=0.9),
            _chunk("a-2", document_id="DOC-A", score=0.8),
            _chunk("b-1", document_id="DOC-B", score=0.7),
        ]

    results = run_document_retrieval_benchmark(
        items=[_item(relevant_document_ids=["DOC-A", "DOC-B"])],
        search_fn=search_fn,
        chunk_top_k=3,
        document_config=_config(),
    )
    metrics = aggregate_document_benchmark_metrics(results)

    assert metrics.document_recall_at_10 == 1.0
    assert metrics.precision_at_10 == 0.2
    assert metrics.candidate_pool_coverage == 1.0
    assert metrics.unique_document_coverage == 1.0
    assert metrics.duplicate_rate == pytest.approx(1 / 3)


def test_zero_relevant_documents_are_supported() -> None:
    def search_fn(query: str, top_k: int) -> list[RetrievedChunk]:
        del query, top_k
        return [_chunk("a-1", document_id="DOC-A", score=0.9)]

    results = run_document_retrieval_benchmark(
        items=[_item(relevant_document_ids=[])],
        search_fn=search_fn,
        chunk_top_k=1,
        document_config=_config(),
    )
    metrics = aggregate_document_benchmark_metrics(results)

    assert metrics.zero_relevant_question_count == 1
    assert metrics.document_recall_at_10 == 0.0


def test_failure_category_relevant_document_never_retrieved() -> None:
    def search_fn(query: str, top_k: int) -> list[RetrievedChunk]:
        del query, top_k
        return [_chunk("x-1", document_id="DOC-X", score=0.9)]

    result = run_document_retrieval_benchmark(
        items=[_item(relevant_document_ids=["DOC-A"])],
        search_fn=search_fn,
        chunk_top_k=1,
        document_config=_config(),
    )[0]

    assert result.metrics.failure_category == FAILURE_RELEVANT_DOCUMENT_NEVER_RETRIEVED


def test_failure_category_removed_by_threshold() -> None:
    def search_fn(query: str, top_k: int) -> list[RetrievedChunk]:
        del query, top_k
        return [_chunk("a-1", document_id="DOC-A", score=0.2)]

    result = run_document_retrieval_benchmark(
        items=[_item(relevant_document_ids=["DOC-A"])],
        search_fn=search_fn,
        chunk_top_k=1,
        document_config=_config(document_relevance_threshold=0.8),
    )[0]

    assert result.metrics.candidate_recall == 1.0
    assert result.metrics.final_recall == 0.0
    assert result.metrics.failure_category == FAILURE_RELEVANT_DOCUMENT_REMOVED_BY_THRESHOLD


def test_failure_category_removed_by_returned_document_limit() -> None:
    def search_fn(query: str, top_k: int) -> list[RetrievedChunk]:
        del query, top_k
        return [
            _chunk("x-1", document_id="DOC-X", score=0.9),
            _chunk("a-1", document_id="DOC-A", score=0.8),
        ]

    result = run_document_retrieval_benchmark(
        items=[_item(relevant_document_ids=["DOC-A"])],
        search_fn=search_fn,
        chunk_top_k=2,
        document_config=_config(max_returned_documents=1),
    )[0]

    assert result.metrics.failure_category == FAILURE_RELEVANT_DOCUMENT_REMOVED_BY_RETURNED_LIMIT


def test_report_generation_writes_json_csv_and_markdown_without_raw_questions(tmp_path: Path) -> None:
    def search_fn(query: str, top_k: int) -> list[RetrievedChunk]:
        del query, top_k
        return [_chunk("a-1", document_id="DOC-A", score=0.9)]

    item = _item(relevant_document_ids=["DOC-A"])
    results = run_document_retrieval_benchmark(
        items=[item],
        search_fn=search_fn,
        chunk_top_k=1,
        document_config=_config(),
    )
    metrics = aggregate_document_benchmark_metrics(results)
    output_dir = tmp_path / "usoud_document_benchmark"
    write_document_benchmark_outputs(
        output_dir=output_dir,
        dataset_path=tmp_path / "dataset.jsonl",
        collection_name="test_collection",
        chunk_top_k=1,
        document_config=_config(),
        results=results,
        metrics=metrics,
    )

    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "per_question.jsonl").exists()
    assert (output_dir / "per_question.csv").exists()
    assert (output_dir / "summary.md").exists()
    assert item.question not in (output_dir / "per_question.jsonl").read_text(encoding="utf-8")
