from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.rag.eval.legal_qa_benchmark import (
    LegalQaItem,
    RetrievedHitRecord,
    SourceConstraints,
    aggregate_metrics,
    build_hybrid_retriever,
    evaluate_question,
    keyword_coverage,
    load_dataset,
    run_retrieval_benchmark,
    validate_dataset_item,
)
from app.rag.retrieval.errors import RetrievalConfigurationError
from app.rag.retrieval.models import RetrievedChunk


def _sample_item(**overrides) -> LegalQaItem:
    payload = {
        "id": "usoud-qa-test-001",
        "corpus": "usoud",
        "question": "Jak Ústavní soud posuzuje právo na spravedlivý proces?",
        "expected_answer_points": ["spravedlivý proces"],
        "expected_source_constraints": {
            "court": None,
            "source": None,
            "case_reference": None,
            "source_document_id": None,
            "decision_date": None,
        },
        "expected_keywords": ["spravedlivý", "proces"],
        "forbidden_answer_patterns": [],
        "difficulty": "easy",
        "legal_topic": "právo na spravedlivý proces",
        "evaluation_type": "retrieval",
        "source_pending": True,
    }
    payload.update(overrides)
    return LegalQaItem.from_dict(payload)


def _hit(text: str, *, chunk_id: str = "1") -> RetrievedHitRecord:
    return RetrievedHitRecord(
        rank=1,
        chunk_id=chunk_id,
        text_snippet=text,
        score=0.9,
        source="hybrid",
        dense_score=0.8,
        bm25_score=0.7,
        rrf_score=0.85,
        metadata={"court": "Ústavní soud"},
    )


def test_dataset_loader_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "usoud-qa-test-001",
                "corpus": "usoud",
                "question": "Otázka o spravedlivém procesu?",
                "expected_answer_points": ["bod"],
                "expected_source_constraints": {
                    "court": None,
                    "source": None,
                    "case_reference": None,
                    "source_document_id": None,
                    "decision_date": None,
                },
                "expected_keywords": ["spravedlivý"],
                "forbidden_answer_patterns": [],
                "difficulty": "easy",
                "legal_topic": "právo na spravedlivý proces",
                "evaluation_type": "retrieval",
                "source_pending": True,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    items = load_dataset(path)
    assert len(items) == 1
    assert items[0].source_pending is True


def test_invalid_item_fails_clearly() -> None:
    with pytest.raises(RetrievalConfigurationError, match="missing required fields"):
        validate_dataset_item({"id": "x"})


def test_source_pending_allowed() -> None:
    item = _sample_item(source_pending=True)
    assert item.source_pending is True


def test_hit_at_k_calculation() -> None:
    item = _sample_item()
    hits = [
        _hit("nesouvislý text", chunk_id="1"),
        _hit("právo na spravedlivý proces", chunk_id="2"),
        _hit("další text", chunk_id="3"),
        _hit("procesní záruky", chunk_id="4"),
    ]
    result = evaluate_question(item, hits)
    assert result.hit_at_1 is False
    assert result.hit_at_3 is True
    assert result.hit_at_5 is True
    assert result.hit_at_10 is True


def test_keyword_coverage() -> None:
    hits = [_hit("právo na spravedlivý proces a procesní záruky")]
    assert keyword_coverage(["spravedlivý", "proces", "dovolání"], hits) == pytest.approx(2 / 3)


def test_fake_retriever_runner() -> None:
    item = _sample_item()

    def search_fn(query: str, top_k: int) -> list[RetrievedChunk]:
        del query
        return [
            RetrievedChunk(
                id="chunk-1",
                text="text o spravedlivém procesu",
                score=0.9,
                source="hybrid",
                metadata={"score_components": {"dense": 0.8, "bm25": 0.7}, "rrf_score": 0.85},
            )
        ][:top_k]

    results = run_retrieval_benchmark(items=[item], search_fn=search_fn, top_k=5)
    assert len(results) == 1
    assert results[0].passed is True
    metrics = aggregate_metrics(results)
    assert metrics.hit_at_1 == 1.0


def test_redis_flag_does_not_connect_to_real_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_CACHE_ENABLED", "0")
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "nalus_us_bge_m3_rag_combined_20260709")
    with patch("app.api.startup._build_production_retrieval") as mocked:
        retriever = MagicMock()
        retriever.search.return_value = []
        mocked.return_value = (retriever, MagicMock())
        with patch("qdrant_client.QdrantClient"):
            search_fn = build_hybrid_retriever(
                collection_name="nalus_us_bge_m3_rag_combined_20260709",
                qdrant_url="http://qdrant:6333",
                use_redis_cache=False,
            )
    search_fn("test", 1)
    mocked.assert_called_once()


def test_source_constraint_match_when_not_pending() -> None:
    item = _sample_item(
        source_pending=False,
        expected_source_constraints={
            "court": "Ústavní soud",
            "source": None,
            "case_reference": None,
            "source_document_id": None,
            "decision_date": None,
        },
        expected_keywords=["spravedlivý"],
    )
    hits = [_hit("spravedlivý proces", chunk_id="1")]
    hits[0].metadata["court"] = "Ústavní soud"
    result = evaluate_question(item, hits)
    assert result.source_constraint_match == 1.0
    assert result.passed is True
