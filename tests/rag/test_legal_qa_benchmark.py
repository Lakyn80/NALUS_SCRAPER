from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.rag.eval.legal_qa_benchmark import (
    LegalQaItem,
    MixedRetrievedHitRecord,
    RetrievedHitRecord,
    SourceConstraints,
    aggregate_metrics,
    aggregate_mixed_metrics,
    build_hybrid_retriever,
    corpus_hit_at_k,
    evaluate_mixed_question,
    evaluate_question,
    keyword_coverage,
    load_dataset,
    merge_two_pass_hits,
    run_mixed_retrieval_benchmark,
    run_retrieval_benchmark,
    validate_dataset_item,
    validate_mixed_dataset_item,
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


def test_bm25_sidecar_path_override_sets_env_and_validates_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EMBEDDING_CACHE_ENABLED", "0")
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "nalus_us_bge_m3_rag_combined_20260709")
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(RetrievalConfigurationError, match="BM25 sidecar not found"):
        build_hybrid_retriever(
            collection_name="nalus_us_bge_m3_rag_combined_20260709",
            qdrant_url="http://qdrant:6333",
            use_redis_cache=False,
            bm25_sidecar_path=missing,
        )


def test_bm25_sidecar_path_override_wires_production_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sidecar = tmp_path / "bm25.sqlite"
    sidecar.write_text("", encoding="utf-8")
    monkeypatch.setenv("EMBEDDING_CACHE_ENABLED", "0")
    with patch("app.api.startup._build_production_retrieval") as mocked:
        retriever = MagicMock()
        retriever.search.return_value = []
        mocked.return_value = (retriever, MagicMock())
        with patch("qdrant_client.QdrantClient"):
            build_hybrid_retriever(
                collection_name="nalus_us_bge_m3_rag_combined_20260709",
                qdrant_url="http://qdrant:6333",
                use_redis_cache=False,
                bm25_sidecar_path=sidecar,
            )
    assert os.environ["BM25_SIDECAR_PATH"] == str(sidecar.resolve())
    mocked.assert_called_once()


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


def _mixed_item(**overrides) -> LegalQaItem:
    payload = {
        "id": "mixed-qa-test-001",
        "corpus": "mixed",
        "expected_target_corpus": "both",
        "question": "Jak se liší ústavní stížnost od dovolání?",
        "expected_answer_points": ["bod"],
        "expected_source_constraints": {
            "court": None,
            "source": None,
            "case_reference": None,
            "source_document_id": None,
            "decision_date": None,
        },
        "expected_keywords": ["ústavní", "dovolání"],
        "forbidden_answer_patterns": [],
        "difficulty": "medium",
        "legal_topic": "rozdíl mezi Ústavním soudem a Nejvyšším soudem",
        "evaluation_type": "retrieval",
        "source_pending": True,
    }
    payload.update(overrides)
    return LegalQaItem.from_dict(payload)


def _mixed_hit(
    *,
    rank: int,
    chunk_id: str,
    retrieved_corpus: str,
    text: str,
    combined_rrf_score: float,
    corpus_rank: int,
) -> MixedRetrievedHitRecord:
    return MixedRetrievedHitRecord(
        rank=rank,
        chunk_id=chunk_id,
        text_snippet=text,
        score=combined_rrf_score,
        source="hybrid",
        retrieved_corpus=retrieved_corpus,
        collection_name=f"{retrieved_corpus}-collection",
        source_document_id=f"doc-{chunk_id}",
        ecli=None,
        case_reference=None,
        dense_score=0.5,
        bm25_score=0.4,
        rrf_score=0.45,
        corpus_rank=corpus_rank,
        combined_rrf_score=combined_rrf_score,
        metadata={},
    )


def test_mixed_dataset_item_requires_expected_target_corpus() -> None:
    with pytest.raises(RetrievalConfigurationError, match="expected_target_corpus"):
        validate_mixed_dataset_item({"id": "mixed-qa-x", "corpus": "mixed"})


def test_expected_target_corpus_validation_rejects_unknown_value() -> None:
    with pytest.raises(RetrievalConfigurationError, match="Unsupported expected_target_corpus"):
        validate_mixed_dataset_item(
            {
                "id": "mixed-qa-x",
                "corpus": "mixed",
                "expected_target_corpus": "invalid",
            }
        )


def test_mixed_dataset_loader_reads_expected_target_corpus(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "mixed-qa-test-001",
                "corpus": "mixed",
                "expected_target_corpus": "both",
                "question": "Otázka?",
                "expected_answer_points": ["bod"],
                "expected_source_constraints": {
                    "court": None,
                    "source": None,
                    "case_reference": None,
                    "source_document_id": None,
                    "decision_date": None,
                },
                "expected_keywords": ["ústavní"],
                "forbidden_answer_patterns": [],
                "difficulty": "medium",
                "legal_topic": "téma",
                "evaluation_type": "retrieval",
                "source_pending": True,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    items = load_dataset(path)
    assert items[0].expected_target_corpus == "both"


def test_corpus_hit_at_k_for_both_expected() -> None:
    hits = [
        _mixed_hit(rank=1, chunk_id="u1", retrieved_corpus="usoud", text="a", combined_rrf_score=0.9, corpus_rank=1),
        _mixed_hit(rank=2, chunk_id="n1", retrieved_corpus="nsoud", text="b", combined_rrf_score=0.8, corpus_rank=1),
    ]
    assert corpus_hit_at_k("both", hits, 1) is False
    assert corpus_hit_at_k("both", hits, 2) is True
    assert corpus_hit_at_k("usoud", hits, 1) is True
    assert corpus_hit_at_k("nsoud", hits, 1) is False


def test_ambiguous_item_does_not_fail_corpus_metric() -> None:
    item = _mixed_item(expected_target_corpus="ambiguous")
    hits = [
        _mixed_hit(rank=1, chunk_id="u1", retrieved_corpus="usoud", text="ústavní", combined_rrf_score=0.9, corpus_rank=1),
    ]
    result = evaluate_mixed_question(item, hits)
    assert result.corpus_hit_at_1 is None
    assert result.corpus_hit_at_3 is None
    assert result.corpus_hit_at_5 is None


def test_deterministic_merge_ordering() -> None:
    usoud_chunks = [
        RetrievedChunk(id="u-1", text="usoud první", score=0.9, source="hybrid", metadata={"rrf_score": 0.9}),
    ]
    nsoud_chunks = [
        RetrievedChunk(id="n-0", text="nsoud padding", score=0.1, source="hybrid", metadata={}),
        RetrievedChunk(id="n-1", text="nsoud první", score=0.95, source="hybrid", metadata={"rrf_score": 0.95}),
    ]
    merged_once = merge_two_pass_hits(
        usoud_hits=usoud_chunks,
        nsoud_hits=nsoud_chunks,
        usoud_collection_name="usoud-collection",
        nsoud_collection_name="nsoud-collection",
        top_k=3,
    )
    merged_twice = merge_two_pass_hits(
        usoud_hits=usoud_chunks,
        nsoud_hits=nsoud_chunks,
        usoud_collection_name="usoud-collection",
        nsoud_collection_name="nsoud-collection",
        top_k=3,
    )
    assert [hit.chunk_id for hit in merged_once] == [hit.chunk_id for hit in merged_twice]
    assert merged_once[0].combined_rrf_score >= merged_once[1].combined_rrf_score
    assert merged_once[1].combined_rrf_score >= merged_once[2].combined_rrf_score
    assert {hit.retrieved_corpus for hit in merged_once} == {"usoud", "nsoud"}


def test_two_pass_fake_retriever_runner() -> None:
    item = _mixed_item()

    def search_fn(query: str, top_k: int) -> list[MixedRetrievedHitRecord]:
        del query
        return [
            _mixed_hit(
                rank=1,
                chunk_id="u-1",
                retrieved_corpus="usoud",
                text="text o ústavní stížnosti",
                combined_rrf_score=0.9,
                corpus_rank=1,
            ),
            _mixed_hit(
                rank=2,
                chunk_id="n-1",
                retrieved_corpus="nsoud",
                text="text o dovolání",
                combined_rrf_score=0.8,
                corpus_rank=1,
            ),
        ][:top_k]

    results = run_mixed_retrieval_benchmark(items=[item], search_fn=search_fn, top_k=5)
    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].corpus_hit_at_3 is True
    metrics = aggregate_mixed_metrics(results)
    assert metrics.retrieval_hit_at_1 == 1.0
    assert metrics.corpus_hit_at_3 == 1.0
    assert metrics.usoud_win_rate_at_1 == 1.0
    assert metrics.ambiguous_count == 0


def test_aggregate_mixed_metrics_excludes_ambiguous_from_corpus_hit() -> None:
    scored = evaluate_mixed_question(
        _mixed_item(id="mixed-qa-test-002", expected_target_corpus="both"),
        [
            _mixed_hit(rank=1, chunk_id="u1", retrieved_corpus="usoud", text="ústavní", combined_rrf_score=0.9, corpus_rank=1),
            _mixed_hit(rank=2, chunk_id="n1", retrieved_corpus="nsoud", text="dovolání", combined_rrf_score=0.8, corpus_rank=1),
        ],
    )
    ambiguous = evaluate_mixed_question(
        _mixed_item(id="mixed-qa-test-003", expected_target_corpus="ambiguous"),
        [
            _mixed_hit(rank=1, chunk_id="u1", retrieved_corpus="usoud", text="ústavní", combined_rrf_score=0.9, corpus_rank=1),
        ],
    )
    metrics = aggregate_mixed_metrics([scored, ambiguous])
    assert metrics.question_count == 2
    assert metrics.ambiguous_count == 1
    assert metrics.corpus_scored_question_count == 1
    assert metrics.corpus_hit_at_3 == 1.0
