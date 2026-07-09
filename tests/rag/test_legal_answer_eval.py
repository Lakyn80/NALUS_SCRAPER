from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.rag.eval.legal_answer_eval import (
    aggregate_answer_metrics,
    build_answer_skeleton,
    evaluate_answer_item,
    gold_source_hit_at_k,
    hit_matches_gold_ecli,
    is_boilerplate_snippet,
    load_gold_registry_from_dataset,
    load_retrieval_results,
    map_support_to_status,
    run_answer_eval,
    validate_gold_review_path,
)
from app.rag.eval.legal_qa_benchmark import LegalQaItem, SourceConstraints, load_dataset
from app.rag.retrieval.errors import RetrievalConfigurationError


def _item(
    *,
    item_id: str = "usoud-qa-test-001",
    corpus: str = "usoud",
    source_pending: bool = True,
    ecli: str | None = None,
) -> LegalQaItem:
    return LegalQaItem(
        id=item_id,
        corpus=corpus,
        question="Jak Ústavní soud posuzuje právo na spravedlivý proces?",
        expected_answer_points=["Právo na spravedlivý proces je ústavně chráněno."],
        expected_source_constraints=SourceConstraints(
            court=None,
            source=None,
            case_reference=None,
            source_document_id=ecli,
            decision_date=None,
        ),
        expected_keywords=["spravedlivý", "proces", "ústavní"],
        forbidden_answer_patterns=[],
        difficulty="medium",
        legal_topic="právo na spravedlivý proces",
        evaluation_type="retrieval",
        source_pending=source_pending,
        expected_target_corpus=None,
    )


def _hit(
    *,
    rank: int,
    chunk_id: str,
    text: str,
    document_id: str | None = None,
) -> dict:
    metadata = {"document_id": document_id} if document_id else {}
    return {
        "rank": rank,
        "chunk_id": chunk_id,
        "text_snippet": text,
        "score": 0.9,
        "source": "hybrid",
        "metadata": metadata,
    }


def test_loads_gold_registry_from_dataset() -> None:
    items = [
        _item(item_id="usoud-qa-001", source_pending=False, ecli="ECLI:CZ:US:2026:1.US.1.1"),
        _item(item_id="usoud-qa-002", source_pending=True),
    ]
    registry = load_gold_registry_from_dataset(items)
    assert registry["usoud-qa-001"].gold_available is True
    assert registry["usoud-qa-001"].expected_ecli == "ECLI:CZ:US:2026:1.US.1.1"
    assert registry["usoud-qa-002"].gold_available is False


def test_matches_retrieval_result_to_gold_source() -> None:
    hits = [
        _hit(
            rank=1,
            chunk_id="1",
            text="právo na spravedlivý proces a ústavní záruky",
            document_id="ECLI:CZ:US:2026:1.US.1.1",
        )
    ]
    assert hit_matches_gold_ecli(hits[0], "ECLI:CZ:US:2026:1.US.1.1")
    assert gold_source_hit_at_k(hits, "ECLI:CZ:US:2026:1.US.1.1", 1) is True


def test_direct_support_produces_pass(tmp_path: Path) -> None:
    item = _item(source_pending=False, ecli="ECLI:CZ:US:2026:1.US.1.1")
    registry = load_gold_registry_from_dataset([item])
    retrieval = {
        "hits": [
            _hit(
                rank=1,
                chunk_id="42",
                text="Ústavní soud posuzuje právo na spravedlivý proces podle článku 36 Listiny.",
                document_id="ECLI:CZ:US:2026:1.US.1.1",
            )
        ]
    }
    result = evaluate_answer_item(
        item=item,
        gold=registry[item.id],
        retrieval=retrieval,
        citation_required=True,
    )
    assert result.support_level == "direct"
    assert result.answer_eval_status == "pass"
    assert result.citation_available is True
    assert "ECLI:CZ:US:2026:1.US.1.1" in result.answer_skeleton


def test_partial_support_produces_partial() -> None:
    item = _item(source_pending=False, ecli="ECLI:CZ:US:2026:1.US.1.1")
    registry = load_gold_registry_from_dataset([item])
    retrieval = {
        "hits": [
            _hit(
                rank=2,
                chunk_id="7",
                text="Zmínka o spravedlivém procesu bez dalšího kontextu.",
                document_id="ECLI:CZ:US:2026:1.US.1.1",
            )
        ]
    }
    result = evaluate_answer_item(item=item, gold=registry[item.id], retrieval=retrieval, citation_required=False)
    assert result.support_level == "partial"
    assert result.answer_eval_status == "partial"


def test_gap_produces_insufficient_support_skeleton() -> None:
    item = _item(source_pending=False, ecli="ECLI:CZ:US:2026:1.US.1.1")
    registry = load_gold_registry_from_dataset([item])
    retrieval = {
        "hits": [
            _hit(
                rank=1,
                chunk_id="9",
                text="nesouvislý text bez klíčových slov",
                document_id="ECLI:CZ:US:2026:9.US.9.9.9",
            )
        ]
    }
    result = evaluate_answer_item(item=item, gold=registry[item.id], retrieval=retrieval, citation_required=False)
    assert result.support_level == "gap"
    assert result.answer_eval_status == "gap"
    assert "Nedostatečná podpora" in result.answer_skeleton


def test_boilerplate_noise_does_not_pass_as_answer() -> None:
    item = _item(source_pending=False, ecli="ECLI:CZ:US:2026:1.US.1.1")
    registry = load_gold_registry_from_dataset([item])
    retrieval = {
        "hits": [
            _hit(
                rank=1,
                chunk_id="1",
                text="takto: Dovolání se odmítá.",
                document_id="ECLI:CZ:US:2026:1.US.1.1",
            )
        ]
    }
    assert is_boilerplate_snippet("takto: Dovolání se odmítá.")
    result = evaluate_answer_item(item=item, gold=registry[item.id], retrieval=retrieval, citation_required=False)
    assert result.support_level == "boilerplate_noise"
    assert result.answer_eval_status == "needs_review"
    assert result.answer_eval_status != "pass"


def test_corpus_only_does_not_claim_document_citation() -> None:
    item = LegalQaItem(
        id="mixed-qa-002",
        corpus="mixed",
        question="Porovnání soudů",
        expected_answer_points=["bod"],
        expected_source_constraints=SourceConstraints(),
        expected_keywords=["ústavní", "nejvyšší"],
        forbidden_answer_patterns=[],
        difficulty="medium",
        legal_topic="mixed",
        evaluation_type="retrieval",
        source_pending=False,
        expected_target_corpus="both",
    )
    registry = load_gold_registry_from_dataset([item])
    retrieval = {"hits": [], "corpus_hit_at_3": True, "corpus_hit_at_5": True}
    result = evaluate_answer_item(item=item, gold=registry[item.id], retrieval=retrieval, citation_required=True)
    assert result.support_level == "corpus_only"
    assert result.citation_available is False
    assert "přesná citace dokumentu není k dispozici" in result.answer_skeleton


def test_citation_required_fails_when_no_citation_available() -> None:
    status = map_support_to_status("direct", citation_required=True, citation_available=False)
    assert status == "needs_review"
    gold = load_gold_registry_from_dataset(
        [_item(source_pending=False, ecli="ECLI:CZ:US:2026:1.US.1.1")]
    )["usoud-qa-test-001"]
    skeleton, available = build_answer_skeleton(
        support_level="gap",
        gold=gold,
        gold_hit=None,
        citation_required=True,
    )
    assert available is False
    assert "Nedostatečná podpora" in skeleton


def test_runner_main_no_llm(tmp_path: Path) -> None:
    dataset = tmp_path / "usoud_qa_test.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "usoud-qa-test-001",
                "corpus": "usoud",
                "question": "Otázka?",
                "expected_answer_points": ["bod"],
                "expected_source_constraints": {
                    "court": None,
                    "source": None,
                    "case_reference": None,
                    "source_document_id": "ECLI:CZ:US:2026:1.US.1.1",
                    "decision_date": None,
                },
                "expected_keywords": ["spravedlivý"],
                "forbidden_answer_patterns": [],
                "difficulty": "medium",
                "legal_topic": "téma",
                "evaluation_type": "retrieval",
                "source_pending": False,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    retrieval = tmp_path / "retrieval.jsonl"
    retrieval.write_text(
        json.dumps(
            {
                "id": "usoud-qa-test-001",
                "hits": [
                    _hit(
                        rank=1,
                        chunk_id="1",
                        text="právo na spravedlivý proces",
                        document_id="ECLI:CZ:US:2026:1.US.1.1",
                    )
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    review = tmp_path / "gold.md"
    review.write_text("# Gold Source Review\n", encoding="utf-8")
    output = tmp_path / "out"

    from scripts.run_legal_answer_eval import main

    assert main(
        [
            "--dataset",
            str(dataset),
            "--retrieval-results",
            str(retrieval),
            "--gold-review",
            str(review),
            "--output-dir",
            str(output),
            "--no-llm",
            "--require-citations",
        ]
    ) == 0
    assert (output / "answer_eval_results.jsonl").exists()


def test_validate_gold_review_path(tmp_path: Path) -> None:
    path = tmp_path / "review.md"
    path.write_text("# Gold Source Review\n", encoding="utf-8")
    validate_gold_review_path(path)
    with pytest.raises(RetrievalConfigurationError):
        validate_gold_review_path(tmp_path / "missing.md")


def test_load_retrieval_results_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text(
        json.dumps({"id": "q1", "hits": []}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    loaded = load_retrieval_results(path)
    assert "q1" in loaded


def test_aggregate_answer_metrics_counts() -> None:
    item = _item(source_pending=False, ecli="ECLI:CZ:US:2026:1.US.1.1")
    registry = load_gold_registry_from_dataset([item, _item(item_id="usoud-qa-002")])
    results = run_answer_eval(
        items=[item, _item(item_id="usoud-qa-002")],
        registry=registry,
        retrieval_by_id={
            item.id: {
                "hits": [
                    _hit(
                        rank=1,
                        chunk_id="1",
                        text="právo na spravedlivý proces a ústavní záruky",
                        document_id="ECLI:CZ:US:2026:1.US.1.1",
                    )
                ]
            },
            "usoud-qa-002": {"hits": []},
        },
        citation_required=True,
    )
    metrics = aggregate_answer_metrics(results)
    assert metrics.gold_available_count == 1
    assert metrics.skipped_count == 1
    assert metrics.direct_support_count == 1
    assert metrics.strict_direct_pass_rate_all == 0.5
    assert metrics.strict_direct_pass_rate_gold == 1.0
    assert metrics.usable_support_rate_gold == 1.0


def test_new_rate_metrics_on_mixed_support_levels() -> None:
    direct = evaluate_answer_item(
        item=_item(source_pending=False, ecli="ECLI:CZ:US:2026:1.US.1.1"),
        gold=load_gold_registry_from_dataset(
            [_item(source_pending=False, ecli="ECLI:CZ:US:2026:1.US.1.1")]
        )["usoud-qa-test-001"],
        retrieval={
            "hits": [
                _hit(
                    rank=1,
                    chunk_id="1",
                    text="Ústavní soud posuzuje právo na spravedlivý proces podle článku 36 Listiny.",
                    document_id="ECLI:CZ:US:2026:1.US.1.1",
                )
            ]
        },
        citation_required=True,
    )
    partial = evaluate_answer_item(
        item=_item(item_id="usoud-qa-002", source_pending=False, ecli="ECLI:CZ:US:2026:2.US.2.2"),
        gold=load_gold_registry_from_dataset(
            [_item(item_id="usoud-qa-002", source_pending=False, ecli="ECLI:CZ:US:2026:2.US.2.2")]
        )["usoud-qa-002"],
        retrieval={
            "hits": [
                _hit(
                    rank=2,
                    chunk_id="2",
                    text="Zmínka o spravedlivém procesu bez dalšího kontextu pro ústavní posouzení.",
                    document_id="ECLI:CZ:US:2026:2.US.2.2",
                )
            ]
        },
        citation_required=True,
    )
    metrics = aggregate_answer_metrics([direct, partial])
    assert metrics.direct_support_count == 1
    assert metrics.partial_support_count == 1
    assert metrics.strict_direct_pass_rate_gold == 0.5
    assert metrics.usable_support_rate_gold == 1.0


def test_runner_module_has_no_nalus_legal_rag_import() -> None:
    source = Path("scripts/run_legal_answer_eval.py").read_text(encoding="utf-8")
    assert "nalus_legal_rag" not in source
    assert "nalus-legal-rag" not in source
