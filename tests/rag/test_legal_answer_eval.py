from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.rag.eval.legal_answer_eval import (
    aggregate_answer_metrics,
    build_failed_case_report_entries,
    build_metric_failure_categories,
    build_nsoud_qa_007_diagnostic,
    build_answer_skeleton,
    determine_final_status,
    evaluate_answer_item,
    gold_source_hit_at_k,
    hit_matches_gold_ecli,
    is_boilerplate_snippet,
    load_gold_registry_from_dataset,
    load_retrieval_results,
    map_support_to_status,
    normalize_answer_text,
    run_answer_eval,
    validate_gold_review_path,
    write_answer_eval_outputs,
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


def test_normalize_answer_text_handles_whitespace_and_currency() -> None:
    assert normalize_answer_text("35 000 RUB") == "35000 rub"
    assert normalize_answer_text("35000 руб.") == "35000 rub"


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
    assert (output / "failed_cases_report.json").exists()
    assert (output / "metrics_summary.json").exists()
    assert (output / "metric_failure_categories.json").exists()


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
    assert metrics.gold_question_count == 1
    assert metrics.missing_gold_count == 1
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


def test_failed_case_classification_marks_retrieval_miss() -> None:
    item = _item(source_pending=False, ecli="ECLI:CZ:US:2026:1.US.1.1")
    registry = load_gold_registry_from_dataset([item])
    retrieval_by_id = {
        item.id: {
            "hits": [
                _hit(
                    rank=1,
                    chunk_id="9",
                    text="nesouvislý text bez klíčových slov",
                    document_id="ECLI:CZ:US:2026:9.US.9.9.9",
                )
            ]
        }
    }
    results = run_answer_eval(
        items=[item],
        registry=registry,
        retrieval_by_id=retrieval_by_id,
        citation_required=True,
    )
    failed_cases = build_failed_case_report_entries(
        run_name="usoud_no_llm_baseline",
        generated_at="2026-07-10T12:00:00Z",
        dataset_path=Path("dataset.jsonl"),
        corpus="usoud",
        items=[item],
        registry=registry,
        retrieval_by_id=retrieval_by_id,
        results=results,
    )
    assert len(failed_cases) == 1
    assert failed_cases[0].failure_category == "true_retrieval_miss"
    assert failed_cases[0].is_real_failure is True
    counts = build_metric_failure_categories(
        failed_cases=failed_cases,
        metrics=aggregate_answer_metrics(results),
    )
    assert counts["true_retrieval_miss"] == 1
    assert counts["metric_denominator_warning"] == 1


def test_write_answer_eval_outputs_writes_failed_case_diagnostics(tmp_path: Path) -> None:
    item = _item(source_pending=False, ecli="ECLI:CZ:US:2026:1.US.1.1")
    items = [item]
    registry = load_gold_registry_from_dataset(items)
    retrieval_by_id = {
        item.id: {
            "hits": [
                _hit(
                    rank=1,
                    chunk_id="9",
                    text="nesouvislý text bez klíčových slov",
                    document_id="ECLI:CZ:US:2026:9.US.9.9.9",
                )
            ]
        }
    }
    results = run_answer_eval(
        items=items,
        registry=registry,
        retrieval_by_id=retrieval_by_id,
        citation_required=True,
    )
    metrics = aggregate_answer_metrics(results)
    output_dir = tmp_path / "usoud_no_llm_baseline"
    write_answer_eval_outputs(
        output_dir=output_dir,
        dataset_path=tmp_path / "dataset.jsonl",
        retrieval_results_path=tmp_path / "retrieval.jsonl",
        gold_review_path=tmp_path / "review.md",
        items=items,
        registry=registry,
        retrieval_by_id=retrieval_by_id,
        results=results,
        metrics=metrics,
        no_llm=True,
        citation_required=True,
        corpus="usoud",
    )
    failed_cases_payload = json.loads(
        (output_dir / "failed_cases_report.json").read_text(encoding="utf-8")
    )
    assert failed_cases_payload["failed_case_count"] == 1
    metrics_summary = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    assert metrics_summary["gold_count"] == 1
    assert metrics_summary["denominator_warning"] is not None
    category_payload = json.loads(
        (output_dir / "metric_failure_categories.json").read_text(encoding="utf-8")
    )
    assert category_payload["failure_category_counts"]["true_retrieval_miss"] == 1


def test_missing_gold_is_not_counted_as_retrieval_failure() -> None:
    item = _item(source_pending=True, ecli=None)
    registry = load_gold_registry_from_dataset([item])
    results = run_answer_eval(
        items=[item],
        registry=registry,
        retrieval_by_id={item.id: {"hits": []}},
        citation_required=True,
    )
    failed_cases = build_failed_case_report_entries(
        run_name="nsoud_no_llm_baseline",
        generated_at="2026-07-10T12:00:00Z",
        dataset_path=Path("dataset.jsonl"),
        corpus="usoud",
        items=[item],
        registry=registry,
        retrieval_by_id={item.id: {"hits": []}},
        results=results,
    )
    assert failed_cases[0].failure_category == "not_evaluable_missing_gold"
    assert failed_cases[0].is_real_failure is False


def test_corpus_only_no_citation_is_not_failure() -> None:
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
    retrieval_by_id = {item.id: {"hits": [], "corpus_hit_at_3": True, "corpus_hit_at_5": True}}
    results = run_answer_eval(
        items=[item],
        registry=registry,
        retrieval_by_id=retrieval_by_id,
        citation_required=True,
    )
    failed_cases = build_failed_case_report_entries(
        run_name="mixed_no_llm_baseline",
        generated_at="2026-07-10T12:00:00Z",
        dataset_path=Path("dataset.jsonl"),
        corpus="mixed",
        items=[item],
        registry=registry,
        retrieval_by_id=retrieval_by_id,
        results=results,
    )
    assert failed_cases[0].failure_category == "corpus_only_no_document_citation_expected"
    assert failed_cases[0].is_real_failure is False


def test_determine_final_status_warn_for_missing_gold_only() -> None:
    status, reason = determine_final_status(
        corpus="mixed",
        failure_category_counts={
            "not_evaluable_missing_gold": 2,
            "metric_denominator_warning": 1,
        },
    )
    assert status == "WARN"
    assert "missing gold coverage" in reason


def test_determine_final_status_fail_with_real_nsoud_risk() -> None:
    status, _reason = determine_final_status(
        corpus="nsoud",
        failure_category_counts={
            "true_retrieval_miss": 1,
            "unsupported_boilerplate_or_gap": 1,
        },
    )
    assert status == "FAIL_WITH_REAL_NSOUD_RISK"


def test_nsoud_qa_007_diagnostic_classifies_true_retrieval_miss() -> None:
    item = LegalQaItem(
        id="nsoud-qa-007",
        corpus="nsoud",
        question="Jak Nejvyšší soud posuzuje dovolací důvod podle § 265b tr. ř.?",
        expected_answer_points=["bod"],
        expected_source_constraints=SourceConstraints(source_document_id="ECLI:CZ:NS:2025:5.TDO.1086.2024.1"),
        expected_keywords=["265b", "trestní", "dovolání"],
        forbidden_answer_patterns=[],
        difficulty="medium",
        legal_topic="trestní dovolání",
        evaluation_type="retrieval",
        source_pending=False,
        expected_target_corpus=None,
    )
    registry = load_gold_registry_from_dataset([item])
    retrieval_by_id = {
        item.id: {
            "keyword_coverage": 1.0,
            "hits": [
                _hit(
                    rank=1,
                    chunk_id="1",
                    text="§ 265b odst. 1 písm. g) tr. ř.",
                    document_id="ECLI:CZ:NS:2024:11.TDO.765.2024.1",
                ),
                _hit(
                    rank=2,
                    chunk_id="2",
                    text="§ 265b odst. 1 písm. h) tr. ř.",
                    document_id="ECLI:CZ:NS:2025:3.TDO.53.2025.1",
                ),
                _hit(
                    rank=3,
                    chunk_id="3",
                    text="trestní dovolání",
                    document_id="ECLI:CZ:NS:2025:6.TDO.21.2025.1",
                ),
            ],
        }
    }
    results = run_answer_eval(
        items=[item],
        registry=registry,
        retrieval_by_id=retrieval_by_id,
        citation_required=True,
    )
    diagnostic = build_nsoud_qa_007_diagnostic(
        items=[item],
        registry=registry,
        retrieval_by_id=retrieval_by_id,
        results=results,
    )
    assert diagnostic is not None
    assert diagnostic["true_retrieval_miss"] is True
    assert diagnostic["matcher_issue"] is False


def test_runner_module_has_no_nalus_legal_rag_import() -> None:
    source = Path("scripts/run_legal_answer_eval.py").read_text(encoding="utf-8")
    assert "nalus_legal_rag" not in source
    assert "nalus-legal-rag" not in source
