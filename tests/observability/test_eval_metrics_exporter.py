from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.observability.eval_metrics_exporter import (
    discover_run_summaries,
    infer_corpus_from_run_name,
    load_run_summary,
    refresh_prometheus_gauges,
    render_metrics,
)
from app.rag.eval.legal_answer_eval import (
    AnswerEvalMetrics,
    aggregate_answer_metrics,
    build_summary_json_payload,
    evaluate_answer_item,
    load_gold_registry_from_dataset,
    write_answer_eval_outputs,
)
from tests.rag.test_legal_answer_eval import _hit, _item


def _sample_summary(run_name: str, corpus: str) -> dict:
    return {
        "generated_at": "2026-07-09T20:00:00Z",
        "run_name": run_name,
        "corpus": corpus,
        "gold": 5,
        "gold_question_count": 5,
        "direct_support_count": 1,
        "partial_support_count": 4,
        "gap_count": 0,
        "boilerplate_noise_count": 0,
        "corpus_only_count": 0,
        "citation_available_count": 5,
        "unsupported_answer_risk_count": 0,
        "unsupported_risk_rate_gold": 0.0,
        "gold_retrieval_miss_count": 0,
        "gold_retrieval_miss_rate": 0.0,
        "corpus_routing_support_rate": 0.0,
        "strict_direct_pass_rate_all": 0.05,
        "strict_direct_pass_rate_gold": 0.2,
        "usable_support_rate_gold": 1.0,
        "citation_available_rate_gold": 1.0,
        "citation_available_rate": 1.0,
    }


def _write_run_dir(base: Path, run_name: str, summary: dict) -> Path:
    run_dir = base / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir


def test_build_summary_json_payload_fields() -> None:
    metrics = AnswerEvalMetrics(
        total_question_count=20,
        gold_question_count=5,
        missing_gold_count=15,
        evaluable_question_count=5,
        not_evaluable_missing_gold_count=15,
        direct_support_count=1,
        partial_support_count=4,
        gap_count=0,
        boilerplate_noise_count=0,
        corpus_only_count=0,
        citation_available_count=5,
        citation_available_rate_gold=1.0,
        corpus_routing_support_rate=0.0,
        strict_direct_pass_rate_all=0.05,
        strict_direct_pass_rate_gold=0.2,
        usable_support_rate_gold=1.0,
        unsupported_risk_rate_gold=0.0,
        gold_retrieval_miss_count=0,
        gold_retrieval_miss_rate=0.0,
        answer_eval_pass_rate=0.05,
        answer_eval_partial_rate=0.2,
        answer_eval_gap_rate=0.0,
        unsupported_answer_risk_count=0,
        skipped_count=15,
        needs_review_count=0,
    )
    payload = build_summary_json_payload(
        run_name="usoud_no_llm_baseline",
        metrics=metrics,
        generated_at="2026-07-09T20:00:00Z",
    )
    assert payload["run_name"] == "usoud_no_llm_baseline"
    assert payload["corpus"] == "usoud"
    assert payload["gold"] == 5
    assert payload["usable_support_rate_gold"] == 1.0
    assert payload["unsupported_answer_risk_count"] == 0
    assert payload["citation_available_rate_gold"] == 1.0


def test_write_answer_eval_outputs_writes_summary_json(tmp_path: Path) -> None:
    item = _item(source_pending=False, ecli="ECLI:CZ:US:2026:1.US.1.1")
    registry = load_gold_registry_from_dataset([item])
    result = evaluate_answer_item(
        item=item,
        gold=registry[item.id],
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
    metrics = aggregate_answer_metrics([result])
    output_dir = tmp_path / "usoud_no_llm_baseline"
    write_answer_eval_outputs(
        output_dir=output_dir,
        dataset_path=tmp_path / "dataset.jsonl",
        retrieval_results_path=tmp_path / "retrieval.jsonl",
        gold_review_path=tmp_path / "review.md",
        items=[item],
        registry=registry,
        retrieval_by_id={
            item.id: {
                "hits": [
                    _hit(
                        rank=1,
                        chunk_id="1",
                        text="Ústavní soud posuzuje právo na spravedlivý proces podle článku 36 Listiny.",
                        document_id="ECLI:CZ:US:2026:1.US.1.1",
                    )
                ]
            }
        },
        results=[result],
        metrics=metrics,
        no_llm=True,
        citation_required=True,
        corpus="usoud",
    )
    summary_path = output_dir / "summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["run_name"] == "usoud_no_llm_baseline"
    assert "strict_direct_pass_rate_gold" in payload
    assert "direct_support_count" in payload


def test_exporter_handles_missing_artifacts_safely(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    assert discover_run_summaries(missing) == []
    rendered = render_metrics(missing)
    assert b"legal_answer_eval_gold" in rendered


def test_exporter_exposes_prometheus_text_format(tmp_path: Path) -> None:
    _write_run_dir(tmp_path, "usoud_no_llm_baseline", _sample_summary("usoud_no_llm_baseline", "usoud"))
    rendered = render_metrics(tmp_path)
    text = rendered.decode("utf-8")
    assert "# HELP legal_answer_eval_usable_support_rate_gold" in text
    assert "# TYPE legal_answer_eval_usable_support_rate_gold gauge" in text
    assert 'run_name="usoud_no_llm_baseline"' in text


def test_exporter_includes_usoud_nsoud_mixed_metrics(tmp_path: Path) -> None:
    _write_run_dir(
        tmp_path,
        "usoud_no_llm_baseline",
        _sample_summary("usoud_no_llm_baseline", "usoud"),
    )
    _write_run_dir(
        tmp_path,
        "nsoud_no_llm_baseline",
        {
            **_sample_summary("nsoud_no_llm_baseline", "nsoud"),
            "usable_support_rate_gold": 0.667,
            "unsupported_answer_risk_count": 1,
        },
    )
    _write_run_dir(
        tmp_path,
        "mixed_no_llm_baseline",
        {
            **_sample_summary("mixed_no_llm_baseline", "mixed"),
            "gold": 2,
            "direct_support_count": 0,
            "partial_support_count": 0,
            "corpus_only_count": 2,
            "usable_support_rate_gold": 1.0,
            "citation_available_rate": 0.0,
        },
    )
    count = refresh_prometheus_gauges(tmp_path)
    assert count == 3
    rendered = render_metrics(tmp_path).decode("utf-8")
    assert 'corpus="usoud"' in rendered
    assert 'corpus="nsoud"' in rendered
    assert 'corpus="mixed"' in rendered
    assert "legal_answer_eval_unsupported_answer_risk_count" in rendered


def test_exporter_does_not_include_question_ids_as_labels(tmp_path: Path) -> None:
    _write_run_dir(tmp_path, "usoud_no_llm_baseline", _sample_summary("usoud_no_llm_baseline", "usoud"))
    rendered = render_metrics(tmp_path).decode("utf-8")
    assert "question_id" not in rendered
    assert "usoud-qa-" not in rendered


def test_load_run_summary_falls_back_to_metrics_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "nsoud_no_llm_baseline"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "gold_available_count": 3,
                "direct_support_count": 0,
                "partial_support_count": 2,
                "gap_count": 0,
                "boilerplate_noise_count": 1,
                "corpus_only_count": 0,
                "unsupported_answer_risk_count": 1,
                "strict_direct_pass_rate_all": 0.0,
                "strict_direct_pass_rate_gold": 0.0,
                "usable_support_rate_gold": 0.667,
                "citation_available_rate": 0.667,
            }
        ),
        encoding="utf-8",
    )
    summary = load_run_summary(run_dir)
    assert summary is not None
    assert summary["run_name"] == "nsoud_no_llm_baseline"
    assert summary["corpus"] == "nsoud"
    assert summary["usable_support_rate_gold"] == 0.667


@pytest.mark.parametrize(
    ("run_name", "expected"),
    [
        ("usoud_no_llm_baseline", "usoud"),
        ("nsoud_full_baseline", "nsoud"),
        ("mixed_two_pass_baseline", "mixed"),
    ],
)
def test_infer_corpus_from_run_name(run_name: str, expected: str) -> None:
    assert infer_corpus_from_run_name(run_name) == expected
