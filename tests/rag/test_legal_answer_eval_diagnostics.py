from __future__ import annotations

import json
from pathlib import Path

from app.rag.eval.legal_answer_eval import (
    aggregate_answer_metrics,
    load_gold_registry_from_dataset,
    run_answer_eval,
    write_answer_eval_outputs,
)
from scripts.generate_legal_answer_eval_diagnostics import main
from tests.rag.test_legal_answer_eval import _hit, _item


def _write_dataset(path: Path, *, source_pending: bool, ecli: str | None) -> None:
    item = _item(source_pending=source_pending, ecli=ecli)
    payload = {
        "id": item.id,
        "corpus": item.corpus,
        "question": item.question,
        "expected_answer_points": item.expected_answer_points,
        "expected_source_constraints": {
            "court": None,
            "source": None,
            "case_reference": None,
            "source_document_id": ecli,
            "decision_date": None,
        },
        "expected_keywords": item.expected_keywords,
        "forbidden_answer_patterns": [],
        "difficulty": item.difficulty,
        "legal_topic": item.legal_topic,
        "evaluation_type": item.evaluation_type,
        "source_pending": source_pending,
        "expected_target_corpus": None,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def test_generate_diagnostics_script_writes_stable_artifacts(tmp_path: Path) -> None:
    runs_dir = tmp_path / "answer_eval"
    run_dir = runs_dir / "usoud_no_llm_baseline"
    run_dir.mkdir(parents=True)

    dataset_path = tmp_path / "dataset.jsonl"
    retrieval_path = tmp_path / "retrieval.jsonl"
    review_path = tmp_path / "gold.md"
    _write_dataset(dataset_path, source_pending=False, ecli="ECLI:CZ:US:2026:1.US.1.1")
    retrieval_path.write_text(
        json.dumps(
            {
                "id": "usoud-qa-test-001",
                "hits": [
                    _hit(
                        rank=1,
                        chunk_id="9",
                        text="nesouvislý text bez klíčových slov",
                        document_id="ECLI:CZ:US:2026:9.US.9.9.9",
                    )
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    review_path.write_text("# Gold Source Review\n", encoding="utf-8")

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
    write_answer_eval_outputs(
        output_dir=run_dir,
        dataset_path=dataset_path,
        retrieval_results_path=retrieval_path,
        gold_review_path=review_path,
        items=items,
        registry=registry,
        retrieval_by_id=retrieval_by_id,
        results=results,
        metrics=metrics,
        no_llm=True,
        citation_required=True,
        corpus="usoud",
    )

    output_dir = tmp_path / "evaluation_quality"
    assert main(
        [
            "--runs-dir",
            str(runs_dir),
            "--output-dir",
            str(output_dir),
        ]
    ) == 0
    assert (output_dir / "failed_cases_report.json").exists()
    assert (output_dir / "failed_cases_report.md").exists()
    assert (output_dir / "metrics_summary.json").exists()
    assert (output_dir / "metric_failure_categories.json").exists()
    category_payload = json.loads(
        (output_dir / "metric_failure_categories.json").read_text(encoding="utf-8")
    )
    assert (
        category_payload["aggregate_failure_category_counts"]["true_retrieval_miss"]
        == 1
    )
    failed_cases_payload = json.loads(
        (output_dir / "failed_cases_report.json").read_text(encoding="utf-8")
    )
    assert failed_cases_payload["final_status"] == "FAIL"


def test_diagnostics_script_does_not_target_production_qdrant() -> None:
    source = Path("scripts/generate_legal_answer_eval_diagnostics.py").read_text(encoding="utf-8")
    assert "nalus_live" not in source
    assert "nalus_stable_20260326" not in source
    assert "alias" not in source.lower()
