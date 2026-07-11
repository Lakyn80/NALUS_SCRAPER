"""Generate stable failed-case diagnostics from offline legal answer-eval runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.eval.legal_answer_eval import (  # noqa: E402
    aggregate_answer_metrics,
    build_nsoud_qa_007_diagnostic,
    build_failed_case_report_entries,
    build_metric_failure_categories,
    build_metrics_summary_payload,
    determine_final_status,
    infer_corpus_from_run_name,
    load_gold_registry_from_dataset,
    load_retrieval_results,
    run_answer_eval,
    validate_gold_review_path,
)
from app.rag.eval.legal_qa_benchmark import load_dataset  # noqa: E402
from app.rag.retrieval.errors import RetrievalConfigurationError  # noqa: E402

DEFAULT_RUNS_DIR = PROJECT_ROOT / "artifacts/rag_eval/legal_qa/answer_eval"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/evaluation_quality"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stable failed-case diagnostics from offline legal answer-eval runs.",
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--run-name",
        action="append",
        default=[],
        help="Optional run directory name to include. Repeatable.",
    )
    return parser.parse_args(argv)


def _load_run_context(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise RetrievalConfigurationError(f"Missing metrics.json in answer-eval run: {run_dir}")

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RetrievalConfigurationError(f"Invalid metrics.json payload in {run_dir}")

    dataset_path = Path(str(payload.get("dataset") or "")).resolve()
    retrieval_results_path = Path(str(payload.get("retrieval_results") or "")).resolve()
    gold_review_path = Path(str(payload.get("gold_review") or "")).resolve()
    corpus = str(payload.get("corpus") or infer_corpus_from_run_name(run_dir.name))
    citation_required = bool(payload.get("citation_required"))

    if not dataset_path.exists():
        raise RetrievalConfigurationError(f"Dataset not found for run {run_dir.name}: {dataset_path}")
    if not retrieval_results_path.exists():
        raise RetrievalConfigurationError(
            f"Retrieval results not found for run {run_dir.name}: {retrieval_results_path}"
        )
    validate_gold_review_path(gold_review_path)

    return {
        "run_name": run_dir.name,
        "corpus": corpus,
        "dataset_path": dataset_path,
        "retrieval_results_path": retrieval_results_path,
        "gold_review_path": gold_review_path,
        "citation_required": citation_required,
    }


def _resolve_run_dirs(runs_dir: Path, selected_run_names: list[str]) -> list[Path]:
    if not runs_dir.exists():
        raise RetrievalConfigurationError(f"Runs directory not found: {runs_dir}")

    all_run_dirs = sorted(child for child in runs_dir.iterdir() if child.is_dir())
    if not selected_run_names:
        return all_run_dirs

    selected = []
    wanted = set(selected_run_names)
    for run_dir in all_run_dirs:
        if run_dir.name in wanted:
            selected.append(run_dir)
    missing = wanted.difference(run_dir.name for run_dir in selected)
    if missing:
        joined = ", ".join(sorted(missing))
        raise RetrievalConfigurationError(f"Requested run(s) not found: {joined}")
    return selected


def _build_run_diagnostic(run_dir: Path) -> dict[str, Any]:
    context = _load_run_context(run_dir)
    items = load_dataset(context["dataset_path"])
    registry = load_gold_registry_from_dataset(items)
    retrieval_by_id = load_retrieval_results(context["retrieval_results_path"])
    results = run_answer_eval(
        items=items,
        registry=registry,
        retrieval_by_id=retrieval_by_id,
        citation_required=context["citation_required"],
    )
    metrics = aggregate_answer_metrics(results)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    failed_cases = build_failed_case_report_entries(
        run_name=context["run_name"],
        generated_at=generated_at,
        dataset_path=context["dataset_path"],
        corpus=context["corpus"],
        items=items,
        registry=registry,
        retrieval_by_id=retrieval_by_id,
        results=results,
    )
    failure_category_counts = build_metric_failure_categories(
        failed_cases=failed_cases,
        metrics=metrics,
    )
    final_status, status_reason = determine_final_status(
        corpus=context["corpus"],
        failure_category_counts=failure_category_counts,
    )
    metrics_summary = build_metrics_summary_payload(
        run_name=context["run_name"],
        corpus=context["corpus"],
        generated_at=generated_at,
        dataset_path=context["dataset_path"],
        retrieval_results_path=context["retrieval_results_path"],
        gold_review_path=context["gold_review_path"],
        metrics=metrics,
        failure_category_counts=failure_category_counts,
        final_status=final_status,
        status_reason=status_reason,
    )
    nsoud_qa_007_diagnostic = None
    if context["corpus"] == "nsoud":
        nsoud_qa_007_diagnostic = build_nsoud_qa_007_diagnostic(
            items=items,
            registry=registry,
            retrieval_by_id=retrieval_by_id,
            results=results,
        )
    return {
        "run_name": context["run_name"],
        "corpus": context["corpus"],
        "dataset_path": str(context["dataset_path"]),
        "retrieval_results_path": str(context["retrieval_results_path"]),
        "gold_review_path": str(context["gold_review_path"]),
        "citation_required": context["citation_required"],
        "metrics": metrics_summary,
        "failure_category_counts": failure_category_counts,
        "failed_cases": [entry.__dict__ for entry in failed_cases],
        "status": final_status,
        "status_reason": status_reason,
        "nsoud_qa_007_diagnostic": nsoud_qa_007_diagnostic,
    }


def _command_string(args: argparse.Namespace) -> str:
    parts = [
        "python",
        "scripts/generate_legal_answer_eval_diagnostics.py",
        "--runs-dir",
        str(args.runs_dir),
        "--output-dir",
        str(args.output_dir),
    ]
    for run_name in args.run_name:
        parts.extend(["--run-name", run_name])
    return " ".join(parts)


def write_aggregate_outputs(
    *,
    output_dir: Path,
    runs: list[dict[str, Any]],
    command_run: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    aggregate_category_counts: dict[str, int] = defaultdict(int)
    all_failed_cases: list[dict[str, Any]] = []
    evaluated_datasets: list[str] = []
    for run in runs:
        evaluated_datasets.append(run["dataset_path"])
        for category, count in run["failure_category_counts"].items():
            aggregate_category_counts[category] += int(count)
        all_failed_cases.extend(run["failed_cases"])

    metrics_summary_payload = {
        "generated_at": generated_at,
        "evaluated_run_count": len(runs),
        "evaluated_datasets": evaluated_datasets,
        "runs": [run["metrics"] for run in runs],
    }
    failed_cases_payload = {
        "generated_at": generated_at,
        "evaluated_run_count": len(runs),
        "failed_case_count": len(all_failed_cases),
        "final_status": None,
        "runs": [
            {
                "run_name": run["run_name"],
                "corpus": run["corpus"],
                "failed_case_count": len(run["failed_cases"]),
                "failed_cases": run["failed_cases"],
                "nsoud_qa_007_diagnostic": run["nsoud_qa_007_diagnostic"],
            }
            for run in runs
        ],
    }
    category_payload = {
        "generated_at": generated_at,
        "aggregate_failure_category_counts": dict(sorted(aggregate_category_counts.items())),
        "final_status": None,
        "runs": [
            {
                "run_name": run["run_name"],
                "corpus": run["corpus"],
                "failure_category_counts": run["failure_category_counts"],
            }
            for run in runs
        ],
    }

    usable_partial = aggregate_category_counts.get("usable_partial_support", 0)
    weak_partial = aggregate_category_counts.get("weak_partial_support", 0)
    retrieval_support = aggregate_category_counts.get("true_retrieval_miss", 0) + aggregate_category_counts.get(
        "unsupported_boilerplate_or_gap", 0
    )
    gold_data = aggregate_category_counts.get("not_evaluable_missing_gold", 0) + aggregate_category_counts.get(
        "invalid_gold_annotation", 0
    )
    reingest_needed = "unknown" if aggregate_category_counts.get("true_retrieval_miss", 0) else "no"
    reingest_reason = (
        "Retrieval misses exist, but this diagnostic does not prove that re-ingest is required."
        if reingest_needed == "unknown"
        else "No evidence in this diagnostic that re-ingest is required."
    )
    final_status = "PASS"
    if any(run["status"] == "FAIL_WITH_REAL_NSOUD_RISK" for run in runs):
        final_status = "FAIL_WITH_REAL_NSOUD_RISK"
    elif any(run["status"] == "FAIL" for run in runs):
        final_status = "FAIL"
    elif any(run["status"] == "WARN" for run in runs):
        final_status = "WARN"
    category_payload["final_status"] = final_status
    failed_cases_payload["final_status"] = final_status

    (output_dir / "failed_cases_report.json").write_text(
        json.dumps(failed_cases_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(metrics_summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metric_failure_categories.json").write_text(
        json.dumps(category_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# NALUS evaluation quality diagnostic",
        "",
        f"- Run timestamp: {generated_at}",
        f"- Evaluated datasets: {', '.join(f'`{dataset}`' for dataset in evaluated_datasets)}",
        f"- Metric names: strict_direct_pass_rate_all, strict_direct_pass_rate_gold, usable_support_rate_gold, citation_available_rate_gold, unsupported_risk_rate_gold, gold_retrieval_miss_rate, support breakdown, gold count",
        "",
        "## Metric values",
        "",
    ]
    for run in runs:
        metrics = run["metrics"]
        metric_values = metrics["metrics"]
        lines.extend(
            [
                f"- {run['run_name']} ({run['corpus']}): strict_all={metric_values['strict_direct_pass_rate_all']:.3f}, strict_gold={metric_values['strict_direct_pass_rate_gold']:.3f}, usable_gold={metric_values['usable_support_rate_gold']:.3f}, citation_rate_gold={metric_values['citation_available_rate_gold']:.3f}, unsupported_risk_gold={metric_values['unsupported_risk_rate_gold']:.3f}, retrieval_miss_gold={metric_values['gold_retrieval_miss_rate']:.3f}, gold={metrics['gold_count']}, total={metrics['total_evaluated_count']}, status={metrics['final_status']}",
            ]
        )
        if metrics["denominator_warning"]:
            lines.append(f"- {run['run_name']} denominator warning: {metrics['denominator_warning']}")
    lines.extend(
        [
            "",
            "## Failure category breakdown",
            "",
        ]
    )
    for category, count in sorted(aggregate_category_counts.items()):
        lines.append(f"- {category}: {count}")
    lines.extend(
        [
            "",
            "## Top diagnostic entries",
            "",
        ]
    )
    top_failed_cases = all_failed_cases[:10]
    if not top_failed_cases:
        lines.append("- None")
    else:
        for case in top_failed_cases:
            lines.append(
                f"- {case['run_id']} / {case['question_id']}: {case['failure_category']} | status={case['answer_eval_status']} | support={case['support_level']}"
            )
    lines.extend(
        [
            "",
            "## Assessment",
            "",
            "- diagnostic entries include real failures, not-evaluable missing-gold items, usable partial support, and corpus-only routing cases",
            "- strict_direct_pass_rate is intentionally conservative",
            "- usable_support_rate_gold is the practical support metric",
            "- missing gold does not mean RAG failure",
            "- corpus_only mixed items are not expected to have document citations",
            "",
            f"- usable_partial_support: {usable_partial}",
            f"- weak_partial_support: {weak_partial}",
            f"- retrieval/support caused failures: {retrieval_support}",
            f"- bad/incomplete gold data caused failures: {gold_data}",
            f"- re-ingest actually needed: {reingest_needed}",
            f"- reason: {reingest_reason}",
            f"- production Qdrant was touched: no",
            f"- exact commands run: `{command_run}`",
            f"- files changed: `artifacts/evaluation_quality/failed_cases_report.json`, `artifacts/evaluation_quality/failed_cases_report.md`, `artifacts/evaluation_quality/metrics_summary.json`, `artifacts/evaluation_quality/metric_failure_categories.json`",
            f"- final status: {final_status}",
            "",
        ]
    )
    nsoud_diag = next(
        (run["nsoud_qa_007_diagnostic"] for run in runs if run["nsoud_qa_007_diagnostic"] is not None),
        None,
    )
    if nsoud_diag is not None:
        lines.extend(
            [
                "## nsoud-qa-007 diagnostic",
                "",
                f"- gold source id: {nsoud_diag['gold_source_id']}",
                f"- retrieved top-k ids: {', '.join(nsoud_diag['retrieved_top_k_ids'])}",
                f"- expected source absent in top-k: {not nsoud_diag['expected_source_present_top_k']}",
                f"- true_retrieval_miss: {nsoud_diag['true_retrieval_miss']}",
                f"- gold_annotation_mismatch: {nsoud_diag['gold_annotation_mismatch']}",
                f"- answer_support_gap: {nsoud_diag['answer_support_gap']}",
                f"- matcher_issue: {nsoud_diag['matcher_issue']}",
                f"- question_too_generic: {nsoud_diag['question_too_generic']}",
                f"- conclusion: {nsoud_diag['conclusion']}",
                "",
            ]
        )
    (output_dir / "failed_cases_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs_dir = args.runs_dir.resolve()
    output_dir = args.output_dir.resolve()
    run_dirs = _resolve_run_dirs(runs_dir, args.run_name)
    if not run_dirs:
        raise RetrievalConfigurationError(f"No answer-eval runs found in {runs_dir}")

    runs = [_build_run_diagnostic(run_dir) for run_dir in run_dirs]
    write_aggregate_outputs(
        output_dir=output_dir,
        runs=runs,
        command_run=_command_string(args),
    )
    print(
        f"[legal-answer-diagnostics] runs={len(runs)} output={output_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RetrievalConfigurationError as exc:
        print(f"[legal-answer-diagnostics] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
