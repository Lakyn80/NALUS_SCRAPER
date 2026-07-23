from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LegalV2EvaluationCase:
    case_id: str
    query: str
    expected_document_ids: list[str] = field(default_factory=list)
    hard_negative_document_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LegalV2PipelineResult:
    case_id: str
    pipeline: str
    retrieved_document_ids: list[str] = field(default_factory=list)
    verified_document_ids: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    chunk_count: int = 0
    average_token_count: float = 0.0
    section_distribution: dict[str, int] = field(default_factory=dict)
    reconstruction_failures: int = 0
    status: str = "pass"
    error: str | None = None


@dataclass(frozen=True)
class LegalV2ComparisonMetrics:
    pipeline: str
    case_count: int
    candidate_recall: float
    exact_precision: float
    hard_negative_false_positives: int
    verified_document_precision: float
    average_latency_ms: float
    chunk_count: int
    average_token_count: float
    section_distribution: dict[str, int]
    reconstruction_failures: int


@dataclass(frozen=True)
class LegalV2EvaluationReport:
    summary: dict[str, Any]
    metrics_by_pipeline: dict[str, LegalV2ComparisonMetrics]
    cases: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": dict(self.summary),
            "metrics_by_pipeline": {
                key: asdict(value) for key, value in self.metrics_by_pipeline.items()
            },
            "cases": list(self.cases),
        }


def run_offline_legal_v2_comparison(
    *,
    cases: list[LegalV2EvaluationCase],
    current_results: list[LegalV2PipelineResult],
    paragraph_child_results: list[LegalV2PipelineResult],
    paragraph_parent_results: list[LegalV2PipelineResult],
    hard_negative_ids: set[str] | None = None,
    status: str = "pass",
) -> LegalV2EvaluationReport:
    started_at = _utc_now()
    pipelines = {
        "current_production_chunks": current_results,
        "paragraph_child_chunks": paragraph_child_results,
        "paragraph_child_parent_windows": paragraph_parent_results,
    }
    normalized_global_negatives = {
        _normalize_document_id(value) for value in hard_negative_ids or set()
    }
    metrics_by_pipeline = {
        name: _metrics_for_pipeline(
            pipeline=name,
            cases=cases,
            results=results,
            global_hard_negative_ids=normalized_global_negatives,
        )
        for name, results in pipelines.items()
    }
    finished_at = _utc_now()
    case_payloads = [
        {
            "case_id": case.case_id,
            "expected_document_count": len(case.expected_document_ids),
            "hard_negative_count": len(case.hard_negative_document_ids),
            "pipelines": {
                name: asdict(result)
                for name, result in _results_by_pipeline_for_case(
                    case.case_id, pipelines
                ).items()
            },
        }
        for case in cases
    ]
    failed = sum(
        1
        for results in pipelines.values()
        for result in results
        if result.status == "failure"
    )
    blocked = sum(
        1 for results in pipelines.values() for result in results if result.status == "blocked"
    )
    return LegalV2EvaluationReport(
        summary={
            "schema": "legal_v2_offline_comparison",
            "status": status,
            "passed": status == "pass" and failed == 0 and blocked == 0,
            "failed": failed,
            "blocked": blocked,
            "total_cases": len(cases),
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": _duration_ms(started_at, finished_at),
            "production_readiness_claimed": False,
        },
        metrics_by_pipeline=metrics_by_pipeline,
        cases=case_payloads,
    )


def write_legal_v2_evaluation_report(
    *,
    output_dir: Path,
    report: LegalV2EvaluationReport,
    status: str | None = None,
    exception: Exception | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    if status is not None:
        payload["summary"]["status"] = status
        payload["summary"]["passed"] = status == "pass" and exception is None
    if exception is not None:
        payload["summary"]["status"] = "exception"
        payload["summary"]["passed"] = False
        payload["summary"]["exception_type"] = exception.__class__.__name__
        payload["summary"]["exception"] = str(exception)

    json_path = output_dir / "legal_v2_evaluation.json"
    markdown_path = output_dir / "legal_v2_evaluation.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown_report(payload), encoding="utf-8")
    return json_path, markdown_path


def _metrics_for_pipeline(
    *,
    pipeline: str,
    cases: list[LegalV2EvaluationCase],
    results: list[LegalV2PipelineResult],
    global_hard_negative_ids: set[str],
) -> LegalV2ComparisonMetrics:
    results_by_case = {result.case_id: result for result in results}
    gold_total = 0
    candidate_gold_hits = 0
    retrieved_total = 0
    retrieved_true_positive_total = 0
    verified_total = 0
    verified_true_positive_total = 0
    hard_negative_false_positives = 0
    latency_values: list[float] = []
    average_token_values: list[float] = []
    chunk_count = 0
    reconstruction_failures = 0
    section_distribution: dict[str, int] = {}

    for case in cases:
        result = results_by_case.get(case.case_id)
        if result is None:
            continue
        gold_ids = {_normalize_document_id(value) for value in case.expected_document_ids}
        case_negative_ids = {
            _normalize_document_id(value) for value in case.hard_negative_document_ids
        }
        hard_negative_ids = global_hard_negative_ids.union(case_negative_ids)
        retrieved_ids = [_normalize_document_id(value) for value in result.retrieved_document_ids]
        verified_ids = [_normalize_document_id(value) for value in result.verified_document_ids]

        gold_total += len(gold_ids)
        candidate_gold_hits += len(gold_ids.intersection(retrieved_ids))
        retrieved_total += len(retrieved_ids)
        retrieved_true_positive_total += len(gold_ids.intersection(retrieved_ids))
        verified_total += len(verified_ids)
        verified_true_positive_total += len(gold_ids.intersection(verified_ids))
        hard_negative_false_positives += len(hard_negative_ids.intersection(verified_ids))
        latency_values.append(result.latency_ms)
        average_token_values.append(result.average_token_count)
        chunk_count += result.chunk_count
        reconstruction_failures += result.reconstruction_failures
        for section, count in result.section_distribution.items():
            section_distribution[section] = section_distribution.get(section, 0) + count

    return LegalV2ComparisonMetrics(
        pipeline=pipeline,
        case_count=len(cases),
        candidate_recall=_ratio(candidate_gold_hits, gold_total),
        exact_precision=_ratio(retrieved_true_positive_total, retrieved_total),
        hard_negative_false_positives=hard_negative_false_positives,
        verified_document_precision=_ratio(verified_true_positive_total, verified_total),
        average_latency_ms=_average(latency_values),
        chunk_count=chunk_count,
        average_token_count=_average(average_token_values),
        section_distribution=section_distribution,
        reconstruction_failures=reconstruction_failures,
    )


def _results_by_pipeline_for_case(
    case_id: str,
    pipelines: dict[str, list[LegalV2PipelineResult]],
) -> dict[str, LegalV2PipelineResult]:
    results: dict[str, LegalV2PipelineResult] = {}
    for pipeline, pipeline_results in pipelines.items():
        for result in pipeline_results:
            if result.case_id == case_id:
                results[pipeline] = result
                break
    return results


def _markdown_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Universal Verified Legal Retrieval v2 evaluation",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Passed: `{summary.get('passed')}`",
        f"- Cases: {summary.get('total_cases')}",
        f"- Started: {summary.get('started_at')}",
        f"- Finished: {summary.get('finished_at')}",
        f"- Duration ms: {summary.get('duration_ms')}",
        f"- Production readiness claimed: `{summary.get('production_readiness_claimed')}`",
        "",
        "## Metrics",
        "",
    ]
    for pipeline, metrics in payload["metrics_by_pipeline"].items():
        lines.extend(
            [
                f"### {pipeline}",
                "",
                f"- candidate_recall: {metrics['candidate_recall']:.3f}",
                f"- exact_precision: {metrics['exact_precision']:.3f}",
                "- hard_negative_false_positives: "
                f"{metrics['hard_negative_false_positives']}",
                "- verified_document_precision: "
                f"{metrics['verified_document_precision']:.3f}",
                f"- average_latency_ms: {metrics['average_latency_ms']:.3f}",
                f"- chunk_count: {metrics['chunk_count']}",
                f"- average_token_count: {metrics['average_token_count']:.3f}",
                f"- reconstruction_failures: {metrics['reconstruction_failures']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Scope",
            "",
            "- This is an offline comparison artifact.",
            "- It does not claim production readiness from a seed dataset.",
            "",
        ]
    )
    return "\n".join(lines)


def _normalize_document_id(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _duration_ms(started_at: str, finished_at: str) -> float:
    started = datetime.strptime(started_at, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    finished = datetime.strptime(finished_at, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    return (finished - started).total_seconds() * 1000
