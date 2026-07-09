"""Read-only Prometheus exporter for offline legal answer eval artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_DIR = Path("artifacts/rag_eval/legal_qa/answer_eval")

SUMMARY_COUNT_FIELDS = (
    "gold",
    "direct_support_count",
    "partial_support_count",
    "gap_count",
    "boilerplate_noise_count",
    "corpus_only_count",
    "unsupported_answer_risk_count",
)

SUMMARY_RATE_FIELDS = (
    "strict_direct_pass_rate_all",
    "strict_direct_pass_rate_gold",
    "usable_support_rate_gold",
    "citation_available_rate",
)

METRIC_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("legal_answer_eval_gold", "Gold items available in the eval run.", SUMMARY_COUNT_FIELDS[:1]),
    ("legal_answer_eval_direct_support_count", "Direct support count (gold items).", SUMMARY_COUNT_FIELDS[1:2]),
    ("legal_answer_eval_partial_support_count", "Partial support count (gold items).", SUMMARY_COUNT_FIELDS[2:3]),
    ("legal_answer_eval_gap_count", "Gap count (gold items).", SUMMARY_COUNT_FIELDS[3:4]),
    ("legal_answer_eval_boilerplate_noise_count", "Boilerplate noise count (gold items).", SUMMARY_COUNT_FIELDS[4:5]),
    ("legal_answer_eval_corpus_only_count", "Corpus-only count (gold items).", SUMMARY_COUNT_FIELDS[5:6]),
    (
        "legal_answer_eval_unsupported_answer_risk_count",
        "Unsupported answer risk count.",
        SUMMARY_COUNT_FIELDS[6:7],
    ),
    (
        "legal_answer_eval_strict_direct_pass_rate_all",
        "Strict direct pass rate over all questions.",
        SUMMARY_RATE_FIELDS[:1],
    ),
    (
        "legal_answer_eval_strict_direct_pass_rate_gold",
        "Strict direct pass rate over gold items.",
        SUMMARY_RATE_FIELDS[1:2],
    ),
    (
        "legal_answer_eval_usable_support_rate_gold",
        "Usable support rate over gold items.",
        SUMMARY_RATE_FIELDS[2:3],
    ),
    (
        "legal_answer_eval_citation_available_rate",
        "Citation available rate over gold items.",
        SUMMARY_RATE_FIELDS[3:4],
    ),
)

_LABEL_NAMES = ("run_name", "corpus")


def infer_corpus_from_run_name(run_name: str) -> str:
    name = run_name.lower()
    if name.startswith("mixed"):
        return "mixed"
    if name.startswith("nsoud"):
        return "nsoud"
    if name.startswith("usoud"):
        return "usoud"
    return "unknown"


_GAUGES: dict[str, Gauge] = {
    metric_name: Gauge(metric_name, description, _LABEL_NAMES)
    for metric_name, description, _ in METRIC_SPECS
}

_FIELD_TO_METRIC: dict[str, str] = {
    field: metric_name
    for metric_name, _, fields in METRIC_SPECS
    for field in fields
}


def _metrics_json_to_summary(payload: dict[str, Any], run_name: str) -> dict[str, Any]:
    corpus = payload.get("corpus")
    if not corpus:
        corpus = infer_corpus_from_run_name(run_name)
    return {
        "generated_at": payload.get("generated_at"),
        "run_name": run_name,
        "corpus": corpus,
        "gold": payload.get("gold_available_count", payload.get("gold", 0)),
        "direct_support_count": payload.get("direct_support_count", 0),
        "partial_support_count": payload.get("partial_support_count", 0),
        "gap_count": payload.get("gap_count", 0),
        "boilerplate_noise_count": payload.get("boilerplate_noise_count", 0),
        "corpus_only_count": payload.get("corpus_only_count", 0),
        "unsupported_answer_risk_count": payload.get("unsupported_answer_risk_count", 0),
        "strict_direct_pass_rate_all": payload.get("strict_direct_pass_rate_all", 0.0),
        "strict_direct_pass_rate_gold": payload.get("strict_direct_pass_rate_gold", 0.0),
        "usable_support_rate_gold": payload.get("usable_support_rate_gold", 0.0),
        "citation_available_rate": payload.get("citation_available_rate", 0.0),
    }


def load_run_summary(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        payload.setdefault("run_name", run_dir.name)
        payload.setdefault("corpus", infer_corpus_from_run_name(str(payload["run_name"])))
        return payload

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        return _metrics_json_to_summary(payload, run_dir.name)

    return None


def discover_run_summaries(artifacts_dir: Path) -> list[dict[str, Any]]:
    if not artifacts_dir.exists():
        logger.warning("Artifacts directory does not exist: %s", artifacts_dir)
        return []

    summaries: list[dict[str, Any]] = []
    for child in sorted(artifacts_dir.iterdir()):
        if not child.is_dir():
            continue
        summary = load_run_summary(child)
        if summary is not None:
            summaries.append(summary)
    return summaries


def refresh_prometheus_gauges(artifacts_dir: Path) -> int:
    for gauge in _GAUGES.values():
        gauge.clear()

    summaries = discover_run_summaries(artifacts_dir)
    for summary in summaries:
        run_name = str(summary["run_name"])
        corpus = str(summary["corpus"])
        labels = (run_name, corpus)
        for field, metric_name in _FIELD_TO_METRIC.items():
            value = summary.get(field, 0)
            _GAUGES[metric_name].labels(*labels).set(float(value))
    return len(summaries)


def render_metrics(artifacts_dir: Path) -> bytes:
    refresh_prometheus_gauges(artifacts_dir)
    return generate_latest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expose offline legal answer eval metrics for Prometheus.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9108)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directory containing per-run answer eval output folders.",
    )
    return parser.parse_args(argv)


class _MetricsHandler(BaseHTTPRequestHandler):
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in {"/metrics", "/metrics/"}:
            self.send_response(404)
            self.end_headers()
            return

        try:
            payload = render_metrics(self.artifacts_dir)
        except Exception:
            logger.exception("Failed to render Prometheus metrics")
            self.send_response(500)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPE_LATEST)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        logger.info("%s - %s", self.address_string(), format % args)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    artifacts_dir = args.artifacts_dir.resolve()
    _MetricsHandler.artifacts_dir = artifacts_dir

    run_count = refresh_prometheus_gauges(artifacts_dir)
    logger.info(
        "Starting legal answer eval metrics exporter on %s:%s (runs=%s, artifacts=%s)",
        args.host,
        args.port,
        run_count,
        artifacts_dir,
    )

    server = HTTPServer((args.host, args.port), _MetricsHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down exporter")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
