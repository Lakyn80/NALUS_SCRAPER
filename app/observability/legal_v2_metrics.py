from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUESTS_TOTAL = Counter(
    "nalus_legal_v2_search_requests_total",
    "Legal Retrieval v2 search requests.",
    ("endpoint", "status"),
)

STAGE_LATENCY_SECONDS = Histogram(
    "nalus_legal_v2_stage_latency_seconds",
    "Legal Retrieval v2 latency by stage.",
    ("stage", "status"),
)

QUERY_INTERPRETATIONS_TOTAL = Counter(
    "nalus_legal_v2_query_interpretations_total",
    "Legal Retrieval v2 query interpretation outcomes.",
    ("status", "provider"),
)

DOCUMENTS_TOTAL = Counter(
    "nalus_legal_v2_documents_total",
    "Legal Retrieval v2 document outcomes.",
    ("decision",),
)

REJECTIONS_TOTAL = Counter(
    "nalus_legal_v2_rejections_total",
    "Legal Retrieval v2 rejection reasons.",
    ("reason",),
)

DEEPSEEK_TOKENS_TOTAL = Counter(
    "nalus_legal_v2_deepseek_tokens_total",
    "Legal Retrieval v2 estimated DeepSeek token usage.",
    ("operation", "token_type"),
)

DEEPSEEK_COST_TOTAL = Counter(
    "nalus_legal_v2_deepseek_cost_total",
    "Legal Retrieval v2 estimated DeepSeek cost.",
    ("operation", "cost_component"),
)

LONG_INPUT_REQUESTS_TOTAL = Counter(
    "nalus_legal_v2_long_input_requests_total",
    "Legal v2 long-input preprocessing requests.",
    ("classification", "status"),
)

CONDENSATION_REQUESTS_TOTAL = Counter(
    "nalus_legal_v2_condensation_requests_total",
    "Legal v2 SearchBrief condensation attempts.",
    ("method", "status"),
)

CONDENSATION_FAILURES_TOTAL = Counter(
    "nalus_legal_v2_condensation_failures_total",
    "Legal v2 SearchBrief condensation failures.",
    ("method", "status"),
)

CONDENSATION_DURATION_SECONDS = Histogram(
    "nalus_legal_v2_condensation_duration_seconds",
    "Legal v2 SearchBrief condensation duration.",
    ("method", "status"),
)

CONDENSED_INPUT_CHARS = Histogram(
    "nalus_legal_v2_condensed_input_chars",
    "Original character counts for condensed long inputs.",
    ("method",),
    buckets=(100, 300, 700, 1500, 3000, 8000, 20000, 50000, 100000),
)

SEARCH_BRIEF_CHARS = Histogram(
    "nalus_legal_v2_search_brief_chars",
    "Generated SearchBrief character counts.",
    ("method",),
    buckets=(50, 100, 200, 500, 1000, 1500, 3000, 8000),
)


def record_request(*, endpoint: str, status: str) -> None:
    REQUESTS_TOTAL.labels(endpoint=endpoint, status=status).inc()


def record_stage_latency(*, stage: str, status: str, latency_ms: float) -> None:
    STAGE_LATENCY_SECONDS.labels(stage=stage, status=status).observe(max(0.0, latency_ms / 1000))


def record_long_input(
    *,
    classification: str,
    method: str,
    status: str,
    latency_ms: float,
    original_chars: int,
    brief_chars: int,
    condensed: bool,
) -> None:
    safe_classification = (classification or "unknown")[:40]
    safe_method = (method or "none")[:40]
    safe_status = (status or "unknown")[:40]
    LONG_INPUT_REQUESTS_TOTAL.labels(
        classification=safe_classification, status=safe_status
    ).inc()
    if condensed or safe_method in {"extractive", "precise"}:
        CONDENSATION_REQUESTS_TOTAL.labels(method=safe_method, status=safe_status).inc()
        CONDENSATION_DURATION_SECONDS.labels(method=safe_method, status=safe_status).observe(
            max(0.0, latency_ms / 1000.0)
        )
        if safe_status != "ok":
            CONDENSATION_FAILURES_TOTAL.labels(method=safe_method, status=safe_status).inc()
        CONDENSED_INPUT_CHARS.labels(method=safe_method).observe(max(0, original_chars))
        SEARCH_BRIEF_CHARS.labels(method=safe_method).observe(max(0, brief_chars))

