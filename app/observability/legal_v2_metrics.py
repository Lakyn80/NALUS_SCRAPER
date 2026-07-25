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


def record_request(*, endpoint: str, status: str) -> None:
    REQUESTS_TOTAL.labels(endpoint=endpoint, status=status).inc()


def record_stage_latency(*, stage: str, status: str, latency_ms: float) -> None:
    STAGE_LATENCY_SECONDS.labels(stage=stage, status=status).observe(max(0.0, latency_ms / 1000))

