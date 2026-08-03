# Retrieval Enterprise Security And Operations

Status: target security and operations specification.

Retrieval changes must preserve confidentiality, controlled side effects, and
operational rollback. Quality work must not weaken security boundaries.

## Secrets

Secrets include:

- API keys;
- authorization headers;
- raw provider request/response bodies when they contain secrets or prompts;
- environment files;
- idempotency keys;
- private prompts;
- direct personal contact/payment identifiers.

Rules:

- Never commit secrets.
- Never print secrets.
- Redact key-like values in examples.
- `.env.example` must contain placeholders only.
- If a key was committed, pushed, or shared, report mandatory rotation.

## Logging And Tracing

Allowed:

- stage name;
- status;
- bounded counts;
- latency;
- provider name;
- model name when non-secret;
- collection/index identifiers when not sensitive;
- redacted error class.

Forbidden:

- raw query text in metric labels;
- document IDs, ECLIs, tenant/user IDs, request IDs, trace IDs, prompts, evidence
  quotes, or error strings in Prometheus labels;
- raw provider content;
- unbounded request/response bodies;
- stack traces in API responses.

Detailed diagnostics may be written only as bounded, redacted artifacts for
explicit debugging tasks.

## Provider Calls

Provider calls are allowed only when the current task explicitly requires them.

Provider-backed operations must record safe diagnostics:

- provider name;
- model name;
- finish reason;
- content length;
- token usage when provided;
- truncation indicators;
- extraction method;
- JSON syntax error if any;
- schema validation paths if any;
- retry count;
- timeout.

Transport/authentication diagnosis stops once connectivity and credentials are
verified. Do not spend additional calls on generic provider checks when the
failure is in a concrete Legal v2 operation.

## Runtime Safety

Disabled endpoints must not initialize:

- Qdrant clients;
- BM25 sidecars;
- embedding models;
- provider credentials;
- expensive caches.

Invalid config fails closed with a controlled error. Runtime code must not lower
thresholds, expand search limits, switch providers, change collections, or
fallback to unrelated results to make a request look successful.

## Resource Policy

Default assumptions:

- CPU only.
- No CUDA/GPU unless explicitly approved.
- No package/model download unless explicitly approved.
- Existing model cache preferred.

Experimental resources must be named and isolated:

- Qdrant collection;
- BM25 sidecar;
- artifact directory;
- model identity;
- index profile;
- source selection.

## Observability

Reuse existing mechanisms:

- logging from `app/core/logging.py`;
- tracing from `app/core/tracing.py`;
- existing Prometheus/Grafana stack under `monitoring/`.

Do not introduce a second observability stack unless an ADR documents why the
existing stack cannot satisfy the requirement.

## Failure Modes

Required fail-closed conditions:

- missing or invalid collection;
- missing or invalid BM25 sidecar;
- model path unavailable when local-only is required;
- provider JSON syntax failure after allowed retries;
- provider schema validation failure after allowed repair/retry;
- verifier truncation when output completeness is required;
- Qdrant/BM25 identity mismatch;
- checkpoint mismatch;
- protected collection target.

User-facing failures should be controlled and concise. Internal diagnostics must
remain bounded and redacted.

