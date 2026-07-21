# Observability, Reliability, Audit, and Failure-Detection Contract

This document adapts the universal production observability prompt to the current NALUS scraper/RAG repository. It is a contract for future implementation work, not a claim that every listed capability is already implemented.

## Current Project Context

- Primary repository: `C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper`
- Current primary branch: `main`
- Application type: Python data pipeline, FastAPI RAG API, offline evaluation runners, and Docker Compose local/demo topology
- Existing observability stack: Prometheus and Grafana under `monitoring/`, plus Python logging and lightweight trace events
- Shared dashboard context: this repo already provisions NALUS dashboards and references the Eternal World dashboard datasource; do not create duplicate Grafana infrastructure
- Existing report roots: `artifacts/`, `app/artifacts/`, and task-specific output directories

## Starting State Observed On Setup

- Branch: `main`
- HEAD: `aa63a3a05b81e8555a3c84da351d6f1ac2faa8e3`
- Existing logging: classic text logging in `app/core/logging.py`
- Existing tracing: lightweight DEBUG trace helper in `app/core/tracing.py`
- Existing metrics: evaluation exporter and constraint retrieval metrics under `app/observability/`
- Existing monitoring: Prometheus and Grafana Compose services, dashboard provisioning, and bounded evaluation metrics
- Pre-existing worktree changes: present before this setup; see `git status --short` before editing any task

## Layer Responsibilities

Logs are operational events and diagnostics. They must be structured for production paths, redact sensitive material, and support correlation drill-down.

Audit is durable business accountability for relevant state transitions. Audit events must answer who or what caused the transition and must not replace authorization checks.

Metrics are low-cardinality numerical signals for dashboards, SLOs, and alerts. They must not include identifiers such as ECLI, document ID, query text, user ID, tenant ID, request ID, correlation ID, or error message as labels.

Traces show causal and timing relationships across request, service, database, external API, queue, worker, verification, reconciliation, and report generation boundaries.

Reports are durable evidence from a specific validation run. JSON reports need stable root summaries, and Markdown reports need human-readable conclusions and caveats.

## Canonical Correlation Context

Future critical workflows should use one canonical context across participating components:

- `correlation_id`: stable across one business operation; validate inbound values and generate securely when absent
- `request_id`: unique per API request
- `operation_id` or `workflow_id`: stable for a critical workflow attempt
- `job_id` or `task_id`: captured after enqueue and inside the worker, if workers are added
- `trace_id` and `span_id`: from the tracing system
- `idempotency_key_fingerprint`: stable fingerprint only; never log the full key
- `external_reference`: safe provider/native reference, where applicable
- `actor_id` and ownership/tenant context: only where needed and never as metric labels

Context must be cleared after each request, task, job, or message. Tests must cover leakage between requests, users, tenants or ownership scopes, and jobs once those concepts are implemented.

## Structured Logging Contract

Production-compatible logs should use stable event names and bounded fields. Recommended fields:

- `timestamp`
- `severity`
- `service`
- `environment`
- `event_name`
- `message`
- `correlation_id`
- `request_id`
- `trace_id`
- `span_id`
- `operation_type`
- `workflow_status`
- `job_id`
- `task_id`
- `adapter_or_provider`
- `idempotency_key_fingerprint`
- `external_reference`
- `http_method`
- `http_route`
- `http_status`
- `duration_ms`
- `safe_error_code`
- `retry_attempt`
- `reconciliation_status`
- `build_version`

Stable event names should include forms such as:

- `workflow.started`
- `workflow.completed`
- `workflow.failed`
- `job.enqueued`
- `job.started`
- `job.completed`
- `external.request.started`
- `external.request.completed`
- `verification.started`
- `verification.succeeded`
- `verification.failed`
- `reconciliation.required`
- `security.authentication_failed`
- `security.authorization_denied`
- `security.tenant_mismatch`
- `security.integrity_mismatch`

Centralized redaction must cover nested dictionaries, lists, dataclasses or Pydantic models, HTTP headers, exceptions, structured log extras, and serialized reports.

Never log secrets, tokens, authorization headers, cookie contents, database URLs with passwords, raw `.env` content, full idempotency keys, unbounded request/response bodies, full private prompts by default, payment-card data, or unnecessary personal data.

## Tracing Contract

OpenTelemetry is the preferred future distributed tracing system unless a project-standard tracing system is established first. Until then, keep `app/core/tracing.py` call sites safe and bounded so they can be migrated.

Critical workflows should include explicit business spans:

- `workflow.prepare`
- `workflow.validate`
- `workflow.execute`
- `job.enqueue`
- `job.execute`
- `external.call`
- `result.verify`
- `reconciliation.run`
- `report.validate`

Allowed span attributes must be bounded, such as `workflow.type`, `workflow.status`, `provider.name`, `operation.result`, `verification.result`, `idempotency.replayed`, `reconciliation.required`, `error.type`, and `http.status_code`.

Do not attach full user content, secrets, tokens, raw SQL with private values, arbitrary payloads, or unbounded document text to spans.

## Metrics Contract

Reuse existing metrics names and exporters where practical. New application metrics should follow bounded-label conventions:

- `application_workflows_total{workflow_type,status}`
- `application_operations_total{operation_type,status}`
- `application_operation_latency_seconds{operation_type,status}`
- `application_idempotency_replays_total{operation_type}`
- `application_reconciliation_total{operation_type,reason}`
- `application_verifications_total{operation_type,result}`
- `application_audit_integrity_checks_total{result}`
- `application_data_quality_checks_total{check,result}`
- `application_security_events_total{event_type,result}`
- `application_provider_calls_total{provider,operation,result}`
- `application_provider_cost_total{provider,operation,cost_component}`
- `application_report_invariant_failures_total{invariant}`

Prohibited labels include query text, document ID, ECLI, tenant ID, user ID, actor ID, workflow ID, operation ID, request ID, correlation ID, trace ID, task ID, customer name, and error message.

Tests for new metrics must inspect registered label names against an allowlist.

## Self-Validating Reports

All critical live, integration, migration, data-quality, security, and workflow runners should persist JSON and Markdown reports on pass, fail, blocked, partial completion, and exception.

Every JSON report should include a stable root-level summary object:

```json
{
  "summary": {
    "status": "pass",
    "passed": 0,
    "failed": 0,
    "blocked": 0,
    "total_cases": 0,
    "created_objects": 0,
    "duplicate_objects": 0,
    "verification_passed": 0,
    "verification_failed": 0,
    "reconciliation_required": 0,
    "audit_integrity_failures": 0,
    "security_invariant_failures": 0,
    "data_quality_failures": 0,
    "provider_calls_total": 0,
    "provider_cost_total": "0",
    "started_at": "",
    "finished_at": "",
    "duration_ms": 0,
    "chronological_report": true
  }
}
```

Report validators must fail the report and return a nonzero process exit when root invariants fail. JSON and Markdown must still be written.

Required invariants include chronological event ordering, summary count consistency, unique created-object counting, duplicate detection, verification before success, terminal state consistency, forbidden side-effect checks after cancellation or denial, idempotent retry consistency, provider-call and provider-cost reconciliation, and secret-redaction checks.

## Audit Integrity Contract

When audit trails are added for business-relevant or security-relevant state changes, events should support:

- `event_id`
- `event_sequence`
- `event_type`
- `timestamp`
- ownership or tenant context
- `actor_type`
- `actor_id`
- `workflow_id`
- `operation_id`
- `correlation_id`
- `previous_status`
- `new_status`
- `payload_hash`
- `previous_event_hash`
- `event_hash`
- `safe_metadata`

Use deterministic canonical serialization for hashing. Scope hash chains per workflow, operation, aggregate, or bounded tenant/ownership scope. Do not put secrets in hashed metadata. Audit hashing does not replace authorization.

Validators should detect sequence gaps, hash mismatches, missing mandatory events, duplicate terminal events, and forbidden execution after cancellation or denial.

## Data-Quality And Domain Invariants

Typed validation should classify every critical invariant result as:

- `pass`
- `fail`
- `insufficient_data`
- `not_applicable`

Material failures must prevent success. Material `insufficient_data` must not be reported as verified success; use controlled states such as `reconciliation_required`, `verification_incomplete`, or `manual_review_required`.

Project-specific invariants should include document identity consistency, source/court consistency, ECLI and document ID normalization, chunk-to-document ownership, full-document reconstruction consistency, retrieval feature-flag behavior, provider-call accounting, evaluation report count consistency, and disabled-by-default experimental paths.

## Security Telemetry

Add structured events and low-cardinality metrics for applicable security conditions:

- invalid or expired credential
- missing scope
- unexpected privileged scope
- ownership or tenant mismatch
- unauthorized state transition
- integrity hash mismatch
- execution of unapproved or invalid operation
- prompt-injection or command-injection bypass attempt
- repeated denied operations
- idempotency conflict
- secret-redaction failure

Never log the secret material involved.

## SLO And Alert Starting Points

Initial operational targets for critical state-changing operations:

- Success rate: at least 99.5 percent where dependencies are healthy
- Duplicate side effects: 0
- Successful operations without required verification: 0
- Audit-integrity failures: 0
- Cross-tenant or cross-owner incidents: 0
- Reconciliation-required operations: reviewed immediately
- p95 latency: documented per workflow before enforcement

Critical alert classes:

- `DuplicateSideEffectDetected`
- `SuccessfulOperationWithoutVerification`
- `CrossTenantOrOwnershipViolation`
- `AuditIntegrityBroken`
- `IntegrityHashMismatch`
- `UnexpectedPrivilegedScope`
- `SecretLeakDetected`

High-severity alert classes:

- `OperationReconciliationRequired`
- `ResultVerificationFailed`
- `CriticalWorkflowFailureRateHigh`
- `ExternalWriteResultUnknown`
- `AuthenticationFailuresHigh`

Warning alert classes:

- `WorkflowP95LatencyHigh`
- `ProviderDailyCostHigh`
- `AuthorizationDenialsHigh`
- `ReportInvariantFailure`
- `QueueBacklogHigh`

Alert labels must stay low-cardinality. Alert annotations should explain what failed, which service or bounded workflow type is involved, which dashboard/query to inspect, and what the operator should do next.

## Controlled Failure Injection

Failure injection must be local/test/demo-only, disabled by default, typed, deterministic, observable, and blocked in production.

Useful failure points for this project:

- `before_retrieval`
- `after_retrieval_before_response`
- `before_external_provider_request`
- `after_external_side_effect_before_response`
- `before_verification`
- `verification_returns_stale_data`
- `verification_returns_insufficient_data`
- `before_terminal_success`
- `database_commit_failure`
- `queue_unavailable`
- `trace_exporter_unavailable`
- `log_backend_unavailable`
- `metrics_backend_unavailable`
- `authentication_failure`
- `authorization_failure`

Never use arbitrary code execution, `eval`, or unvalidated strings. Production activation must fail closed.

## Reproducibility Metadata

Expose safe metadata through startup logs, build-info metrics, diagnostics, test reports, and trace resource attributes where available:

- Git commit SHA
- dirty flag at build time
- application version
- adapter/provider version
- policy or calculation version
- runtime configuration version
- database migration revision, if introduced
- test runner version
- container image identifier or digest
- language/runtime version
- environment

Do not expose private repository paths or secret environment values in production-facing surfaces.

## Documentation And Runbooks

Keep these documents current as implementation proceeds:

- `AGENTS.md`
- `docs/OBSERVABILITY_CONTRACT.md`
- `docs/TEST_REPORTING.md`, if report schemas are changed
- `docs/incident-response.md` or files under `docs/runbooks/`, when alerts are added
- `PROJECT_PROGRESS.md`, when a task changes behavior or validation expectations

## Phased Implementation Backlog

Phase A: correlation context, structured logging, and redaction.

Phase B: OpenTelemetry-compatible tracing and trace-context propagation.

Phase C: centralized log integration, reusing the deployment standard.

Phase D: self-validating reports with stable summaries and invariant validators.

Phase E: audit integrity for business-relevant operations.

Phase F: typed domain and data-quality invariant validation.

Phase G: metrics, recording rules, SLOs, alerts, and dashboards.

Phase H: controlled failure injection and live observability runner.

Phase I: documentation, runbooks, and full validation.

Each phase must keep the system runnable and include focused tests. Do not claim success for validations that were not executed.
