# Observability Incident Runbook

This runbook is the starting point for future alerts. It intentionally avoids high-cardinality identifiers and secret material in alert labels.

## Duplicate Side Effect

Severity: critical

Check:

- Recent workflow or operation metrics by bounded `workflow_type`, `operation_type`, and `status`
- Report summaries for `duplicate_objects`
- Audit validator output for duplicate terminal events
- Logs filtered by a specific `correlation_id` only after obtaining it from a safe report or incident record

Operator action:

- Stop automatic retry for the affected operation type if it can create additional side effects.
- Identify whether the native external result already exists.
- Move unknown outcomes to `reconciliation_required` or `manual_review_required`.
- Do not mark the operation succeeded until verification passes.

## Reconciliation Required

Severity: high

Check:

- `application_reconciliation_total` by operation type and reason, once implemented
- Structured events named `reconciliation.required`
- Report fields `reconciliation_required`, `verification_failed`, and `provider_calls_total`

Operator action:

- Verify the external/native state before retrying.
- Confirm idempotency fingerprint consistency.
- Record the final controlled outcome as succeeded, failed, reconciliation required, or manual review required.

## Verification Failure

Severity: high

Check:

- `verification.failed` events
- Report invariant failures and per-case verification fields
- Retrieval or provider evidence used for verification

Operator action:

- Treat failed or insufficient verification as non-success.
- Inspect source data freshness and document identity normalization.
- Re-run with isolated synthetic data when possible.

## Audit Integrity Failure

Severity: critical

Check:

- Audit validator output for hash mismatch, missing mandatory events, sequence gaps, or duplicate terminal events
- Related report summaries for `audit_integrity_failures`
- Recent deployment or migration changes

Operator action:

- Preserve existing audit storage before remediation.
- Stop affected workflow writes if integrity cannot be trusted.
- Rebuild derived views only from verified append-only source events.

## Authentication Or Authorization Failure Spike

Severity: high

Check:

- Security telemetry by bounded `event_type` and `result`
- Structured events such as `security.authentication_failed` and `security.authorization_denied`
- Recent configuration and credential rotation changes

Operator action:

- Do not log or paste tokens, authorization headers, cookies, or raw `.env` values.
- Verify whether the spike is expected after deployment, expiry, or configuration change.
- Keep denied operations denied; do not bypass authorization to recover traffic.

## Provider Cost Spike

Severity: warning or high depending on budget impact

Check:

- Provider call and cost totals by bounded provider, operation, and cost component
- Evaluation runner report summaries
- Recent task scope changes that enabled paid providers

Operator action:

- Disable nonessential paid-provider runners.
- Confirm local/offline mode for tests that should not incur cost.
- Preserve reports that explain call counts and cost reconciliation.

## Queue Backlog

Severity: warning

Check:

- Queue backlog metric, once workers are introduced
- Job started/completed/failed events
- Worker health and dependency readiness

Operator action:

- Do not blindly redeliver non-idempotent jobs.
- Verify idempotency and external side-effect state before retry.
- Scale workers only after confirming dependency health.

## Trace Exporter Outage

Severity: warning unless paired with workflow failures

Check:

- Observability diagnostics endpoint or command, once implemented
- Collector/exporter logs
- Application readiness separately from observability backend status

Operator action:

- Keep application safety controls active.
- Do not convert unknown business outcomes into success because tracing is unavailable.
- Restore tracing backend independently of primary readiness.

## Log Backend Outage

Severity: warning unless audit or reports are affected

Check:

- Local structured stdout/stderr availability
- Collector or log backend status
- Report persistence status

Operator action:

- Keep audit and report persistence policies intact.
- Do not disable redaction or validation to restore log flow.
- Preserve local evidence needed for post-incident review.
