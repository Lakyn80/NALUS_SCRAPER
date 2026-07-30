# NALUS Scraper Agent Rules

These rules apply to all coding-agent work in this repository.

## Project Variables

- Primary repository: `C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper`
- Additional repositories: none unless explicitly named by the user
- Primary branch expectation: `main`
- Application type: Python RAG/data pipeline with FastAPI API, CLI runners, Qdrant, Redis, Prometheus, Grafana, and offline evaluation/reporting artifacts
- Production environment identifier: `production` or `prod`
- Test/demo identifiers: `test`, `local`, `demo`, `staging`
- Generated report directories: `artifacts/`, `app/artifacts/`, `test-results/`, and task-specific runner output directories
- Push allowed: no, unless the user explicitly asks for a push

## Critical Workflows

- NALUS and NSoud scraping, extraction, normalization, and batch persistence
- RAG ingestion into Qdrant, including deterministic chunking and idempotent point IDs
- Retrieval and answer workflows exposed through `app/api/rag_router.py`
- Document-level and constraint-aware retrieval, both gated by explicit feature flags
- Offline legal QA, answer evaluation, and document-retrieval benchmark runners
- Evaluation metrics export through the existing Prometheus/Grafana stack
- Docker Compose startup for API, Qdrant, Redis, exporter, Prometheus, and Grafana

## Critical Invariants

- Do not weaken authentication, authorization, tenant or ownership checks, validation, or feature-flag guards to make a task pass.
- Existing stable retrieval paths must remain backward compatible unless the user explicitly asks for a behavior change.
- Disabled-by-default experimental paths must stay disabled by default.
- No critical state-changing operation may be reported as successful without required verification.
- Unknown external side-effect results must not be retried blindly; use reconciliation or a controlled failure state.
- Generated reports must include stable machine-readable summaries and must not be marked pass when their own invariants fail.
- Metric labels must remain low-cardinality. Never use query text, document IDs, ECLIs, tenant IDs, user IDs, correlation IDs, request IDs, trace IDs, job IDs, or error messages as labels.
- Secrets, tokens, authorization headers, raw environment values, full idempotency keys, private prompts, unbounded request/response bodies, and unnecessary personal data must not be logged, traced, reported, or exported.
- Observability backend outages must not bypass application safety controls.
- Business-relevant audit records, when added, must be append-only and integrity-verifiable.
- Failure injection must be disabled by default and impossible to enable accidentally in production.

## Canonical Documentation

Keep project handoff information in exactly two root files:

- `AGENTS.md` is the canonical coding-agent rulebook: workflow rules, invariants, validation policy, and git policy.
- `PROJECT_PROGRESS.md` is the canonical chronological handoff log: current state, completed tasks, validation results, known limitations, and the next recommended task.

Do not create or revive parallel root handoff files such as `AGENT.md`, `readme.dev`, `HANDOFF.md`, or task-specific status files for the same purpose. If a task changes durable behavior, update `PROJECT_PROGRESS.md`. If a task changes agent operating rules, update `AGENTS.md`. Technical documentation for a feature belongs under `docs/`.

## Mandatory Pre-Task Audit

Before changing code for non-trivial work, inspect:

- `README.md`
- `PROJECT_PROGRESS.md`
- relevant files under `docs/`
- relevant API, RAG, observability, monitoring, Docker, and test files

Record the current state with:

```powershell
git branch --show-current
git rev-parse HEAD
git status --short
git log -8 --oneline --decorate
```

Report pre-existing changes before editing. Do not overwrite, discard, stage, or commit unrelated changes.

## Observability Implementation Policy

Reuse existing mechanisms before adding new infrastructure:

- Logging starts at `app/core/logging.py`.
- Lightweight trace events start at `app/core/tracing.py`.
- Evaluation metrics are exported from `app/observability/eval_metrics_exporter.py`.
- Constraint retrieval metrics are in `app/observability/constraint_retrieval_metrics.py`.
- Prometheus and Grafana configuration lives under `monitoring/`.

Do not introduce a second Grafana, Prometheus, collector, log backend, tracing backend, dashboard stack, or reporting framework unless the existing stack cannot satisfy the requirement and the reason is documented.

Implement production observability in atomic phases:

1. Correlation context, structured logging, and redaction.
2. Distributed tracing or a documented OpenTelemetry migration path.
3. Centralized logs, if not already provided by the deployment environment.
4. Self-validating reports with stable summaries.
5. Audit integrity for business-relevant state transitions.
6. Domain and data-quality invariant validation.
7. Metrics, recording rules, SLOs, alerts, and dashboards.
8. Controlled failure injection and live failure scenarios.
9. Documentation and full validation.

Keep the system runnable after each phase. Do not leave partially wired production dependencies.

## Validation Expectations

Use the repository-standard focused tests first, then broader validation when the blast radius justifies it. For Python changes, prefer:

```powershell
python -m compileall app tests scripts
python -m pytest -q
git diff --check
```

Run narrower test selections when a full test run would be unnecessarily slow or dependent on unavailable local services, and clearly report what was and was not executed.

## Git Policy

Use focused commits only when the user asks for commits. Before any commit, inspect:

```powershell
git status --short
git diff --stat
git diff --check
git diff --cached --stat
git diff --cached --check
```

Do not commit generated reports, runtime data, environment files, logs, caches, PID/state files, database data, queue data, metrics data, trace data, or container artifacts.

Do not use `git reset --hard`, `git clean`, force push, or automatic rebase unless the user explicitly asks for that operation.
