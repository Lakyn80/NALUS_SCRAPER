# Project Progress

## 2026-07-23 Europe/Moscow - Task: Universal Verified Legal Retrieval v2 foundation

- Goal:
  Add disabled-by-default runtime and evaluation foundations for Universal Verified Legal Retrieval v2: paragraph-aware legal document structure, deterministic parsing, hierarchical child chunks, parent evidence windows, versioned indexing contract, universal QuerySpec v2, final semantic verifier interface, deterministic fail-closed gate, diagnostics, tests, hard-negative fixtures, and an offline comparison report writer.
- Scope:
  Additive backend/evaluation code only. No production frontend switch, no active production retrieval profile change, no current Qdrant collection or BM25 sidecar overwrite, no external or paid LLM provider calls, no commit, and no push.
- Starting audit:
  Branch `main`, HEAD `017c1957935cf1ab71a7eedaa479122a284ffcfb`.
  Pre-existing untracked generated artifacts were present under `artifacts/evaluation_quality/citizenship_query_retrieval_audit_20260712.*` and `artifacts/rag_eval/legal_qa/answer_eval/{mixed_document_gold_default,nsoud_document_gold_default,usoud_document_gold_default}/`.
  Recent HEAD history started with `017c195 feat(observability): add correlation context and structured logging`, followed by `333ce2f Add observability engineering guardrails`, `9b29ad5 Add Docker registry publishing support`, `2e45e98 Add verified and full document retrieval APIs`, `069d7ca Harden production retrieval candidate selection`, `aa63a3a Add offline document retrieval benchmark`, `33b1711 Add document-level retrieval pipeline`, and `71755ca Enable evidence windows for document-gold evaluation`.
- What changed:
  Added `app/rag/legal_v2/models.py` with stable paragraph/chunk IDs, section enum, paragraph metadata provenance, document reconstruction, and parsing diagnostics.
  Added `app/rag/legal_v2/parser.py` with deterministic line-ending normalization, numbered-paragraph detection, heading and section transitions, damaged-format fallback segmentation, boilerplate/citation classification, source offsets, source order, and diagnostics.
  Added `app/rag/legal_v2/chunking.py` with paragraph-aware child chunks, sentence-aware splitting for overlong paragraphs, complete paragraph/sentence overlap, no incompatible section crossing, parent evidence windows, deterministic IDs, source spans, paragraph text maps, and reconstruction.
  Added `app/rag/legal_v2/indexing.py` with disabled v2 indexing contract and proposed collection/profile `nalus_legal_paragraph_chunks_v2` plus BM25 sidecar id `nalus_legal_paragraph_bm25_v2`.
  Added `app/rag/legal_v2/query_spec.py` with a universal typed QuerySpec v2 contract preserving `original_query`, `normalized_query`, `structured_query`, and entity-preserving `retrieval_queries`.
  Added `app/rag/legal_v2/verifier.py` with provider-agnostic structured verifier interface, deterministic fake verifier, strict output validation, evidence paragraph validation, and deterministic gate.
  Added `app/rag/legal_v2/diagnostics.py` with bounded runtime diagnostic payloads and explicit Prometheus label-safety flags.
  Added `app/rag/legal_v2/evaluation.py` with offline comparison metrics and JSON/Markdown report writer for pass/failure/blocked/exception states.
  Added focused tests under `tests/rag/test_legal_v2_*.py` and hard-negative fixture `tests/fixtures/legal_v2_hard_negatives.jsonl`.
  Generated seed offline comparison artifact under `artifacts/rag_eval/legal_v2_seed_comparison_20260723/`.
- Tests and validation run:
  `python -m compileall app tests` -> passed.
  `python -m pytest -q tests\rag\test_legal_v2_parser_chunking.py tests\rag\test_legal_v2_query_spec.py tests\rag\test_legal_v2_verifier.py tests\rag\test_legal_v2_evaluation.py` -> `17 passed`.
  `python -m pytest -q tests\rag\test_full_document_retrieval.py tests\rag\test_document_retrieval.py tests\rag\test_constraint_pipeline.py tests\rag\test_constraint_verification.py tests\rag\test_production_bge_m3_profile.py` -> `45 passed`.
  `ruff check app\rag\legal_v2 tests\rag\test_legal_v2_parser_chunking.py tests\rag\test_legal_v2_query_spec.py tests\rag\test_legal_v2_verifier.py tests\rag\test_legal_v2_evaluation.py --no-cache` -> passed.
  `mypy app\rag\legal_v2` -> passed with no issues in 9 source files.
  `git diff --check` -> passed.
- Offline seed comparison:
  Tiny deterministic seed only; no production readiness claimed.
  `current_production_chunks`: candidate_recall `1.0`, exact_precision `0.25`, hard_negative_false_positives `1`, verified_document_precision `0.333`.
  `paragraph_child_chunks`: candidate_recall `1.0`, exact_precision `0.5`, hard_negative_false_positives `0`, verified_document_precision `0.0`.
  `paragraph_child_parent_windows`: candidate_recall `1.0`, exact_precision `0.5`, hard_negative_false_positives `0`, verified_document_precision `1.0`.
- Behavior preserved:
  Existing production retrieval config remains on `nalus_bge_m3_dense_bm25_rrf_v1` / `nalus_bge_m3_chunks_v1`. No API route, frontend, active collection alias, embedding model, Qdrant data, BM25 scoring, RRF behavior, Redis/cache behavior, or provider configuration was changed.
- Known limitations:
  The parser and QuerySpec extraction are conservative deterministic first-pass helpers, not a complete legal NLP system.
  The LLM verifier is an interface plus deterministic fake provider; no paid/external provider is enabled.
  The seed comparison is intentionally small and synthetic, useful for contract validation only.
- Next recommended task:
  Add an offline v2 index builder that reads reviewed legal documents, writes only the new `nalus_legal_paragraph_chunks_v2` payload format, and runs the comparison runner on a curated gold dataset before any production activation discussion.

## 2026-07-22 Europe/Moscow - Task: Implement Observability Phase A runtime foundation

- Goal:
  Implement executable Phase A observability primitives: canonical correlation context, request ID generation, structured JSON-compatible logging, centralized redaction, FastAPI middleware, context cleanup, and tests.
- Scope:
  Runtime observability only. No OpenTelemetry, Loki, Tempo, Grafana service, alerting, audit hash chain, failure injection, retrieval ranking, query rewriting, API response contract, or business logic changes were intended.
- What changed:
  Added `app/core/context.py` with contextvars-backed `correlation_id`, `request_id`, `operation_id`, `workflow_id`, `job_id`, and `task_id`, secure generated IDs, inbound ID validation, explicit binding/reset helpers for future workers, and cleanup.
  Added `app/core/redaction.py` with recursive redaction for mappings, nested structures, lists, tuples, Pydantic models, dataclasses, headers, exceptions, structured extras, string secret patterns, and stable idempotency-key fingerprinting.
  Extended `app/core/logging.py` with automatic context enrichment through a LogRecordFactory, JSON log formatting controlled by `LOG_FORMAT=json` or `LOG_JSON=1`, structured fields, redacted extras, and duplicate-handler protection.
  Updated `app/core/tracing.py` so trace payloads are redacted while preserving existing trace formatting compatibility.
  Added `app/api/middleware.py` and installed it in `app/api/main.py` and `app/api_app.py`. The middleware accepts valid `X-Correlation-ID`, replaces invalid or missing values, always generates `X-Request-ID`, returns both response headers, logs request start/completion/failure events, and clears context in `finally`.
  Added deterministic Phase A tests for context validation/binding/cleanup, middleware response headers and leakage protection, concurrent request isolation, structured log enrichment, JSON log validity, redaction coverage, idempotency key safety, and duplicate handler protection.
  Made `tests/api/test_main_startup.py` timing deterministic under the new middleware and annotated `app/api/rag_router.py` query-cache state for the touched `app/api/main.py` mypy check.
  Updated `docs/OBSERVABILITY_CONTRACT.md` with concise Phase A runtime usage.
- Tests and validation run:
  `python -m pytest -q tests/test_observability_context.py tests/test_redaction.py tests/test_structured_logging.py tests/api/test_observability_middleware.py` -> `34 passed`, one Starlette/httpx deprecation warning.
  `python -m pytest -q tests/test_core_tracing.py tests/api/test_main_startup.py tests/api/test_rag_api.py` -> `70 passed`, one Starlette/httpx deprecation warning.
  `python -m compileall app tests` -> passed.
  `ruff check app/core/context.py app/core/redaction.py app/core/logging.py app/core/tracing.py app/api/middleware.py app/api/main.py app/api_app.py tests/test_observability_context.py tests/test_redaction.py tests/test_structured_logging.py tests/api/test_observability_middleware.py tests/api/test_main_startup.py --no-cache` -> passed.
  `mypy app/core/context.py app/core/redaction.py app/core/logging.py app/core/tracing.py app/api/middleware.py app/api_app.py` -> passed.
  `mypy app/api/main.py` -> passed after annotating query-cache state.
- Runtime smoke:
  Inline `TestClient(app.api_app.app)` smoke hit `/docs` twice. First request used inbound `X-Correlation-ID=smoke-12345678` and authorization/cookie header secrets; response returned the same correlation ID and an `X-Request-ID`. Second request generated a different correlation ID and a fresh request ID. Captured middleware events were `http.request.started` and `http.request.completed` for both requests. Middleware logs did not contain the header secret, and `get_context()` was fully empty after both requests.
- Blocked broader validation:
  `ruff check app tests --no-cache` currently fails on unrelated pre-existing lint findings across older modules/tests.
  `mypy app` currently fails on unrelated pre-existing missing stubs and type issues across older modules; the new Phase A modules pass narrow mypy.
- Known limitations:
  Phase A does not add distributed tracing, collector/exporter backends, central log aggregation, alert rules, audit chains, failure injection, or diagnostics endpoints.
  Middleware logs safe path/route only and intentionally does not log request bodies, raw query strings, authorization headers, cookies, or request payloads.
- Next recommended task:
  Start Phase B by designing an OpenTelemetry-compatible tracing plan that maps the existing lightweight `trace_event` call sites to bounded spans without adding a duplicate monitoring stack.

## 2026-07-22 Europe/Moscow - Task: Set production observability and reliability guardrails

- Goal:
  Adapt the supplied universal production observability, reliability, audit, and failure-detection prompt into durable repository rules before continuing with implementation work.
- Scope:
  Documentation and agent guardrails only. No runtime code, API behavior, retrieval ranking, Qdrant data, Docker topology, Prometheus scrape config, Grafana dashboards, generated reports, or feature flags were changed.
- What changed:
  Added root `AGENTS.md` with project variables, critical workflows, critical invariants, mandatory pre-task audit steps, observability implementation policy, validation expectations, and git policy.
  Added `docs/OBSERVABILITY_CONTRACT.md` with the project-specific observability contract, current stack inventory, correlation/logging/tracing/metrics/reporting/audit/failure-injection requirements, and a phased implementation backlog.
  Added `.cursor/rules/observability-reliability.mdc` so Cursor also picks up the same guardrails without duplicating the full contract.
  Added `docs/runbooks/observability-incidents.md` with initial operator guidance for duplicate side effects, reconciliation, verification failures, audit integrity failures, authentication/authorization spikes, provider-cost spikes, queue backlog, trace exporter outage, and log backend outage.
- Starting state:
  Branch `main`, HEAD `aa63a3a05b81e8555a3c84da351d6f1ac2faa8e3`.
  The worktree already had pre-existing modified and untracked files before this setup.
  Existing stack inspection found classic text logging in `app/core/logging.py`, lightweight trace events in `app/core/tracing.py`, Prometheus/Grafana under `monitoring/`, and existing observability tests/exporters under `app/observability/` and `tests/observability/`.
- Tests run:
  Documentation-only setup; no application tests were run.
  `git diff --check` should be run before committing this together with any future implementation phase.
- Known limitations:
  This setup does not implement structured JSON logging, OpenTelemetry, audit hash chains, report validators, alert rules, controlled failure injection, or live observability runners. Those remain phased implementation work and must be added with focused tests.
- Next recommended task:
  Start Phase A: canonical correlation context, structured logging, and centralized redaction, reusing `app/core/logging.py` and existing API boundaries.

## 2026-07-13 11:20 Europe/Moscow — Task: Add disabled constraint-aware verified document retrieval

- Goal:
  Add an additive backend retrieval path that can interpret structured legal query constraints, verify candidate documents against bounded full-document evidence, and reject partial/contradictory matches without changing the stable MVP chunk-level retrieval flow.
- Scope:
  Backend retrieval/config/API/observability/tests/docs only. The frontend, BGE-M3 embeddings, BM25, RRF, Qdrant collection/data, ingestion, Redis/cache behavior, query rewrite, answer generation, and existing `/api/rag/retrieve` and `/api/rag/query` response contracts were not changed.
- What changed:
  Added typed constraint models, validated config, deterministic structured-query interpretation, deterministic constraint verification, and an additive pipeline in `app/rag/retrieval/`.
  Added `POST /api/rag/retrieve-verified`, disabled by default through `NALUS_CONSTRAINT_RETRIEVAL_ENABLED=0`.
  The new endpoint groups candidate chunks by canonical document id, reconstructs bounded full-document text using the existing read-only full-document store, verifies hard constraints, and returns only verified unique documents.
  Added strict behavior for hard constraints: mismatch or not-proven excludes a document in strict mode, and no hidden threshold lowering or unrelated fallback is applied.
  Added Prometheus metrics through the existing metrics stack with bounded labels only: endpoint, status, decision status, constraint category, verification status, and method.
  Added config examples to `.env.example` and Docker environment defaults to keep the feature disabled unless explicitly enabled.
  Added `docs/CONSTRAINT_AWARE_RETRIEVAL.md` and a manually reviewable seed dataset fixture for future evaluation work.
- Why it changed:
  Previous failures showed that lexical/chunk retrieval can return partial matches such as citizenship mentions without the requested nationality relation, or child-abduction country mentions without the requested destination/actor relation. The new module verifies material constraints at document level before returning results.
- Files changed:
  `app/rag/retrieval/constraint_models.py`
  `app/rag/retrieval/constraint_config.py`
  `app/rag/retrieval/structured_query.py`
  `app/rag/retrieval/constraint_verification.py`
  `app/rag/retrieval/constraint_pipeline.py`
  `app/observability/constraint_retrieval_metrics.py`
  `app/api/rag_router.py`
  `.env.example`
  `docker-compose.yml`
  `docs/CONSTRAINT_AWARE_RETRIEVAL.md`
  `tests/rag/test_structured_query.py`
  `tests/rag/test_constraint_verification.py`
  `tests/rag/test_constraint_pipeline.py`
  `tests/observability/test_constraint_retrieval_metrics.py`
  `tests/api/test_rag_api.py`
  `tests/fixtures/constraint_retrieval_seed_dataset.jsonl`
- Tests run:
  `python -m pytest tests/rag/test_structured_query.py tests/rag/test_constraint_verification.py tests/rag/test_constraint_pipeline.py tests/observability/test_constraint_retrieval_metrics.py -q` -> initial `1 failed, 11 passed`; fixed parent-role detection for forms such as `matkou`.
  `python -m pytest tests/rag/test_structured_query.py tests/rag/test_constraint_verification.py tests/rag/test_constraint_pipeline.py tests/observability/test_constraint_retrieval_metrics.py -q` -> `12 passed`.
  `python -m pytest tests/api/test_rag_api.py -q` -> `44 passed`.
  `python -m pytest tests/rag/test_document_retrieval.py tests/rag/test_full_document_retrieval.py -q` -> `19 passed`.
  `python -m pytest tests/api/test_rag_api.py tests/rag/test_structured_query.py tests/rag/test_constraint_verification.py tests/rag/test_constraint_pipeline.py tests/rag/test_document_retrieval.py tests/rag/test_full_document_retrieval.py tests/observability/test_constraint_retrieval_metrics.py -q` -> `75 passed`.
  `python -m compileall app\rag\retrieval\constraint_config.py app\rag\retrieval\constraint_models.py app\rag\retrieval\structured_query.py app\rag\retrieval\constraint_verification.py app\rag\retrieval\constraint_pipeline.py app\observability\constraint_retrieval_metrics.py app\api\rag_router.py` -> passed.
- Smoke result:
  Runtime Docker smoke was not run in this task. The endpoint is disabled by default, and focused API tests verify disabled behavior, successful verified retrieval when explicitly enabled, empty verified result without fallback, and provider failure as 503.
- Behavior preserved:
  Existing `/api/rag/retrieve` remains chunk-level and backward compatible. Existing `/api/rag/query` remains unchanged. Document-level ranking endpoint `/api/rag/retrieve-documents` remains separately gated. No embedding, ranking, Qdrant, BM25, RRF, Redis/cache, model-provider, or frontend behavior was changed.
- Known limitations:
  The first rollout is deterministic and conservative. It does not use an LLM verifier and does not claim absolute legal relevance. The seed dataset is for manual review and is not a gold benchmark. Full-document verification depends on reconstructable same-document chunks.
- Next recommended task:
  Run the disabled endpoint in an isolated environment on manually reviewed citizenship and child-abduction queries, then build a curated gold dataset before considering frontend exposure or production activation.

## 2026-07-13 09:45 Europe/Moscow — Task: Fix court filters and full-judgment result presentation for MVP search

- Goal:
  Fix broken Ústavní soud / Nejvyšší soud filters, stop presenting repeated chunk hits as separate decisions, and make full judgments accessible directly from each frontend result card while keeping MVP search on the original stable chunk-level retrieval path.
- Scope:
  Backend metadata/source filter recognition, frontend chunk-result grouping by document id, frontend inline full-judgment loading, focused tests, runtime smoke, and documentation only. Retrieval ranking, embeddings, BM25/RRF scoring, Qdrant data, Redis/cache behavior, and the disabled additive document-level ranking endpoint were not changed.
- What changed:
  Updated `app/api/rag_router.py` so court/source filtering recognizes `source`, `court`, `court_name`, `document_id`, `source_document_id`, `case_reference`, `reference`, and ECLI prefixes such as `ECLI:CZ:US` / `ECLI:CZ:NS`.
  Updated retrieved chunk response projection to infer `court_name` from metadata/ECLI when explicit court metadata is missing.
  Added endpoint regression tests for `usoud / nalus`, ECLI-only ÚS results, `nsoud`, and ECLI-only NS results.
  Updated NalusFE chunk mapping to group chunk-level results by canonical `documentId`, merge supporting passages, preserve best score, and fill court/ECLI from document identity when metadata is incomplete.
  Added inline full-judgment loading in each result card through the existing read-only `GET /api/retrieval/documents/{document_id}` proxy.
  Changed the results heading from "Nalezená relevantní rozhodnutí" to "Nalezená rozhodnutí" and clarified that ordering is technical relevance, not a legal-relevance guarantee.
  Updated NalusFE README.
- Why it changed:
  The attached frontend output showed repeated chunk hits from the same decisions, missing court/ECLI labels, broken court filters returning zero results, and result cards showing only passages while full judgment text required opening a separate detail page.
- Tests run:
  Backend: `python -m pytest tests/api/test_rag_api.py tests/rag/test_full_document_retrieval.py tests/rag/test_document_retrieval.py -q` -> `59 passed`.
  Frontend: `npm run typecheck` -> passed.
  Frontend: `npm run lint` -> passed.
  Frontend Docker build during `docker compose up -d --build frontend` -> passed.
- Smoke result:
  Backend direct `POST /api/rag/retrieve` for `udělení českého občanstvi ruskému občanu`: no source filter -> 50 chunks; `sources=["constitutional"]` -> 50 chunks; `sources=["supreme"]` -> 0 chunks in the current ÚS/NALUS collection.
  Frontend proxy `POST /api/retrieval/documents`: `court=all` -> 19 unique documents; `court=usoud` -> 22 unique documents; `court=nsoud` -> 0 results.
  Frontend `/vyhledavani?q=udělení českého občanstvi ruskému občanu` returned HTTP 200, rendered `Nalezená rozhodnutí`, included `Zobrazit celý rozsudek zde`, and included the known document `ECLI:CZ:US:2023:3.US.3469.22.1`.
  Frontend full-document proxy `GET /api/retrieval/documents/ECLI%3ACZ%3AUS%3A2023%3A3.US.3469.22.1` returned HTTP 200, `full_text_availability_status=available`, `chunk_count=12`, and full text length `15734`.
  Backend `POST /api/rag/retrieve-documents` returned HTTP 404, confirming the unfinished document-level ranking endpoint remains disabled.
- Behavior preserved:
  Search still uses the original MVP `/api/rag/retrieve` flow. BGE-M3 embeddings, BM25, RRF, top-k request size, Qdrant collection/data, Redis/cache behavior, query rewrite behavior, answer generation, and document-level ranking feature flag default were not changed.
- Known limitations:
  The remaining irrelevant/weak results are a retrieval-quality issue in the original chunk-level ranking, not a frontend rendering bug. This task reduces duplicate clutter and fixes filters/metadata but does not add calibrated legal relevance filtering.
  Current runtime collection appears to be ÚS/NALUS-focused for this query; `court=nsoud` correctly returns no results under the current source filter.
- Next recommended task:
  Add a separate relevance-calibration task for the citizenship/Russian-citizen query: reviewed gold set, query expansion/synonyms, safe thresholding, duplicate-aware document ranking, and explicit rollout criteria before changing production ranking.

## 2026-07-13 03:40 Europe/Moscow — Task: Show full judgments in frontend while keeping MVP chunk-level search stable

- Goal:
  Keep the unfinished additive document-level ranking module disabled for MVP search, but allow the frontend to display full judgments instead of only citations/passages when a user opens a result detail.
- Scope:
  Added a read-only document-by-id reconstruction endpoint, frontend full-document proxy/detail rendering, focused tests, and documentation. Search ranking remains the stable chunk-level `/api/rag/retrieve` path.
- What changed:
  Added `app/rag/retrieval/full_document.py` with typed read-only Qdrant full-document reconstruction from same-document chunks ordered by `chunk_index`, document id validation, bounded chunk count, metadata normalization, explicit availability status, and diagnostics.
  Added `GET /api/rag/documents/{document_id}` to `app/api/rag_router.py`.
  Kept `POST /api/rag/retrieve-documents` disabled by default; smoke verified it returns HTTP 404 with `NALUS_DOCUMENT_RETRIEVAL_ENABLED=0`.
  Updated API logging for `/search`, `/retrieve`, and `/query` to log query length instead of raw query text.
  Added backend tests for full-document reconstruction and endpoint success/error paths.
  Updated NalusFE types/parsing and added `GET /api/retrieval/documents/{id}` as a Next.js proxy to the backend full-document endpoint.
  Updated NalusFE search mapping so result detail links use the canonical `documentId`, not `documentId#chunkId`.
  Updated the result detail tab and `/rozhodnuti/[id]` page to render full judgment text from the backend full-document endpoint.
  Increased NalusFE search proxy timeout to 60 seconds to tolerate cold start of the existing backend retrieval stack.
  Updated `docs/DOCUMENT_LEVEL_RETRIEVAL.md` and NalusFE `frontend/README.md`.
- Why it changed:
  Stable MVP search must not switch to the unfinished document-level ranking module, but the UI still needs to show complete judgments. The safe boundary is chunk-level retrieval for search plus read-only full-document reconstruction by already-known document id.
- Files changed:
  `app/rag/retrieval/full_document.py`
  `app/api/rag_router.py`
  `tests/rag/test_full_document_retrieval.py`
  `tests/api/test_rag_api.py`
  `docs/DOCUMENT_LEVEL_RETRIEVAL.md`
  NalusFE `frontend/src/types/retrieval.ts`
  NalusFE `frontend/src/lib/api/responseValidation.ts`
  NalusFE `frontend/src/lib/api/fullDocumentServer.ts`
  NalusFE `frontend/src/app/api/retrieval/documents/[id]/route.ts`
  NalusFE `frontend/src/lib/api/documentSearchServer.ts`
  NalusFE `frontend/src/lib/api/judgmentMapping.ts`
  NalusFE `frontend/src/components/ResultDetailTabs.tsx`
  NalusFE `frontend/src/app/rozhodnuti/[id]/page.tsx`
  NalusFE `frontend/README.md`
- Tests run:
  Backend: `python -m pytest tests/rag/test_full_document_retrieval.py tests/api/test_rag_api.py tests/rag/test_document_retrieval.py -q` -> `57 passed`.
  Frontend: `npm run typecheck` -> passed.
  Frontend: `npm run lint` -> passed.
  Frontend Docker build during `docker compose up -d --build frontend` -> passed.
- Smoke result:
  Backend `/health` returned `status=ok` and `orchestrator_ready=true`.
  Backend `GET /api/rag/documents/ECLI%3ACZ%3AUS%3A2026%3A3.US.446.26.1` returned HTTP 200, `full_text_availability_status=available`, `chunk_count=16`, and `full_text` length `21768`.
  Frontend proxy `GET /api/retrieval/documents/ECLI%3ACZ%3AUS%3A2026%3A3.US.446.26.1` returned the same full text length and chunk count.
  Frontend detail page `/rozhodnuti/ECLI%3ACZ%3AUS%3A2026%3A3.US.446.26.1` returned HTTP 200 and rendered full judgment content.
  Frontend search proxy `POST /api/retrieval/documents` returned HTTP 200 with 50 stable chunk-level results after backend restart.
- Behavior preserved:
  Search retrieval ranking, top-k, BGE-M3 embeddings, BM25 scoring, RRF, Qdrant collection/data, Redis/cache behavior, query rewrite behavior, answer generation, and the disabled document-level ranking endpoint default were not changed.
- Known limitations:
  Full-document reconstruction depends on same-document chunks having reliable document identifiers and preferably contiguous `chunk_index` values. If indexes are missing or duplicated, the endpoint returns `partial` with diagnostics instead of hiding the issue.
  Uvicorn access logs include the request path, so document ids in the URL can appear in access logs. No full document text is logged and the updated application search logs do not include raw queries.
- Next recommended task:
  Add a small frontend component test or Playwright smoke for opening "Celý dokument" from a search result once the frontend test harness is introduced.

## 2026-07-13 03:35 Europe/Moscow — Task: Restore stable MVP chunk-level retrieval flow

- Goal:
  Disable runtime use of the unfinished additive document-level retrieval module and return the NALUS MVP/frontend path to the stable chunk-level retrieval endpoint while keeping the new module available for separate tuning.
- Scope:
  Runtime/configuration and frontend proxy routing only. The document-level module code, BGE-M3 embeddings, Qdrant collections/data, BM25 scoring, RRF fusion, Redis/cache behavior, ingestion, thresholds, and retrieval algorithms were not changed.
- What changed:
  Set the Docker default `NALUS_DOCUMENT_RETRIEVAL_ENABLED` back to `0`.
  Kept document-level limit variables documented/configured as disabled-by-default knobs, so the module can be enabled later through an explicit rollout task.
  Updated `docs/DOCUMENT_LEVEL_RETRIEVAL.md` to state that MVP runtime should use the stable chunk-level flow while document-level retrieval is tuned separately.
  Updated NalusFE server-side proxy to call `POST /api/rag/retrieve` with `top_k=50` instead of `POST /api/rag/retrieve-documents`.
  Added frontend parsing/mapping for the stable chunk-level backend response `{ "results": [...] }`.
  Updated NalusFE README to document the stable MVP backend flow.
- Why it changed:
  The forensic audit showed that the document-level endpoint was already active in MVP runtime although the module still lacks calibrated relevance policy, metadata normalization, and full-document endpoint support. With threshold `0.0`, unrelated documents could appear in the top 50 aggregated results.
- Tests run:
  Backend: `python -m pytest tests/rag/test_document_retrieval.py tests/api/test_rag_api.py -q` -> `44 passed`.
  Frontend: `npm run typecheck` -> passed.
  Frontend: `npm run lint` -> passed.
  Frontend Docker build during `docker compose up -d --build frontend` -> passed.
- Smoke result:
  Recreated backend API container and verified `NALUS_DOCUMENT_RETRIEVAL_ENABLED=0`.
  Direct backend `POST /api/rag/retrieve-documents` returned HTTP 404 as expected.
  Direct backend `POST /api/rag/retrieve` returned 50 stable chunk-level results for the citizenship query.
  Rebuilt/recreated the NalusFE Docker container and verified `POST /api/retrieval/documents` returned 50 frontend results mapped from chunk-level backend results, with no document-level diagnostics payload.
- Known limitations:
  The first backend request after container restart can exceed the current frontend timeout because the existing retrieval path may cold-load BGE-M3 and attempt the configured query rewrite provider. A warm request succeeded. This task did not change model loading, query rewrite, or timeout policy.
  The frontend route name remains `/api/retrieval/documents` for UI compatibility, but its backend source is now chunk-level MVP retrieval.
- Next recommended task:
  Continue tuning document-level retrieval behind the disabled feature flag: relevance policy, metadata normalization, full-document endpoint, benchmark calibration, and explicit rollout criteria.

## 2026-07-13 03:10 Europe/Moscow — Task: Citizenship query retrieval forensic audit

- Goal:
  Perform a read-only forensic audit of the current NALUS document retrieval quality, metadata normalization, query rewrite behavior, ranking diagnostics, and full-document availability for the query `najdi rozsudek ústavního soudu o udělování českého občanství ruským občanům`.
- Scope:
  Audit only. Retrieval algorithms, BGE-M3 embeddings, BM25, RRF, Qdrant collections/data, aliases, ingestion, Redis, thresholds, backend API behavior, and frontend files were not changed.
- What changed:
  Created `artifacts/evaluation_quality/citizenship_query_retrieval_audit_20260712.md`.
  Created `artifacts/evaluation_quality/citizenship_query_retrieval_audit_20260712.json`.
  Updated this progress file with the audit result.
- Findings:
  Direct backend reproduction returned 50 final document-level results from 200 RRF candidate chunks and 142 unique documents before the final `max_returned_documents=50` cap.
  Query rewrite is wired in runtime through `QueryRewriteService`, but the configured DeepSeek provider returned 401/empty output during reproduction, so the effective retrieval query remained the original query.
  The configured document relevance threshold is `0.0`; therefore unrelated documents are not filtered by relevance threshold and can remain in the top 50 if their RRF/document score is high enough.
  Deterministic content classification found 7 potentially relevant citizenship-related results and 43 clearly irrelevant results; no result was classified as clearly relevant to the narrower Russian-citizen citizenship query.
  `ECLI:CZ:US:2026:3.US.446.26.1` was returned because BM25 ranked same-document chunks at rank 21; the matching text is about municipal referendum/spatial planning and contains weak lexical overlap such as `občanům`, not citizenship granting.
  `ECLI:CZ:US:2026:4.US.893.26.1` was returned because dense retrieval ranked the opening chunk at rank 27; the document concerns parental responsibility/care of a minor child, not citizenship granting.
  Final document IDs were unique and no evidence of cross-judgment chunk mixing was found in final aggregation.
- Metadata audit:
  The observed `Neuvedený soud` / `ECLI: neuvedeno` defect for `ECLI:CZ:US:2026:3.US.446.26.1` originates in backend response metadata availability/projection: the best returned chunks expose `document_id`, `source_document_id`, and `decision_date`, but not `court`, `ecli`, or `case_reference`.
  The Next.js proxy did not corrupt the data. The frontend mapper deterministically falls back to `Neuvedený soud`, undefined ECLI, and `document_id` as case reference when backend metadata is missing.
- Full-text availability:
  For both unrelated examples and inspected final documents, complete judgment text is available as same-document Qdrant chunks mirrored in the BM25 sidecar when chunk indexes are contiguous.
  `ECLI:CZ:US:2026:3.US.446.26.1` has 16 Qdrant/BM25 chunks, indexes 0-15, no missing/duplicate indexes, and deterministic reconstruction is possible.
  `ECLI:CZ:US:2026:4.US.893.26.1` has 10 Qdrant/BM25 chunks, indexes 0-9, no missing/duplicate indexes, and deterministic reconstruction is possible.
- Tests run:
  `python -m pytest tests/rag/test_document_retrieval.py tests/api/test_rag_api.py -q` -> `44 passed`.
  Repeated `pytest-asyncio` loop-scope deprecation warning is non-blocking and unrelated.
- Smoke result:
  `docker compose ps` showed backend, Qdrant, Redis, Prometheus, Grafana, and the eval metrics exporter running.
  Direct backend `POST /api/rag/retrieve-documents` reproduced 50 results.
  Read-only frontend proxy `POST /api/retrieval/documents` confirmed the metadata fallback behavior for the two known unrelated examples.
- Behavior preserved:
  No retrieval/ranking change, no threshold change, no Qdrant write, no Redis/cache behavior change, no embedding change, no model download, no frontend file modification, and no new fallback was introduced.
- Known limitations:
  The classification is deterministic and human-auditable but conservative. It does not use an LLM and does not assert absolute legal relevance beyond observable document content and query-term/topic evidence.
  The audit generated runtime artifacts and intentionally did not commit them unless a later repository policy task requests that.
- Next recommended task:
  Implement a separate read-only full-document endpoint with canonical metadata normalization, ordered same-document chunk reconstruction, explicit full-text availability status, and regression tests before exposing document detail deep links.

## 2026-07-13 02:04 Europe/Moscow — Task: NalusFE document retrieval frontend integration

- Goal:
  Connect the existing NalusFE Next.js search interface to the additive document-level FastAPI retrieval endpoint without changing retrieval ranking, embeddings, Qdrant data, BM25, RRF, Redis, ingestion, or answer generation.
- What changed:
  Added Docker environment wiring so the existing `POST /api/rag/retrieve-documents` endpoint is enabled for the local integrated runtime.
  Updated `.env.example` to show `NALUS_DOCUMENT_RETRIEVAL_ENABLED=1` for the frontend integration path.
  Clarified `docs/DOCUMENT_LEVEL_RETRIEVAL.md` so the code default remains disabled while the integrated Docker runtime can explicitly enable the endpoint.
- Why it changed:
  The frontend integration uses a server-side Next.js proxy and must call the real document-level endpoint. The running backend container needs the existing feature flag enabled for end-to-end smoke tests and local presentation use.
- Files changed:
  `docker-compose.yml`
  `.env.example`
  `docs/DOCUMENT_LEVEL_RETRIEVAL.md`
  `PROJECT_PROGRESS.md`
- Tests run:
  NalusFE frontend: `npm run lint`, `npm run typecheck`, `npm run build`, and `npm audit --audit-level=moderate` all passed after adding a safe PostCSS override for the latest Next.js transitive dependency tree.
  Backend: `python -m pytest tests/rag/test_document_retrieval.py tests/api/test_rag_api.py -q` -> `44 passed`.
  Compose config validation passed for both NalusFE and nalus-scraper.
- Smoke result:
  Backend `/health` returned `status=ok` with `orchestrator_ready=true`.
  Backend `POST /api/rag/retrieve-documents` returned 50 real unique document-level results for a Czech legal query.
  NalusFE dev server and Docker-served frontend `POST /api/retrieval/documents` both returned 50 real backend-backed results with first document `ECLI:CZ:US:2026:2.US.98.26.1`.
  Docker-served `GET /vyhledavani?q=...` returned HTTP 200 and rendered backend-backed search content, not mock data.
  Invalid frontend API input checks returned controlled HTTP 400 errors for empty query and invalid filter values.
  The smoke run showed the existing backend query-rewrite path attempted the configured text LLM and fell back to the original query after a provider 401; the frontend did not call answer-generation or chat endpoints.
- Behavior preserved:
  Retrieval ranking, document scoring, BGE-M3 embeddings, BM25, RRF, Qdrant collections/data, Redis/cache behavior, ingestion, LLM/DeepSeek calls, and existing `/api/rag/retrieve` and `/api/rag/query` behavior were not changed.
- Known limitations:
  The backend still does not expose a separate document-detail-by-id endpoint; the frontend can render document details only from search results returned by `retrieve-documents`.
  The document aggregation module is no-LLM, but the endpoint obtains candidate chunks through the existing orchestrator retrieval path, including optional query rewrite when the backend is configured with a real text LLM provider.
- Next recommended task:
  Add a dedicated read-only document detail endpoint if the product needs stable deep links to individual judgments independent of a search response.

## 2026-07-10 15:25 Europe/Moscow — Task: NSoud provenance checker + conservative single gold annotation

- Goal:
  Build a read-only NSoud provenance checker for pending legal QA items, then apply only the single conservative NSoud gold annotation that passed the check.
- What changed:
  Added `scripts/check_nsoud_gold_provenance.py` and `tests/test_check_nsoud_gold_provenance.py`.
  Added `artifacts/rag_eval/legal_qa/nsoud_provenance_check_20260710.md`.
  Added `artifacts/rag_eval/legal_qa/annotations/nsoud_provenance_candidates_20260710.jsonl`.
  Updated `scripts/apply_gold_source_annotations.py` to annotate only `nsoud-qa-007`.
  Regenerated `artifacts/rag_eval/legal_qa/datasets/nsoud_qa_v1.jsonl`.
  Refreshed `artifacts/rag_eval/legal_qa/answer_eval/nsoud_no_llm_baseline/*`.
  Updated `artifacts/rag_eval/legal_qa/gold_source_review_20260709.md`.
  Updated `artifacts/rag_eval/legal_qa/answer_eval_report_20260709.md`.
  Added `PROJECT_EXECUTION_PROTOCOL.md` as the local execution protocol for this repo.
- Why it changed:
  Provenance extraction was no longer the blocker for NSoud pending questions. The checker was needed to separate technical provenance availability from true legal relevance. Only `nsoud-qa-007` met the conservative bar for direct gold annotation.
- Files changed:
  `PROJECT_EXECUTION_PROTOCOL.md`
  `PROJECT_PROGRESS.md`
  `scripts/check_nsoud_gold_provenance.py`
  `tests/test_check_nsoud_gold_provenance.py`
  `scripts/apply_gold_source_annotations.py`
  `artifacts/rag_eval/legal_qa/gold_source_review_20260709.md`
  `artifacts/rag_eval/legal_qa/datasets/nsoud_qa_v1.jsonl`
  `artifacts/rag_eval/legal_qa/answer_eval/nsoud_no_llm_baseline/*`
  `artifacts/rag_eval/legal_qa/answer_eval_report_20260709.md`
  `artifacts/rag_eval/legal_qa/nsoud_provenance_check_20260710.md`
  `artifacts/rag_eval/legal_qa/annotations/nsoud_provenance_candidates_20260710.jsonl`
- Tests run:
  `python -m pytest tests/test_check_nsoud_gold_provenance.py tests/rag/test_legal_qa_benchmark.py tests/rag/test_legal_answer_eval.py -q`
- Smoke result:
  Read-only Qdrant lookup succeeded via `docker compose exec -T api`.
  NSoud no-LLM answer eval rerun completed and refreshed `summary.json`.
- Known limitations:
  `nsoud-qa-007` increased NSoud gold coverage from `3/10` to `4/10`, but answer-support quality for that item still evaluates as `gap`.
  The remaining pending NSoud items still need manual relevance review before any further annotation.
  Existing uncommitted generated ÚS/mixed answer eval artifacts remain in the worktree and were not part of this task.
- Next recommended task:
  Review `nsoud-qa-001`, `002`, `005`, `006`, `008`, and `009` manually against `nsoud_provenance_check_20260710.md` and decide whether any should stay pending, be reformulated, or be rejected as benchmark questions.

## 2026-07-10 20:30 Europe/Moscow — Task: Legal answer eval metric semantics repair after failed-case diagnostics

- Goal:
  Repair the interpretation of offline legal answer-eval metrics so that reports clearly separate real failures, missing-gold non-evaluable items, usable partial support, corpus-only routing support, and true retrieval misses.
- What changed:
  Updated `app/rag/eval/legal_answer_eval.py` with explicit total/gold/missing-gold/evaluable fields, gold retrieval miss metrics, unsupported-risk rate, citation-available rate over gold, and corpus-routing support rate.
  Reworked failed-case categorization to use `not_evaluable_missing_gold`, `invalid_gold_annotation`, `true_retrieval_miss`, `usable_partial_support`, `weak_partial_support`, `unsupported_boilerplate_or_gap`, `corpus_only_no_document_citation_expected`, and `metric_denominator_warning`.
  Added conservative final-status logic (`PASS` / `WARN` / `FAIL` / `FAIL_WITH_REAL_NSOUD_RISK`) driven by real failure categories instead of strict-rate thresholds alone.
  Added dedicated `nsoud-qa-007` diagnostics with expected source, retrieved top-k ids, and conservative conclusion.
  Updated the Prometheus summary compatibility path in `app/observability/eval_metrics_exporter.py`.
  Regenerated `artifacts/evaluation_quality/*` and refreshed `artifacts/rag_eval/legal_qa/answer_eval_report_20260709.md`.
- Why it changed:
  The previous diagnostic outputs overstated failure severity by treating missing gold and corpus-only mixed cases as ordinary failures. The new semantics make the reports usable for decision-making without hiding the real NSoud risks.
- Files changed:
  `app/rag/eval/legal_answer_eval.py`
  `app/observability/eval_metrics_exporter.py`
  `scripts/run_legal_answer_eval.py`
  `scripts/generate_legal_answer_eval_diagnostics.py`
  `tests/rag/test_legal_answer_eval.py`
  `tests/rag/test_legal_answer_eval_diagnostics.py`
  `tests/observability/test_eval_metrics_exporter.py`
  `artifacts/evaluation_quality/*`
  `artifacts/rag_eval/legal_qa/answer_eval_report_20260709.md`
- Tests run:
  `python -m pytest tests/rag/test_legal_answer_eval.py -q`
  `python -m pytest tests/rag/test_legal_answer_eval_diagnostics.py -q`
  `python -m pytest tests/observability/test_eval_metrics_exporter.py -q`
- Smoke result:
  `python scripts/generate_legal_answer_eval_diagnostics.py --runs-dir artifacts/rag_eval/legal_qa/answer_eval --output-dir artifacts/evaluation_quality` completed successfully and produced updated JSON/Markdown diagnostics.
- Known limitations:
  The worktree still contains pre-existing dirty offline answer-eval artifacts under `artifacts/rag_eval/legal_qa/answer_eval/usoud_no_llm_baseline/*` and `mixed_no_llm_baseline/*`.
  No new commit was created in this task.
  `nsoud-qa-007` remains a conservative true retrieval miss in the current frozen baseline.
- Next recommended task:
  Review the NSoud criminal-dovolani benchmark questions around § 265b tr. ř., especially `nsoud-qa-007` and `nsoud-qa-010`, and decide whether the next action is query reformulation, gold refinement, or a separate retrieval-quality investigation.

## 2026-07-10 21:10 Europe/Moscow — Task: Read-only NSoud retrieval risk investigation for `nsoud-qa-007` and `nsoud-qa-010`

- Goal:
  Verify whether the post-diagnostics NSoud risk cases are true retrieval misses, provenance/export artifacts, or benchmark-design issues, without changing retrieval logic or retrieval data.
- What changed:
  Added `artifacts/evaluation_quality/nsoud_retrieval_risk_investigation_20260710.md`.
  Added `artifacts/evaluation_quality/nsoud_retrieval_risk_investigation_20260710.json`.
- Why it changed:
  The repaired diagnostics still flagged `FAIL_WITH_REAL_NSOUD_RISK`, but `nsoud-qa-007` and `nsoud-qa-010` needed direct read-only verification against Qdrant, BM25 sidecar contents, and current top-50 retrieval behavior.
- Files changed:
  `PROJECT_PROGRESS.md`
  `artifacts/evaluation_quality/nsoud_retrieval_risk_investigation_20260710.md`
  `artifacts/evaluation_quality/nsoud_retrieval_risk_investigation_20260710.json`
- Tests run:
  `python -m pytest tests/rag/test_legal_answer_eval.py tests/rag/test_legal_answer_eval_diagnostics.py -q`
- Smoke result:
  Read-only Qdrant inspection via `docker compose exec -T api` confirmed that `nsoud-qa-007` expected source `ECLI:CZ:NS:2025:5.TDO.1086.2024.1` is present in the collection and already corresponds to frozen baseline chunk `735`.
  BM25 sidecar inspection confirmed `1862/1862` rows have blank `document_id` and `source_document_id`, which explains provenance loss in BM25-backed frozen hits.
- Known limitations:
  No code or retrieval data was changed in this task, so the existing diagnostics artifacts remain unchanged until a future provenance/export fix or benchmark-item reformulation is executed.
  `nsoud-qa-010` remains a benchmark-quality risk because the current expected source is mostly operative `Dovolání se odmítá` boilerplate and does not cleanly support the doctrinal distinction in the question.
- Next recommended task:
  Remove `nsoud-qa-007` from the “true retrieval miss” bucket by fixing provenance/export visibility for BM25-backed NSoud hits, then reformulate or replace `nsoud-qa-010` before using it as a hard retrieval-quality signal.

## 2026-07-11 09:50 Europe/Moscow — Task: NSoud BM25 sidecar provenance repair without scoring changes

- Goal:
  Repair the NSoud BM25 sidecar so BM25 and hybrid retrieval artifacts expose correct provenance metadata, while preserving BM25 scoring, dense scoring, and RRF behavior.
- What changed:
  Updated `scripts/build_bm25_sidecar_from_qdrant.py` to flatten and export richer provenance fields from Qdrant payloads.
  Updated `app/rag/retrieval/bm25_sidecar.py` so BM25 retrieval results hydrate provenance metadata from explicit sidecar columns.
  Added `scripts/repair_nsoud_bm25_sidecar_provenance.py` with `--dry-run` and `--execute` modes and strict `chunk_id`-based mapping to read-only Qdrant payloads.
  Added `tests/test_repair_nsoud_bm25_sidecar_provenance.py`.
  Wrote candidate repaired sidecar `storage/rag/bm25/nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1.provenance_repaired.sqlite`.
  Created candidate run `artifacts/rag_eval/legal_qa/runs/nsoud_sidecar_provenance_repaired/` and candidate answer eval `artifacts/rag_eval/legal_qa/answer_eval/nsoud_sidecar_provenance_repaired/`.
  Added repair reports `artifacts/evaluation_quality/nsoud_bm25_sidecar_provenance_repair_20260710.md` and `.json`.
- Why it changed:
  The original NSoud sidecar had blank provenance in `1862/1862` rows, which made frozen BM25-backed hits lose usable `document_id` and `source_document_id` metadata even though the corresponding Qdrant points already had correct provenance.
- Files changed:
  `PROJECT_PROGRESS.md`
  `app/rag/retrieval/bm25_sidecar.py`
  `scripts/build_bm25_sidecar_from_qdrant.py`
  `scripts/repair_nsoud_bm25_sidecar_provenance.py`
  `tests/test_repair_nsoud_bm25_sidecar_provenance.py`
  `artifacts/evaluation_quality/nsoud_bm25_sidecar_provenance_repair_20260710.md`
  `artifacts/evaluation_quality/nsoud_bm25_sidecar_provenance_repair_20260710.json`
- Tests run:
  `python -m pytest tests/test_repair_nsoud_bm25_sidecar_provenance.py -q`
  `python -m pytest tests/rag/test_production_bge_m3_profile.py tests/test_merge_bge_m3_candidate_collections.py tests/rag/test_legal_qa_benchmark.py -q`
  `python -m pytest tests/rag/test_legal_answer_eval.py tests/rag/test_legal_answer_eval_diagnostics.py -q`
- Smoke result:
  `docker compose exec -T api python scripts/repair_nsoud_bm25_sidecar_provenance.py ... --dry-run` confirmed `1862/1862` deterministic matches and zero text mismatches.
  `docker compose exec -T api python scripts/repair_nsoud_bm25_sidecar_provenance.py ... --execute` produced a repaired candidate sidecar with `0` blank `document_id`, `source_document_id`, `ecli`, `case_number`, and `source`.
  Candidate retrieval benchmark kept `hit@1=0.700`, `hit@5=1.000`, `pass_rate=1.000`, while `nsoud-qa-007` now exposes rank-1 ECLI metadata directly from the retrieval artifact.
- Known limitations:
  `court` and `spisova_znacka` remain blank where they are absent in Qdrant payloads; the repair does not invent fields.
  `nsoud-qa-010` remains a real answer-support / boilerplate benchmark risk and still drives the candidate-only diagnostic final status to `FAIL_WITH_REAL_NSOUD_RISK`.
  Existing dirty generated ÚS/mixed answer-eval artifacts in the worktree remain unrelated and untouched.
- Next recommended task:
  Use the repaired sidecar/export path as the NSoud benchmark candidate, then either update the diagnostics status wording to distinguish answer-support risk from retrieval-miss risk more explicitly, or reformulate `nsoud-qa-010` before treating NSoud as fully green.

## 2026-07-11 12:40 Europe/Moscow — Task: NSoud strict direct pass audit

- Goal:
  Explain why `nsoud_sidecar_provenance_repaired` still has `strict_direct_pass_rate_gold=0.0` after provenance repair, and verify that the Grafana/Prometheus metrics path is reading the intended artifacts.
- What changed:
  Added `artifacts/evaluation_quality/nsoud_strict_direct_audit_20260711_124021.md`.
  Added `artifacts/evaluation_quality/nsoud_strict_direct_audit_20260711_124021.json`.
- Why it changed:
  The repaired NSoud run improved citation availability and reduced unsupported answer risk, but the dashboard still showed weak strict-direct performance. A per-question audit was needed to separate benchmark/gold misalignment, same-document wrong-chunk retrieval, and any possible dashboard mapping issue.
- Files changed:
  `PROJECT_PROGRESS.md`
  `artifacts/evaluation_quality/nsoud_strict_direct_audit_20260711_124021.md`
  `artifacts/evaluation_quality/nsoud_strict_direct_audit_20260711_124021.json`
- Tests run:
  `python -m pytest tests/rag/test_legal_answer_eval.py tests/rag/test_legal_answer_eval_diagnostics.py tests/observability/test_eval_metrics_exporter.py tests/test_repair_nsoud_bm25_sidecar_provenance.py -q`
- Smoke result:
  Read-only inspection confirmed the dashboard exporter is reading per-run `summary.json` files from `artifacts/rag_eval/legal_qa/answer_eval/*` with labels `(run_name, corpus)`.
  No dashboard query/label bug was needed to explain the NSoud strict-direct weakness.
- Known limitations:
  The audit is intentionally read-only; no retrieval logic, evaluator behavior, or benchmark source data was changed in this task.
  `nsoud-qa-004` and `nsoud-qa-010` still look like benchmark/gold alignment risks rather than clean retrieval regressions.
  `nsoud-qa-007` still needs a focused same-document chunk-selection follow-up before it can become a strict-direct pass.
- Next recommended task:
  Re-annotate or replace `nsoud-qa-004` and `nsoud-qa-010`, then run a narrowly scoped follow-up on `nsoud-qa-007` to test whether a better same-document chunk can be surfaced without changing global BM25/dense/RRF scoring.

## 2026-07-11 13:10 Europe/Moscow — Task: NALUS Production Task Validator

- Goal:
  Add a reusable deterministic validator for NALUS production tasks that checks dirty-file scope, risky diffs, documentation/test expectations, and task-safety signals before commit or final reporting.
- What changed:
  Added `app/project_validation/` with git-state parsing, file classification, diff scanning, reporting, and orchestration modules.
  Added CLI entrypoint `scripts/validate_nalus_task.py`.
  Added `tests/test_nalus_task_validator.py`.
  Added `docs/NALUS_TASK_VALIDATOR.md`.
- Why it changed:
  The repo needed a project-specific equivalent of the Memorial/Eternal World task validator so future NALUS tasks can detect accidental baseline-artifact staging, risky retrieval/Qdrant/model changes, missing progress updates, and missing tests before commits.
- Files changed:
  `PROJECT_PROGRESS.md`
  `app/project_validation/__init__.py`
  `app/project_validation/schemas.py`
  `app/project_validation/git_status.py`
  `app/project_validation/file_classifier.py`
  `app/project_validation/diff_scanner.py`
  `app/project_validation/report.py`
  `app/project_validation/validator.py`
  `scripts/validate_nalus_task.py`
  `tests/test_nalus_task_validator.py`
  `docs/NALUS_TASK_VALIDATOR.md`
- Tests run:
  `python -m pytest tests/test_nalus_task_validator.py -q`
  `python -m pytest tests/rag/test_legal_answer_eval.py tests/rag/test_legal_answer_eval_diagnostics.py tests/observability/test_eval_metrics_exporter.py tests/test_repair_nsoud_bm25_sidecar_provenance.py -q`
  `python scripts/validate_nalus_task.py --task-name "NALUS Production Task Validator" --mode implementation --expected-branch main --no-write`
- Known limitations:
  The validator is intentionally heuristic and diff-based; it does not understand semantic intent beyond configured patterns.
  Risk detection is intentionally conservative and currently scans changed source/test diffs, not full repository history.
  Generated validation reports are optional runtime artifacts and are not committed by default.
- Next recommended task:
  Run the validator before future NALUS commits and extend allowlists/risk rules only when an intentional change type repeatedly appears in real workflow.

## 2026-07-12 00:36 Europe/Moscow — Task: Refresh ÚS and Mixed no-LLM canonical answer-eval baselines

- Goal:
  Persist the intentionally regenerated canonical ÚS and Mixed no-LLM answer-eval artifacts so a clean checkout and exporter restart preserve the current verified monitoring values.
- What changed:
  Refreshed the canonical `usoud_no_llm_baseline` artifacts to represent `10/20` gold questions with `1` direct and `9` partial support results.
  Refreshed the canonical `mixed_no_llm_baseline` artifacts to represent `8/10` corpus-only gold questions with successful corpus routing.
  Persisted the generated diagnostics files emitted alongside both canonical runs.
- Why it changed:
  Gold annotation coverage was expanded after the prior canonical artifacts were committed. Persisting the regenerated outputs prevents Grafana and Prometheus values from reverting after checkout or restart.
- Expected metrics:
  ÚS: `gold=10`, `usable_support_rate_gold=1.0`, `citation_available_rate=1.0`, `unsupported_answer_risk_count=0`.
  Mixed: `gold=8`, `corpus_only_count=8`, `usable_support_rate_gold=1.0`, `corpus_routing_support_rate=1.0`, `citation_available_rate=0.0`, `unsupported_answer_risk_count=0`.
- Exporter/Grafana verification:
  Restarted `nalus-eval-metrics-exporter` and confirmed the expected `legal_answer_eval_gold`, `legal_answer_eval_usable_support_rate_gold`, and `legal_answer_eval_citation_available_rate` series for both named runs at `http://localhost:9108/metrics`.
  The exporter uses `legal_answer_eval_citation_available_rate`; no Grafana query change was required.
- Files changed:
  `PROJECT_PROGRESS.md`
  `artifacts/rag_eval/legal_qa/answer_eval/usoud_no_llm_baseline/*`
  `artifacts/rag_eval/legal_qa/answer_eval/mixed_no_llm_baseline/*`
- Tests run:
  `python -m pytest tests/rag/test_legal_answer_eval.py tests/observability/test_eval_metrics_exporter.py -q` -> `32 passed` with one non-blocking `pytest-asyncio` deprecation warning.
- Behavior preserved:
  Retrieval, BGE-M3, embedding dimensions/provider, dense scoring, BM25 scoring, RRF, global `top_k`, Qdrant collections/aliases/data, Redis/cache behavior, model loading, and LLM/DeepSeek behavior were not changed.
- Known limitations:
  Mixed citation availability remains `0.0` by design because all eight Mixed gold items are corpus-only and do not require document citations.
- Next recommended task:
  Complete the evidence-backed NSoud QA dataset/gold repair and regenerate an isolated `nsoud_dataset_repaired` candidate without changing retrieval scoring.

## 2026-07-12 00:45 Europe/Moscow — Task: NSoud QA dataset and gold repair

- Goal:
  Conservatively repair the four NSoud benchmark/gold issues identified by the strict-direct audit, regenerate an isolated retrieval/no-LLM candidate, and verify monitoring compatibility without changing retrieval scoring.
- Original issues and decisions:
  `nsoud-qa-003`: `evaluator_followup_needed` — corrected the inflection-specific expected keyword `občanské` to source form `občanský`; retained question and ECLI.
  `nsoud-qa-004`: `safe_gold_reannotation` — replaced the mismatched criminal `8 Tdo` gold with civil rank-1 `ECLI:CZ:NS:2025:33.CDO.79.2024.1` and reformulated the item to the § 237 o. s. ř. criteria explicitly supported by chunk `1000`.
  `nsoud-qa-007`: `safe_same_document_chunk_refinement` — retained the verified ECLI and query; replaced the tautological answer point with doctrine from same-document chunks `732–733`, while recording weaker rank-1 closing-summary chunk `735`.
  `nsoud-qa-010`: `safe_question_reformulation` — removed the unsupported odmítnutí-versus-zamítnutí comparison and asked the narrower admissibility question directly supported by existing-gold chunk `1644`.
- Dataset/gold changes:
  Updated only `nsoud-qa-003`, `004`, `007`, and `010` in `nsoud_qa_v1.jsonl`.
  Updated the reproducible NSoud ECLI map in `scripts/apply_gold_source_annotations.py` and the human gold review table.
  Added idempotence, evidence-alignment, unchanged-item, and no-invented-provenance regression coverage in `tests/test_nsoud_dataset_repair.py`.
- Candidate artifacts:
  Retrieval: `artifacts/rag_eval/legal_qa/runs/nsoud_dataset_repaired/` using the existing repaired sidecar and read-only Qdrant search.
  Answer eval/diagnostics: `artifacts/rag_eval/legal_qa/answer_eval/nsoud_dataset_repaired/` with `--no-llm --require-citations`.
  Repair audit: `artifacts/evaluation_quality/nsoud_dataset_repair_20260711.md` and `.json`.
- Metrics before (`nsoud_sidecar_provenance_repaired`):
  `gold=4`, `direct=0`, `partial=3`, `gap=0`, `boilerplate_noise=1`, `citation_available_rate=0.75`, `usable_support_rate_gold=0.75`, `unsupported_answer_risk_count=1`, `strict_direct_pass_rate_gold=0.0`.
- Metrics after (`nsoud_dataset_repaired`):
  `gold=4`, `direct=0`, `partial=3`, `gap=1`, `boilerplate_noise=0`, `citation_available_rate=0.75`, `usable_support_rate_gold=0.75`, `unsupported_answer_risk_count=1`, `strict_direct_pass_rate_gold=0.0`.
  Retrieval candidate: `pass_rate=0.9`, `source_hit@1=0.75`, `source_hit@3=0.75`, `source_hit@5=1.0`, `mean_source_constraint_match=1.0`.
- Monitoring verification:
  Restarted `nalus-eval-metrics-exporter`; all requested `legal_answer_eval_*` metrics for `run_name="nsoud_dataset_repaired"` were exposed with actual values.
  Prometheus query for `legal_answer_eval_gold{run_name="nsoud_dataset_repaired"}` returned `4`; metric names remain Grafana-compatible and no dashboard query changed.
- Files changed:
  `PROJECT_PROGRESS.md`
  `artifacts/rag_eval/legal_qa/datasets/nsoud_qa_v1.jsonl`
  `artifacts/rag_eval/legal_qa/gold_source_review_20260709.md`
  `scripts/apply_gold_source_annotations.py`
  `tests/test_nsoud_dataset_repair.py`
  `artifacts/evaluation_quality/nsoud_dataset_repair_20260711.md`
  `artifacts/evaluation_quality/nsoud_dataset_repair_20260711.json`
  `artifacts/rag_eval/legal_qa/runs/nsoud_dataset_repaired/*`
  `artifacts/rag_eval/legal_qa/answer_eval/nsoud_dataset_repaired/*`
- Tests run:
  `python -m pytest tests/rag/test_legal_answer_eval.py -q` -> `22 passed`.
  `python -m pytest tests/rag/test_legal_answer_eval_diagnostics.py -q` -> `2 passed`.
  `python -m pytest tests/rag/test_legal_qa_benchmark.py -q` -> `19 passed`.
  `python -m pytest tests/observability/test_eval_metrics_exporter.py -q` -> `10 passed`.
  `python -m pytest tests/test_repair_nsoud_bm25_sidecar_provenance.py -q` -> `5 passed`.
  `python -m pytest tests/test_nalus_task_validator.py -q` -> `9 passed`.
  `python -m pytest tests/test_nsoud_dataset_repair.py -q` -> `3 passed`.
  Repeated `pytest-asyncio` default-loop-scope deprecation warning is non-blocking and unrelated to this task.
- Runtime/infra safety:
  Qdrant access was read-only search; no ingest, collection rebuild, write, or alias switch occurred.
  BGE-M3 loaded from the existing local cache; no model download occurred.
  Redis was not enabled or used; no LLM or DeepSeek call occurred.
  Dense scoring, BM25 scoring, RRF, global `top_k`, embeddings, cache behavior, and fallback behavior were unchanged.
- Validator result:
  `python scripts/validate_nalus_task.py --task-name "NSoud QA dataset repair" --mode eval_change --expected-branch main --no-write` -> understood `WARN` with exactly two `unknown_dirty_file` findings.
  Both warnings are intentional classifier limitations for the explicitly allowed task files `artifacts/rag_eval/legal_qa/datasets/nsoud_qa_v1.jsonl` and `artifacts/rag_eval/legal_qa/gold_source_review_20260709.md`; documentation/test checks passed and all safety summaries remained `no`.
- Known limitations:
  `nsoud-qa-010` remains an honest unsupported risk: the correct doctrinal gold chunk is rank 4, but its fixed 240-character exported snippet ends before the supporting sentences.
  `nsoud-qa-003` remains at exported-snippet coverage `2/3 = 0.6667`, below the unchanged `>= 0.67` strict gate.
- Next recommended task:
  Add and test deterministic evidence-window handling for gold chunks whose relevant doctrine lies beyond the exported snippet, without lowering the strict threshold or changing global retrieval scoring.

## 2026-07-12 — Task: Shared Grafana — Add Eternal World to NALUS Grafana

- Goal:
  Use the existing Grafana on `http://localhost:3002` as one UI for NALUS and Eternal World while retaining two independent Prometheus instances and TSDBs.
- Architecture:
  Preserved NALUS datasource `Prometheus` / UID `prometheus` / internal URL `http://prometheus:9090` as the only default datasource.
  Added `Eternal World Prometheus` / UID `eternal-world-prometheus`, with URL supplied through `ETERNAL_WORLD_PROMETHEUS_URL` and local Docker default `http://host.docker.internal:9090`.
  NALUS Prometheus remains on host port `9091`; Eternal World Prometheus remains on `9090`.
  Separated dashboard provider paths into `/var/lib/grafana/dashboards/nalus` and `/var/lib/grafana/dashboards/eternal-world` to prevent overlapping scans and duplicate UIDs.
- Dashboard source-of-truth:
  Eternal World dashboard files are mounted read-only from the sibling Eternal World repository. No dashboard JSON copy is maintained in NALUS.
  Provider folders are `NALUS` and `Eternal World`.
- Configuration:
  Added environment overrides for the Eternal World Prometheus URL and dashboard directory.
  Added `host.docker.internal:host-gateway` for portable local host routing where Docker supports `host-gateway`.
  Bind mounts use `create_host_path: false`, so a missing sibling checkout fails explicitly.
- Validator support:
  Added an explicit `infra_config` classification for Compose, monitoring provisioning, and `.env.example` files.
  Fixed `--allow-risk infra_or_dependency_change` so an explicitly authorized infrastructure task can pass without weakening Qdrant/model/retrieval safety rules.
- Tests and validation:
  `docker compose config --quiet` passed.
  `python -m json.tool monitoring/grafana/dashboards/legal_answer_eval_dashboard.json` passed.
  `python -m pytest tests/test_nalus_task_validator.py tests/observability/test_shared_grafana_provisioning.py tests/observability/test_eval_metrics_exporter.py -q` -> `25 passed` with the existing non-blocking `pytest-asyncio` warning.
  Task validator in implementation mode returned `PASS` with zero findings after explicitly authorizing the requested Compose infrastructure change and the unchanged Redis context line in `.env.example`.
  Shared provisioning tests verify datasource preservation, unique datasource UIDs/default, non-overlapping provider paths, read-only mounts, and the unchanged NALUS dashboard UID bindings.
- Runtime smoke:
  Recreated only `grafana`; Grafana `11.4.0` became healthy on `3002`.
  Datasource health returned `OK` for both `prometheus` and `eternal-world-prometheus`.
  NALUS dashboard loaded in folder `NALUS`; Eternal World dashboard loaded in folder `Eternal World` with UID `eternal-world-fa-chat`.
  Grafana proxy isolation check returned NALUS `legal_answer_eval_gold` only through UID `prometheus`, and Eternal World `fa_chat_requests_total` only through UID `eternal-world-prometheus`.
  Shared Grafana provisioning logs contained no blocking datasource, dashboard, duplicate UID, or permission error.
- Behavior preserved:
  NALUS application metrics, Prometheus scrape config, exporter, retrieval, BGE-M3, BM25, RRF, Qdrant, Redis, API behavior, and production aliases were not changed.
  Eternal World application metrics and Prometheus storage were not changed.
- Known limitations:
  The local default relies on the host gateway. Linux/server deployments must override `ETERNAL_WORLD_PROMETHEUS_URL` with an address reachable from the Grafana container.
  Shared Grafana currently remains owned by the NALUS Compose stack; a dedicated observability repository is deferred until more projects require integration.
- Next recommended task:
  Move shared Grafana into a dedicated observability-stack repository only when more projects need to be added.

## 2026-07-12 17:59 Europe/Moscow — Task: Deterministic same-document evidence windows for legal answer evaluation

- Goal:
  Allow deterministic no-LLM legal answer evaluation to inspect a bounded same-document evidence window for a verified gold hit, without changing retrieval ranking, evaluator thresholds, model behavior, Qdrant state, or LLM behavior.
- Architecture:
  Added `app/rag/eval/evidence_window.py` as the typed evidence-window layer. The evaluator validates `source_document_id`, `document_id`, `ecli`, and `chunk_index`, loads same-document adjacent chunks, orders by `chunk_index`, enforces chunk and character bounds, preserves provenance diagnostics, and reports failures explicitly. The existing evaluator behavior remains the default unless `--evidence-window` is passed.
- What changed:
  Updated `app/rag/eval/legal_answer_eval.py` so enabled evidence windows evaluate keyword support against combined evidence text while source/citation matching still depends on verified document provenance.
  Updated `scripts/run_legal_answer_eval.py` with explicit evidence-window CLI options and an explicit local sidecar path option.
  Updated `scripts/generate_legal_answer_eval_diagnostics.py` so diagnostics replay the evidence-window configuration recorded in `metrics.json`.
  Added `tests/rag/test_legal_evidence_window.py` with focused unit/integration coverage for ordering, bounds, same-document enforcement, diagnostics, summary counters, and NSoud-style cases.
  Created candidate answer-eval artifacts under `artifacts/rag_eval/legal_qa/answer_eval/nsoud_evidence_window_candidate/`.
  Added `artifacts/evaluation_quality/nsoud_evidence_window_evaluation_20260712.md` and `.json`.
- Configuration:
  `--evidence-window --evidence-neighbors-before 1 --evidence-neighbors-after 1 --evidence-max-chunks 3 --evidence-max-characters 6000 --evidence-sidecar storage/rag/bm25/nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1.provenance_repaired.sqlite`.
- Evidence source:
  The candidate used the repaired local NSoud BM25 sidecar in read-only SQLite mode. Qdrant lookup was not needed and Qdrant was not contacted for this candidate evaluation.
- Candidate metrics:
  Before (`nsoud_dataset_repaired`): `gold=4`, `direct=0`, `partial=3`, `gap=1`, `boilerplate_noise=0`, `usable_support_rate_gold=0.75`, `citation_available_rate=0.75`, `unsupported_answer_risk_count=1`, `strict_direct_pass_rate_gold=0.0`.
  After (`nsoud_evidence_window_candidate`): `gold=4`, `direct=3`, `partial=1`, `gap=0`, `boilerplate_noise=0`, `usable_support_rate_gold=1.0`, `citation_available_rate=1.0`, `unsupported_answer_risk_count=0`, `strict_direct_pass_rate_gold=0.75`, `evidence_window_used_count=4`, `evidence_window_failed_count=0`, `evidence_window_truncated_count=1`, `same_document_neighbor_count=8`.
- `nsoud-qa-010` result:
  Anchor chunk `1644` remained rank `4`; chunks `1643`, `1644`, and `1645` were included from the same document. Combined evidence length was `3952`. The relevant doctrine became visible, support changed from `gap` to `partial`, citation became available, and unsupported risk cleared. This confirms exported snippet truncation rather than retrieval ranking as the issue.
- `nsoud-qa-003` result:
  Original keyword coverage was `2/3 = 0.6667`; evidence-window coverage became `3/3 = 1.0`, and the item became `direct`. The strict threshold and morphology rules were not changed. The evidence window for this item was truncated at the configured `6000` characters and reports that truncation explicitly.
- Tests run:
  `python -m pytest tests/rag/test_legal_evidence_window.py -q` -> `20 passed`.
  `python -m pytest tests/rag/test_legal_answer_eval.py -q` -> `22 passed`.
  `python -m pytest tests/rag/test_legal_answer_eval_diagnostics.py -q` -> `2 passed`.
  `python -m pytest tests/observability/test_eval_metrics_exporter.py -q` -> `10 passed`.
  `python -m pytest tests/test_nalus_task_validator.py -q` -> `11 passed`.
  `python -m pytest tests/rag/test_legal_evidence_window.py tests/rag/test_legal_answer_eval.py tests/rag/test_legal_answer_eval_diagnostics.py tests/observability/test_eval_metrics_exporter.py tests/test_nalus_task_validator.py -q` -> `65 passed`.
  Repeated `pytest-asyncio` loop-scope deprecation warning is non-blocking and unrelated to this task.
- Monitoring verification:
  Restarted `nalus-eval-metrics-exporter`. `curl.exe -s http://localhost:9108/metrics | Select-String 'run_name="nsoud_evidence_window_candidate"'` exposed the expected existing bounded metrics for the new run: `gold=4`, `direct=3`, `partial=1`, `gap=0`, `unsupported=0`, `strict_direct_pass_rate_gold=0.75`, `usable_support_rate_gold=1.0`, and `citation_available_rate=1.0`.
- Validator:
  Exact validator command without allowlist returned `WARN` for two intentional `bm25_change` findings because the evidence-window evaluator reads the local BM25 sidecar as an evidence source. The follow-up validator run with `--allow-risk bm25_change` returned `PASS` with zero findings. No BM25 scoring changed.
- Behavior preserved:
  Retrieval ranking, retrieved hit order, global `top_k`, dense scoring, BM25 scoring, RRF, BGE-M3, embedding dimensions, Qdrant collections/aliases/data, Redis/cache behavior, Grafana queries, strict-direct threshold, and LLM/DeepSeek behavior were not changed.
- Known limitations:
  Evidence windows improve evaluator visibility only and do not change retrieval ranking. `nsoud-qa-010` remains `partial` because the verified gold hit is rank `4`, and the strict-direct definition still requires rank `1`.
- Next recommended task:
  Validate evidence-window mode across ÚS and Mixed before deciding whether it should become the default no-LLM answer-eval behavior.

## 2026-07-12 22:23 Europe/Moscow — Task: Cross-corpus evidence-window validation

- Goal:
  Validate deterministic evidence-window evaluation across ÚS and Mixed corpora before considering any default behavior change, while keeping evidence windows opt-in.
- What changed:
  Extended `app/rag/eval/evidence_window.py` so the read-only BM25 sidecar evidence loader supports both known sidecar schemas: NSoud with explicit `ecli` and ÚS without `ecli` but with `document_id` / `source_document_id`.
  Fixed `evidence_window_failed_count` so corpus-only skips (`provenance_valid=None`) are not counted as failed evidence windows.
  Added focused regression tests for sidecars without `ecli` and Mixed corpus-only skip behavior.
  Created `usoud_evidence_window_candidate` and `mixed_evidence_window_candidate` answer-eval artifact directories.
  Added `artifacts/evaluation_quality/cross_corpus_evidence_window_validation_20260712.md` and `.json`.
- Candidate runs:
  ÚS: `artifacts/rag_eval/legal_qa/answer_eval/usoud_evidence_window_candidate/`.
  Mixed: `artifacts/rag_eval/legal_qa/answer_eval/mixed_evidence_window_candidate/`.
- Evidence sources:
  ÚS used `storage/rag/bm25/nalus_us_bge_m3_rag_combined_20260709.sqlite` in SQLite read-only mode.
  Mixed used no document evidence source because all gold items are corpus-only and evidence windows are skipped by design.
- ÚS before/after:
  Baseline `usoud_no_llm_baseline`: `gold=10`, `direct=1`, `partial=9`, `gap=0`, `boilerplate=0`, `usable_support_rate_gold=1.0`, `citation_available_rate=1.0`, `unsupported_answer_risk_count=0`, `strict_direct_pass_rate_gold=0.1`.
  Candidate `usoud_evidence_window_candidate`: `gold=10`, `direct=7`, `partial=3`, `gap=0`, `boilerplate=0`, `usable_support_rate_gold=1.0`, `citation_available_rate=1.0`, `unsupported_answer_risk_count=0`, `strict_direct_pass_rate_gold=0.7`, `evidence_window_used_count=10`, `evidence_window_failed_count=0`, `evidence_window_truncated_count=0`, `same_document_neighbor_count=20`.
- Mixed before/after:
  Baseline `mixed_no_llm_baseline`: `gold=8`, `corpus_only_count=8`, `usable_support_rate_gold=1.0`, `citation_available_rate=0.0`, `corpus_routing_support_rate=1.0`, `unsupported_answer_risk_count=0`.
  Candidate `mixed_evidence_window_candidate`: `gold=8`, `corpus_only_count=8`, `usable_support_rate_gold=1.0`, `citation_available_rate=0.0`, `corpus_routing_support_rate=1.0`, `unsupported_answer_risk_count=0`, `evidence_window_used_count=0`, `evidence_window_failed_count=0`, `evidence_window_truncated_count=0`.
- NSoud reference:
  `nsoud_evidence_window_candidate` remains green as the reference document-gold candidate: `gold=4`, `direct=3`, `partial=1`, `gap=0`, `usable_support_rate_gold=1.0`, `citation_available_rate=1.0`, `unsupported_answer_risk_count=0`, `strict_direct_pass_rate_gold=0.75`.
- Safety verification:
  ÚS per-row validation found no cross-document mismatch, no invalid evidence windows, and no fabricated citations.
  Mixed per-row validation found no valid or failed document evidence windows, no corpus-only citation, and no corpus-only row with evidence-window chunks.
- Tests run:
  `python -m pytest tests/rag/test_legal_evidence_window.py -q` -> `22 passed`.
  `python -m pytest tests/rag/test_legal_answer_eval.py -q` -> `22 passed`.
  `python -m pytest tests/rag/test_legal_answer_eval_diagnostics.py -q` -> `2 passed`.
  `python -m pytest tests/observability/test_eval_metrics_exporter.py -q` -> `10 passed`.
  `python -m pytest tests/test_nalus_task_validator.py -q` -> `11 passed`.
  `python -m pytest tests/rag/test_legal_evidence_window.py tests/rag/test_legal_answer_eval.py tests/rag/test_legal_answer_eval_diagnostics.py tests/observability/test_eval_metrics_exporter.py tests/test_nalus_task_validator.py -q` -> `67 passed`.
  Repeated `pytest-asyncio` loop-scope deprecation warning is non-blocking and unrelated.
- Monitoring verification:
  Restarted `nalus-eval-metrics-exporter`. The new `usoud_evidence_window_candidate` and `mixed_evidence_window_candidate` runs were visible at `http://localhost:9108/metrics` through existing `legal_answer_eval_*` metrics and bounded labels `(run_name, corpus)`. No Grafana query changed.
- Validator:
  Exact validator command without allowlist returned `WARN` for the intentional `bm25_change` sidecar-read diff. The validator run with `--allow-risk bm25_change` returned `PASS` with zero findings. No BM25 scoring changed.
- Default-mode recommendation:
  Keep evidence windows opt-in for now. Future default activation is recommended only for document-gold no-LLM answer evaluation, not globally and not for Mixed corpus-only routing evaluation.
- Known limitations:
  This validates offline no-LLM answer-eval artifacts only. It does not change or validate live generation behavior.
- Next recommended task:
  Prepare a separate default-policy task that enables evidence windows only for document-gold no-LLM evaluation, with corpus-only skip behavior explicitly documented and tested.

## 2026-07-12 23:32 Europe/Moscow — Task: Document-gold evidence-window default policy

- Goal:
  Make deterministic same-document evidence windows the default only for offline no-LLM document-gold legal answer evaluation, while keeping corpus-only routing, live runtime retrieval, LLM generation, retrieval benchmarks, model behavior, Qdrant, Redis, scoring, thresholds, and Grafana queries unchanged.
- What changed:
  Added an explicit typed evidence-window policy layer in `app/rag/eval/evidence_window.py` with `off`, `document_gold`, and `explicit_all` behavior.
  Updated `app/rag/eval/legal_answer_eval.py` so policy decisions are recorded per result with configured/effective policy, activation reason, skip reason, document-gold presence, default activation, explicit activation, and aggregate counters.
  Updated `scripts/run_legal_answer_eval.py` so the no-LLM CLI defaults to `document_gold`, preserves existing `--evidence-window`, adds `--evidence-window-policy off|document-gold|all`, adds `--no-evidence-window`, and rejects conflicting combinations.
  Updated `scripts/generate_legal_answer_eval_diagnostics.py` so diagnostics replay the recorded evidence-window policy.
  Added regression coverage for default activation, corpus-only skip, explicit off, explicit enable, LLM-mode skip, missing provenance safety, CLI conflicts, default policy mapping, counters, threshold preservation, and retrieval immutability.
  Created local candidate output directories `usoud_document_gold_default`, `nsoud_document_gold_default`, and `mixed_document_gold_default`.
  Added `artifacts/evaluation_quality/document_gold_evidence_window_policy_20260712.md` and `.json`.
- Policy behavior:
  `document_gold` activates only when `no_llm=true`, gold is available, the item is not corpus-only, and a document gold id is present. Invalid provenance still fails safely at construction time as `missing_or_invalid_provenance`; no neighboring chunks are guessed.
  Corpus-only gold is skipped with `corpus_only_gold`, citation remains unavailable by design, and the skip is not counted as an evidence-window failure.
  LLM-mode evaluation does not silently activate the document-gold default; explicit policy is required.
- Candidate runs:
  ÚS `usoud_document_gold_default`: `gold=10`, `direct=7`, `partial=3`, `usable_support_rate_gold=1.0`, `citation_available_rate_gold=1.0`, `unsupported_answer_risk_count=0`, `strict_direct_pass_rate_gold=0.7`, `evidence_window_used_count=10`, `evidence_window_failed_count=0`, `evidence_window_default_activated_count=10`.
  NSoud `nsoud_document_gold_default`: `gold=4`, `direct=3`, `partial=1`, `usable_support_rate_gold=1.0`, `citation_available_rate_gold=1.0`, `unsupported_answer_risk_count=0`, `strict_direct_pass_rate_gold=0.75`, `evidence_window_used_count=4`, `evidence_window_failed_count=0`, `evidence_window_default_activated_count=4`.
  Mixed `mixed_document_gold_default`: `gold=8`, `corpus_only_count=8`, `usable_support_rate_gold=1.0`, `citation_available_rate_gold=0.0`, `corpus_routing_support_rate=1.0`, `unsupported_answer_risk_count=0`, `evidence_window_used_count=0`, `evidence_window_failed_count=0`, `evidence_window_corpus_only_skipped_count=8`.
- Tests run:
  `python -m pytest tests/rag/test_legal_evidence_window.py -q` -> `28 passed`.
  `python -m pytest tests/rag/test_legal_answer_eval.py -q` -> `24 passed`.
  `python -m pytest tests/rag/test_legal_answer_eval_diagnostics.py -q` -> `2 passed`.
  `python -m pytest tests/observability/test_eval_metrics_exporter.py -q` -> `10 passed`.
  `python -m pytest tests/test_nalus_task_validator.py -q` -> `11 passed`.
  `python -m pytest tests/rag/test_legal_evidence_window.py tests/rag/test_legal_answer_eval.py tests/rag/test_legal_answer_eval_diagnostics.py tests/observability/test_eval_metrics_exporter.py tests/test_nalus_task_validator.py -q` -> `75 passed`.
  Repeated `pytest-asyncio` loop-scope deprecation warning is non-blocking and unrelated.
- Monitoring verification:
  Recreated only `nalus-eval-metrics-exporter`. `http://localhost:9108/metrics` exposed all three new run names through the existing `legal_answer_eval_*` bounded metrics: `usoud_document_gold_default`, `nsoud_document_gold_default`, and `mixed_document_gold_default`.
- Validator:
  Initial exact validator run returned `WARN` only because the three requested candidate run output directories were new unknown dirty files.
  Follow-up validator run with explicit `--allow-candidate-run usoud_document_gold_default --allow-candidate-run nsoud_document_gold_default --allow-candidate-run mixed_document_gold_default` returned `PASS` with zero findings.
- Behavior preserved:
  Retrieval rank/order/scores, top_k, strict thresholds, dense scoring, BM25 scoring, RRF, BGE-M3, embedding dimensions, Qdrant collections/aliases/data, Redis/cache behavior, Grafana queries, and LLM/DeepSeek behavior were not changed.
- Known limitations:
  The new policy affects offline deterministic no-LLM answer evaluation only. Candidate run directories are generated artifacts for local review and are not part of the application runtime.
- Next recommended task:
  Use the new `document_gold` policy for future offline no-LLM legal answer-eval runs, and keep live generation unchanged until a separate runtime evidence policy is explicitly designed and reviewed.

## 2026-07-13 00:18 Europe/Moscow — Task: Add document-level exhaustive retrieval pipeline

- Goal:
  Add a production-grade document-level retrieval path that returns bounded unique court decisions identified from candidate chunks, while preserving the existing chunk-level retrieval path and API compatibility.
- Scope:
  Implemented an additive module and endpoint only. Existing `/api/rag/retrieve`, `/api/rag/query`, hybrid retrieval, dense retrieval, BM25 sidecar scoring, RRF fusion, BGE-M3 embeddings, Qdrant collections, Redis/cache behavior, ingest, LLM behavior, and frontend behavior remain unchanged.
- What changed:
  Added `app/rag/retrieval/document_retrieval.py` with typed configuration, canonical document grouping, duplicate removal, deterministic document scoring, dynamic threshold filtering, best supporting passages, safe document metadata projection, and bounded diagnostics.
  Added `POST /api/rag/retrieve-documents` as an explicit additive endpoint in `app/api/rag_router.py`.
  Added disabled-by-default document retrieval configuration to `.env.example`.
  Added `docs/DOCUMENT_LEVEL_RETRIEVAL.md` describing the pipeline, config, scoring strategy, API response, safety properties, and future extension points.
  Added `tests/rag/test_document_retrieval.py` and expanded `tests/api/test_rag_api.py`.
- Configuration:
  `NALUS_DOCUMENT_RETRIEVAL_ENABLED=0` keeps the new endpoint disabled by default.
  `NALUS_DOCUMENT_MAX_CANDIDATE_CHUNKS`, `NALUS_DOCUMENT_MAX_RETURNED_DOCUMENTS`, `NALUS_DOCUMENT_MAX_SUPPORTING_CHUNKS_PER_DOCUMENT`, `NALUS_DOCUMENT_RELEVANCE_THRESHOLD`, `NALUS_DOCUMENT_SCORING_STRATEGY`, and optional `NALUS_DOCUMENT_LATENCY_BUDGET_MS` centralize document-level retrieval behavior.
- Scoring:
  The first deterministic strategy is `best_plus_average_top_chunks`, combining best chunk score with average top supporting chunk score. The strategy is explicit and can be extended without changing grouping or API contracts.
- API behavior:
  Existing `/api/rag/retrieve` response remains chunk-oriented and unchanged.
  New `/api/rag/retrieve-documents` returns `documents` and `diagnostics`. If the configured threshold filters all documents, the endpoint returns an empty `documents` list with diagnostics and does not silently lower thresholds or fall back to unrelated documents.
- Tests run:
  `python -m pytest tests/rag/test_document_retrieval.py -q` -> `10 passed`.
  `python -m pytest tests/api/test_rag_api.py -q` -> `34 passed`.
  `python -m pytest tests/rag/test_production_bge_m3_profile.py tests/rag/test_retrieval_service.py -q` -> `39 passed`.
  `python -m pytest tests/rag/test_document_retrieval.py tests/api/test_rag_api.py tests/rag/test_production_bge_m3_profile.py tests/rag/test_retrieval_service.py tests/test_nalus_task_validator.py -q` -> `94 passed`.
  Repeated `pytest-asyncio` loop-scope deprecation warning is non-blocking and unrelated.
- Validator:
  Initial validator run failed only because `PROJECT_PROGRESS.md` had not yet been updated; diff-scan warnings matched intentional runtime/API/config terms and existing generated candidate output directories from the previous task.
  Follow-up validator run with explicit allowlist for the intentional `top_k_change`, `logger_change`, `bm25_change`, `rrf_change`, `dense_change`, and existing generated candidate run directories returned `PASS` with zero findings.
- Runtime/API smoke:
  `docker compose ps` showed `api`, `qdrant`, `redis`, `prometheus`, `grafana`, and `nalus-eval-metrics-exporter` running.
  Focused API smoke `python -m pytest tests/api/test_rag_api.py::TestRawRetrieveEndpoint::test_document_retrieve_returns_unique_documents_with_diagnostics tests/api/test_rag_api.py::TestRawRetrieveEndpoint::test_existing_retrieve_response_shape_remains_backward_compatible -q` -> `2 passed`.
- Behavior preserved:
  No ingest, no Qdrant write, no embedding regeneration, no model download, no Redis enablement, no LLM/DeepSeek call, no BM25 scoring change, no RRF change, no default API behavior change, and no hidden threshold fallback.
- Known limitations:
  This first implementation groups and scores already retrieved candidates. It does not yet benchmark document-level recall against legal QA datasets and does not implement document-level reranking or follow-up retrieval.
- Next recommended task:
  Add an offline document-level retrieval benchmark that compares unique-document recall against the existing chunk-level benchmark under controlled candidate pool and threshold settings.

## 2026-07-13 01:52 Europe/Moscow — Task: Offline document-level retrieval benchmark

- Goal:
  Add a production-quality offline benchmark for the additive document-level retrieval pipeline, measuring multi-document recall and diagnostics without changing retrieval, ranking, embeddings, Qdrant, BM25, RRF, Redis, LLM behavior, APIs, or frontend behavior.
- Scope:
  Added a separate benchmark only. Existing legal QA benchmark, answer evaluation, document-level retrieval runtime endpoint, hybrid retrieval, and all production retrieval components remain unchanged.
- What changed:
  Added `app/rag/eval/document_retrieval_benchmark.py` with typed JSONL dataset support for multiple relevant documents per question, deterministic candidate/final recall metrics, precision@K, duplicate rate, zero-result rate, latency metrics, failure classification, and report writing.
  Added `scripts/run_document_retrieval_benchmark.py` as a read-only runner using the existing `build_hybrid_retriever` search function without modifying retrieval behavior.
  Extended `app/observability/eval_metrics_exporter.py` to expose document benchmark summaries through the existing Prometheus exporter and conventions. New metrics use bounded labels only: `run_name` and `corpus`.
  Added `docs/DOCUMENT_LEVEL_RETRIEVAL_BENCHMARK.md` documenting dataset format, metrics, failure categories, reports, runner usage, Prometheus label safety, and extension points.
  Added tests for dataset loading, duplicate gold normalization, candidate recall, final recall, precision, large/multiple gold sets, zero relevant documents, failure categories, report generation, runner config, and exporter metrics.
- Dataset format:
  JSONL items include `id`, `corpus`, `question`, and `relevant_document_ids`. Optional metadata includes `legal_topic` and `difficulty`.
  `relevant_document_ids` supports arbitrary counts. Duplicate identifiers are normalized and deduplicated deterministically.
- Metrics implemented:
  Chunk recall@10/20/50/100, document recall@10/20/50/100, precision@10/20/50/100, candidate pool coverage, unique document coverage, duplicate rate, zero result rate, average retrieved documents, average candidate chunks, average latency, and document aggregation latency.
- Failure diagnostics:
  `relevant_document_never_retrieved`, `relevant_document_removed_by_aggregation`, `relevant_document_removed_by_threshold`, `relevant_document_removed_by_returned_document_limit`, `duplicate_handling_issue`, `metadata_issue`, and `unknown`.
- Reports:
  Writer produces `metrics.json`, `summary.json`, `per_question.jsonl`, `per_question.csv`, and `summary.md`.
  No real benchmark output artifact was generated or committed in this task.
- Observability:
  Reused the existing Prometheus exporter. No second metrics system was added.
  Prometheus labels remain bounded to `run_name` and `corpus`; tests verify raw query text, document ids, and ECLI values are not emitted as labels.
- Tests run:
  `python -m pytest tests/rag/test_document_retrieval_benchmark.py -q` -> `9 passed`.
  `python -m pytest tests/test_run_document_retrieval_benchmark.py -q` -> `2 passed`.
  `python -m pytest tests/observability/test_eval_metrics_exporter.py -q` -> `11 passed`.
  `python -m pytest tests/rag/test_document_retrieval_benchmark.py tests/test_run_document_retrieval_benchmark.py tests/observability/test_eval_metrics_exporter.py tests/rag/test_document_retrieval.py tests/rag/test_legal_qa_benchmark.py -q` -> `51 passed`.
  Repeated `pytest-asyncio` loop-scope deprecation warning is non-blocking and unrelated.
- Validator:
  Initial validator run failed only because `PROJECT_PROGRESS.md` had not yet been updated. Diff-scan warnings were for intentional evaluation terms (`top_k`, BM25/RRF mentions in safety documentation, Redis rejection, and logger calls without raw query logging).
- Behavior preserved:
  Retrieval logic, ranking, embeddings, Qdrant collections/data, BM25 scoring, RRF fusion, Redis behavior, DeepSeek/LLM prompts, API behavior, and frontend behavior were not changed.
- Known limitations:
  This task implements and tests the framework. It does not create a curated multi-document gold dataset and does not run a real corpus benchmark artifact.
- Next recommended task:
  Build a reviewed multi-document gold dataset for ÚS/NSoud and run the new benchmark once the gold set is approved.
