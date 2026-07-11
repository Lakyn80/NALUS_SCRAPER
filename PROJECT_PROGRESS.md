# Project Progress

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
