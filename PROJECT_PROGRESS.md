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
