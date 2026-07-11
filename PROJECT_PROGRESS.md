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
