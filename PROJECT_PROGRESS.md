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
