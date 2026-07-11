# NALUS Task Validator

Purpose: provide a deterministic pre-commit / pre-report validator for NALUS legal RAG tasks.

It is designed to catch:
- accidental staging of unrelated generated artifacts
- risky Qdrant write or alias-change code in read-only tasks
- retrieval / BM25 / RRF / Redis / model-download risk terms in diffs
- missing `PROJECT_PROGRESS.md` updates
- missing test changes for implementation work
- vague task hygiene before commit or final reporting

## How To Run

Implementation task:

```powershell
python scripts/validate_nalus_task.py `
  --task-name "Repair NSoud BM25 sidecar provenance" `
  --mode implementation `
  --expected-branch main `
  --write-report artifacts/evaluation_quality/nalus_task_validation_example.md `
  --write-json artifacts/evaluation_quality/nalus_task_validation_example.json
```

Dry run:

```powershell
python scripts/validate_nalus_task.py `
  --task-name "Audit NSoud strict direct pass" `
  --mode audit `
  --expected-branch main `
  --no-write
```

## Modes

- `audit`
- `implementation`
- `artifact_only`
- `runtime_change`
- `eval_change`

## Exit Codes

- `0` = `PASS`
- `1` = `FAIL`
- `2` = `WARN`

## What It Checks

- branch vs `--expected-branch`
- dirty/staged/untracked git files
- file classification:
  `source_code`, `tests`, `docs`, `project_progress`, `eval_reports`, `candidate_eval_artifacts`, `generated_baseline_artifacts`, `local_noise`, `model_cache`, `unknown`
- risky diff terms:
  Qdrant writes, alias changes, protected alias references, model downloads, fallback embeddings, forbidden repo imports, DeepSeek/OpenAI key references
- warning diff terms:
  `top_k`, `rrf`, `bm25`, `dense`, `redis`, logger changes, infra/dependency file changes
- source-code tasks without test changes
- source-code tasks without `PROJECT_PROGRESS.md` updates

## Allowing Intentional Risk

Use `--allow-risk` to suppress specific rule ids or literal matched terms when a future task intentionally changes something risky and that intent is explicitly documented.

Examples:

```powershell
python scripts/validate_nalus_task.py `
  --task-name "Intentional retrieval tuning" `
  --mode eval_change `
  --expected-branch main `
  --allow-risk bm25_change `
  --allow-risk rrf_change `
  --no-write
```

Use `--allow-no-test-change "reason"` only when the change is genuinely docs-only or artifact-only and you want that reason recorded in the report.

## Relation To PROJECT_PROGRESS.md And Final Reports

- if `app/**` or `scripts/**` change, the validator expects `PROJECT_PROGRESS.md` to change too
- if implementation code changes without test changes, the validator fails by default in implementation-like modes
- the Markdown/JSON outputs are intended to make final reporting concrete and auditable

## Limitations

- the validator uses git diff and file-pattern heuristics; it does not fully understand intent
- risk detection is substring/regex based and intentionally conservative
- it does not inspect Qdrant, Redis, model caches, or the network
- it reports dirty baseline-generated artifacts but does not mutate or clean them
