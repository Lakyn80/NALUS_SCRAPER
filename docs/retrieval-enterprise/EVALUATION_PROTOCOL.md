# Retrieval Enterprise Evaluation Protocol

Status: target evaluation specification.

Evaluation determines whether a change improves judgment retrieval. It must not
be mixed with prompt tuning, benchmark correction, or production rollout unless
the task explicitly scopes those actions.

## Evaluation Principles

- Measure retrieval separately from answer generation.
- Compare against the current baseline.
- Use deterministic manifests.
- Keep gold-data edits as separate reviewed tasks.
- Report failures as failures.
- Do not call DeepSeek for Stage A retrieval comparisons unless the task is
  explicitly about verifier or query interpretation behavior.

## Baseline Parity Gate

Before a new retrieval module can replace or wrap baseline candidate generation,
it must prove parity with the current path:

- same query set;
- same source corpus;
- same Qdrant collection/BM25 sidecar;
- same candidate limits;
- same profile constants;
- same local BGE-M3 model identity;
- deterministic ordering and tie-breakers documented.

Allowed tolerance must be declared before the run. Example tolerances:

- exact same top 10 document IDs for smoke parity; or
- same gold coverage at K plus bounded rank deltas for broader benchmark parity.

## Required Metrics

Candidate retrieval:

- Recall@K.
- MRR.
- NDCG.
- gold coverage.
- rank deltas.
- hard-negative rate.
- unique candidate documents.
- chunk-to-document aggregation counts.

Verifier-aware retrieval:

- verified_match count.
- no_verified_results count.
- false approval count.
- false rejection count.
- classification distribution.
- hard constraint proof coverage.
- related-only count.

Performance:

- per-stage latency.
- total latency.
- CPU execution speed.
- peak RAM estimate or measurement.
- disk usage.
- build duration.
- provider calls and cost when providers are explicitly in scope.

## Benchmark Data Rules

- Gold data must be versioned.
- Benchmark corrections must be reviewed and explained.
- A pipeline tuning task must not silently edit benchmark labels.
- Hard negatives are protected evidence. Changing them requires a benchmark
  correction task and progress entry.
- Holdout sets must remain holdout sets.

## Report Requirements

Each evaluation writes:

- JSON summary with config, git commit, dirty status, profile, corpus, limits,
  query count, metrics, and failures.
- Markdown summary with decision, metrics, caveats, and next action.

Reports must state:

- what was tested;
- what was not tested;
- whether the gate passed;
- why the next phase is allowed or blocked;
- rollback implications.

## ColBERT Advancement Gate

`baseline + ColBERT` can advance only if it proves:

- better gold recall/ranking than baseline under declared metrics;
- no material hard-negative regression;
- CPU latency is acceptable for the target mode;
- RAM and disk use fit the target deployment;
- index build is resumable and reproducible;
- model/schema/index isolation is proven.

If gains are ambiguous, the experiment remains isolated.

## Smoke Tests

Smoke tests are not proof of quality. They are used to detect:

- broken endpoint wiring;
- invalid config;
- missing collection/sidecar;
- model path problems;
- response schema regressions;
- fail-open verifier behavior.

Passing smoke does not permit rollout without benchmark gates.

