# Failed cases diagnostic

- Run timestamp: 2026-07-12T19:22:27Z
- Run name: `usoud_evidence_window_candidate`
- Evaluated datasets: `C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper\artifacts\rag_eval\legal_qa\datasets\usoud_qa_v1.jsonl`
- Metric names: strict_direct_pass_rate_all, strict_direct_pass_rate_gold, usable_support_rate_gold, citation_available_rate_gold, unsupported_risk_rate_gold, gold_retrieval_miss_rate, support breakdown, gold count
- Metric values: strict_all=0.350, strict_gold=0.700, usable_gold=1.000, citation_rate_gold=1.000, unsupported_risk_gold=0.000, retrieval_miss_gold=0.000
- Gold count: 10
- Total evaluated count: 20

## Failure category breakdown

- not_evaluable_missing_gold: 10
- usable_partial_support: 3

## Top failed cases

- usoud-qa-002: usable_partial_support | status=partial | support=partial
- usoud-qa-005: not_evaluable_missing_gold | status=skipped | support=gap
- usoud-qa-006: not_evaluable_missing_gold | status=skipped | support=gap
- usoud-qa-008: not_evaluable_missing_gold | status=skipped | support=gap
- usoud-qa-012: usable_partial_support | status=partial | support=partial

## Interpretation

- strict_direct_pass_rate is intentionally conservative
- usable_support_rate_gold is the practical support metric
- missing gold does not mean retrieval failure
- corpus_only mixed items are not expected to have document citations
- NSoud unsupported-risk items require manual review

## Cause assessment

- usable partial support diagnostics: 3
- retrieval/support caused failures: 0
- bad/incomplete gold data caused failures: 10
- re-ingest needed: no
- production Qdrant touched: no
- final status: WARN
- status reason: Main issues are missing gold coverage, conservative strict-pass gating, or small denominators.
