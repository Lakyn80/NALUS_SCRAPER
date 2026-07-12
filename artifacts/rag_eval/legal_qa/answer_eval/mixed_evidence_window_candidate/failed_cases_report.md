# Failed cases diagnostic

- Run timestamp: 2026-07-12T19:22:26Z
- Run name: `mixed_evidence_window_candidate`
- Evaluated datasets: `C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper\artifacts\rag_eval\legal_qa\datasets\mixed_qa_v1.jsonl`
- Metric names: strict_direct_pass_rate_all, strict_direct_pass_rate_gold, usable_support_rate_gold, citation_available_rate_gold, unsupported_risk_rate_gold, gold_retrieval_miss_rate, support breakdown, gold count
- Metric values: strict_all=0.000, strict_gold=0.000, usable_gold=1.000, citation_rate_gold=0.000, unsupported_risk_gold=0.000, retrieval_miss_gold=0.000
- Gold count: 8
- Total evaluated count: 10

## Failure category breakdown

- corpus_only_no_document_citation_expected: 8
- metric_denominator_warning: 1
- not_evaluable_missing_gold: 2

## Top failed cases

- mixed-qa-001: corpus_only_no_document_citation_expected | status=partial | support=corpus_only
- mixed-qa-002: corpus_only_no_document_citation_expected | status=partial | support=corpus_only
- mixed-qa-003: corpus_only_no_document_citation_expected | status=partial | support=corpus_only
- mixed-qa-004: not_evaluable_missing_gold | status=skipped | support=gap
- mixed-qa-005: corpus_only_no_document_citation_expected | status=partial | support=corpus_only

## Interpretation

- strict_direct_pass_rate is intentionally conservative
- usable_support_rate_gold is the practical support metric
- missing gold does not mean retrieval failure
- corpus_only mixed items are not expected to have document citations
- NSoud unsupported-risk items require manual review

## Cause assessment

- usable partial support diagnostics: 0
- retrieval/support caused failures: 0
- bad/incomplete gold data caused failures: 2
- re-ingest needed: no
- production Qdrant touched: no
- final status: WARN
- status reason: Main issues are missing gold coverage, conservative strict-pass gating, or small denominators.

## Notes

- Gold denominator is small (8); percentage metrics are unstable.
