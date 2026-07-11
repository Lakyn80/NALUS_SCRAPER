# Failed cases diagnostic

- Run timestamp: 2026-07-11T06:49:40Z
- Run name: `nsoud_sidecar_provenance_repaired`
- Evaluated datasets: `C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper\artifacts\rag_eval\legal_qa\datasets\nsoud_qa_v1.jsonl`
- Metric names: strict_direct_pass_rate_all, strict_direct_pass_rate_gold, usable_support_rate_gold, citation_available_rate_gold, unsupported_risk_rate_gold, gold_retrieval_miss_rate, support breakdown, gold count
- Metric values: strict_all=0.000, strict_gold=0.000, usable_gold=0.750, citation_rate_gold=0.750, unsupported_risk_gold=0.250, retrieval_miss_gold=0.250
- Gold count: 4
- Total evaluated count: 10

## Failure category breakdown

- metric_denominator_warning: 1
- not_evaluable_missing_gold: 6
- unsupported_boilerplate_or_gap: 1
- usable_partial_support: 3

## Top failed cases

- nsoud-qa-001: not_evaluable_missing_gold | status=skipped | support=gap
- nsoud-qa-002: not_evaluable_missing_gold | status=skipped | support=gap
- nsoud-qa-003: usable_partial_support | status=partial | support=partial
- nsoud-qa-004: usable_partial_support | status=partial | support=partial
- nsoud-qa-005: not_evaluable_missing_gold | status=skipped | support=gap

## Interpretation

- strict_direct_pass_rate is intentionally conservative
- usable_support_rate_gold is the practical support metric
- missing gold does not mean retrieval failure
- corpus_only mixed items are not expected to have document citations
- NSoud unsupported-risk items require manual review

## Cause assessment

- usable partial support diagnostics: 3
- retrieval/support caused failures: 1
- bad/incomplete gold data caused failures: 6
- re-ingest needed: unknown
- production Qdrant touched: no
- final status: FAIL_WITH_REAL_NSOUD_RISK
- status reason: NSoud contains real unsupported or retrieval-risk items that require manual review.

## Notes

- Gold denominator is small (4); percentage metrics are unstable.

## nsoud-qa-007 diagnostic

- gold source id: ECLI:CZ:NS:2025:5.TDO.1086.2024.1
- retrieved top-k ids: ECLI:CZ:NS:2025:5.TDO.1086.2024.1, ECLI:CZ:NS:2024:11.TDO.765.2024.1, ECLI:CZ:NS:2025:4.TDO.1137.2024.1, ECLI:CZ:NS:2025:3.TDO.53.2025.1, ECLI:CZ:NS:2025:11.TDO.75.2025.1, ECLI:CZ:NS:2025:6.TDO.21.2025.1, ECLI:CZ:NS:2025:3.TDO.1120.2024.1, ECLI:CZ:NS:2025:11.TDO.875.2024.1, ECLI:CZ:NS:2024:8.TDO.760.2024.1, ECLI:CZ:NS:2024:6.TDO.976.2024.1
- expected source absent in top-k: False
- true_retrieval_miss: False
- gold_annotation_mismatch: False
- answer_support_gap: False
- matcher_issue: False
- question_too_generic: False
- conclusion: Conservative conclusion: expected source is present; this is not a true retrieval miss.
