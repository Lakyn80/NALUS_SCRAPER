# NALUS evaluation quality diagnostic

- Run timestamp: 2026-07-11T21:44:59Z
- Evaluated datasets: `C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper\artifacts\rag_eval\legal_qa\datasets\nsoud_qa_v1.jsonl`
- Metric names: strict_direct_pass_rate_all, strict_direct_pass_rate_gold, usable_support_rate_gold, citation_available_rate_gold, unsupported_risk_rate_gold, gold_retrieval_miss_rate, support breakdown, gold count

## Metric values

- nsoud_dataset_repaired (nsoud): strict_all=0.000, strict_gold=0.000, usable_gold=0.750, citation_rate_gold=0.750, unsupported_risk_gold=0.250, retrieval_miss_gold=0.000, gold=4, total=10, status=FAIL_WITH_REAL_NSOUD_RISK
- nsoud_dataset_repaired denominator warning: Gold denominator is small (4); percentage metrics are unstable.

## Failure category breakdown

- metric_denominator_warning: 1
- not_evaluable_missing_gold: 6
- unsupported_boilerplate_or_gap: 1
- usable_partial_support: 2
- weak_partial_support: 1

## Top diagnostic entries

- nsoud_dataset_repaired / nsoud-qa-001: not_evaluable_missing_gold | status=skipped | support=gap
- nsoud_dataset_repaired / nsoud-qa-002: not_evaluable_missing_gold | status=skipped | support=gap
- nsoud_dataset_repaired / nsoud-qa-003: usable_partial_support | status=partial | support=partial
- nsoud_dataset_repaired / nsoud-qa-004: weak_partial_support | status=partial | support=partial
- nsoud_dataset_repaired / nsoud-qa-005: not_evaluable_missing_gold | status=skipped | support=gap
- nsoud_dataset_repaired / nsoud-qa-006: not_evaluable_missing_gold | status=skipped | support=gap
- nsoud_dataset_repaired / nsoud-qa-007: usable_partial_support | status=partial | support=partial
- nsoud_dataset_repaired / nsoud-qa-008: not_evaluable_missing_gold | status=skipped | support=gap
- nsoud_dataset_repaired / nsoud-qa-009: not_evaluable_missing_gold | status=skipped | support=gap
- nsoud_dataset_repaired / nsoud-qa-010: unsupported_boilerplate_or_gap | status=gap | support=gap

## Assessment

- diagnostic entries include real failures, not-evaluable missing-gold items, usable partial support, and corpus-only routing cases
- strict_direct_pass_rate is intentionally conservative
- usable_support_rate_gold is the practical support metric
- missing gold does not mean RAG failure
- corpus_only mixed items are not expected to have document citations

- usable_partial_support: 2
- weak_partial_support: 1
- retrieval/support caused failures: 1
- bad/incomplete gold data caused failures: 6
- re-ingest actually needed: no
- reason: No evidence in this diagnostic that re-ingest is required.
- production Qdrant was touched: no
- exact commands run: `python scripts/generate_legal_answer_eval_diagnostics.py --runs-dir artifacts\rag_eval\legal_qa\answer_eval --output-dir artifacts\rag_eval\legal_qa\answer_eval\nsoud_dataset_repaired --run-name nsoud_dataset_repaired`
- files changed: `artifacts/evaluation_quality/failed_cases_report.json`, `artifacts/evaluation_quality/failed_cases_report.md`, `artifacts/evaluation_quality/metrics_summary.json`, `artifacts/evaluation_quality/metric_failure_categories.json`
- final status: FAIL_WITH_REAL_NSOUD_RISK

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
