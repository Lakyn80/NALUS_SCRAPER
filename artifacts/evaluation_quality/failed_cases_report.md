# NALUS evaluation quality diagnostic

- Run timestamp: 2026-07-12T15:01:28Z
- Evaluated datasets: `C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper\artifacts\rag_eval\legal_qa\datasets\mixed_qa_v1.jsonl`, `C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper\artifacts\rag_eval\legal_qa\datasets\nsoud_qa_v1.jsonl`, `C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper\artifacts\rag_eval\legal_qa\datasets\nsoud_qa_v1.jsonl`, `C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper\artifacts\rag_eval\legal_qa\datasets\nsoud_qa_v1.jsonl`, `C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper\artifacts\rag_eval\legal_qa\datasets\nsoud_qa_v1.jsonl`, `C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper\artifacts\rag_eval\legal_qa\datasets\usoud_qa_v1.jsonl`
- Metric names: strict_direct_pass_rate_all, strict_direct_pass_rate_gold, usable_support_rate_gold, citation_available_rate_gold, unsupported_risk_rate_gold, gold_retrieval_miss_rate, support breakdown, gold count

## Metric values

- mixed_no_llm_baseline (mixed): strict_all=0.000, strict_gold=0.000, usable_gold=1.000, citation_rate_gold=0.000, unsupported_risk_gold=0.000, retrieval_miss_gold=0.000, gold=8, total=10, status=WARN
- mixed_no_llm_baseline denominator warning: Gold denominator is small (8); percentage metrics are unstable.
- nsoud_dataset_repaired (nsoud): strict_all=0.000, strict_gold=0.000, usable_gold=0.750, citation_rate_gold=0.750, unsupported_risk_gold=0.250, retrieval_miss_gold=0.000, gold=4, total=10, status=FAIL_WITH_REAL_NSOUD_RISK
- nsoud_dataset_repaired denominator warning: Gold denominator is small (4); percentage metrics are unstable.
- nsoud_evidence_window_candidate (nsoud): strict_all=0.300, strict_gold=0.750, usable_gold=1.000, citation_rate_gold=1.000, unsupported_risk_gold=0.000, retrieval_miss_gold=0.000, gold=4, total=10, status=WARN
- nsoud_evidence_window_candidate denominator warning: Gold denominator is small (4); percentage metrics are unstable.
- nsoud_no_llm_baseline (nsoud): strict_all=0.000, strict_gold=0.000, usable_gold=0.250, citation_rate_gold=0.250, unsupported_risk_gold=0.750, retrieval_miss_gold=0.500, gold=4, total=10, status=FAIL_WITH_REAL_NSOUD_RISK
- nsoud_no_llm_baseline denominator warning: Gold denominator is small (4); percentage metrics are unstable.
- nsoud_sidecar_provenance_repaired (nsoud): strict_all=0.000, strict_gold=0.000, usable_gold=0.500, citation_rate_gold=0.500, unsupported_risk_gold=0.500, retrieval_miss_gold=0.500, gold=4, total=10, status=FAIL_WITH_REAL_NSOUD_RISK
- nsoud_sidecar_provenance_repaired denominator warning: Gold denominator is small (4); percentage metrics are unstable.
- usoud_no_llm_baseline (usoud): strict_all=0.050, strict_gold=0.100, usable_gold=1.000, citation_rate_gold=1.000, unsupported_risk_gold=0.000, retrieval_miss_gold=0.000, gold=10, total=20, status=WARN

## Failure category breakdown

- corpus_only_no_document_citation_expected: 8
- metric_denominator_warning: 5
- not_evaluable_missing_gold: 36
- true_retrieval_miss: 3
- unsupported_boilerplate_or_gap: 3
- usable_partial_support: 12
- weak_partial_support: 4

## Top diagnostic entries

- mixed_no_llm_baseline / mixed-qa-001: corpus_only_no_document_citation_expected | status=partial | support=corpus_only
- mixed_no_llm_baseline / mixed-qa-002: corpus_only_no_document_citation_expected | status=partial | support=corpus_only
- mixed_no_llm_baseline / mixed-qa-003: corpus_only_no_document_citation_expected | status=partial | support=corpus_only
- mixed_no_llm_baseline / mixed-qa-004: not_evaluable_missing_gold | status=skipped | support=gap
- mixed_no_llm_baseline / mixed-qa-005: corpus_only_no_document_citation_expected | status=partial | support=corpus_only
- mixed_no_llm_baseline / mixed-qa-006: corpus_only_no_document_citation_expected | status=partial | support=corpus_only
- mixed_no_llm_baseline / mixed-qa-007: corpus_only_no_document_citation_expected | status=partial | support=corpus_only
- mixed_no_llm_baseline / mixed-qa-008: corpus_only_no_document_citation_expected | status=partial | support=corpus_only
- mixed_no_llm_baseline / mixed-qa-009: corpus_only_no_document_citation_expected | status=partial | support=corpus_only
- mixed_no_llm_baseline / mixed-qa-010: not_evaluable_missing_gold | status=skipped | support=gap

## Assessment

- diagnostic entries include real failures, not-evaluable missing-gold items, usable partial support, and corpus-only routing cases
- strict_direct_pass_rate is intentionally conservative
- usable_support_rate_gold is the practical support metric
- missing gold does not mean RAG failure
- corpus_only mixed items are not expected to have document citations

- usable_partial_support: 12
- weak_partial_support: 4
- retrieval/support caused failures: 6
- bad/incomplete gold data caused failures: 36
- re-ingest actually needed: unknown
- reason: Retrieval misses exist, but this diagnostic does not prove that re-ingest is required.
- production Qdrant was touched: no
- exact commands run: `python scripts/generate_legal_answer_eval_diagnostics.py --runs-dir artifacts\rag_eval\legal_qa\answer_eval --output-dir artifacts\evaluation_quality`
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
