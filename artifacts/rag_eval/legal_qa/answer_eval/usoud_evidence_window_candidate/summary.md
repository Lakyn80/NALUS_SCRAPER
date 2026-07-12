# No-LLM answer eval summary

- Generated: 2026-07-12T19:22:27Z
- Dataset: `C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper\artifacts\rag_eval\legal_qa\datasets\usoud_qa_v1.jsonl`
- Retrieval results: `C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper\artifacts\rag_eval\legal_qa\runs\usoud_full_baseline\retrieval_results.jsonl`
- Gold review: `C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper\artifacts\rag_eval\legal_qa\gold_source_review_20260709.md`
- Mode: deterministic no-LLM
- Citation required: True
- Evidence window enabled: True

## Interpretation

- `direct` = strict pass (document gold + snippet support)
- `partial` = usable support, not full direct answer pass
- `gap` / `boilerplate_noise` = must not generate a confident answer
- `corpus_only` = corpus routing only, no document citation

## Support breakdown (gold items)

- direct_support_count: 7
- partial_support_count: 3
- gap_count: 0
- boilerplate_noise_count: 0
- corpus_only_count: 0

## Rates

- strict_direct_pass_rate_all: 0.350
- strict_direct_pass_rate_gold: 0.700
- usable_support_rate_gold: 1.000
- citation_available_rate_gold: 1.000
- corpus_routing_support_rate: 0.000
- unsupported_risk_rate_gold: 0.000
- gold_retrieval_miss_rate: 0.000
- answer_eval_pass_rate (alias): 0.350
- answer_eval_partial_rate: 0.150
- answer_eval_gap_rate: 0.000

## Risk / coverage

- total questions: 20
- gold available: 10
- missing gold: 10
- evaluable questions: 10
- not evaluable (missing gold): 10
- citation available count: 10
- unsupported_answer_risk_count: 0
- gold_retrieval_miss_count: 0
- skipped: 10
- needs review: 0
- evidence_window_used_count: 10
- evidence_window_failed_count: 0
- evidence_window_truncated_count: 0
- same_document_neighbor_count: 20

