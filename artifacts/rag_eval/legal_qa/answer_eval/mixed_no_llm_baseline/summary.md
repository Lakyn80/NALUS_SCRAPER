# No-LLM answer eval summary

- Generated: 2026-07-11T21:18:40Z
- Dataset: `C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper\artifacts\rag_eval\legal_qa\datasets\mixed_qa_v1.jsonl`
- Retrieval results: `C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper\artifacts\rag_eval\legal_qa\runs\mixed_two_pass_baseline\retrieval_results.jsonl`
- Gold review: `C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper\artifacts\rag_eval\legal_qa\gold_source_review_20260709.md`
- Mode: deterministic no-LLM
- Citation required: True

## Interpretation

- `direct` = strict pass (document gold + snippet support)
- `partial` = usable support, not full direct answer pass
- `gap` / `boilerplate_noise` = must not generate a confident answer
- `corpus_only` = corpus routing only, no document citation

## Support breakdown (gold items)

- direct_support_count: 0
- partial_support_count: 0
- gap_count: 0
- boilerplate_noise_count: 0
- corpus_only_count: 8

## Rates

- strict_direct_pass_rate_all: 0.000
- strict_direct_pass_rate_gold: 0.000
- usable_support_rate_gold: 1.000
- citation_available_rate_gold: 0.000
- corpus_routing_support_rate: 1.000
- unsupported_risk_rate_gold: 0.000
- gold_retrieval_miss_rate: 0.000
- answer_eval_pass_rate (alias): 0.000
- answer_eval_partial_rate: 0.800
- answer_eval_gap_rate: 0.000

## Risk / coverage

- total questions: 10
- gold available: 8
- missing gold: 2
- evaluable questions: 8
- not evaluable (missing gold): 2
- citation available count: 0
- unsupported_answer_risk_count: 0
- gold_retrieval_miss_count: 0
- skipped: 2
- needs review: 0

