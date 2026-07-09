# No-LLM answer eval summary

- Generated: 2026-07-09T19:57:57Z
- Dataset: `C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper\artifacts\rag_eval\legal_qa\datasets\usoud_qa_v1.jsonl`
- Retrieval results: `C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper\artifacts\rag_eval\legal_qa\runs\usoud_gold_eval\retrieval_results.jsonl`
- Gold review: `C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper\artifacts\rag_eval\legal_qa\gold_source_review_20260709.md`
- Mode: deterministic no-LLM
- Citation required: True

## Interpretation

- `direct` = strict pass (document gold + snippet support)
- `partial` = usable support, not full direct answer pass
- `gap` / `boilerplate_noise` = must not generate a confident answer
- `corpus_only` = corpus routing only, no document citation

## Support breakdown (gold items)

- direct_support_count: 1
- partial_support_count: 4
- gap_count: 0
- boilerplate_noise_count: 0
- corpus_only_count: 0

## Rates

- strict_direct_pass_rate_all: 0.050
- strict_direct_pass_rate_gold: 0.200
- usable_support_rate_gold: 1.000
- citation_available_rate: 1.000
- answer_eval_pass_rate (alias): 0.050
- answer_eval_partial_rate: 0.200
- answer_eval_gap_rate: 0.000

## Risk / coverage

- total questions: 20
- gold available: 5
- unsupported_answer_risk_count: 0
- skipped: 15
- needs review: 0

