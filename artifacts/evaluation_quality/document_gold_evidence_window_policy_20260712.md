# Document-gold evidence-window default policy

Generated: 2026-07-12

## Scope

This change makes deterministic same-document evidence windows the default only for offline legal answer evaluation when the item is document-gold and the run is no-LLM.

Evidence windows remain skipped or disabled for corpus-only gold, missing gold, LLM mode without explicit request, retrieval-only benchmarks, live RAG runtime, and invalid/missing provenance.

## Policy behavior

- Default CLI policy: `document_gold`.
- Explicit disable: `--no-evidence-window` or `--evidence-window-policy off`.
- Explicit enable: existing `--evidence-window` or `--evidence-window-policy all`.
- Conflicting flag combinations are rejected before evaluation.
- Corpus-only skips are recorded as `corpus_only_gold` and are not counted as failures.
- Invalid provenance produces `missing_or_invalid_provenance`; no neighboring chunks are guessed.

## Candidate runs

All commands were run without the old explicit `--evidence-window` flag.

### ÚS — `usoud_document_gold_default`

- direct_support_count: 7
- partial_support_count: 3
- usable_support_rate_gold: 1.0
- citation_available_rate_gold: 1.0
- unsupported_answer_risk_count: 0
- strict_direct_pass_rate_gold: 0.7
- evidence_window_used_count: 10
- evidence_window_failed_count: 0
- evidence_window_default_activated_count: 10

### NSoud — `nsoud_document_gold_default`

- direct_support_count: 3
- partial_support_count: 1
- usable_support_rate_gold: 1.0
- citation_available_rate_gold: 1.0
- unsupported_answer_risk_count: 0
- strict_direct_pass_rate_gold: 0.75
- evidence_window_used_count: 4
- evidence_window_failed_count: 0
- evidence_window_default_activated_count: 4

### Mixed — `mixed_document_gold_default`

- corpus_only_count: 8
- usable_support_rate_gold: 1.0
- citation_available_rate_gold: 0.0
- corpus_routing_support_rate: 1.0
- unsupported_answer_risk_count: 0
- evidence_window_used_count: 0
- evidence_window_failed_count: 0
- evidence_window_corpus_only_skipped_count: 8

## Safety checks

- Retrieval ranking/order/scores changed: no.
- Strict thresholds changed: no.
- Dense scoring changed: no.
- BM25 scoring changed: no.
- RRF changed: no.
- BGE-M3/model behavior changed: no.
- Qdrant modified: no.
- Redis enabled or used: no.
- LLM/DeepSeek called: no.
- Grafana queries changed: no.
- Cross-document leakage found: no.
- Corpus-only citation fabrication found: no.

## Conclusion

The `document_gold` policy is safe as the default for offline no-LLM document-gold answer evaluation. It preserves the validated ÚS and NSoud improvements while keeping Mixed corpus-only routing citation-free and failure-free.
