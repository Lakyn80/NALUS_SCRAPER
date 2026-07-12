# Cross-Corpus Evidence-Window Validation - 2026-07-12

- Generated: `2026-07-12T22:23:16+03:00`
- Task: `Cross-corpus evidence-window validation`
- Default behavior changed: no
- Retrieval/scoring/threshold changes: none
- Qdrant access: none
- Redis/LLM/model downloads: none

## Reference: NSoud

`nsoud_evidence_window_candidate` remains the reference document-gold result from the prior task.

| Metric | Value |
| --- | ---: |
| gold | 4 |
| direct_support_count | 3 |
| partial_support_count | 1 |
| gap_count | 0 |
| boilerplate_noise_count | 0 |
| usable_support_rate_gold | 1.0 |
| citation_available_rate | 1.0 |
| unsupported_answer_risk_count | 0 |
| strict_direct_pass_rate_gold | 0.75 |
| evidence_window_used_count | 4 |
| evidence_window_failed_count | 0 |
| evidence_window_truncated_count | 1 |

## ÚS Candidate

- Baseline: `usoud_no_llm_baseline`
- Candidate: `usoud_evidence_window_candidate`
- Retrieval artifact: `artifacts/rag_eval/legal_qa/runs/usoud_full_baseline/retrieval_results.jsonl`
- Dataset: `artifacts/rag_eval/legal_qa/datasets/usoud_qa_v1.jsonl`
- Evidence source: `storage/rag/bm25/nalus_us_bge_m3_rag_combined_20260709.sqlite`
- Evidence source mode: SQLite read-only

| Metric | Baseline | Candidate |
| --- | ---: | ---: |
| gold | 10 | 10 |
| direct_support_count | 1 | 7 |
| partial_support_count | 9 | 3 |
| gap_count | 0 | 0 |
| boilerplate_noise_count | 0 | 0 |
| usable_support_rate_gold | 1.0 | 1.0 |
| citation_available_rate | 1.0 | 1.0 |
| unsupported_answer_risk_count | 0 | 0 |
| strict_direct_pass_rate_gold | 0.1 | 0.7 |
| evidence_window_used_count | n/a | 10 |
| evidence_window_failed_count | n/a | 0 |
| evidence_window_truncated_count | n/a | 0 |
| same_document_neighbor_count | n/a | 20 |

Acceptance result: passed. Usable support did not regress, citation availability did not regress, unsupported risk remained zero, and all 10 document-gold evidence windows had valid same-document provenance.

Direct-support improvements came from same-document windows for `usoud-qa-001`, `usoud-qa-003`, `usoud-qa-007`, `usoud-qa-009`, `usoud-qa-010`, and `usoud-qa-011`. No cross-document mismatch was found.

## Mixed Candidate

- Baseline: `mixed_no_llm_baseline`
- Candidate: `mixed_evidence_window_candidate`
- Retrieval artifact: `artifacts/rag_eval/legal_qa/runs/mixed_two_pass_baseline/retrieval_results.jsonl`
- Dataset: `artifacts/rag_eval/legal_qa/datasets/mixed_qa_v1.jsonl`
- Evidence source: none, by design

Mixed gold remains corpus-only. Evidence-window mode was enabled at the CLI level, but document-level evidence windows were skipped for corpus-only gold items.

| Metric | Baseline | Candidate |
| --- | ---: | ---: |
| gold | 8 | 8 |
| direct_support_count | 0 | 0 |
| partial_support_count | 0 | 0 |
| gap_count | 0 | 0 |
| boilerplate_noise_count | 0 | 0 |
| corpus_only_count | 8 | 8 |
| usable_support_rate_gold | 1.0 | 1.0 |
| citation_available_rate | 0.0 | 0.0 |
| corpus_routing_support_rate | 1.0 | 1.0 |
| unsupported_answer_risk_count | 0 | 0 |
| strict_direct_pass_rate_gold | 0.0 | 0.0 |
| evidence_window_used_count | n/a | 0 |
| evidence_window_failed_count | n/a | 0 |
| evidence_window_truncated_count | n/a | 0 |

Acceptance result: passed. `corpus_only_count` remained `8`, `corpus_routing_support_rate` remained `1.0`, `usable_support_rate_gold` remained `1.0`, and `citation_available_rate` remained `0.0`. No document-level direct pass or citation was fabricated.

## Safety Assessment

- Same-document validation: passed for ÚS and NSoud document-gold candidates.
- Corpus-only behavior: passed for Mixed; document evidence windows were skipped, not used or failed.
- Cross-document leakage: none detected.
- Fabricated citations: none detected.
- Raw legal text in Prometheus labels: none; exporter continues to expose bounded aggregate labels only.
- Retrieval implication: none. Retrieval artifacts were reused and no retrieval benchmark was rerun.
- Threshold implication: none. The strict support gate remains unchanged.

## Regressions

No metric regressions were observed in ÚS, NSoud, or Mixed candidate runs.

## Recommendation

Do not enable evidence windows globally.

Recommended future default: enable deterministic evidence windows only for document-gold no-LLM answer evaluation, while keeping corpus-only routing evaluation in its current non-document mode. Mixed corpus-only behavior should continue to skip document evidence windows.

For now, keep the implementation opt-in until a separate scoped default-change task updates CLI defaults, tests, docs, and monitoring expectations.

## Known Limitations

- This validation covers offline no-LLM answer evaluation artifacts, not live retrieval or generation.
- Mixed corpus-only validation confirms skip behavior, not document-window quality.
- ÚS and NSoud improvements show better evaluator visibility; they do not imply retrieval ranking changes.

## Next Recommended Task

Prepare a separate default-policy task that enables evidence windows only for document-gold no-LLM evaluation, with corpus-only skip behavior explicitly documented and tested.
