# NSoud Evidence-Window Evaluation - 2026-07-12

- Generated: `2026-07-12T17:59:12+03:00`
- Candidate run: `nsoud_evidence_window_candidate`
- Baseline run: `nsoud_dataset_repaired`
- Retrieval source: `artifacts/rag_eval/legal_qa/runs/nsoud_dataset_repaired/retrieval_results.jsonl`
- Dataset: `artifacts/rag_eval/legal_qa/datasets/nsoud_qa_v1.jsonl`
- Neighbor source: `storage/rag/bm25/nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1.provenance_repaired.sqlite`
- Qdrant access: none for this candidate evaluation
- Retrieval/embedding/BM25/RRF changes: none

## Architecture

Evidence-window handling was added in the evaluation layer. Retrieval results are read as before, and the evaluator optionally constructs a bounded same-document window for the verified gold hit. Retrieval ranks, scores, hit order, `top_k`, dense scoring, BM25 scoring, RRF, BGE-M3, Qdrant collections, Redis, and LLM behavior are unchanged.

The implementation is isolated in `app/rag/eval/evidence_window.py`. `legal_answer_eval.py` calls it only when an explicit `EvidenceWindowConfig.enabled=true` configuration is supplied by the CLI.

## Configuration

- `--evidence-window`
- previous chunks: `1`
- following chunks: `1`
- max chunks: `3`
- max characters: `6000`
- same-document boundary required: `true`
- sidecar path: repaired NSoud BM25 sidecar above

## Deterministic Ordering And Safety

The evaluator identifies the anchor by verified gold ECLI/source document, then validates `source_document_id`, `document_id`, `ecli`, and `chunk_index`. Chunks are selected by numeric adjacency and ordered by `chunk_index`, with `chunk_id` retained in diagnostics. Any contradictory provenance fails the window for that hit. No fuzzy matching is used and no cross-document chunks are included.

Text assembly is bounded by configured chunk count and character count. Truncation is reported explicitly in result rows and summary counters.

## Metrics

| Metric | `nsoud_dataset_repaired` | `nsoud_evidence_window_candidate` |
| --- | ---: | ---: |
| gold | 4 | 4 |
| direct_support_count | 0 | 3 |
| partial_support_count | 3 | 1 |
| gap_count | 1 | 0 |
| boilerplate_noise_count | 0 | 0 |
| usable_support_rate_gold | 0.75 | 1.0 |
| citation_available_rate | 0.75 | 1.0 |
| unsupported_answer_risk_count | 1 | 0 |
| strict_direct_pass_rate_gold | 0.0 | 0.75 |
| evidence_window_used_count | n/a | 4 |
| evidence_window_failed_count | n/a | 0 |
| evidence_window_truncated_count | n/a | 1 |
| same_document_neighbor_count | n/a | 8 |

## nsoud-qa-010

- Original anchor chunk: `1644`, rank `4`, ECLI `ECLI:CZ:NS:2025:29.NSCR.1.2025.1`
- Original 240-character snippet: starts at the factual/procedural opening of the reasons and ends before the doctrine.
- Included same-document chunks: `1643`, `1644`, `1645`
- Included chunk indexes: `1`, `2`, `3`
- Combined evidence length: `3952`
- Evidence source: `bm25_sidecar`
- Relevant doctrine visible after window: yes. Chunk `1644` contains that dovolani against a decision rejecting appeal is not objectively admissible under `§ 238 odst. 1 písm. e/ o. s. ř.` and that the remedy is `žaloba pro zmatečnost podle § 229 odst. 4 o. s. ř.`
- Support before: `gap`, keyword coverage `0.0`, citation unavailable, unsupported risk `true`
- Support after: `partial`, keyword coverage `1.0`, citation available, unsupported risk `false`
- Conclusion: the issue was evidence export truncation, not retrieval ranking. The gold source was already present at rank 4.

## nsoud-qa-003

- Support keywords: `přípustnost`, `dovolání`, `občanský`
- Original matching keywords: `přípustnost`, `dovolání`
- Evidence-window matching keywords: all three
- Coverage before: `2/3 = 0.6667`
- Coverage after: `3/3 = 1.0`
- Result before: `partial`
- Result after: `direct`
- Threshold: unchanged at the existing strict gate. No morphology or threshold rule was changed.

## Other Gold Items

- `nsoud-qa-004`: `partial -> direct`, keyword coverage `0.0 -> 1.0`, citation remained available.
- `nsoud-qa-007`: `partial -> direct`, keyword coverage `0.3333 -> 1.0`, citation remained available.
- No NSoud gold item regressed.

## Limitations

- Evidence windows improve evaluator visibility only; they do not make retrieval select a different rank.
- `nsoud-qa-010` remains `partial`, not `direct`, because the verified gold hit is rank 4 and the strict-direct definition still requires rank 1.
- `nsoud-qa-003` is the only truncated evidence window in the candidate; truncation is explicit and bounded at 6000 characters.
- The candidate used the repaired local BM25 sidecar and did not need Qdrant lookup.

## Threshold And Retrieval Assessment

Retrieval is not implicated by this candidate. The previously weak cases were caused by exported evidence visibility, not missing gold documents. No threshold change is needed for this task, and no threshold was changed.

## Next Recommended Task

Decide whether evidence-window mode should become the default for offline no-LLM legal answer evaluation runs after one more corpus-level validation pass for ÚS and Mixed, still without changing retrieval ranking or evaluator thresholds.
