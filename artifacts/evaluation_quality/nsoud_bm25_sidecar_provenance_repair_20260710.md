# NSoud BM25 Sidecar Provenance Repair — 2026-07-10

## Scope and Safety

- Task: repair NSoud BM25 sidecar provenance/export without changing retrieval scoring.
- Collection read: `nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1`
- Original sidecar: `storage/rag/bm25/nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1.sqlite`
- Candidate repaired sidecar: `storage/rag/bm25/nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1.provenance_repaired.sqlite`
- Qdrant access: read-only
- Qdrant writes: none
- Dense scoring changed: no
- BM25 scoring formula changed: no
- RRF changed: no
- Re-ingest: not run
- LLM/DeepSeek: not called

## Current Sidecar Audit

- Tables: `bm25_chunks`
- Row count: `1862`
- Columns:
  `chunk_id`, `text`, `document_id`, `source_document_id`, `decision_date`, `chunk_index`, `qdrant_collection`, `retrieval_profile`, `bm25_index_id`
- Provenance columns physically present before repair:
  `document_id`, `source_document_id`, `chunk_index`
- Provenance columns missing before repair:
  `ecli`, `case_number`, `source`

### Blank / Null Audit Before Repair

| field | blank/null rows |
| --- | ---: |
| `document_id` | 1862 |
| `source_document_id` | 1862 |
| `ecli` | 1862 |
| `case_number` | 1862 |
| `chunk_index` | 0 |
| `source` | 1862 |

Additional note:
- `chunk_index = -1` placeholder rows before repair: `1862`

## Repair Method

Implemented:

- `scripts/repair_nsoud_bm25_sidecar_provenance.py`
- richer provenance extraction in `scripts/build_bm25_sidecar_from_qdrant.py`
- richer sidecar metadata hydration in `app/rag/retrieval/bm25_sidecar.py`

Deterministic mapping strategy:

1. read current sidecar rows
2. read Qdrant payloads from the known NSoud collection in read-only mode
3. map rows to Qdrant points strictly by `chunk_id`
4. fail if `chunk_id` is missing or ambiguous
5. preserve original sidecar `text` for BM25 scoring stability
6. enrich only provenance/export fields:
   `document_id`, `source_document_id`, `ecli`, `case_number`, `spisova_znacka`, `court`, `source`, `decision_date`, `chunk_index`
7. write a new candidate sidecar instead of overwriting the original

No fuzzy matching, no guessed ECLI, no Qdrant mutation.

## Repaired Sidecar Audit

- Row count after repair: `1862`
- Qdrant payloads matched by `chunk_id`: `1862 / 1862`
- Text mismatches during repair: `0`

### Columns After Repair

`chunk_id`, `text`, `document_id`, `source_document_id`, `ecli`, `case_number`, `spisova_znacka`, `court`, `source`, `decision_date`, `chunk_index`, `qdrant_collection`, `retrieval_profile`, `bm25_index_id`

### Blank / Null Audit After Repair

| field | blank/null rows |
| --- | ---: |
| `document_id` | 0 |
| `source_document_id` | 0 |
| `ecli` | 0 |
| `case_number` | 0 |
| `chunk_index` | 0 |
| `source` | 0 |

Additional note:
- `chunk_index = -1` placeholder rows after repair: `0`

## Expected Source Verification

Expected `nsoud-qa-007` source:
`ECLI:CZ:NS:2025:5.TDO.1086.2024.1`

Verification result:

- searchable in repaired sidecar metadata: yes
- matching sidecar rows by `document_id` / `source_document_id` / `ecli`: `12`
- BM25/hybrid retrieval artifacts can now expose this source directly

## Benchmark Comparison

Frozen baseline:

- run: `artifacts/rag_eval/legal_qa/runs/nsoud_full_baseline`
- `hit@1 = 0.700`
- `hit@3 = 0.900`
- `hit@5 = 1.000`
- `hit@10 = 1.000`
- `mean_keyword_coverage = 0.833`
- `pass_rate = 1.000`
- `mean_source_constraint_match = null` in the frozen metrics payload

Candidate repaired sidecar run:

- run: `artifacts/rag_eval/legal_qa/runs/nsoud_sidecar_provenance_repaired`
- `hit@1 = 0.700`
- `hit@3 = 0.900`
- `hit@5 = 1.000`
- `hit@10 = 1.000`
- `mean_keyword_coverage = 0.833`
- `pass_rate = 1.000`
- `mean_source_constraint_match = 1.000`
- `source_hit@1 = 0.750`

Interpretation:

- retrieval scoring behavior did not regress
- top-k benchmark rates stayed unchanged
- the main change is provenance visibility, not score/rank logic

## `nsoud-qa-007` Impact

Candidate retrieval run now exposes:

- `rank 1`
- `chunk_id = 735`
- `document_id = ECLI:CZ:NS:2025:5.TDO.1086.2024.1`
- `source_document_id = ECLI:CZ:NS:2025:5.TDO.1086.2024.1`
- `ecli = ECLI:CZ:NS:2025:5.TDO.1086.2024.1`
- `case_reference = 5 Tdo 1086/2024`

Candidate-only diagnostics conclusion:

- `expected_source_present_top_k = true`
- `true_retrieval_miss = false`
- conclusion:
  `Conservative conclusion: expected source is present; this is not a true retrieval miss.`

After repair, `nsoud-qa-007` is no longer a provenance-driven retrieval miss. It remains only a `partial support` answer-eval case.

## Answer Eval Impact

Baseline NSoud no-LLM answer eval:

- `usable_support_rate_gold = 0.50`
- `partial_support_count = 2`
- `boilerplate_noise_count = 1`
- `unsupported_answer_risk_count = 2`

Candidate repaired-sidecar answer eval:

- run: `artifacts/rag_eval/legal_qa/answer_eval/nsoud_sidecar_provenance_repaired`
- `gold_question_count = 4`
- `partial_support_count = 3`
- `boilerplate_noise_count = 1`
- `citation_available_count = 3`
- `citation_available_rate_gold = 0.75`
- `usable_support_rate_gold = 0.75`
- `gold_retrieval_miss_count = 1`
- `gold_retrieval_miss_rate = 0.25`
- `unsupported_answer_risk_count = 1`

Important detail:

- candidate-only diagnostics for `nsoud-qa-007` show `true_retrieval_miss = false`
- the remaining real risk is `nsoud-qa-010`, which stays `unsupported_boilerplate_or_gap`
- current diagnostics final-status enum still resolves to `FAIL_WITH_REAL_NSOUD_RISK` because `nsoud-qa-010` remains a real NSoud answer-support risk

## Re-ingest Assessment

- Re-ingest needed: `no`
- Reason:
  the expected NSoud sources were already in Qdrant; the failure was sidecar provenance/export, not missing collection content

## Limitations

- `court` and `spisova_znacka` remain blank where they are not present in Qdrant payloads; the repair does not invent them
- this task repaired candidate sidecar/export visibility only; it did not attempt to redesign `nsoud-qa-010`
- frozen baseline artifacts were not overwritten
