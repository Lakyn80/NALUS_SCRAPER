# Mixed Retrieval Baseline — 2026-07-09

**Frozen baseline** for two-pass mixed retrieval-only evaluation over Ústavní soud and Nejvyšší soud corpora.

---

## Run metadata

| Field | Value |
|-------|-------|
| Generated | `2026-07-09T16:49:27Z` |
| Dataset | `artifacts/rag_eval/legal_qa/datasets/mixed_qa_v1.jsonl` |
| Mode | two-pass mixed retrieval (eval merge only, not production router) |
| ÚS collection | `nalus_us_bge_m3_rag_combined_20260709` |
| NSoud collection | `nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1` |
| ÚS BM25 sidecar | `storage/rag/bm25/nalus_us_bge_m3_rag_combined_20260709.sqlite` |
| NSoud BM25 sidecar | `storage/rag/bm25/nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1.sqlite` |
| Question count | 10 |
| Corpus-scored questions | 8 (`expected_target_corpus=both`) |
| Ambiguous questions | 2 |
| Top-k | 10 |
| Retrieval only | **true** |
| Redis embedding cache | **false** |
| Mixed merge RRF k | 60 (benchmark-only cross-corpus rank merge) |
| Run output | `artifacts/rag_eval/legal_qa/runs/mixed_two_pass_baseline/` |

---

## Metrics

| Metric | Value |
|--------|-------|
| corpus_hit@1 | **0.000** |
| corpus_hit@3 | **1.000** |
| corpus_hit@5 | **1.000** |
| retrieval_hit@1 | **1.000** |
| retrieval_hit@3 | **1.000** |
| retrieval_hit@5 | **1.000** |
| retrieval_hit@10 | **1.000** |
| mean keyword coverage | **1.000** |
| pass rate | **1.000** |
| usoud_win_rate@1 | **0.000** |
| nsoud_win_rate@1 | **1.000** |
| source_pending_count | 10 |

---

## Interpretation notes

- `corpus_hit@1 = 0` is expected for `expected_target_corpus=both`: a single rank-1 slot cannot contain both corpora.
- `corpus_hit@3` and `corpus_hit@5` are the meaningful corpus-routing proxy metrics for comparative questions.
- `usoud_win_rate@1 = 0` reflects benchmark merge tie-breaking (corpus rank-1 ties resolve to `nsoud` before `usoud` alphabetically), not production retrieval quality.
- All items remain `source_pending=true`; keyword proxy only.

---

## Evaluation scope

This baseline measures **retrieval quality only** with per-corpus two-pass search and benchmark-only cross-corpus RRF rank merge.

It does **not** measure:

- LLM answer correctness
- production corpus routing
- citation accuracy to specific spisová značka / ECLI

---

## Production safety (this run)

| Check | Status |
|-------|--------|
| Qdrant writes | None (read-only search) |
| Ingest | Not run |
| `nalus_live` | Untouched |
| Alias switch | None |
| Model download | None (offline HF cache) |
| Redis cache | Disabled |
| Production retrieval logic | Unchanged |

---

## Use as reference

Compare future mixed two-pass runs against this file. Next phase: gold-source annotation, optional production corpus router, then answer synthesis eval.
