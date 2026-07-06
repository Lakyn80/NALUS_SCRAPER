# NALUS RAG Eval — Final Combined Comparison

- Benchmark complete: **True**
- Dataset: **8 questions**, **1862 chunks**
- Completed candidates: **7 / 7**

## Environment fixes applied

- torch upgraded from 2.5.1+cpu to 2.6.0+cpu for bge_m3 (CVE-2025-32434)
- failed hybrid candidates rerun one-by-one (separate artifact_dir per model)
- BM25 package 0.2.0 installed from /tmp copy (writable, not read-only mount)

## Reruns applied

- multilingual_e5_base__dense_plus_bm25 -> 20260705_211457Z
- paraphrase_multilingual_mpnet_base_v2__dense_plus_bm25 -> 20260705_215018Z
- multilingual_e5_large__dense_plus_bm25 -> 20260705_215820Z
- bge_m3__dense_plus_bm25 -> 20260706_092545Z

## Overall winner (among completed)

- `bge_m3__dense_plus_bm25`
- hit_rate: `1.0`
- evidence_marker_coverage: `1.0`
- mrr: `0.9375`
- source: `rerun` / `20260706_092545Z`

## BM25 vs best dense/hybrid: **no**


## Final ranking (completed only)

1. `bge_m3__dense_plus_bm25` (hit_rate=1.0, coverage=1.0, mrr=0.9375, source=rerun)
2. `bm25__bm25` (hit_rate=1.0, coverage=1.0, mrr=0.71875, source=bm25_batch)
3. `multilingual_e5_small__dense_plus_bm25` (hit_rate=1.0, coverage=1.0, mrr=0.6354166666666666, source=bm25_batch)
4. `multilingual_e5_base__dense_plus_bm25` (hit_rate=1.0, coverage=1.0, mrr=0.6354166666666666, source=rerun)
5. `multilingual_e5_large__dense_plus_bm25` (hit_rate=1.0, coverage=1.0, mrr=0.47916666666666663, source=rerun)
6. `paraphrase_multilingual_mpnet_base_v2__dense_plus_bm25` (hit_rate=0.875, coverage=0.875, mrr=0.53125, source=rerun)
7. `multilingual_e5_small__dense` (hit_rate=0.5, coverage=0.5, mrr=0.5, source=dense_baseline)

## Failures / not run

- `multilingual_e5_base__unknown` [FAILED]: timed out
- `paraphrase_multilingual_mpnet_base_v2__unknown` [FAILED]: timed out
