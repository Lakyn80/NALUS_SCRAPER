# Legal Q&A Retrieval Baseline Report

**Date:** 2026-07-09  
**Scope:** Retrieval-only benchmark baselines for ÚS, NSoud, and mixed corpora.

---

## 1. ÚS baseline (complete)

| Field | Value |
|-------|-------|
| Status | **COMPLETE** |
| Dataset | `artifacts/rag_eval/legal_qa/datasets/usoud_qa_v1.jsonl` |
| Collection | `nalus_us_bge_m3_rag_combined_20260709` |
| Questions | 20 |
| BM25 sidecar | `storage/rag/bm25/nalus_us_bge_m3_rag_combined_20260709.sqlite` |
| Redis cache | false |

### Metrics

| hit@1 | hit@3 | hit@5 | hit@10 | keyword coverage | pass rate |
|-------|-------|-------|--------|------------------|-----------|
| 0.750 | 1.000 | 1.000 | 1.000 | 0.883 | 1.000 |

Frozen baseline: `artifacts/rag_eval/legal_qa/baselines/usoud_retrieval_baseline_20260709.md`  
Run artifacts: `artifacts/rag_eval/legal_qa/runs/usoud_full_baseline/`

---

## 2. NSoud baseline (complete)

| Field | Value |
|-------|-------|
| Status | **COMPLETE** |
| Dataset | `artifacts/rag_eval/legal_qa/datasets/nsoud_qa_v1.jsonl` |
| Collection | `nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1` (1,862 points) |
| BM25 sidecar | `storage/rag/bm25/nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1.sqlite` |
| Provenance backfill | `nsoud-bge-m3-provenance-backfill-v1` (1,862 points updated) |
| Redis cache | false |

### Metrics

| hit@1 | hit@3 | hit@5 | hit@10 | keyword coverage | pass rate |
|-------|-------|-------|--------|------------------|-----------|
| 0.700 | 0.900 | 1.000 | 1.000 | 0.833 | 1.000 |

Frozen baseline: `artifacts/rag_eval/legal_qa/baselines/nsoud_retrieval_baseline_20260709.md`  
Run artifacts: `artifacts/rag_eval/legal_qa/runs/nsoud_full_baseline/`

### Cross-corpus snapshot (keyword proxy)

| Corpus | Questions | hit@1 | hit@3 | hit@5 | pass rate |
|--------|-----------|-------|-------|-------|-----------|
| ÚS | 20 | 0.750 | 1.000 | 1.000 | 1.000 |
| NSoud | 10 | 0.700 | 0.900 | 1.000 | 1.000 |

---

## 3. Mixed baseline (complete)

| Field | Value |
|-------|-------|
| Status | **COMPLETE** |
| Dataset | `artifacts/rag_eval/legal_qa/datasets/mixed_qa_v1.jsonl` (10 questions) |
| Mode | two-pass mixed retrieval (`--mixed-two-pass`) |
| ÚS collection | `nalus_us_bge_m3_rag_combined_20260709` |
| NSoud collection | `nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1` |
| ÚS BM25 sidecar | `storage/rag/bm25/nalus_us_bge_m3_rag_combined_20260709.sqlite` (from production config) |
| NSoud BM25 sidecar | `storage/rag/bm25/nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1.sqlite` |
| Corpus-scored questions | 8 (`expected_target_corpus=both`) |
| Ambiguous questions | 2 |
| Redis cache | false |

### Metrics

| corpus_hit@1 | corpus_hit@3 | corpus_hit@5 | retrieval_hit@1 | retrieval_hit@10 | keyword coverage | pass rate |
|--------------|--------------|--------------|-----------------|------------------|------------------|-----------|
| 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

| usoud_win_rate@1 | nsoud_win_rate@1 | source_pending |
|------------------|------------------|----------------|
| 0.000 | 1.000 | 10 / 10 |

Frozen baseline: `artifacts/rag_eval/legal_qa/baselines/mixed_retrieval_baseline_20260709.md`  
Run artifacts: `artifacts/rag_eval/legal_qa/runs/mixed_two_pass_baseline/`

### Mixed eval notes

- Benchmark uses **eval-only** cross-corpus RRF rank merge (k=60); production hybrid RRF/BM25/BGE unchanged.
- `corpus_hit@1=0` is expected for `expected_target_corpus=both` (one rank-1 slot cannot hold both corpora).
- `corpus_hit@3/5=1.0` is the meaningful corpus coverage metric for comparative questions.
- `nsoud_win_rate@1=1.0` is a merge tie-break artifact (rank-1 ties → `nsoud` before `usoud` alphabetically), not a retrieval-quality verdict.

---

## 4. Limitations (all datasets)

| Limitation | Impact |
|------------|--------|
| `source_pending=true` on all 40 seed items | hit@k uses keyword proxy, not gold case match |
| No LLM synthesis / answer eval | Cannot assess answer quality yet |
| ÚS collection partial (~13k / full 5y window) | Recall ceiling for older ÚS decisions |
| NSoud collection partial (eval longform subset) | Recall ceiling for older NS decisions |
| Mixed merge is benchmark-only | No production corpus router yet |
| No gold ECLI / spisová značka constraints | corpus_hit@k is routing proxy only |

---

## 5. Next steps

1. Manually verify top-3 hits for representative questions → set `source_pending=false` + gold constraints
2. Re-run single-corpus and mixed baselines with strict source-constraint metrics
3. Design production corpus router (optional) based on frozen benchmark evidence
4. Add LLM answer synthesis evaluation (separate phase)
5. Optional: Redis cache A/B (`EMBEDDING_CACHE_ENABLED=1`) after gold-source review

---

## 6. Should retrieval logic change?

**No production retrieval change recommended yet.**

All three baselines show pass rate 1.0 on keyword proxy. Single-corpus hit@1 gaps (ÚS 0.75, NSoud 0.70) and mixed `corpus_hit@1=0` need gold-source interpretation before RRF/BM25/BGE tuning.

---

## 7. Safety summary

| Check | Status |
|-------|--------|
| Redis used | **false** |
| Qdrant access | **read-only** (search only) |
| Production aliases touched | **false** |
| `nalus_live` / `nalus_stable_20260326` | **untouched** |
| Model downloaded | **false** |
| `nalus-legal-rag` imported/modified | **false** |
| Production retrieval logic changed | **false** (benchmark-only mixed merge added) |
