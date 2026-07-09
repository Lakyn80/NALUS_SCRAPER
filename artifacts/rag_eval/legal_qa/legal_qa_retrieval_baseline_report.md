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

## 3. Mixed baseline (skipped)

| Field | Value |
|-------|-------|
| Status | **SKIPPED — single-collection runner** |
| Dataset | `artifacts/rag_eval/legal_qa/datasets/mixed_qa_v1.jsonl` (10 questions) |

### Why skipped

Mixed questions compare Ústavní soud vs Nejvyšší soud and cross-court legal concepts. The current runner supports **one Qdrant collection + one BM25 sidecar per run**.

Running mixed dataset against only `nalus_us_bge_m3_rag_combined_20260709` would:

- answer ÚS-flavored questions from ÚS corpus (partially valid)
- answer NSoud-flavored questions from ÚS corpus (invalid for routing test)
- fail comparative/routing questions by design

### Required for valid mixed evaluation

Choose one:

**Option A — Corpus router**

- Route `corpus=usoud` → ÚS collection
- Route `corpus=nsoud` → NSoud collection
- Route `corpus=mixed` → two-pass retrieval with result labeling

**Option B — Two-pass retrieval**

- Retrieve top_k from ÚS collection and NSoud collection separately
- Label hits with `corpus_origin`
- Fuse or present side-by-side for mixed questions

Do not hack single-collection runner for mixed eval.

---

## 4. Limitations (all datasets)

| Limitation | Impact |
|------------|--------|
| `source_pending=true` on all 40 seed items | hit@k uses keyword proxy, not gold case match |
| No LLM synthesis | Cannot assess answer quality yet |
| ÚS collection partial (~13k / full 5y window) | Recall ceiling for older ÚS decisions |
| NSoud collection partial (eval longform subset) | Recall ceiling for older NS decisions |

---

## 5. Next steps

1. Implement corpus router or two-pass mixed retrieval
2. Run `mixed_qa_v1.jsonl`
3. Manually verify top-3 hits for 10 questions → set `source_pending=false` + gold constraints
4. Re-run with strict source-constraint hit@k
5. Optional: Redis cache A/B (`EMBEDDING_CACHE_ENABLED=1`) after all baselines frozen

---

## 6. Should retrieval logic change?

**No change recommended yet.**

ÚS and NSoud baselines show strong keyword retrieval (pass rate 1.0, hit@5–10 = 1.0). The hit@1 gaps (ÚS 0.75, NSoud 0.70) may improve with gold-source eval or RRF tuning, but mixed baseline is still missing — changing RRF/BM25/BGE now would invalidate comparison.

Wait for mixed baseline before tuning.

---

## 7. Safety summary

| Check | Status |
|-------|--------|
| Redis used | **false** |
| Qdrant access | **read-only** (ÚS run only) |
| Production aliases touched | **false** |
| `nalus_live` / `nalus_stable_20260326` | **untouched** |
| Model downloaded | **false** |
| `nalus-legal-rag` imported/modified | **false** |
| Retrieval logic changed | **false** |
