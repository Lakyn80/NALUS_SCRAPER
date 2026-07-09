# ÚS Retrieval Baseline — 2026-07-09

**Frozen baseline** for BGE-M3 hybrid retrieval-only evaluation over Ústavní soud judgments.

---

## Run metadata

| Field | Value |
|-------|-------|
| Generated | `2026-07-09T15:43:13Z` |
| Dataset | `artifacts/rag_eval/legal_qa/datasets/usoud_qa_v1.jsonl` |
| Collection | `nalus_us_bge_m3_rag_combined_20260709` |
| Question count | 20 |
| Top-k | 10 |
| Retrieval only | **true** (no LLM answer synthesis) |
| Redis embedding cache | **false** |
| Run output | `artifacts/rag_eval/legal_qa/runs/usoud_full_baseline/` |

---

## Metrics

| Metric | Value |
|--------|-------|
| hit@1 | **0.750** |
| hit@3 | **1.000** |
| hit@5 | **1.000** |
| hit@10 | **1.000** |
| mean keyword coverage | **0.883** |
| pass rate | **1.000** |
| mean source constraint match | n/a (all items `source_pending=true`) |

---

## Evaluation scope

This baseline measures **retrieval quality only**:

- question → BGE-M3 dense + BM25 + RRF → top_k chunks
- pass = at least one `expected_keyword` found in top-k hit text/metadata

It does **not** measure:

- LLM answer correctness
- citation accuracy to a specific spisová značka / ECLI
- synthesis quality

---

## Source constraints

All 20 items have `source_pending=true`. No gold `case_reference`, `source_document_id`, or `decision_date` constraints are enforced yet.

Keyword-based hit@k is a **proxy** until human-verified gold sources are annotated.

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

---

## Use as reference

Compare future runs against this file. Do not change retrieval logic before completing NSoud + mixed baselines and gold-source annotation review.
