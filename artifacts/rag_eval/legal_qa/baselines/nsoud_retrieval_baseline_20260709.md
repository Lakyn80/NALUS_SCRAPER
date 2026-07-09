# NSoud Retrieval Baseline — 2026-07-09

**Frozen baseline** for BGE-M3 hybrid retrieval-only evaluation over Nejvyšší soud judgments.

---

## Run metadata

| Field | Value |
|-------|-------|
| Generated | `2026-07-09T16:34:10Z` |
| Dataset | `artifacts/rag_eval/legal_qa/datasets/nsoud_qa_v1.jsonl` |
| Collection | `nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1` |
| BM25 sidecar | `storage/rag/bm25/nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1.sqlite` |
| Question count | 10 |
| Top-k | 10 |
| Retrieval only | **true** (no LLM answer synthesis) |
| Redis embedding cache | **false** |
| Provenance backfill | `nsoud-bge-m3-provenance-backfill-v1` (1,862 points) |
| Run output | `artifacts/rag_eval/legal_qa/runs/nsoud_full_baseline/` |

---

## Metrics

| Metric | Value |
|--------|-------|
| hit@1 | **0.700** |
| hit@3 | **0.900** |
| hit@5 | **1.000** |
| hit@10 | **1.000** |
| mean keyword coverage | **0.833** |
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

All 10 items have `source_pending=true`. No gold `case_reference`, `source_document_id`, or `decision_date` constraints are enforced yet.

Keyword-based hit@k is a **proxy** until human-verified gold sources are annotated.

---

## Production safety (this run)

| Check | Status |
|-------|--------|
| Qdrant writes | Provenance backfill only on eval collection (not production aliases) |
| Ingest | Not run |
| `nalus_live` | Untouched |
| Alias switch | None |
| Model download | None (offline HF cache) |
| Redis cache | Disabled |

---

## Use as reference

Compare future NSoud runs against this file. Cross-corpus comparison with ÚS baseline is now possible for retrieval-only keyword proxy metrics.
