# Legal Q&A Retrieval Benchmark Plan

**Date:** 2026-07-09  
**Scope:** Retrieval-only benchmark harness over BGE-M3 hybrid RAG candidate (no LLM synthesis in v1).

---

## 1. Goal

Measure hybrid retrieval quality before adding answer synthesis:

```text
question → BGE-M3 dense + BM25 + RRF → top_k chunks → metrics
```

Phase 2 (later): retrieval → synthesis → answer quality scoring.

---

## 2. Target runtime

| Item | Value |
|------|-------|
| Collection | `nalus_us_bge_m3_rag_combined_20260709` |
| Profile | `nalus_bge_m3_dense_bm25_rrf_v1` |
| Retrieval | Local `HybridBgeM3Retriever` (no `nalus-legal-rag`) |
| Embedding cache | **Disabled** for baseline (`EMBEDDING_CACHE_ENABLED=0`) |
| Production | `nalus_live` / aliases **not touched** |

---

## 3. Dataset layout

```
artifacts/rag_eval/legal_qa/datasets/
  usoud_qa_v1.jsonl      # 20 ÚS questions
  nsoud_qa_v1.jsonl      # 10 NSoud questions
  mixed_qa_v1.jsonl      # 10 cross-court / comparative questions
```

### Item schema

```json
{
  "id": "usoud-qa-001",
  "corpus": "usoud",
  "question": "...",
  "expected_answer_points": ["..."],
  "expected_source_constraints": {
    "court": null,
    "source": null,
    "case_reference": null,
    "source_document_id": null,
    "decision_date": null
  },
  "expected_keywords": ["..."],
  "forbidden_answer_patterns": [],
  "difficulty": "easy",
  "legal_topic": "...",
  "evaluation_type": "retrieval",
  "source_pending": true
}
```

**v1 rule:** No invented case references. `source_pending=true` until human-verified gold sources exist.

---

## 4. Metrics (retrieval-only v1)

| Metric | Definition |
|--------|------------|
| `hit@k` | Question passes if any top-k hit matches keywords **or** source constraints (when not pending) |
| `keyword_coverage` | Share of `expected_keywords` found in top-k hit texts (case-insensitive) |
| `source_constraint_match` | Share of non-null constraints satisfied by any top-k hit (only when `source_pending=false`) |

Pass condition per hit (v1, `source_pending=true`):

- At least **one** `expected_keyword` appears in chunk text or metadata string fields.

---

## 5. Runner

```powershell
python scripts/run_legal_qa_benchmark.py `
  --dataset artifacts/rag_eval/legal_qa/datasets/usoud_qa_v1.jsonl `
  --collection-name nalus_us_bge_m3_rag_combined_20260709 `
  --top-k 10 `
  --retrieval-only `
  --limit 3 `
  --output-dir artifacts/rag_eval/legal_qa/runs/smoke
```

Outputs per run:

```
artifacts/rag_eval/legal_qa/runs/<timestamp>/
  retrieval_results.jsonl
  metrics.json
  failures.md
  summary.md
```

---

## 6. Redis A/B plan

| Run | `EMBEDDING_CACHE_ENABLED` | Purpose |
|-----|---------------------------|---------|
| Baseline | `0` | Quality + latency without cache |
| Cached | `1` + `--use-redis-cache` | Same questions, measure cache hits / speed |

Do not compare until baseline metrics exist.

---

## 7. Safety rules

- Read-only Qdrant (search/scroll only)
- No ingest, no alias switch
- No model download
- No MPNet/hash/mock fallback
- Tests use fake retriever / in-memory cache only

---

## 8. Next steps

1. Run full `usoud_qa_v1` (20) retrieval-only benchmark
2. Run `nsoud_qa_v1` + `mixed_qa_v1`
3. Manually verify top hits for 5 questions → set `source_pending=false` + gold constraints
4. Re-run with verified sources for strict hit@k
5. Enable Redis cache and compare latency
6. Phase 2: add synthesis + answer scoring
