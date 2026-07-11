# Legal Q&A No-LLM Answer Eval Report — 2026-07-09

Deterministic answer-support evaluation over frozen gold-source annotations. **No LLM was called.**

Update: refreshed on 2026-07-10 after conservative gold expansion for ÚS and mixed.
Update: refreshed again on 2026-07-10 after adding only `nsoud-qa-007` as a conservative NSoud gold item from the read-only provenance check.

---

## Metric interpretation

| Metric | Meaning |
|--------|---------|
| `strict_direct_pass_rate_all` | `direct_support_count / total_question_count`; intentionally conservative strict gate |
| `strict_direct_pass_rate_gold` | `direct_support_count / gold_question_count`; strict only over evaluable gold items |
| `usable_support_rate_gold` | Gold items with `direct`, `partial`, or `corpus_only` support |
| `citation_available_rate_gold` | Gold items where ECLI/chunk citation is present in skeleton; `corpus_only` mixed items are not treated as citation failures |
| `unsupported_risk_rate_gold` | `unsupported_answer_risk_count / gold_question_count` |
| `gold_retrieval_miss_rate` | `gold_retrieval_miss_count / gold_question_count` |
| `not_evaluable_missing_gold_count` | Questions still pending gold; these are **not** retrieval failures |

**Support levels**

| Level | Meaning |
|-------|---------|
| `direct` | Strict pass — document gold + rank-1 snippet with ≥67% keyword overlap |
| `partial` | Usable support, not a full direct answer pass |
| `gap` / `boilerplate_noise` | Must **not** generate a confident answer |
| `corpus_only` | Corpus routing only — no document citation |

**Status interpretation**

- `strict_direct_pass_rate_*` is intentionally conservative.
- `usable_support_rate_gold` is the practical support metric.
- Missing gold does **not** mean RAG failure.
- Mixed `corpus_only` items are expected to have `citation_available_rate_gold=0.0`.
- Real failures are limited to true retrieval misses, invalid gold annotations, or unsupported boilerplate/gap risk on gold items.

---

## Retrieval inputs

| Corpus | Retrieval results used |
|--------|------------------------|
| ÚS | `artifacts/rag_eval/legal_qa/runs/usoud_gold_eval/retrieval_results.jsonl` |
| NSoud | `artifacts/rag_eval/legal_qa/runs/nsoud_full_baseline/retrieval_results.jsonl` |
| Mixed | `artifacts/rag_eval/legal_qa/runs/mixed_two_pass_baseline/retrieval_results.jsonl` |

Gold review: `artifacts/rag_eval/legal_qa/gold_source_review_20260709.md`

---

## ÚS results (`answer_eval/usoud_no_llm_baseline/`)

| Metric | Value |
|--------|-------|
| Total questions | 20 |
| Gold available | 10 |
| Missing gold | 10 |
| direct_support_count | 1 |
| partial_support_count | 9 |
| gap_count | 0 |
| boilerplate_noise_count | 0 |
| corpus_only_count | 0 |
| strict_direct_pass_rate_all | **0.050** |
| strict_direct_pass_rate_gold | **0.100** |
| usable_support_rate_gold | **1.000** |
| citation_available_rate_gold | **1.000** |
| unsupported_risk_rate_gold | **0.000** |
| gold_retrieval_miss_rate | **0.000** |
| unsupported_answer_risk_count | 0 |
| skipped_count | 10 |
| Status | **WARN** |

Gold item breakdown:
- `usoud-qa-004` → **direct / pass**
- `usoud-qa-001, 002, 003, 007, 009, 010, 011, 012, 015` → partial
- Interpretation: ÚS currently shows **usable support**, but the strict direct gate remains low and half of the dataset is still not evaluable due to missing gold.

---

## NSoud results (`answer_eval/nsoud_no_llm_baseline/`)

| Metric | Value |
|--------|-------|
| Total questions | 10 |
| Gold available | 4 |
| Missing gold | 6 |
| direct_support_count | 0 |
| partial_support_count | 2 |
| gap_count | 1 |
| boilerplate_noise_count | 1 |
| corpus_only_count | 0 |
| strict_direct_pass_rate_all | **0.000** |
| strict_direct_pass_rate_gold | **0.000** |
| usable_support_rate_gold | **0.500** |
| citation_available_rate_gold | **0.500** |
| unsupported_risk_rate_gold | **0.500** |
| gold_retrieval_miss_rate | **0.250** |
| unsupported_answer_risk_count | **2** |
| needs_review_count | 1 |
| skipped_count | 6 |
| Status | **FAIL_WITH_REAL_NSOUD_RISK** |

Gold item breakdown:
- `nsoud-qa-003, 004` → usable partial support
- `nsoud-qa-007` → **true retrieval miss** at answer-eval layer; expected ECLI absent from retrieved top-k
- `nsoud-qa-010` → **unsupported_boilerplate_or_gap / needs_review** (operative snippet “Dovolání se odmítá.”)
- Interpretation: NSoud is the only corpus with confirmed real risk in the current diagnostics.

---

## Mixed results (`answer_eval/mixed_no_llm_baseline/`)

| Metric | Value |
|--------|-------|
| Total questions | 10 |
| Gold available | 8 (corpus-only) |
| Missing gold | 2 |
| direct_support_count | 0 |
| partial_support_count | 0 |
| gap_count | 0 |
| boilerplate_noise_count | 0 |
| corpus_only_count | 8 |
| strict_direct_pass_rate_all | **0.000** |
| strict_direct_pass_rate_gold | **0.000** |
| usable_support_rate_gold | **1.000** |
| citation_available_rate_gold | **0.000** (by design — corpus-only gold) |
| corpus_routing_support_rate | **1.000** |
| unsupported_risk_rate_gold | **0.000** |
| gold_retrieval_miss_rate | **0.000** |
| unsupported_answer_risk_count | 0 |
| skipped_count | 2 |
| Status | **WARN** |

Mixed rules enforced:
- `corpus_only` skeleton does not claim a document citation
- `citation_available=false` without ECLI/source_document_id
- No direct pass without document gold

Gold item breakdown:
- `mixed-qa-001, 002, 003, 005, 006, 007, 008, 009` → corpus-only routing support
- `mixed-qa-004, 010` → not evaluable yet because gold is still missing

---

## Cross-corpus summary

| Corpus | Gold | Missing gold | Direct | Partial | Gap | Boilerplate | Corpus-only | strict_direct_pass_rate_gold | usable_support_rate_gold | citation_available_rate_gold | unsupported_risk_rate_gold | gold_retrieval_miss_rate | Status |
|--------|------|--------------|--------|---------|-----|-------------|-------------|------------------------------|--------------------------|------------------------------|----------------------------|--------------------------|--------|
| ÚS | 10 | 10 | 1 | 9 | 0 | 0 | 0 | 0.100 | 1.000 | 1.000 | 0.000 | 0.000 | WARN |
| NSoud | 4 | 6 | 0 | 2 | 1 | 1 | 0 | 0.000 | 0.500 | 0.500 | 0.500 | 0.250 | FAIL_WITH_REAL_NSOUD_RISK |
| Mixed | 8 | 2 | 0 | 0 | 0 | 0 | 8 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | WARN |

---

## Limitations

1. **Strict direct threshold** — rank-1 gold ECLI + ≥67% keyword overlap on snippet; many gold retrieval passes become `partial` at answer layer.
2. **Snippet-only** — does not read full chunk text from Qdrant.
3. **Boilerplate detection** — snippets &lt;40 chars or operative lines (NSoud dovolání) flagged as `needs_review`.
4. **10/20 ÚS, 6/10 NSoud, and 2/10 mixed remain not evaluable** — remaining `source_pending=true` items still need provenance-safe gold expansion or manual relevance review.
5. **Verified provenance is not enough for NSoud gold quality** — `nsoud-qa-007` currently evaluates as a true retrieval miss at the answer-support layer.
6. **No generative answers** — skeleton only.

---

## Safety

| Check | Status |
|-------|--------|
| Real LLM called | **No** |
| DeepSeek called | **No** |
| Redis | **Off** (not used) |
| Qdrant writes | **None** (read-only via frozen retrieval JSONL) |
| Aliases touched | **No** |
| Retrieval logic changed | **No** |
| `nalus-legal-rag` | **Not imported/modified** |

---

## Run commands

```powershell
# ÚS
python scripts/run_legal_answer_eval.py `
  --dataset artifacts/rag_eval/legal_qa/datasets/usoud_qa_v1.jsonl `
  --retrieval-results artifacts/rag_eval/legal_qa/runs/usoud_gold_eval/retrieval_results.jsonl `
  --gold-review artifacts/rag_eval/legal_qa/gold_source_review_20260709.md `
  --output-dir artifacts/rag_eval/legal_qa/answer_eval/usoud_no_llm_baseline `
  --no-llm --require-citations

# NSoud
python scripts/run_legal_answer_eval.py `
  --dataset artifacts/rag_eval/legal_qa/datasets/nsoud_qa_v1.jsonl `
  --retrieval-results artifacts/rag_eval/legal_qa/runs/nsoud_full_baseline/retrieval_results.jsonl `
  --gold-review artifacts/rag_eval/legal_qa/gold_source_review_20260709.md `
  --output-dir artifacts/rag_eval/legal_qa/answer_eval/nsoud_no_llm_baseline `
  --no-llm --require-citations

# Mixed
python scripts/run_legal_answer_eval.py `
  --dataset artifacts/rag_eval/legal_qa/datasets/mixed_qa_v1.jsonl `
  --retrieval-results artifacts/rag_eval/legal_qa/runs/mixed_two_pass_baseline/retrieval_results.jsonl `
  --gold-review artifacts/rag_eval/legal_qa/gold_source_review_20260709.md `
  --output-dir artifacts/rag_eval/legal_qa/answer_eval/mixed_no_llm_baseline `
  --no-llm --require-citations
```
