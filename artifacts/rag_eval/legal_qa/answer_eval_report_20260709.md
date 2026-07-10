# Legal Q&A No-LLM Answer Eval Report — 2026-07-09

Deterministic answer-support evaluation over frozen gold-source annotations. **No LLM was called.**

Update: refreshed on 2026-07-10 after conservative gold expansion for ÚS and mixed.

---

## Metric interpretation

| Metric | Meaning |
|--------|---------|
| `strict_direct_pass_rate_all` | Share of **all** questions with `answer_eval_status=pass` (direct + citation when required) |
| `strict_direct_pass_rate_gold` | Same, but only over gold-available items |
| `usable_support_rate_gold` | Gold items with `direct`, `partial`, or `corpus_only` support |
| `answer_eval_pass_rate` | Alias for `strict_direct_pass_rate_all` (backward compat) |
| `citation_available_rate` | Gold items where ECLI/chunk citation is present in skeleton |

**Support levels**

| Level | Meaning |
|-------|---------|
| `direct` | Strict pass — document gold + rank-1 snippet with ≥67% keyword overlap |
| `partial` | Usable support, not a full direct answer pass |
| `gap` / `boilerplate_noise` | Must **not** generate a confident answer |
| `corpus_only` | Corpus routing only — no document citation |

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
| direct_support_count | 1 |
| partial_support_count | 9 |
| gap_count | 0 |
| boilerplate_noise_count | 0 |
| corpus_only_count | 0 |
| strict_direct_pass_rate_all | **0.050** |
| strict_direct_pass_rate_gold | **0.100** |
| usable_support_rate_gold | **1.000** |
| citation_available_rate | **1.000** |
| unsupported_answer_risk_count | 0 |
| skipped_count | 10 |

Gold item breakdown:
- `usoud-qa-004` → **direct / pass**
- `usoud-qa-001, 002, 003, 007, 009, 010, 011, 012, 015` → partial (rank-1 gold ECLI present but keyword coverage on snippet &lt; direct threshold)

---

## NSoud results (`answer_eval/nsoud_no_llm_baseline/`)

| Metric | Value |
|--------|-------|
| Total questions | 10 |
| Gold available | 3 |
| direct_support_count | 0 |
| partial_support_count | 2 |
| gap_count | 0 |
| boilerplate_noise_count | 1 |
| corpus_only_count | 0 |
| strict_direct_pass_rate_all | **0.000** |
| strict_direct_pass_rate_gold | **0.000** |
| usable_support_rate_gold | **0.667** |
| citation_available_rate | **0.667** |
| unsupported_answer_risk_count | **1** |
| needs_review_count | 1 |
| skipped_count | 7 |

Gold item breakdown:
- `nsoud-qa-003, 004` → partial
- `nsoud-qa-010` → **boilerplate_noise / needs_review** (operative snippet “Dovolání se odmítá.”)

---

## Mixed results (`answer_eval/mixed_no_llm_baseline/`)

| Metric | Value |
|--------|-------|
| Total questions | 10 |
| Gold available | 8 (corpus-only) |
| direct_support_count | 0 |
| partial_support_count | 0 |
| gap_count | 0 |
| boilerplate_noise_count | 0 |
| corpus_only_count | 8 |
| strict_direct_pass_rate_all | **0.000** |
| strict_direct_pass_rate_gold | **0.000** |
| usable_support_rate_gold | **1.000** |
| citation_available_rate | **0.000** (by design — no document gold) |
| unsupported_answer_risk_count | 0 |
| skipped_count | 2 |

Mixed rules enforced:
- `corpus_only` skeleton does not claim a document citation
- `citation_available=false` without ECLI/source_document_id
- No direct pass without document gold

Gold item breakdown:
- `mixed-qa-001, 002, 003, 005, 006, 007, 008, 009` → corpus_only / partial — no fabricated document citation

---

## Cross-corpus summary

| Corpus | Gold | Direct | Partial | Gap | Boilerplate | Corpus-only | strict_direct_pass_rate_gold | usable_support_rate_gold | citation_available_rate | unsupported_risk |
|--------|------|--------|---------|-----|-------------|-------------|------------------------------|--------------------------|-------------------------|------------------|
| ÚS | 10 | 1 | 9 | 0 | 0 | 0 | 0.100 | 1.000 | 1.000 | 0 |
| NSoud | 3 | 0 | 2 | 0 | 1 | 0 | 0.000 | 0.667 | 0.667 | 1 |
| Mixed | 8 | 0 | 0 | 0 | 0 | 8 | 0.000 | 1.000 | 0.000 | 0 |

---

## Limitations

1. **Strict direct threshold** — rank-1 gold ECLI + ≥67% keyword overlap on snippet; many gold retrieval passes become `partial` at answer layer.
2. **Snippet-only** — does not read full chunk text from Qdrant.
3. **Boilerplate detection** — snippets &lt;40 chars or operative lines (NSoud dovolání) flagged as `needs_review`.
4. **10/20 ÚS, 7/10 NSoud, and 2/10 mixed remain skipped** — remaining `source_pending=true` items still need provenance-safe gold expansion.
5. **No generative answers** — skeleton only.

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
