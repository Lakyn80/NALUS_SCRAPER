# Legal Q&A No-LLM Answer Eval Report — 2026-07-09

Deterministic answer-support evaluation over frozen gold-source annotations. **No LLM was called.**

---

## What this checks

| Check | Description |
|-------|-------------|
| Gold source presence | ECLI/document gold hit in top-1/3/5 |
| Support level | `direct`, `partial`, `gap`, `boilerplate_noise`, `corpus_only` |
| Answer skeleton | Built only from `expected_answer_points` + verified snippet context |
| Citation availability | ECLI + chunk_id when `--require-citations` |
| Unsupported answer risk | Gap/boilerplate or missing citation on would-be direct answer |

## What this does **not** check

- LLM answer quality or fluency
- Legal correctness beyond gold snippet keyword overlap
- Synthesis across multiple chunks
- DeepSeek / any generative model output

---

## Retrieval inputs (detected paths)

| Corpus | Retrieval results used |
|--------|------------------------|
| ÚS | `artifacts/rag_eval/legal_qa/runs/usoud_gold_eval/retrieval_results.jsonl` |
| NSoud | `artifacts/rag_eval/legal_qa/runs/nsoud_gold_eval/retrieval_results.jsonl` |
| Mixed | `artifacts/rag_eval/legal_qa/runs/mixed_gold_eval/retrieval_results.jsonl` |

Fallback order in runner: `*_gold_eval` → `*_full_baseline` / `mixed_two_pass_baseline`.

Gold review: `artifacts/rag_eval/legal_qa/gold_source_review_20260709.md`

---

## ÚS results (`answer_eval/usoud_no_llm_baseline/`)

| Metric | Value |
|--------|-------|
| Total questions | 20 |
| Gold available | 5 |
| Direct support | 1 |
| Partial support | 4 |
| Pass rate (all Q) | 0.050 |
| Partial rate | 0.200 |
| Citation available rate | **1.000** |
| Unsupported answer risk | 0 |
| Skipped | 15 |

Gold item breakdown:
- `usoud-qa-004` → **direct / pass**
- `usoud-qa-001, 003, 009, 012` → partial (rank-1 gold ECLI present but keyword coverage on snippet &lt; direct threshold)

---

## NSoud results (`answer_eval/nsoud_no_llm_baseline/`)

| Metric | Value |
|--------|-------|
| Total questions | 10 |
| Gold available | 3 |
| Direct support | 0 |
| Partial support | 2 |
| Boilerplate noise | 1 |
| Pass rate | 0.000 |
| Partial rate | 0.200 |
| Citation available rate | **0.667** |
| Unsupported answer risk | **1** |
| Needs review | 1 |
| Skipped | 7 |

Gold item breakdown:
- `nsoud-qa-003, 004` → partial
- `nsoud-qa-010` → **boilerplate_noise / needs_review** (operative snippet “Dovolání se odmítá.”)

---

## Mixed results (`answer_eval/mixed_no_llm_baseline/`)

| Metric | Value |
|--------|-------|
| Total questions | 10 |
| Gold available | 2 (corpus-only) |
| Corpus-only | 2 |
| Pass rate | 0.000 |
| Partial rate | 0.200 |
| Citation available rate | **0.000** (by design — no document gold) |
| Unsupported answer risk | 0 |
| Skipped | 8 |

Gold item breakdown:
- `mixed-qa-002, 005` → corpus_only / partial — no fabricated document citation

---

## Cross-corpus summary

| Corpus | Gold | Direct | Partial | Boilerplate | Corpus-only | Citation rate | Unsupported risk |
|--------|------|--------|---------|-------------|-------------|---------------|------------------|
| ÚS | 5 | 1 | 4 | 0 | 0 | 1.000 | 0 |
| NSoud | 3 | 0 | 2 | 1 | 0 | 0.667 | 1 |
| Mixed | 2 | 0 | 0 | 0 | 2 | 0.000 | 0 |

---

## Limitations

1. **Strict direct threshold** — rank-1 gold ECLI + ≥67% keyword overlap on snippet; many gold retrieval passes become `partial` at answer layer.
2. **Snippet-only** — does not read full chunk text from Qdrant.
3. **Boilerplate detection** — short operative lines (NSoud dovolání) flagged as `needs_review`.
4. **15/20 ÚS and 7/10 NSoud skipped** — still `source_pending=true`; expand gold before corpus-wide answer eval.
5. **No generative answers** — skeleton only.

---

## Safety

| Check | Status |
|-------|--------|
| Real LLM called | **No** |
| DeepSeek called | **No** |
| Redis | **Off** |
| Qdrant writes | **None** |
| Aliases touched | **No** |
| Retrieval logic changed | **No** |
| `nalus-legal-rag` | **Not imported/modified** |

---

## Next step

Optional **LLM answer generation** behind an explicit flag (e.g. `--enable-llm` / DeepSeek), using the same gold-support gate before accepting generated text. Not implemented in this phase.

Run command:

```powershell
python scripts/run_legal_answer_eval.py `
  --dataset artifacts/rag_eval/legal_qa/datasets/usoud_qa_v1.jsonl `
  --retrieval-results artifacts/rag_eval/legal_qa/runs/usoud_gold_eval/retrieval_results.jsonl `
  --gold-review artifacts/rag_eval/legal_qa/gold_source_review_20260709.md `
  --output-dir artifacts/rag_eval/legal_qa/answer_eval/usoud_no_llm_baseline `
  --no-llm --require-citations
```
