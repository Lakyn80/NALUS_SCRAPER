# Case Similarity Retrieval Golden v1 (document-level pilot)

**Status:** development pilot (20 case descriptions)

**Dataset:** [`benchmarks/legal_v2/case_similarity_golden_v1_pilot.jsonl`](../../benchmarks/legal_v2/case_similarity_golden_v1_pilot.jsonl)

**Schema / validator:** `app/rag/legal_v2/benchmark/case_similarity_golden.py`

**Reviewed-pool corpus loader:** `load_reviewed_pool_corpus()` in `app/rag/legal_v2/benchmark/corpus.py`

**Case-similarity corpus (pilot + supplemental hard negatives):** `load_case_similarity_corpus()` — the 20 reviewed primaries plus offline supplemental criminal appeals from `artifacts/legal_v2/court_format_study/raw_sources` used only as hard negatives for `nalus-cs-pilot-016`.

**Hard-negative evaluability:** each row carries explicit fields:

- `hard_negative_evaluable` (default `true` for older rows)
- `hard_negative_blocker` (`null` when evaluable; required when blocked)

`nalus-cs-pilot-007` is blocked with `insufficient_same_domain_corpus`. Blocked rows remain in Hit@K / MRR but are excluded from hard-negative outrank denominators.

**Retrieval baseline runner:** `scripts/legal_v2/evaluate_case_similarity_golden_v1.py` (Legal v2 hybrid retriever, offline, no LLM).

**Binding plan:** [`NALUS_LEGAL_RAG_MASTER_PLAN.md`](./NALUS_LEGAL_RAG_MASTER_PLAN.md)

---

## What this is

This is the **primary pilot** for NALUS case-similarity search:

```text
user legal situation description
→ most similar whole judgment
→ supporting passages
→ hard negatives that look similar but are wrong
```

It is **not** a question-to-passage quiz. Step 4A
([`RETRIEVAL_GOLDEN_V1.md`](./RETRIEVAL_GOLDEN_V1.md)) remains a **secondary**
passage-retrieval benchmark and stays frozen.

| Count | Role |
|---|---|
| 20 | Realistic Czech case descriptions |
| 20 | Reviewed parser-v7 judgments used exactly once as primary expected documents |
| all | `split = development` |

Court distribution in the pilot pool:

- 10 Constitutional Court
- 5 High Court Prague
- 5 High Court Olomouc

---

## Retrieval target

- **Primary target:** the complete judgment (`expected_document_ids`)
- **Supporting blocks:** 2–5 canonical blocks that explain and verify relevance
- **Accepted alternatives:** only when another reviewed judgment genuinely fits the whole described situation
- **Hard negatives:** 1–3 similar-but-wrong judgments with explicit rationales

Production ranking will later be evaluated with document-level metrics. This step
adds **no** production retrieval implementation.

---

## Query styles

Exact distribution:

| Style | Count |
|---|---:|
| `client_narrative` | 8 |
| `noisy_client_narrative` | 4 |
| `multi_issue_client_narrative` | 4 |
| `concise_case_description` | 4 |

Queries are paraphrases (60–180 words, 3–8 sentences). They must not leak
document IDs, block IDs, case numbers, party names, judge names, or verbatim
supporting-block sentences.

---

## Document identity (ECLI)

ECLI is the immutable canonical identifier of a judicial decision in Legal v2.
Source-specific IDs (`doc-*`, import IDs, review IDs) are secondary traceability
metadata only.

Identity contract for indexed judgments:

```text
document_id == canonical_document_id == ecli
source_document_id = secondary traceability only
```

Pilot mapping artifact:

[`benchmarks/legal_v2/case_similarity_document_identity_v1.json`](../../benchmarks/legal_v2/case_similarity_document_identity_v1.json)

Each golden row carries:

- `source_document_id` / `expected_document_ids` (benchmark/source `doc-*`)
- `expected_primary_ecli` / `expected_primary_canonical_document_id`
- ECLI fields on accepted-alternative and hard-negative rationales

Judgments without a verified ECLI are explicitly marked
`blocked_missing_verified_ecli` and must not enter the production index.

Evaluation and corpus compatibility match on ECLI, never on `doc-*` as the
production document ID.

---

## Intended future metrics

Do **not** run these yet; no retrieval configuration is evaluated in this step.

- Recall@1
- Recall@3
- Recall@5
- MRR@10
- nDCG@10
- hard-negative outrank rate

---

## Scripts

```powershell
python scripts/legal_v2/build_case_similarity_golden_v1_pilot.py
python scripts/legal_v2/validate_case_similarity_golden_v1_pilot.py
python scripts/legal_v2/export_case_similarity_golden_v1_manual_review.py
```

Manual review export (local / gitignored):

`artifacts/legal_v2/case_similarity_golden_v1_pilot/manual_review.md`

---

## Safety

- Does not modify Step 4A rows
- Does not change parser v7 or canonical schema semantics
- Offline only: no provider API, Qdrant, BM25, Redis, Docker, or embeddings
