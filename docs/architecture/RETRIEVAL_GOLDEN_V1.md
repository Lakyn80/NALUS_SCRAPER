# Retrieval Golden v1 (Step 4A pilot)

**Status:** development pilot (30 queries)

**Dataset:** [`benchmarks/legal_v2/retrieval_golden_v1_pilot.jsonl`](../../benchmarks/legal_v2/retrieval_golden_v1_pilot.jsonl)

**Schema / validator:** `app/rag/legal_v2/benchmark/retrieval_golden.py`

**Development corpus loader:** `app/rag/legal_v2/benchmark/corpus.py`

**Binding plan:** [`NALUS_LEGAL_RAG_MASTER_PLAN.md`](./NALUS_LEGAL_RAG_MASTER_PLAN.md) §4.2 / Step 4

---

## What this is

An **evidence-first, block-grounded** retrieval-golden **pilot**:

| Count | Role |
|---|---|
| 29 | Positive queries grounded in a real canonical block from the **development** archetype documents |
| 1 | Corpus-negative query (legally plausible; unanswered by the pilot corpus) |
| **30** | Total |

This is **not** the final 100–150 query retrieval benchmark.

Queries were authored in an editing session by reading canonical block text first, then writing the question. They are **not** generated from general legal knowledge and then searched for supporting evidence.

---

## Grounding rules

For every positive item:

1. Select a legally meaningful source block from a development document.
2. Copy an `evidence_excerpt` **verbatim** from that block.
3. Store exact `source_document_id`, `primary_expected_block_id`, and `expected_block_ids`.
4. Validator checks that the excerpt is a whitespace-normalized substring of the block `raw_text`.
5. `accepted_alternative_block_ids` only when another block genuinely answers the same query.
6. `hard_negative_block_ids` only when a similar/confusable block does **not** answer the query.

### Hard negatives vs corpus-negative

| Concept | Meaning |
|---|---|
| **Hard negative block** | A real block that is textually or legally similar but is **wrong** for a **positive** query |
| **Corpus-negative query** | A query that should retrieve **nothing** useful from the corpus (`is_negative=true`, empty expected IDs) |

Do not confuse these.

---

## Split support

All pilot rows currently use `split = "development"`.

The schema also allows:

- `validation`
- `locked_holdout`

Locked holdout must not be used for tuning chunking, `top_k`, fusion, rerankers, prompts, or models.

This pilot uses **only development-role documents** from [`parser_benchmark/archetypes_v1.json`](./parser_benchmark/archetypes_v1.json). It does **not** consume locked parser holdouts as retrieval holdout.

---

## What this pilot must not be used for

- Selecting a **production chunking winner** (A/B/C/D)
- Claiming the retrieval benchmark is complete
- Tuning production retrieval parameters as if the set were frozen holdout

Chunking variants may be implemented later, but a winner may be declared only against an expanded frozen retrieval benchmark (target 100–150 queries), not against this 30-query pilot alone.

---

## Offline validation

```powershell
python scripts/legal_v2/validate_retrieval_golden_v1_pilot.py
python -m pytest -q tests/rag/test_legal_v2_retrieval_golden_v1_pilot.py tests/rag/test_legal_v2_canonical_schema_v1.py
```

Rebuild (deterministic, no providers):

```powershell
python scripts/legal_v2/build_retrieval_golden_v1_pilot.py
```

Optional local report path (gitignored):

```text
artifacts/legal_v2/retrieval_golden_v1_pilot/
```

---

## Safety

- No parser v7 changes
- No canonical schema v1 semantic changes
- No Qdrant / BM25 / Redis / Docker / provider calls
- Canonical block identities come from the Phase 2 mapper (`a2ce8ba`)
