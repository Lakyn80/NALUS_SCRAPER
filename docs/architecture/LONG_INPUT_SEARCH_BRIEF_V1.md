# Legal v2 Long-Input / SearchBrief Preprocessing

Status: implemented, **disabled by default** (`NALUS_LEGAL_V2_LONG_INPUT_ENABLED=0`).

## Architecture

```text
Raw Query
   │
   ▼
QueryInputService
   ├── short → passthrough
   └── long
         ↓
     normalization
         ↓
     segmentation
         ↓
     boilerplate control
         ↓
     extractive scoring
         ↓
     SearchBrief
         │
         ▼
     QuerySpec (unchanged)
         │
         ▼
 BGE-M3 + BM25 + RRF + ECLI aggregation
```

## Boundaries

### Stage 1 remains

```text
QuerySpec + BGE-M3 + BM25 + RRF + ECLI aggregation
```

### New component

```text
Long Input Preprocessing / SearchBrief
app/rag/legal_v2/query_input/
```

### Future component

```text
PRECISE LLM SearchBrief provider
(PreciseLLMSearchBriefProvider stub — no external calls)
```

### Not included

```text
ColBERT
cross-encoder
LLM condensation runtime
```

## Modes

| Mode | Behavior |
| --- | --- |
| OFF (default) | identical to prior Stage 1; raw query max 8000 |
| EXTRACTIVE | long paste → deterministic SearchBrief → Stage 1 |
| PRECISE | reserved; raises `UnsupportedCondensationModeError` |

## Flags

```env
NALUS_LEGAL_V2_LONG_INPUT_ENABLED=0
NALUS_LEGAL_V2_LONG_INPUT_METHOD=extractive
NALUS_LEGAL_V2_LONG_INPUT_CHAR_THRESHOLD=700
NALUS_LEGAL_V2_LONG_INPUT_HARD_LIMIT=100000
NALUS_LEGAL_V2_SEARCH_BRIEF_TARGET_CHARS=1000
NALUS_LEGAL_V2_CONDENSATION_POLICY_VERSION=extractive_v1
```

## Invariants

- Short queries: `retrieval_query == original_query` (whitespace-normalized).
- SearchBrief must preserve negation/contrast (`nehledám X, ale Y`).
- ECLI / sp. zn. / č. j. / `doc-*` / `nalus-cs-pilot-*` are suppressed from `brief_text`.
- No paid LLM calls in the active path.
- Stage 1 retrieval knobs are unchanged.
