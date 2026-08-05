# Retrieval Enterprise Documentation

Status: entry point for the retrieval-enterprise track.

This directory is the shared architecture baseline for future retrieval
modernization work. It is documentation only; it does not activate any runtime
path.

## Required Reading Order

1. [`../architecture/NALUS_LEGAL_RAG_MASTER_PLAN.md`](../architecture/NALUS_LEGAL_RAG_MASTER_PLAN.md) — **controlling** post-v7 end-to-end plan
2. [`../architecture/PARSER_V7_BASELINE_DECISION.md`](../architecture/PARSER_V7_BASELINE_DECISION.md) — `ACCEPT_V7_WITH_KNOWN_LIMITATIONS`
3. [`../architecture/parser_benchmark/archetypes_v1.json`](../architecture/parser_benchmark/archetypes_v1.json) — Phase 1 archetype manifest
4. [`../architecture/CANONICAL_BLOCK_CHUNK_SCHEMA_V1.md`](../architecture/CANONICAL_BLOCK_CHUNK_SCHEMA_V1.md) — Phase 2 document/block/child/parent contract
5. [`../architecture/RETRIEVAL_GOLDEN_V1.md`](../architecture/RETRIEVAL_GOLDEN_V1.md) — Step 4A retrieval-golden pilot (30 queries)
6. `SYSTEM_ARCHITECTURE.md`
7. `PACKAGE_BOUNDARIES.md`
8. `CONTRACTS.md`
9. `DATA_AND_INDEX_LIFECYCLE.md`
10. `EVALUATION_PROTOCOL.md`
11. `SECURITY_AND_OPERATIONS.md`
12. `MIGRATION_AND_ROLLBACK.md`
13. `IMPLEMENTATION_ROADMAP.md`
14. accepted ADRs under `adr/`

If `IMPLEMENTATION_ROADMAP.md` or any older note conflicts with
`NALUS_LEGAL_RAG_MASTER_PLAN.md` on sequencing, the master plan wins for
post-v7 product work.

`NALUS_SYSTEM_BUILD_PLAN.md` in this directory is only a pointer to the master
plan.

## Mandatory Prompt Clause

Every implementation prompt in this track must include:

```text
Read and comply with all documents in docs/retrieval-enterprise/.
Follow docs/architecture/NALUS_LEGAL_RAG_MASTER_PLAN.md for post-v7 sequencing.
Do not silently deviate from an accepted ADR.
If implementation reality conflicts with the architecture, stop and report it.
```

## Current Next Step

Parser v7 is accepted as baseline with known limitations. Do not open parser v8
for non-blocking label noise.

Next according to the master plan:

1. Step 4A retrieval-golden pilot (30 queries) is available under `benchmarks/legal_v2/retrieval_golden_v1_pilot.jsonl`.
2. Expand retrieval golden toward 100–150 queries with validation / locked_holdout splits.
3. Only then evaluate chunking A/B/C/D for a production winner against the frozen benchmark.
4. Fill remaining `pending_external` parser archetype holdouts when new unseen documents are available.

Do not begin broad parser polishing, ColBERT/cross-encoder product branches, or
uncontrolled multi-layer tuning before that benchmark exists.
Do not treat the 30-query pilot alone as sufficient to select a chunking winner.
