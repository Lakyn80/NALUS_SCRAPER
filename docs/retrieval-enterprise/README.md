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
5. [`../architecture/RETRIEVAL_GOLDEN_V1.md`](../architecture/RETRIEVAL_GOLDEN_V1.md) — Step 4A passage retrieval-golden pilot (30 queries)
6. [`../architecture/CASE_SIMILARITY_RETRIEVAL_GOLDEN_V1.md`](../architecture/CASE_SIMILARITY_RETRIEVAL_GOLDEN_V1.md) — document-level case-similarity pilot (20 cases)
7. `SYSTEM_ARCHITECTURE.md`
8. `PACKAGE_BOUNDARIES.md`
9. `CONTRACTS.md`
10. `DATA_AND_INDEX_LIFECYCLE.md`
11. `EVALUATION_PROTOCOL.md`
12. `SECURITY_AND_OPERATIONS.md`
13. `MIGRATION_AND_ROLLBACK.md`
14. `IMPLEMENTATION_ROADMAP.md`
15. accepted ADRs under `adr/`

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

1. Step 4A passage pilot (30 queries) and case-similarity pilot (20 docs) are available under `benchmarks/legal_v2/`.
2. Complete human audit of the case-similarity pilot, then expand retrieval goldens toward larger validation / locked_holdout splits.
3. Only then evaluate chunking A/B/C/D for a production winner against frozen document-level benchmarks.
4. Fill remaining `pending_external` parser archetype holdouts when new unseen documents are available.

Do not begin broad parser polishing, ColBERT/cross-encoder product branches, or
uncontrolled multi-layer tuning before that benchmark exists.
Do not treat the current pilots alone as sufficient to select a chunking winner.
