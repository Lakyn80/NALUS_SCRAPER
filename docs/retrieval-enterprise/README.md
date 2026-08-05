# Retrieval Enterprise Documentation

Status: entry point for the retrieval-enterprise track.

This directory is the shared architecture baseline for future retrieval
modernization work. It is documentation only; it does not activate any runtime
path.

## Required Reading Order

1. [`../architecture/NALUS_LEGAL_RAG_MASTER_PLAN.md`](../architecture/NALUS_LEGAL_RAG_MASTER_PLAN.md) — **controlling** post-v7 end-to-end plan
2. [`../architecture/PARSER_V7_BASELINE_DECISION.md`](../architecture/PARSER_V7_BASELINE_DECISION.md) — `ACCEPT_V7_WITH_KNOWN_LIMITATIONS`
3. [`../architecture/parser_benchmark/archetypes_v1.json`](../architecture/parser_benchmark/archetypes_v1.json) — Phase 1 archetype manifest
4. `SYSTEM_ARCHITECTURE.md`
5. `PACKAGE_BOUNDARIES.md`
6. `CONTRACTS.md`
7. `DATA_AND_INDEX_LIFECYCLE.md`
8. `EVALUATION_PROTOCOL.md`
9. `SECURITY_AND_OPERATIONS.md`
10. `MIGRATION_AND_ROLLBACK.md`
11. `IMPLEMENTATION_ROADMAP.md`
12. accepted ADRs under `adr/`

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

1. Fill remaining `pending_external` holdout slots in `archetypes_v1.json` when new unseen documents are available.
2. Design the canonical block/child/parent chunk schema (Phase 2).
3. Build the retrieval golden (100–150 span-level queries) with locked holdout (Phase 4).

Do not begin broad parser polishing, ColBERT/cross-encoder product branches, or
uncontrolled multi-layer tuning before that benchmark exists.
