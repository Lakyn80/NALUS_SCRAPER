# Retrieval Enterprise Documentation

Status: entry point for the retrieval-enterprise track.

This directory is the shared architecture baseline for future retrieval
modernization work. It is documentation only; it does not activate any runtime
path.

## Required Reading Order

1. `SYSTEM_ARCHITECTURE.md`
2. `PACKAGE_BOUNDARIES.md`
3. `CONTRACTS.md`
4. `DATA_AND_INDEX_LIFECYCLE.md`
5. `EVALUATION_PROTOCOL.md`
6. `SECURITY_AND_OPERATIONS.md`
7. `MIGRATION_AND_ROLLBACK.md`
8. `IMPLEMENTATION_ROADMAP.md`
9. accepted ADRs under `adr/`

## Mandatory Prompt Clause

Every implementation prompt in this track must include:

```text
Read and comply with all documents in docs/retrieval-enterprise/.
Do not silently deviate from an accepted ADR.
If implementation reality conflicts with the architecture, stop and report it.
```

## Current Next Step

Run Prompt 0: audit and definitive architecture validation only. Do not
implement runtime code, download models, install packages, call paid providers,
create Qdrant collections, commit, or push.

