# ADR 0002 - Phase-Gated Additive Rollout

Status: accepted

Date: 2026-08-03

## Context

Future retrieval work may introduce installable package boundaries, a new
orchestration layer, baseline adapters, ingestion changes, ColBERT-style
late-interaction reranking, and configurable profiles.

These changes affect quality, cost, CPU latency, RAM, disk usage, operational
rollback, and data safety.

## Decision

All major retrieval changes are additive and phase-gated.

The sequence is:

1. audit and definitive architecture;
2. retrieval core;
3. pipeline orchestration with fake adapters;
4. baseline adapters with parity gate;
5. ingestion subsystem with checkpoint gate;
6. ColBERT adapter and isolated experimental Qdrant;
7. configurable profiles;
8. evaluation and comparison;
9. optional Legal v2 integration.

No phase can rely on hidden work from a later phase.

## Consequences

Benefits:

- current production behavior remains stable;
- each phase has a clear rollback;
- ColBERT only advances after measured benefit;
- FE changes wait for backend proof.

Costs:

- integration happens later;
- temporary compatibility shims may exist during migration.

## Guardrails

- No protected Qdrant collection writes.
- No default profile changes before rollout.
- No frontend changes before backend experiment passes.
- No benchmark gold edits inside retrieval tuning tasks.
- No downloads or paid provider calls unless explicitly approved.

