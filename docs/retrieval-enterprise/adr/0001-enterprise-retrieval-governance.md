# ADR 0001 - Enterprise Retrieval Governance

Status: accepted

Date: 2026-08-03

## Context

NALUS retrieval work has grown from scraping and RAG ingestion into a multi-stage
legal judgment finder with parser audits, isolated Legal v2 indexing, BGE-M3,
BM25, RRF, QuerySpec interpretation, semantic verification, benchmarks, and FE
integration concerns.

A single large implementation prompt would create an uncontrolled diff and make
it hard to prove which change improved or harmed retrieval quality.

## Decision

Use `docs/retrieval-enterprise/` as the controlling architecture document set
for future enterprise retrieval work.

Every future implementation prompt in this track must:

- read and comply with the document set;
- implement one bounded phase;
- end with tests, report, rollback notes, and an explicit gate decision;
- avoid silent deviations from accepted ADRs.

## Consequences

Benefits:

- package boundaries are explicit before large refactors;
- experiments are isolated from production resources;
- benchmark and runtime changes are separated;
- future agents share the same architecture.

Costs:

- early work is slower because architecture and gates must be documented;
- deviations require ADR updates instead of ad hoc implementation choices.

## Non-Goals

This ADR does not implement ColBERT, change Legal v2 runtime behavior, alter
Qdrant collections, update the frontend, or run provider calls.

