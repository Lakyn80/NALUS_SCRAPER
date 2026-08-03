# Retrieval Enterprise System Architecture

Status: controlling specification for future retrieval work.

This document set governs the planned enterprise retrieval architecture for
NALUS/NSoud legal judgment search. It does not change the current runtime by
itself. Future implementation prompts must read and comply with every document
in `docs/retrieval-enterprise/` before editing code.

If implementation reality conflicts with this architecture, stop and report the
conflict. Do not silently deviate from an accepted ADR.

## Purpose

The system is a judgment finder: a user query should return court decisions that
are supported by the indexed corpus and verification policy. It is not a legal
advice generator, and retrieval quality must be measured independently from any
answer-generation model.

The architecture must allow the current baseline and future ColBERT-style
late-interaction experiments to coexist without changing production resources
or weakening the Legal v2 gate.

## Current Repository Baseline

Observed baseline as of 2026-08-03:

- API entrypoint: `POST /api/rag/search-v2` in `app/api/rag_router.py`.
- Feature flag: `NALUS_LEGAL_V2_SEARCH_ENABLED`.
- Current Legal v2 package: `app/rag/legal_v2/`.
- Newer implementation boundaries already exist under:
  - `app/rag/legal_v2/ingest/`
  - `app/rag/legal_v2/query/`
  - `app/rag/legal_v2/retrieve/`
  - `app/rag/legal_v2/verify/`
- Several top-level `app/rag/legal_v2/*.py` modules are compatibility
  re-export shims. New work should target the real subpackages, not deepen the
  shim layer.
- Current retrieval path:
  1. QuerySpec interpretation.
  2. BGE-M3 dense search against isolated Legal v2 Qdrant collection.
  3. BM25 sidecar search.
  4. RRF fusion.
  5. Document aggregation.
  6. Paragraph-aware evidence window selection.
  7. Semantic verifier.
  8. Deterministic terminal gate.
- Current ingest path has parser audit, chunking, checkpoint/resume, Qdrant
  write safety, BM25 sidecar identity validation, and manifests.

## Target Architecture

The target system is built from isolated layers:

```text
API / CLI composition root
  -> pipeline orchestration
    -> retrieval-core contracts and domain models
    -> adapters through ports only

Adapters:
  Qdrant dense reader/writer
  BM25 sidecar reader/writer
  BGE-M3 embedding provider
  future late-interaction provider
  source document readers
  evaluation/report writers
```

The dependency direction is one-way:

```text
api, scripts, composition
  depend on pipeline and adapters

pipeline
  depends on retrieval-core contracts

adapters
  depend on retrieval-core contracts and external libraries

retrieval-core
  depends only on Python standard library and approved typing/model packages
```

`retrieval-core` must never import Qdrant, FastEmbed, sentence-transformers,
SQLite BM25 implementation details, FastAPI, Docker, provider SDKs, or project
runtime settings.

## Core Concepts

The core vocabulary is stable across profiles:

- `QuerySpec`: typed interpretation of the user's retrieval intent.
- `Constraint`: hard, soft, or negative condition extracted from the query.
- `CandidateChunk`: a retrievable indexed text unit with stable provenance.
- `CandidateDocument`: a unique court decision aggregated from chunks.
- `EvidenceWindow`: bounded paragraphs used to justify verification.
- `Retriever`: a port that returns candidate chunks/documents.
- `FusionStrategy`: combines candidate lists without knowing adapter details.
- `Verifier`: classifies whether evidence proves the requested judgment match.
- `RetrievalProfile`: declarative runtime composition of enabled strategies.
- `IndexManifest`: immutable description of one index build.
- `EvaluationManifest`: immutable description of one benchmark run.

## Runtime Profiles

Profiles are configuration, not orchestration rewrites:

- `baseline`: current BGE-M3 dense + BM25 + RRF + verifier path.
- `baseline_colbert`: baseline candidate generation with a late-interaction
  reranker over candidate documents.
- `qdrant_native_late_interaction`: future profile using native multivector
  search after separate proof of quality and operational safety.

The default production profile remains unchanged until a separate rollout task
explicitly enables another profile behind a feature flag.

## Non-Negotiable Safety Rules

- No production Qdrant collection is modified by experiments.
- No existing stable endpoint is broken or renamed.
- `search-v2` remains explicitly feature-gated.
- Frontend behavior is not changed until backend gates pass and a separate FE
  task is approved.
- Model downloads, package installs, GPU/CUDA usage, and paid provider calls are
  prohibited unless the current prompt explicitly allows them.
- Benchmark gold data must not be edited during a model or pipeline comparison
  task unless that task is explicitly a benchmark correction task.
- `related_only` and other non-verified classifications must not be promoted to
  verified results by presentation code.
- Logs, traces, metrics, artifacts, and docs must not contain secrets, raw
  provider responses, full prompts, unbounded user text, or full judgments.

## Composition Root

The composition root owns:

- environment parsing and validation;
- profile selection;
- adapter construction;
- feature flag enforcement;
- runtime capability checks;
- redaction policy;
- timeout and budget policy;
- rollback behavior.

Core and pipeline packages must receive already-validated dependencies through
constructors or protocol interfaces.

## Architecture Gates

Each implementation phase must end with:

- focused tests;
- `git diff --check`;
- a written report in the requested handoff location;
- an explicit list of unchanged production resources;
- an explicit rollback statement;
- a statement of whether the next phase is allowed.

No phase may use the next phase's technology as a hidden dependency.

