# Retrieval Enterprise Package Boundaries

Status: target boundaries for future implementation.

This document defines where new code belongs. It does not require immediately
renaming current modules. During migration, compatibility shims are allowed only
when they preserve existing imports and do not hide cross-layer coupling.

## Current Boundary Notes

Current Legal v2 code already has useful subpackages:

- `app/rag/legal_v2/ingest/`
- `app/rag/legal_v2/query/`
- `app/rag/legal_v2/retrieve/`
- `app/rag/legal_v2/verify/`
- `app/rag/legal_v2/evidence/`

Top-level modules such as `app/rag/legal_v2/retriever.py` may be shims. New code
should prefer the concrete subpackage implementation unless an explicit
compatibility task says otherwise.

## Target Packages

### retrieval-core

Owns:

- domain models;
- protocol/port definitions;
- core exceptions;
- score/result value objects;
- deterministic serialization contracts;
- dependency rules.

Allowed dependencies:

- Python standard library.
- `typing_extensions` if already available and needed.
- A project-approved validation library only if Prompt 0 accepts it.

Forbidden imports:

- Qdrant clients.
- SQLite implementation details.
- FastAPI.
- Docker/runtime env parsing.
- provider SDKs.
- sentence-transformers/FastEmbed/transformers.
- app-level logging and metrics backends.

### retrieval-pipeline

Owns:

- orchestration;
- candidate generation workflow;
- fusion workflow;
- document aggregation workflow;
- evidence selection workflow;
- verifier/reranker invocation through ports;
- result assembly.

Allowed dependencies:

- `retrieval-core`.
- Pure in-memory helpers.

Forbidden:

- Direct construction of infrastructure clients.
- Direct environment parsing.
- Direct model loading.
- Direct Qdrant/BM25/SQLite calls.

### retrieval-adapters-baseline

Owns:

- BGE-M3 dense adapter.
- Qdrant read adapter.
- BM25 SQLite sidecar adapter.
- RRF strategy adapter if it stays adapter-specific.

Rules:

- Reads and writes are separate ports.
- Query-time adapters must be read-only.
- Model loading must be local-files-only unless a later prompt explicitly
  allows downloads.
- Adapter outputs must be converted into core value objects before leaving the
  adapter boundary.

### retrieval-ingestion

Owns:

- source adapters;
- parsing;
- transformation;
- hierarchical chunking;
- embedding batches;
- Qdrant writes;
- BM25 writes;
- checkpoint/resume;
- manifests;
- identity validation.

Rules:

- Ingest writes only to explicitly isolated and validated targets.
- Builder must refuse protected collections.
- BM25 and Qdrant identities must be reconciled before a manifest can pass.
- Checkpoint files must be validated against corpus hash, collection, sidecar,
  model, profile, and source selection.

### retrieval-late-interaction

Owns:

- ColBERT/FastEmbed adapter;
- multivector schema adapter;
- late-interaction query embedding;
- candidate document reranking;
- related capability checks.

Rules:

- Experimental by default.
- No dependency from core or baseline pipeline to ColBERT classes.
- No production profile can depend on this package until evaluation gates pass.

### retrieval-evaluation

Owns:

- benchmark runners;
- metrics;
- report generation;
- manifest validation;
- parity comparison;
- smoke gates.

Rules:

- Evaluation data changes are separate tasks.
- Reports must be machine-readable and human-readable when persisted.
- A failed invariant must make the report fail; it must not be described as
  pass with caveats.

### api-composition

Owns:

- endpoint request/response mapping;
- feature flags;
- runtime dependency construction;
- profile selection;
- safe payload shaping and redaction;
- HTTP error behavior.

Rules:

- API layer may not contain ranking logic.
- API layer may not promote related candidates to verified documents.
- Disabled endpoints must avoid initializing heavy clients or credentials.

## Dependency Graph

Allowed direction:

```text
api-composition
  -> retrieval-pipeline
  -> retrieval-core

retrieval-adapters-*
  -> retrieval-core

retrieval-ingestion
  -> retrieval-core
  -> source-specific adapters

retrieval-evaluation
  -> retrieval-core
  -> selected pipeline/adapters under test
```

Forbidden direction:

```text
retrieval-core -> adapters
retrieval-core -> API
retrieval-pipeline -> concrete Qdrant/BM25/model clients
adapters -> API endpoint models
evaluation -> production runtime mutation
frontend -> backend verification semantics
```

## Migration Rule

When moving existing code into a target package:

1. Add tests around current behavior first.
2. Move or wrap the smallest coherent unit.
3. Preserve import compatibility if external callers still depend on old paths.
4. Keep old path shims thin and documented.
5. Remove shims only in an explicit cleanup task.

