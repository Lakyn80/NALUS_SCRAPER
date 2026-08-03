# Retrieval Enterprise Implementation Roadmap

Status: controlling phased roadmap.

Every phase is intentionally small enough to review and roll back. A later phase
must not start until the previous phase's gate is documented as passed.

## Global Rules For Every Prompt

Each implementation prompt must include:

```text
Read and comply with all documents in docs/retrieval-enterprise/.
Do not silently deviate from an accepted ADR.
If implementation reality conflicts with the architecture, stop and report it.
```

Each prompt must define:

1. Scope.
2. Explicit non-goals.
3. Allowed files and packages.
4. Forbidden changes.
5. Dependency boundaries.
6. Backward compatibility.
7. Data safety.
8. Security.
9. Performance boundaries.
10. Test plan.
11. Acceptance criteria.
12. Rollback.
13. Output report.
14. No automatic commit or push.
15. No paid provider calls unless explicitly approved.

## Prompt 0 - Audit And Definitive Architecture

Scope:

- Audit current repository structure.
- Map current BGE-M3, BM25, Qdrant, RRF, Legal v2, parser, ingestion,
  checkpointing, and evaluation paths.
- Confirm whether proposed installable/package boundaries fit the existing repo.
- Produce final contracts, dependency graph, package boundaries, ADRs, and
  acceptance criteria.

Non-goals:

- No runtime code changes.
- No new runtime components.
- No new Qdrant collection.
- No model download.
- No package install.
- No paid provider calls.
- No commit or push.

Gate:

- Architecture report is internally consistent with the real repo.
- ADRs identify accepted decisions and open questions.
- Prompt 1 scope is precise enough to implement without touching adapters.

## Prompt 1 - Workspace And Retrieval Core

Scope:

- Package workspace.
- Domain models.
- Ports and protocols.
- Error taxonomy.
- Typed contracts.
- Dependency rules.
- Architecture tests.

Forbidden:

- Qdrant, BM25, ColBERT, FastAPI, provider SDK, embedding model, and API wiring.

Gate:

- Core imports no infrastructure libraries.
- Architecture tests fail if forbidden imports are introduced.
- Existing runtime behavior is unchanged.

## Prompt 2 - Pipeline Orchestration

Scope:

- Candidate generation orchestration.
- Fusion contract.
- Document aggregation.
- Evidence selection contract.
- Reranker contract.
- Response assembly.
- Composition interfaces.

Adapters:

- Fake or in-memory only.

Gate:

- Complete pipeline can be tested without Qdrant, models, external providers,
  Docker, or network.
- No existing Legal v2 runtime path is changed unless the prompt explicitly
  states an adapter-free compatibility shim is in scope.

## Prompt 3 - Baseline Adapters

Scope:

- Current BGE-M3 dense retrieval adapter.
- Current BM25 SQLite sidecar adapter.
- Current RRF strategy adapter.
- Current Qdrant read-only access adapter.

Gate:

- New adapter path returns the same Stage A candidate set as the existing
  baseline within a documented tolerance.
- No Qdrant writes.
- Current pipeline remains default and unchanged.

## Prompt 4 - Ingestion Subsystem

Scope:

- Source readers.
- Transformations.
- Embedding provider contracts.
- Batch jobs.
- Checkpointing and resumability.
- Manifests and checksums.
- Idempotent writes.
- Dead-letter/error reporting.

Adapters:

- Use fake late-interaction provider if needed.

Gate:

- Interrupted ingest resumes without duplicate chunks or mixed identities.
- Manifest validates Qdrant and BM25 identity.
- Protected collections cannot be written.

## Prompt 5 - ColBERT Adapter And Experimental Qdrant

Scope:

- Local FastEmbed.
- `answerai-colbert-small-v1`.
- Multivector schema.
- `MAX_SIM`.
- Isolated experimental collection or volume.
- Document batch ingest.
- Query embedding.
- Reranking limited to candidate documents.

Forbidden:

- Current pilot collection writes.
- Production alias changes.
- Default profile changes.

Gate:

- No protected collection is modified.
- Experimental index manifest proves collection, model, schema, and corpus
  isolation.
- Local model availability and no-download behavior are verified before use, or
  the phase stops.

## Prompt 6 - Configurable Pipeline Profiles

Scope:

- Baseline profile.
- Baseline + ColBERT profile.
- Future Qdrant-native profile.
- Config validation.
- Model registry.
- Capability checks.
- Composition root.

Gate:

- Technology changes are selected by configuration.
- Orchestration code is profile-agnostic.
- Invalid profiles fail closed at startup or request initialization.

## Prompt 7 - Evaluation And Comparison

Scope:

- Unified benchmark runner.
- Recall@K, MRR, NDCG.
- Gold coverage.
- Hard-negative rate.
- Rank deltas.
- Latency.
- RAM.
- Disk usage.
- Build duration.
- Reproducible manifests.

Comparison:

```text
baseline
vs.
baseline + ColBERT
```

Forbidden:

- DeepSeek or any semantic verifier in retrieval-quality comparison unless the
  task explicitly evaluates verifier behavior.

Gate:

- ColBERT does not advance unless the result proves statistically and
  substantively useful quality improvement.
- Cost, RAM, disk, and latency are realistic for CPU operation.

## Prompt 8 - Observability And Operational Hardening

Scope:

- Structured logging.
- Trace/correlation IDs.
- Per-stage latency.
- Counters.
- Error taxonomy.
- Health/readiness.
- Resource limits.
- Timeouts.
- Retry policy.
- Circuit-breaker boundaries.
- Safe logging.
- Snapshot/backup procedure.

Gate:

- Metrics labels remain low cardinality.
- No raw query, document id, ECLI, prompt, evidence quote, secret, request id, or
  error string is used as a metric label.
- Runtime failure modes fail closed and are visible.

## Prompt 9 - Optional Legal v2 Integration

Scope:

- Feature flag.
- Experimental endpoint or explicit mode.
- Rollback path.
- Shadow evaluation.
- API contract.

Forbidden:

- Default production behavior changes before benchmark approval.
- Frontend changes before backend experiment passes.

Gate:

- Rollback is one config change.
- Existing stable retrieval endpoints remain backward compatible.
- Product-facing related/verified semantics remain truthful.

