# Retrieval Enterprise Migration And Rollback

Status: target migration specification.

Migration is additive and reversible. The current baseline remains available
until a separate rollout decision changes default behavior.

## Migration Strategy

1. Document architecture and ADRs.
2. Add core contracts without adapters.
3. Add adapter-free orchestration.
4. Add baseline adapters and prove parity.
5. Add ingestion subsystem with checkpoint safety.
6. Add late-interaction experiment in isolated resources.
7. Add profile-based composition.
8. Evaluate baseline versus experiment.
9. Roll out only through feature flags and explicit mode selection.

## Backward Compatibility

Must remain backward compatible unless a task explicitly says otherwise:

- existing API routes;
- existing request/response fields;
- current Legal v2 disabled guard behavior;
- current production retrieval path;
- current Qdrant production collections;
- current BM25 production sidecars;
- current frontend behavior.

New fields may be additive if they are optional and safely ignored by old
clients.

## Feature Flags

Every experimental runtime path needs:

- default-off flag;
- explicit profile name;
- config validation;
- diagnostics showing selected profile;
- rollback instruction.

The default value must preserve current production behavior.

## Rollout Stages

### Local

- fake adapters and unit tests;
- no external providers unless scoped;
- no production resources.

### Isolated Runtime

- isolated Qdrant collection;
- isolated BM25 sidecar;
- no frontend traffic by default;
- bounded smoke tests.

### Shadow Evaluation

- run benchmark/profile comparison;
- capture latency, RAM, disk, and quality metrics;
- do not alter user-facing results.

### Controlled Pilot

- explicit endpoint or mode;
- clear related/verified semantics;
- rollback through one config change.

### Production Candidate

- benchmark gates passed;
- operations gates passed;
- rollback rehearsed;
- documentation updated.

## Rollback

Rollback must avoid data deletion when possible:

- disable feature flag;
- switch profile back to baseline;
- stop using experimental collection/sidecar;
- leave artifacts and manifests for audit;
- do not delete experimental data unless a cleanup task explicitly approves it.

Emergency rollback must be one config change for runtime behavior.

## Blockers

Stop and report before continuing when:

- protected resources would be written;
- package boundaries require a forbidden dependency;
- benchmark data conflicts with source evidence;
- provider output cannot be parsed safely;
- an experiment requires model/package download not authorized by the prompt;
- current runtime behavior cannot be preserved;
- rollback cannot be made explicit.

