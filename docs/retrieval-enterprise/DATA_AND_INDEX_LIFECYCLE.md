# Retrieval Enterprise Data And Index Lifecycle

Status: target lifecycle specification.

The data lifecycle must be auditable from source files through retrieval results.
No experiment may modify production indexes, aliases, sidecars, caches, or
frontend behavior unless a rollout task explicitly approves it.

## Source Intake

Sources:

- NALUS Constitutional Court documents.
- NSoud Supreme Court documents when explicitly selected.
- Future sources only through source adapters with provenance.

Every source document must have:

- stable `document_id`;
- source name;
- source URL or source file provenance when available;
- decision date when available;
- court;
- raw or normalized full text;
- metadata extraction status.

Incomplete sources are not silently repaired. They are either excluded or
included under an explicit partial-document policy recorded in the manifest.

## Parse And Audit

Parsing produces:

- normalized document text;
- ordered legal paragraphs;
- section type;
- paragraph IDs;
- source offsets;
- diagnostics.

Audit must detect:

- empty or tiny documents;
- missing identifiers;
- duplicate identifiers;
- damaged formatting;
- reconstruction mismatch;
- paragraph boundary issues;
- cross-document mixing;
- unsupported source format;
- incomplete source records.

Parser audit is read-only. It does not embed, write Qdrant, write BM25, call
providers, or download models.

## Chunking

Chunking produces child retrieval chunks and parent evidence windows.

Rules:

- Chunk IDs are deterministic.
- Chunk payloads include source document ID and paragraph provenance.
- Chunks must not mix documents.
- Parent windows must be reconstructable from paragraph IDs.
- Chunk config version is recorded in index manifest.

## Embedding

Rules:

- CPU is the default execution target.
- Existing cached BGE-M3 is used unless a prompt explicitly approves a different
  local model.
- No model download is allowed in audit or baseline parity phases.
- Embedding dimension is validated before writes.
- Embedding cache is an optimization only; it must not change ranking semantics.

## Qdrant And BM25 Writes

Writes are allowed only in ingestion phases and only to validated isolated
targets.

Protected resources include at minimum:

- `nalus`
- `nalus_live`
- `nalus_bge_m3_chunks_v1`
- production aliases and volumes
- any collection not matching the accepted experiment naming policy.

BM25 sidecars must use an index ID matching the collection/profile. A pilot or
experiment collection must not reuse the canonical BM25 index ID unless an ADR
explicitly accepts that mapping.

## Checkpoint And Resume

Checkpoint files must record:

- collection;
- BM25 path and index ID;
- source corpus hash;
- source selection;
- model identity and dimension;
- parser/chunker/builder versions;
- completed document batches;
- failed document IDs and reasons.

Resume must validate the checkpoint before writing. A mismatch fails closed.

Required resume properties:

- no duplicate Qdrant points;
- no duplicate BM25 chunk rows;
- no skipped approved source documents;
- no mixed source corpus;
- no deletion of already-valid unrelated resources.

## Date-Range Builds

When a build is scoped by date range:

- the date range must be present in CLI args/config;
- the range must be recorded in `source_selection`;
- source document counts must be computed before write;
- documents outside the range must be excluded before chunking;
- validation must prove no out-of-range document was indexed.

## Manifests

Every build writes a machine-readable JSON manifest and a human-readable summary.

A manifest marked `pass` must prove:

- parser-approved documents were indexed or explicitly excluded;
- Qdrant point IDs match expected chunk IDs;
- BM25 chunk IDs match expected chunk IDs;
- model/dimension/profile match config;
- protected resources were not targeted;
- checkpoint was cleared after successful completion.

## Retention

Generated build artifacts belong under `artifacts/` and are not committed unless
a task explicitly says a small durable report should be committed. Technical
documentation belongs under `docs/`. Chronological handoff belongs only in
`PROJECT_PROGRESS.md`.

