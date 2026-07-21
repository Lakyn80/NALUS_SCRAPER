# Constraint-aware verified retrieval

This is an additive backend module for document-level legal retrieval. It is
disabled by default and does not replace the stable MVP chunk-level endpoints.

## Runtime boundary

- Existing `/api/rag/retrieve` remains chunk-level and backward compatible.
- Existing `/api/rag/query` remains unchanged.
- New endpoint: `POST /api/rag/retrieve-verified`.
- Enable only with `NALUS_CONSTRAINT_RETRIEVAL_ENABLED=1`.
- No embeddings, BGE-M3 model, BM25, RRF, Qdrant collection, index data, or cache
  behavior are changed by this module.

## Pipeline

1. Interpret the query into a typed `StructuredQuery`.
2. Retrieve bounded candidate chunks through the existing orchestrator.
3. Group candidates by canonical document id using the existing document-level
   aggregation code.
4. Reconstruct bounded full-document text through the read-only full-document
   store.
5. Verify hard constraints against trusted metadata and bounded text evidence.
6. Return only documents whose hard constraints are proven.

If strict mode is enabled and a hard constraint is not proven, the candidate is
rejected. The endpoint does not lower thresholds and does not fall back to
unrelated documents.

## Deterministic first rollout

The first rollout intentionally uses `deterministic_v1` interpretation and
verification. It does not send full document text to an external LLM provider.
The typed model can support a future strictly validated LLM interpreter/verifier,
but that must be enabled in a separate reviewed task.

Covered deterministic constraints include:

- court identity from trusted metadata/document id,
- Czech citizenship application/grant/refusal event,
- applicant/person nationality,
- international child abduction / wrongful removal or retention,
- destination-country relation,
- parent actor role.

## Configuration

```env
NALUS_CONSTRAINT_RETRIEVAL_ENABLED=0
NALUS_CONSTRAINT_RETRIEVAL_STRICT_MODE=1
NALUS_CONSTRAINT_MAX_CANDIDATE_CHUNKS=200
NALUS_CONSTRAINT_MAX_CANDIDATE_DOCUMENTS=50
NALUS_CONSTRAINT_MAX_RETURNED_DOCUMENTS=20
NALUS_CONSTRAINT_MAX_SUPPORTING_CHUNKS=3
NALUS_CONSTRAINT_MAX_CHUNKS_PER_DOCUMENT_FOR_VERIFICATION=24
NALUS_CONSTRAINT_MAX_DOCUMENT_CHARACTERS_FOR_VERIFICATION=40000
NALUS_CONSTRAINT_TOTAL_LATENCY_BUDGET_MS=10000
NALUS_CONSTRAINT_DOCUMENT_VERIFICATION_TIMEOUT_MS=1500
NALUS_CONSTRAINT_STRUCTURED_QUERY_STRATEGY=deterministic_v1
NALUS_CONSTRAINT_VERIFICATION_STRATEGY=deterministic_v1
NALUS_CONSTRAINT_RANKING_STRATEGY=retrieval_plus_constraint_coverage_v1
NALUS_CONSTRAINT_INCLUDE_REJECTED_DOCUMENTS=0
```

All limits are validated at startup/request time. Invalid config returns a clear
503 error for the additive endpoint.

## API response

`POST /api/rag/retrieve-verified`

Request:

```json
{
  "query": "udělení českého občanství ruskému občanu",
  "sources": ["constitutional"],
  "debug": false
}
```

Response includes:

- `structured_query`: extracted constraints, entities, relations, ambiguities,
  and retrieval expansions;
- `documents`: verified unique documents only;
- `rejected_documents`: omitted unless `debug=true` or explicitly enabled;
- `diagnostics`: bounded counts, latency, and exclusion counts.

Evidence quotes are bounded snippets. They are intended for auditability, not as
full-document output.

## Observability

The module reuses the existing Prometheus client conventions. It does not add a
second metrics system.

Metric labels are bounded:

- endpoint,
- status,
- decision status,
- constraint category,
- verification status,
- verification method.

Prometheus labels must never include raw queries, document ids, ECLI values,
chunk ids, evidence quotes, or other sensitive content.

## Rollout and rollback

Rollout:

1. Keep `NALUS_CONSTRAINT_RETRIEVAL_ENABLED=0` in MVP runtime.
2. Run endpoint tests and offline seed evaluation.
3. Enable only in an isolated environment.
4. Compare verified results against manual review.
5. Route frontend traffic only after explicit product/quality approval.

Rollback:

1. Set `NALUS_CONSTRAINT_RETRIEVAL_ENABLED=0`.
2. Existing MVP endpoints remain available.
3. No Qdrant/index/cache migration is required.

## Known limitations

- Deterministic Czech legal-language patterns are conservative.
- The seed dataset is not a gold benchmark until manually reviewed.
- Full-document verification depends on indexed chunks being reconstructable by
  canonical document id.
