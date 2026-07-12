# Document-level exhaustive retrieval

Status: additive module, disabled by default.

## Goal

NALUS document-level retrieval is a retrieval architecture feature. It does not generate legal advice, perform legal reasoning, summarize decisions, or call an LLM.

The goal is to return relevant unique court decisions that can be identified within the indexed corpus under configured retrieval limits and relevance policy.

## Pipeline

The existing chunk-level retrieval path remains unchanged.

The additive document-level path is:

1. retrieve a bounded candidate chunk pool through the existing hybrid retriever;
2. group candidate chunks by canonical document id;
3. deduplicate repeated chunk/document hits;
4. compute deterministic document-level relevance scores;
5. filter documents by configured relevance threshold;
6. return bounded unique documents with strongest supporting passages.

## Configuration

The endpoint is disabled by default.

Environment variables:

- `NALUS_DOCUMENT_RETRIEVAL_ENABLED`
- `NALUS_DOCUMENT_MAX_CANDIDATE_CHUNKS`
- `NALUS_DOCUMENT_MAX_RETURNED_DOCUMENTS`
- `NALUS_DOCUMENT_MAX_SUPPORTING_CHUNKS_PER_DOCUMENT`
- `NALUS_DOCUMENT_RELEVANCE_THRESHOLD`
- `NALUS_DOCUMENT_SCORING_STRATEGY`
- `NALUS_DOCUMENT_LATENCY_BUDGET_MS`

The latency budget is diagnostic-only. The implementation does not silently lower thresholds, expand limits, or fall back to unrelated documents.

## Document grouping

Grouping uses the existing canonical identifiers in this order:

1. `source_document_id`
2. `document_id`
3. `ecli`
4. `case_reference`
5. `reference`

Chunks without a usable document identifier are skipped and counted in diagnostics.

## Scoring

The first supported strategy is `best_plus_average_top_chunks`.

It combines:

- the best chunk score;
- the average of the top supporting chunk scores.

This keeps scoring deterministic while avoiding an irreversible one-chunk-only design. Additional strategies can be added behind the scoring interface.

## API

New endpoint:

`POST /api/rag/retrieve-documents`

The existing `/api/rag/retrieve` endpoint remains unchanged and continues returning chunk-level `results`.

Response shape:

- `documents`
  - `document_id`
  - `score`
  - `best_passages`
  - `metadata`
  - `candidate_chunk_count`
  - `best_chunk_score`
- `diagnostics`
  - candidate chunks retrieved
  - unique documents produced
  - duplicate hits removed
  - documents filtered
  - final document count
  - threshold and configured limits
  - retrieval and aggregation latency

## Safety properties

- No ingest.
- No Qdrant writes.
- No embedding regeneration.
- No model download.
- No LLM call.
- No BM25 scoring change.
- No RRF change.
- No existing API field removal or rename.
- No hidden threshold fallback.

## Extension points

Future work can add:

- additional scoring strategies;
- document-level reranking;
- follow-up retrieval over selected document ids;
- offline benchmark scripts for recall/precision tuning.

These should be implemented as explicit additive changes with tests and without changing the default chunk-level retrieval path.
