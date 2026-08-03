# Retrieval Enterprise Contracts

Status: target contract specification.

Contracts are stable interfaces between layers. Implementations may evolve, but
they must not change contract semantics without an ADR and migration plan.

## Contract Rules

- Contracts are typed and deterministic.
- Runtime adapters return core contract objects, not raw SDK objects.
- Contract objects do not contain secrets, provider raw responses, full prompts,
  or unbounded full judgment text.
- All identifiers used for persistence are stable and reproducible.
- Query-time results are bounded.
- Verification status is not presentation status.

## QuerySpec

Represents the user's legal-document search intent.

Required fields:

- `original_query`
- `normalized_query`
- `intent`
- `retrieval_queries`
- `hard_constraints`
- `soft_constraints`
- `negative_constraints`
- `ambiguities`
- `requires_verification`

Rules:

- Hard constraints are dispositive legal requirements that must be proven.
- Soft constraints improve retrieval and ranking but do not alone reject a
  document.
- Negative constraints reject contradictory results when proven.
- Structural fact slots, such as actor, object, origin, and destination, must
  not become hard constraints unless the prompt/ADR defines why they are
  dispositive for the legal task.
- QuerySpec extraction failures fail closed or use an explicitly declared
  deterministic fallback.

## CandidateChunk

Represents one indexed retrieval unit.

Required fields:

- `chunk_id`
- `document_id`
- `text`
- `score`
- `metadata`
- `source`
- `chunk_index` or `source_order`
- provenance fields sufficient to reconstruct the indexed source.

Rules:

- Chunk IDs are deterministic.
- Chunks without a document ID may be counted in diagnostics but must not become
  verified documents.
- Chunk payloads must not cross document boundaries.

## CandidateDocument

Represents one unique court decision after aggregation.

Required fields:

- `document_id`
- `score`
- `metadata`
- `paragraphs`
- `chunk_ids`
- optional rank provenance, such as dense rank, BM25 rank, and fused score.

Rules:

- A document is unique by canonical document ID.
- Aggregation must preserve strongest supporting passages.
- Aggregation must not invent text or metadata.
- Candidate count limits must be explicit and reported in diagnostics.

## EvidenceWindow

Represents bounded evidence submitted to a verifier or reranker.

Required fields:

- `constraint_id`
- `paragraph_ids`
- `text`
- `section_types`
- `source_of_claim`
- `heading_context`

Rules:

- Evidence text is bounded.
- Evidence paragraphs must belong to the candidate document.
- Evidence should prefer current-case holdings/reasoning over cited-case
  summaries when the query asks for the current decision.
- Full judgments are not sent as evidence unless a prompt explicitly approves a
  bounded full-document verification experiment.

## FusionStrategy

Combines multiple candidate lists.

Required behavior:

- deterministic ordering;
- tie-breakers documented;
- source rank provenance retained;
- no silent threshold fallback.

Current baseline:

- RRF over dense and BM25 chunk lists.

## Reranker

Optional contract used after candidate generation.

Rules:

- Reranker operates over bounded candidate documents or evidence windows.
- Reranker cannot add candidates that were not supplied by candidate generation.
- Reranker cannot promote a document to verified status.
- Late-interaction reranking must report model, schema, input count, output
  count, latency, and truncation/limit decisions.

## Verifier

Classifies whether the candidate evidence satisfies the QuerySpec.

Required output:

- `document_id`
- `decision`
- `relevance_classification`
- `constraint_results`
- `evidence_references`
- `reason`
- bounded diagnostics.

Allowed terminal decisions:

- `verified_match`
- `hard_mismatch`
- `not_proven`
- `ambiguous`
- `unverifiable_query`
- `verifier_error`

Rules:

- `verified_match` requires all active hard constraints to be proven by allowed
  evidence.
- `related_only`, `partial_match`, `incidental_overlap`,
  `insufficient_evidence`, and `not_relevant` must not be returned as verified
  documents.
- Provider JSON syntax failures and schema validation failures must be recorded
  separately.
- `finish_reason`, content length, token usage, truncation indicators,
  extraction method, JSON parse error, and schema error paths must be available
  in safe diagnostics for provider-backed verification.

## SearchResult

Represents one search operation.

Required fields:

- `status`
- `interpretation_status`
- `query_spec_summary`
- `verified_documents`
- `related_documents`
- `rejected_documents` when debug or explicitly enabled
- `rejection_counts`
- `latency_ms_by_stage`
- `provider`
- `index`
- `diagnostics`

Rules:

- `verified_documents` contains only verified matches.
- `related_documents` can help UX but must remain semantically separate.
- A successful search with no verified result returns `status=no_verified_results`
  and an empty verified list, not an HTTP error.
- Diagnostics are bounded and redacted.

## IndexManifest

Required fields:

- collection name;
- BM25 path and index ID;
- source corpus and document count;
- indexed/excluded counts;
- chunk count;
- model identity and dimension;
- parser/chunker/builder versions;
- git commit and dirty flag;
- corpus hash;
- checkpoint/resume summary;
- write validation status.

Rules:

- Manifest cannot pass when Qdrant and BM25 identities differ.
- Manifest cannot pass when protected resources were targeted.
- Manifest must describe source selection and date range when used.

