# Canonical block / chunk schema v1

**Status:** Phase 2 data contract (experiment / pilot)  
**Parser baseline:** `legal-decision-parser.cz-courts.v7` @ `a53bf53`  
**Binding plan:** [`NALUS_LEGAL_RAG_MASTER_PLAN.md`](./NALUS_LEGAL_RAG_MASTER_PLAN.md) §5 and Phase 2  
**Typed models:** `app/rag/legal_v2/schema/canonical_v1.py`  
**Mapper:** `app/rag/legal_v2/schema/map_from_legal_v2.py`

---

## Purpose

Lock a stable document → block → child → parent identity contract so later
chunking and retrieval experiments can compare variants without renaming
payloads ad hoc.

This schema does **not** cut over the production Qdrant / BM25 write path.
Production ingest continues to use `LegalParagraph` / `RetrievalChildChunk` /
`ParentEvidenceWindow` until a later migration after a Phase 3 chunking winner.

---

## Entities

### Document

| Field | Type | Notes |
|---|---|---|
| `document_id` | string | Stable NALUS document id |
| `source_document_id` | string \| null | Upstream source id when known |
| `ecli` | string \| null | |
| `case_number` | string \| null | |
| `court` | string \| null | |
| `court_chamber` | string \| null | |
| `decision_type` | string \| null | |
| `decision_date` | string \| null | ISO date preferred |
| `jurisdiction` | string | default `CZ` |
| `language` | string | default `cs` |
| `source_url` | string \| null | |
| `source_checksum` | string \| null | SHA-256 of source bytes/text when known |
| `parser_profile` | string | e.g. `legal-decision-parser.cz-courts.v7` |

### Parser block

| Field | Type | Notes |
|---|---|---|
| `block_id` | string | Stable; when mapped from legal_v2 equals `paragraph_id` |
| `document_id` | string | |
| `block_index` | int | Zero-based document order |
| `line_start` | int \| null | 1-based inclusive when line inventory available |
| `line_end` | int \| null | 1-based inclusive |
| `start_offset` | int \| null | Char offset in document text (legacy ingest) |
| `end_offset` | int \| null | |
| `raw_text` | string | Original reconstruction unit |
| `normalized_text` | string | |
| `primary_class` | string | Dominant structural / section class |
| `all_line_classes` | string[] | Per-line classes when known |
| `section_path` | string[] | Hierarchical section labels |
| `heading_context` | string[] | |
| `paragraph_number` | string \| null | Display numbering when present |
| `hierarchy_level` | int \| null | |
| `parent_block_id` | string \| null | Soft structural parent when known |
| `citations` | string[] | Reserved; may be empty in v1 |
| `statutes` | string[] | Reserved |
| `case_references` | string[] | Reserved |
| `dates` | string[] | Reserved |
| `source_checksum` | string | Checksum of `raw_text` |

### Child retrieval chunk

| Field | Type | Notes |
|---|---|---|
| `chunk_id` | string | Stable retrieval unit id |
| `document_id` | string | |
| `source_block_ids` | string[] | Ordered block ids composing the child |
| `line_start` | int \| null | Min line over source blocks when known |
| `line_end` | int \| null | Max line over source blocks when known |
| `start_offset` | int \| null | Char span when known |
| `end_offset` | int \| null | |
| `chunk_text` | string | Indexed / displayed child text |
| `embedding_text` | string | Text used for embedding (may equal `chunk_text`) |
| `section_path` | string[] | |
| `heading_context` | string[] | |
| `primary_paragraph_number` | string \| null | |
| `parent_id` | string \| null | Owning parent context id |
| `token_count` | int | |
| `chunking_profile` | string | Profile id string (A–D later); see below |
| `content_checksum` | string | Checksum of `chunk_text` |

### Parent context

| Field | Type | Notes |
|---|---|---|
| `parent_id` | string | Stable; when mapped from legal_v2 equals `window_id` |
| `document_id` | string | |
| `child_ids` | string[] | Children belonging to this parent |
| `line_start` | int \| null | |
| `line_end` | int \| null | |
| `start_offset` | int \| null | |
| `end_offset` | int \| null | |
| `parent_text` | string | Generation / expansion context |
| `section_path` | string[] | |
| `context_type` | string | e.g. section evidence window |
| `token_count` | int | |
| `content_checksum` | string | Checksum of `parent_text` |

---

## Stable identity and checksum rules

- Identities must be deterministic for the same document text, parser profile,
  block order, and chunking profile.
- When mapping from current legal_v2 ingest, preserve existing
  `paragraph_id` / `chunk_id` / `window_id` values as
  `block_id` / `chunk_id` / `parent_id` so experiments can join to old payloads.
- Greenfield helpers use SHA-256 truncated digests with typed prefixes
  (`:b:`, `:c:`, `:p:`).
- Content checksums are lowercase hex SHA-256 of UTF-8 payload bytes.
- Stable identity must support chunking-variant comparison, source
  reconstruction, exact citation, deduplication, and reindex regression tracking.

### Default chunking profile (Phase 2)

Current hierarchical mapper emits:

```text
legal_v2_hierarchical_parent_child_v1
```

Phase 3 will introduce explicit A/B/C/D profile ids without renaming this
contract field.

---

## Legacy alias map

| Canonical v1 | Current legal_v2 / Qdrant |
|---|---|
| `block_id` | `paragraph_id`, review `stable_block_id` / `parser_block_id` |
| `source_block_ids` | `paragraph_ids` |
| `parent_id` | `window_id`, payload `parent_window_id` |
| `child_ids` | `child_chunk_ids` |
| `chunk_text` | `RetrievalChildChunk.text` / payload `text` |
| `parent_text` | `ParentEvidenceWindow.text` |
| `chunking_profile` | (absent; closest: `chunker_version`) |
| `line_start` / `line_end` | review export only today |
| `start_offset` / `end_offset` | char offsets on paragraphs/chunks |
| `parser_profile` | `parser_version` constant / audit field |

Production payload writer remains unchanged in Phase 2.

---

## Invariants (Phase 2 gate)

1. Every child reconstructs to the concatenation of its source blocks’ `raw_text`
   without loss or duplication of those blocks’ content for the declared ids.
2. Every parent’s `child_ids` reference children that declare the same `parent_id`
   (when `parent_id` is set).
3. A parent must not claim children from another document.
4. Block / child / parent ids are unique within a document bundle.
5. Round-trip JSON serialize → deserialize preserves the contract fields.

---

## Out of scope

- Parser rule changes and parser v8
- Full corpus indexing / Qdrant upsert
- Chunking A/B/C/D winner selection
- Retrieval golden annotation (100–150 queries)
