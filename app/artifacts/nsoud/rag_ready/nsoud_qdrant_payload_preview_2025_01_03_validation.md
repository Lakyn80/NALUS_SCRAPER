# NSoud Qdrant Payload Preview Validation

- Input: `app/artifacts/nsoud/rag_ready/nsoud_chunks_2025_01_03.parquet`
- Output Parquet: `app/artifacts/nsoud/rag_ready/nsoud_qdrant_payload_preview_2025_01_03.parquet`
- Output JSONL: `app/artifacts/nsoud/rag_ready/nsoud_qdrant_payload_preview_2025_01_03.jsonl`
- Validation status: **PASS**
- Total payload rows: **1862**
- Duplicate point_id count: **0**
- Duplicate chunk_id count: **0**
- Empty text count: **0**
- Missing required metadata count: **0**
- Document sequence validation passed/failed: **150/0**
- Section sequence validation passed/failed: **729/0**
- Document neighbor validation passed/failed: **150/0**
- Section neighbor validation passed/failed: **729/0**

## Status
- Payload preview validation passed.

## Missing Field Counts

| Field | Missing Count |
| --- | ---: |
| `point_id` | 0 |
| `text` | 0 |
| `source` | 0 |
| `provider` | 0 |
| `court` | 0 |
| `authority_level` | 0 |
| `case_number` | 0 |
| `ecli` | 0 |
| `decision_date` | 0 |
| `publication_date` | 0 |
| `document_type` | 0 |
| `legal_area` | 15 |
| `title` | 0 |
| `url` | 0 |
| `source_attribution` | 0 |
| `content_hash` | 0 |
| `document_id` | 0 |
| `chunk_id` | 0 |
| `chunk_index` | 0 |
| `total_chunks_in_document` | 0 |
| `section_id` | 0 |
| `section_type` | 0 |
| `section_index` | 0 |
| `chunk_index_in_section` | 0 |
| `total_chunks_in_section` | 0 |
| `previous_chunk_id` | 150 |
| `next_chunk_id` | 150 |
| `previous_section_chunk_id` | 729 |
| `next_section_chunk_id` | 729 |
| `chunk_text_length` | 0 |
| `paragraph_count` | 0 |
| `chunk_warning` | 1849 |
| `ns_section_hint` | 0 |
| `structure_confidence` | 0 |
| `structure_status` | 0 |
| `structure_needs_review` | 0 |
| `detected_section_order` | 0 |
| `detected_markers` | 0 |
| `section_source` | 0 |
| `chunking_strategy` | 0 |

## Text Lengths
- min: 25
- max: 8155
- avg: 1384.16

## Document Type Distribution

| Value | Count |
| --- | ---: |
| USNESENÍ | 1664 |
| ROZSUDEK | 198 |

## Legal Area Distribution

| Value | Count |
| --- | ---: |
| civil | 981 |
| criminal | 866 |
| <missing> | 15 |

## Section Type Distribution

| Value | Count |
| --- | ---: |
| reasoning | 1194 |
| signature | 192 |
| operative_part | 165 |
| appeal_instruction | 161 |
| header | 150 |

## Structure Status Distribution

| Value | Count |
| --- | ---: |
| strong | 1696 |
| medium | 166 |

## Chunk Warning Distribution

| Value | Count |
| --- | ---: |
| <missing> | 1849 |
| overlong_ns_paragraph | 13 |
