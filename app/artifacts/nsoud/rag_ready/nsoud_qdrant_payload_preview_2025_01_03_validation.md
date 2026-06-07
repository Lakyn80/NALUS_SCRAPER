# NSoud Qdrant Payload Preview Validation

- Input: `app\artifacts\nsoud\rag_ready\nsoud_chunks_2025_01_03.parquet`
- Output Parquet: `app\artifacts\nsoud\rag_ready\nsoud_qdrant_payload_preview_2025_01_03.parquet`
- Output JSONL: `app\artifacts\nsoud\rag_ready\nsoud_qdrant_payload_preview_2025_01_03.jsonl`
- Validation status: **WARN**
- Total payload rows: **1785**
- Duplicate point_id count: **0**
- Duplicate chunk_id count: **0**
- Empty text count: **0**

## Status
- Some optional metadata fields are missing.

## Missing Required Field Counts

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
| `legal_area` | 13 |
| `title` | 0 |
| `url` | 0 |
| `source_attribution` | 0 |
| `content_hash` | 0 |
| `document_id` | 0 |
| `chunk_id` | 0 |
| `chunk_index` | 0 |
| `chunk_text_length` | 0 |
| `paragraph_count` | 0 |
| `chunk_warning` | 0 |
| `ns_section_hint` | 0 |

## Chunk Text Lengths
- min: 13
- max: 8154
- avg: 1442.95

## Source Distribution

| Value | Count |
| --- | ---: |
| nsoud | 1785 |

## Authority Level Distribution

| Value | Count |
| --- | ---: |
| supreme | 1785 |

## Document Type Distribution

| Value | Count |
| --- | ---: |
| USNESENÍ | 1592 |
| ROZSUDEK | 193 |

## Legal Area Distribution

| Value | Count |
| --- | ---: |
| civil | 932 |
| criminal | 840 |
| <missing> | 13 |

## Chunk Warning Distribution

| Value | Count |
| --- | ---: |
| <missing> | 1771 |
| overlong_ns_paragraph | 14 |

## NS Section Hint Distribution

| Value | Count |
| --- | ---: |
| oduvodneni | 1162 |
| closing | 150 |
| pouceni | 146 |
| header | 142 |
| vyrok | 126 |
| unknown | 59 |
