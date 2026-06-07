# NSoud Qdrant Upload Dry Run

- Input: `app/artifacts/nsoud/rag_ready/nsoud_embeddings_2025_01_03.parquet`
- Collection name: `nsoud_chunks_test_2025_01_03`
- Validation status: **WARN**
- QDRANT_UPLOAD_READY: **true**
- Total points: **1785**
- Vector size: **768**
- Duplicate point_id count: **0**
- Missing embedding count: **0**
- Inconsistent embedding_dim count: **0**
- Empty text count: **0**
- Report path: `app/artifacts/nsoud/rag_ready/nsoud_qdrant_upload_dry_run_2025_01_03.md`
- Upload plan path: `app/artifacts/nsoud/rag_ready/nsoud_qdrant_upload_plan_2025_01_03.json`

## Status
- Some optional payload metadata fields are missing.

## Missing Required Metadata Counts

| Field | Missing Count |
| --- | ---: |
| `point_id` | 0 |
| `text` | 0 |
| `embedding` | 0 |
| `embedding_dim` | 0 |
| `provider` | 0 |
| `source` | 0 |
| `court` | 0 |
| `authority_level` | 0 |
| `case_number` | 0 |
| `document_id` | 0 |
| `chunk_id` | 0 |
| `chunk_index` | 0 |
| `url` | 0 |

## Missing Optional Metadata Counts

| Field | Missing Count |
| --- | ---: |
| `ecli` | 0 |
| `decision_date` | 0 |
| `publication_date` | 0 |
| `document_type` | 0 |
| `legal_area` | 13 |
| `title` | 0 |
| `source_attribution` | 0 |
| `content_hash` | 0 |
| `chunk_text_length` | 0 |
| `paragraph_count` | 0 |
| `chunk_warning` | 0 |
| `ns_section_hint` | 0 |

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

## Recommended Next Docker Command

`docker compose exec api python app/nsoud/qdrant_upload_dry_run.py --input app/artifacts/nsoud/rag_ready/nsoud_embeddings_2025_01_03.parquet --collection nsoud_chunks_test_2025_01_03 --out-report app/artifacts/nsoud/rag_ready/nsoud_qdrant_upload_dry_run_2025_01_03.md --out-plan app/artifacts/nsoud/rag_ready/nsoud_qdrant_upload_plan_2025_01_03.json`
