# NSoud JSONL to Parquet Validation

- Input: `app\artifacts\nsoud\nsoud_consolidated_2025_01_03.jsonl`
- Output Parquet: `app\artifacts\nsoud\rag_ready\nsoud_documents_2025_01_03.parquet`
- Validation status: **WARN**
- Total records: **150**
- Duplicate content_hash count: **0**
- Duplicate URL count: **0**

## Status
- Some optional metadata fields are missing.

## Columns
source, court, authority_level, case_number, ecli, decision_date, publication_date, document_type, legal_area, title, url, full_text, source_attribution, scraped_at, content_hash, full_text_length, has_ecli, has_decision_date, has_publication_date, has_legal_area

## Missing Value Counts

| Column | Missing Count |
| --- | ---: |
| `source` | 0 |
| `court` | 0 |
| `authority_level` | 0 |
| `case_number` | 0 |
| `ecli` | 0 |
| `decision_date` | 0 |
| `publication_date` | 0 |
| `document_type` | 0 |
| `legal_area` | 3 |
| `title` | 0 |
| `url` | 0 |
| `full_text` | 0 |
| `source_attribution` | 0 |
| `scraped_at` | 0 |
| `content_hash` | 0 |
| `full_text_length` | 0 |
| `has_ecli` | 0 |
| `has_decision_date` | 0 |
| `has_publication_date` | 0 |
| `has_legal_area` | 0 |

## Full Text Lengths
- min: 2091
- max: 86652
- avg: 17181.99

## Source Distribution

| Value | Count |
| --- | ---: |
| nsoud | 150 |

## Document Type Distribution

| Value | Count |
| --- | ---: |
| USNESENÍ | 135 |
| ROZSUDEK | 15 |

## Legal Area Distribution

| Value | Count |
| --- | ---: |
| civil | 107 |
| criminal | 40 |
| <missing> | 3 |
