# NSoud Embeddings Validation

- Input: `app/artifacts/nsoud/rag_ready/nsoud_qdrant_payload_preview_2025_01_03.parquet`
- Output: `app/artifacts/nsoud/rag_ready/nsoud_embeddings_2025_01_03.parquet`
- Validation status: **PASS**
- Input rows: **1862**
- Output rows: **1862**
- Embedding dim: **768**
- Missing embeddings count: **0**
- Duplicate point_id count: **0**
- Duplicate chunk_id count: **0**
- Empty text count: **0**
- Missing required metadata count: **0**
- Metadata preservation status: **PASS**
- Document sequence validation passed/failed: **150/0**
- Section sequence validation passed/failed: **729/0**
- Document neighbor validation passed/failed: **150/0**
- Section neighbor validation passed/failed: **729/0**
- Model name: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- Device: `cpu`
- Backend: `local`

## Status
- Embedding validation passed.

## Provider Distribution

| Value | Count |
| --- | ---: |
| nsoud | 1862 |

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

## Text Lengths

- min: 25
- max: 8155
- avg: 1384.16

## Recommended Docker Command

`docker compose exec api python app/nsoud/generate_embeddings.py --input app/artifacts/nsoud/rag_ready/nsoud_qdrant_payload_preview_2025_01_03.parquet --out app/artifacts/nsoud/rag_ready/nsoud_embeddings_2025_01_03.parquet --batch-size 32 --device auto`
