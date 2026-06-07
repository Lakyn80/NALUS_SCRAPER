# NSoud Embeddings Validation

- Input: `app\artifacts\nsoud\rag_ready\nsoud_qdrant_payload_preview_2025_01_03.parquet`
- Output: `app\artifacts\nsoud\rag_ready\nsoud_embeddings_2025_01_03_test.parquet`
- Validation status: **PASS**
- Total rows: **20**
- Embedding dim: **768**
- Missing embeddings count: **0**
- Duplicate point_id count: **0**
- Empty text count: **0**
- Model name: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- Device: `cpu`
- Backend: `docker-api`
- Legal area missing count: **0**

## Status
- Embedding validation passed.

## Source Distribution

| Value | Count |
| --- | ---: |
| nsoud | 20 |

## Document Type Distribution

| Value | Count |
| --- | ---: |
| USNESENÍ | 10 |
| ROZSUDEK | 10 |

## Legal Area Distribution

| Value | Count |
| --- | ---: |
| civil | 20 |

## Recommended Docker Command

`docker compose exec api python app/nsoud/generate_embeddings.py --input app/artifacts/nsoud/rag_ready/nsoud_qdrant_payload_preview_2025_01_03.parquet --out app/artifacts/nsoud/rag_ready/nsoud_embeddings_2025_01_03_test.parquet --batch-size 32 --device auto`
