# NSoud Embeddings Validation

- Input: `app/artifacts/nsoud/rag_ready/nsoud_qdrant_payload_preview_2025_01_03.parquet`
- Output: `app/artifacts/nsoud/rag_ready/nsoud_embeddings_2025_01_03.parquet`
- Validation status: **WARN**
- Total rows: **1785**
- Embedding dim: **768**
- Missing embeddings count: **0**
- Duplicate point_id count: **0**
- Empty text count: **0**
- Model name: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- Device: `cpu`
- Backend: `local`
- Legal area missing count: **13**

## Status
- Some optional metadata fields are missing.

## Source Distribution

| Value | Count |
| --- | ---: |
| nsoud | 1785 |

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

## Recommended Docker Command

`docker compose exec api python app/nsoud/generate_embeddings.py --input app/artifacts/nsoud/rag_ready/nsoud_qdrant_payload_preview_2025_01_03.parquet --out app/artifacts/nsoud/rag_ready/nsoud_embeddings_2025_01_03.parquet --batch-size 32 --device auto`
