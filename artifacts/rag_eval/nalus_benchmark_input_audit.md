# NALUS rag-embedding-benchmark input audit

Generated: 2026-07-04

## Summary

The NALUS repository already contains a complete **NSOud (Nejvyšší soud)** pilot corpus in `app/artifacts/nsoud/rag_ready/` with an uploaded Qdrant collection and a categorized relevance eval dataset. The benchmark will **re-embed all chunks per candidate model** into new Qdrant collections via `rag-embedding-benchmark` — it does not query the existing production collection directly.

## Input data found

| Asset | Path | Format | Count / status |
|-------|------|--------|----------------|
| Documents | `app/artifacts/nsoud/rag_ready/nsoud_documents_2025_01_03.parquet` | Parquet | 150 documents |
| Chunks | `app/artifacts/nsoud/rag_ready/nsoud_chunks_2025_01_03.parquet` | Parquet | 1862 chunks (`document_section_aware`) |
| Qdrant payload preview | `app/artifacts/nsoud/rag_ready/nsoud_qdrant_payload_preview_2025_01_03.parquet` | Parquet | 1862 rows |
| Embeddings (baseline) | `app/artifacts/nsoud/rag_ready/nsoud_embeddings_2025_01_03.parquet` | Parquet | 1862 × 768-dim (`paraphrase-multilingual-mpnet-base-v2`) |
| Existing eval dataset | `app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/relevance_eval_dataset.json` | JSON | 14 positive, 5 negative, 6 underspecified |
| Qdrant upload report | `app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/upload_report.md` | Markdown | PASS, 1862 points |

**Court source:** Nejvyšší soud ČR (`rozhodnuti.nsoud.cz`), batch label `2025_01_03` (Jan–Mar 2025 scrape).

## Qdrant usability

| Item | Value |
|------|-------|
| Docker service | `qdrant` (`qdrant/qdrant:v1.13.6`) |
| Internal URL (from `api` container) | `http://qdrant:6333` |
| Existing collection | `nsoud_chunks_section_aware_test_2025_01_03` (1862 points, 768-dim cosine) |
| Usable for benchmark | **Yes** — Qdrant is reachable from the `api` container on the Docker network |
| Benchmark behavior | Creates new per-model collections (`nalus_rag_eval__<model>__...`); does not reuse the existing indexed vectors |

## Chunks usability

| Item | Status |
|------|--------|
| Chunk text available | **Yes** — `chunk_text` column in parquet |
| Document linkage | **Yes** — `document_id` (ECLI) per chunk |
| SQL store required by package | **Prepared** — `artifacts/rag_eval/nalus_chunks.sqlite` loaded from parquet via `scripts/prepare_rag_eval_input.py` |
| Chunk metadata for preflight | `source_document_id` mapped from `document_id` |

NALUS does **not** have Postgres. The pilot uses **SQLite** as the `sql_qdrant` chunk source — minimal adapter surface, no new database service.

## Eval dataset

| Item | Value |
|------|-------|
| Existing categorized dataset | `relevance_eval_dataset.json` (14 positive cases with `source_terms` + `source_chunk_ids`) |
| Pilot rag-eval dataset | `artifacts/rag_eval/nalus_eval.json` (8 cases derived from positive_answerable, markers verified against chunk text) |
| Explicit `source_documents` | Included (real ECLI ids + chunk excerpts) for preflight alignment |

## Benchmark config

| File | Purpose |
|------|---------|
| `artifacts/rag_eval/nalus.rag_eval.yaml` | `sql_qdrant` backend, SQLite corpus, Qdrant URL, 3 embedding models |
| `artifacts/rag_eval/nalus_chunks.sqlite` | Chunk store for `rag-embedding-benchmark` |
| `artifacts/rag_eval/nalus_eval.json` | Pilot eval cases |
| `artifacts/rag_eval/out/` | Benchmark outputs (`ranking.json`, `report.md`, run artifacts) |

## Models to compare

1. `multilingual_e5_small` (384-dim)
2. `multilingual_e5_base` (768-dim)
3. `paraphrase_multilingual_mpnet_base_v2` (768-dim, current NALUS baseline)

## Reproduce commands

```bash
# 1. Prepare SQLite corpus + eval JSON (inside api container)
docker compose exec api python scripts/prepare_rag_eval_input.py

# 2. Validate
docker compose exec api rag-eval validate --config /app/artifacts/rag_eval/nalus.rag_eval.yaml

# 3. Run benchmark
docker compose exec api rag-eval run --config /app/artifacts/rag_eval/nalus.rag_eval.yaml
```

## Limitations (pilot-level)

- Corpus size: 150 documents / 1862 chunks — suitable for regression, not statistically definitive.
- Questions: 8 positive retrieval cases — pilot only, not full 14-case suite.
- Single dense retrieval mode — no BGE hybrid (deferred v2).
- CPU inference — first run downloads HuggingFace models into `huggingface_cache` volume.
