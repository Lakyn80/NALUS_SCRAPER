# Ustavni soud / NALUS - BGE-M3 Full-Corpus Rebuild Plan

Status: **PLAN ONLY** - no ingest, no embedding run, no API change, no commit.

Generated: 2026-07-08 16:38:22 +03:00

## 1. Hard safety boundary

Do **not** update `nalus_live`.

Do **not** write to `nalus_stable_20260326`.

Do **not** run the old production ingest path for this task.

Do **not** run full embedding until the smoke and pilot stages below pass.

Do **not** modify API routing, retrieval ranking, clarification gate code, or benchmark winner artifacts while preparing this rebuild.

This rebuild must create a **parallel candidate** collection and artifacts only. Any future production switch must be a separate, explicit decision after read-only validation.

## 2. Current raw corpus status

Verified read-only through Docker from `/app/batches`.

| Item | Current value |
|---|---:|
| Data JSON files in `batches/` excluding `manifest.json` | 44 |
| Manifest entries | 44 |
| JSON files including `manifest.json` | 45 |
| Manifest `doc_count` sum | 104,045 raw records |
| Data size | 1,160.1 MiB |
| Earliest parsed `decision_date` | 1993-09-14 |
| Latest parsed `decision_date` | 2026-06-25 |
| Raw records with empty `full_text` across all batches | 55 |

Latest incremental scrape already merged:

| Item | Current value |
|---|---:|
| New file | `batches/year_2026_20260708_124949.json` |
| New records | 1,486 |
| New records with empty `full_text` | 0 |
| New file date range | 2026-01-06 to 2026-06-25 |
| Duplicate records skipped during 2026 re-walk | 245 |

The 2026 incremental scrape is raw corpus only: text was fetched and stored, but no chunking, embedding, Qdrant ingest, alias switch, or API/retrieval change was performed.

Note: the all-corpus `55` empty `full_text` records are legacy corpus records. The new 2026 file has `0` empty texts.

## 3. Current Qdrant and API status

Verified read-only through Docker with `QdrantClient(url="http://qdrant:6333", check_compatibility=False)`.

| Target | Count / status |
|---|---:|
| `nalus_live` alias | points to `nalus_stable_20260326` |
| `nalus_stable_20260326` | 776,424 points |
| `nalus_live` | 776,424 points |
| `nalus` | 770,776 points |
| Existing client long-form BGE-M3 eval collection | `nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1` |

The API currently uses `QDRANT_COLLECTION_NAME=${QDRANT_COLLECTION_NAME:-nalus}` in `docker-compose.yml`. Production-live selection is via the `nalus_live` alias, which currently points to `nalus_stable_20260326`.

No write operation is required for this planning stage. A candidate rebuild must not depend on changing `QDRANT_COLLECTION_NAME`, and must not repoint `nalus_live`.

## 4. Ingestion and benchmark scripts found

| Path | What it does | Suitability for BGE-M3 full rebuild |
|---|---|---|
| `scripts/scrape_all_nalus.py` | Year-by-year NALUS scraper. Can save batch JSON. Has optional `_ingest_file()` path. | Use only for scrape with `--no-ingest`. Its `_ingest_file()` creates collections with `_MOCK_VECTOR_DIM=10` and uses `QdrantIngestor` without a real BGE-M3/BM25 plan. Not suitable for this rebuild. |
| `scripts/ingest_batch.py` | Standalone batch JSON -> runtime chunks -> Qdrant ingest. Uses default `SentenceTransformersEmbedder()` when ingesting. | **Not suitable** for BGE-M3 candidate. It creates missing collections with `_MOCK_VECTOR_DIM=10`, defaults to mpnet, has no BGE-M3 model arg, no BM25/RRF support, no full-corpus resume plan, and updates `batches/manifest.json`. |
| `app/main.py` `_run_ingest()` | Legacy auto-ingest after ad-hoc scrape. | Not suitable. It creates missing collections with `_MOCK_VECTOR_DIM=10` and uses default `QdrantIngestor` behavior. Keep `NALUS_AUTO_INGEST` off for update work. |
| `app/rag/ingest/qdrant_ingest.py` | Reusable dense point ingestor with deterministic point IDs and injectable embedder. | Useful lower-level component only. It does not build BM25, RRF, full-corpus artifacts, or guarded candidate collection lifecycle by itself. |
| `app/api/startup.py` | Builds live orchestrator, runtime corpus, dense retriever, keyword retriever, and possible background sync. | Do not use as rebuild mechanism. It is live API startup code, not a controlled offline full-corpus candidate builder. |
| `app/rag/retrieval/*` | Current API retrieval: dense Qdrant + in-memory keyword scoring + score-sort fusion. | Must remain unchanged. It is not the benchmark-winner `dense_plus_bm25`/RRF implementation. |
| `scripts/prepare_rag_eval_input.py` | Builds rag-eval SQLite and eval JSON from NSoud parquet artifacts. | NSoud-specific, not a NALUS/US full-corpus builder. Useful as a schema reference only. |
| `artifacts/rag_eval/client_longform_v1/run_benchmark.sh` | Installs `rag-embedding-benchmark[sql-qdrant-bm25]`, validates configs, runs client-longform benchmark candidates, merges ranking, finalizes winner. | Suitable for small benchmark evaluation, not directly for full-corpus production candidate build. |
| `artifacts/rag_eval/client_longform_v1/configs/hybrid_bge_m3.yaml` | Winner config: `sql_qdrant`, `bge_m3`, `dense_plus_bm25`, RRF `k=60`, BM25 `k1=1.5`, `b=0.75`. | Good source of winning retrieval settings. Needs a safe full-corpus build wrapper before use on all NALUS data. |
| `../eternal-world/packages/rag-embedding-benchmark/rag_eval/adapters/sql_qdrant.py` | Generic SQLite/Postgres chunks + SentenceTransformers + Qdrant adapter. | Supports BGE-M3 dense and BM25/RRF retrieval, but `index_source()` calls `recreate_collection`, encodes all chunks in one run, and is not resumable. Unsafe as-is for full rebuild. |
| `../eternal-world/packages/rag-embedding-benchmark/rag_eval/retrieval/bm25.py` | Legal text tokenization and in-memory `bm25s` index. | Works for benchmark-sized corpora. Full corpus may need memory testing or persistent BM25 artifacts. |
| `app/nsoud/*` Qdrant scripts | NSoud chunk/upload/eval tooling. | Different court/data model. Do not use for US/NALUS rebuild except as conceptual reference. |

## 5. BGE-M3 / dense_plus_bm25 support found

The benchmark package registry maps:

| Model code | Provider model | Dimension | Notes |
|---|---|---:|---|
| `bge_m3` | `BAAI/bge-m3` | 1024 | Local SentenceTransformers, normalized vectors, high-resource/manual real eval |

The winner config from `client_longform_v1` is:

| Setting | Value |
|---|---|
| Retrieval mode | `dense_plus_bm25` |
| Fusion | `rrf` |
| `rrf_k` | 60 |
| `bm25_k1` | 1.5 |
| `bm25_b` | 0.75 |
| Top K | 5 |
| Benchmark dataset chunks | 1,862 |
| Winner collection | `nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1` |
| Winner hit rate / MRR | 0.875 / 0.6354166667 |

Important distinction: this is benchmark support, not production API support. Current API retrieval still uses the app's dense + simple keyword scorer + score-sort fusion. It does not use the rag-eval BM25/RRF path.

## 6. Gaps before a safe full-corpus candidate

1. No guarded US/NALUS full-corpus builder exists.
2. No script currently materializes all `batches/*.json` into a canonical deduplicated SQLite chunk table for `sql_qdrant`.
3. Existing `sql_qdrant.index_source()` uses `recreate_collection`; that is acceptable for isolated benchmark collections but too risky for a large candidate unless guarded by collection-name allow/deny checks.
4. Existing `sql_qdrant.index_source()` is not resumable; a full BGE-M3 CPU run over hundreds of thousands of chunks could fail late and require restarting.
5. Existing BM25 support is in-memory. Full corpus BM25 memory/time must be measured during smoke/pilot before a full run.
6. Old NALUS ingest paths use the wrong collection-dimension behavior for this task and do not implement BM25/RRF.
7. The raw corpus contains legacy overlap and 55 legacy empty-text records, so canonical dedup and invalid-row handling must be explicit.
8. No API integration exists for a BGE-M3 + BM25/RRF candidate. That is fine for Stage 1; production integration must be separate.

## 7. Proposed names

Use names that cannot be confused with live/stable production collections.

| Purpose | Proposed name |
|---|---|
| Smoke Qdrant collection | `nalus_us_bge_m3_smoke_20260708` |
| Pilot Qdrant collection | `nalus_us_bge_m3_pilot_20260708` |
| Full candidate Qdrant collection | `nalus_us_bge_m3_full_20260708` |
| Artifact root | `/app/artifacts/nalus_update/usoud_bge_m3_full_20260708/` |
| Smoke SQLite | `/app/artifacts/nalus_update/usoud_bge_m3_full_20260708/smoke/nalus_us_chunks.sqlite` |
| Full SQLite | `/app/artifacts/nalus_update/usoud_bge_m3_full_20260708/full/nalus_us_chunks.sqlite` |
| BM25 artifacts, if persisted later | `/app/artifacts/nalus_update/usoud_bge_m3_full_20260708/bm25/` |

Explicit denylist for any candidate builder:

```text
nalus
nalus_live
nalus_stable_20260326
```

Also deny any collection name that starts with `nalus_stable_` unless an explicit future production-switch task asks for it.

## 8. Estimates

The best current point-count baseline is `nalus_stable_20260326` with 776,424 Qdrant points before the latest raw 2026 scrape.

The new 2026 file has 1,486 decisions. At roughly 10 chunks per decision, expect about 15k additional chunks. Full BGE-M3 candidate size is therefore likely around **790k-820k dense vectors**, depending on dedup and chunking.

Resource estimate for BGE-M3 dense vectors:

| Component | Rough estimate |
|---|---:|
| Dense vectors only, 800k x 1024 x float32 | ~3.1 GiB raw vector data |
| Qdrant storage with HNSW/payload overhead | likely 8-15+ GiB |
| SQLite chunk store | likely 1-2+ GiB |
| BM25 in-memory/full index | must be measured; potentially several GiB |
| CPU-only BGE-M3 embedding time | many hours to days until measured |

Do not start full ingest until smoke and pilot runs measure throughput, memory, disk growth, and restart/resume behavior.

## 9. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Accidentally writing to `nalus_live` or `nalus_stable_20260326` | Production corruption | Hard denylist in builder; no alias updates; print target and require candidate prefix. |
| Old 10-dimensional ingest path | Broken collection or dimension mismatch | Do not use `scripts/ingest_batch.py`, `scrape_all_nalus.py` ingest, or `app/main.py` auto-ingest for this task. |
| `recreate_collection` on wrong name | Data loss | Only allow names matching `nalus_us_bge_m3_(smoke|pilot|full)_YYYYMMDD`. |
| CPU BGE-M3 full run too slow | Multi-day job or partial output | Smoke first; pilot next; implement checkpoint/resume before full. |
| BM25 in-memory index too large | OOM | Measure on smoke/pilot; consider persisted/sharded BM25 artifacts before full. |
| Duplicate legacy records | Duplicate chunks / bad ranking | Build canonical dedup by `ecli`, then `case_reference`, then URL/result ID; keep best full-text record. |
| Empty legacy texts | Invalid chunks | Exclude empty `full_text` rows from SQLite with `validation_status='invalid'` or skip entirely. |
| API retrieval mismatch | Misleading production assumptions | Keep candidate offline/read-only. Do not claim production parity until API retrieval is separately wired to BM25/RRF. |

## 10. Staged execution plan

### Stage 0 - Guardrails and builder

Implement a new guarded script, for example `scripts/build_usoud_bge_m3_candidate.py`, with:

- Docker-only execution.
- Required `--collection` argument.
- Hard denylist for `nalus`, `nalus_live`, `nalus_stable_20260326`.
- Required candidate prefix `nalus_us_bge_m3_`.
- `--dry-run` mode that builds corpus stats but does not create or write Qdrant collections.
- Canonical dedup over all selected batch files.
- SQLite output compatible with `rag_eval.adapters.sql_qdrant`.
- Resume/checkpoint support for embedding/upsert.
- Explicit `--limit-docs` / `--limit-chunks` for smoke and pilot.
- No alias update command.

### Stage 1 - Smoke ingest only

Purpose: prove corpus conversion, BGE-M3 model load, collection creation, point count, BM25 retrieval, and read-only validation on a tiny subset.

Use only the new 2026 file and a small limit. Delete/recreate only the smoke collection.

### Stage 2 - Pilot

Run on `batches/year_2026_20260708_124949.json` or a 5k-20k chunk subset. Measure:

- chunks/sec embedding throughput,
- Qdrant upsert throughput,
- disk growth,
- memory during BM25 index build,
- restart/resume behavior,
- retrieval sanity on 5-10 legal queries.

### Stage 3 - Full corpus materialization

Build canonical full SQLite from all 44 data files. Output corpus summary:

- raw records,
- unique canonical decisions,
- skipped duplicates,
- skipped/invalid empty texts,
- chunk count,
- max/avg chunk length,
- per-year coverage.

No Qdrant write in this stage unless explicitly requested after report review.

### Stage 4 - Full dense candidate build

Only after Stage 3 is reviewed, run BGE-M3 embedding/upsert into `nalus_us_bge_m3_full_20260708`. It must be resumable and must never call alias update.

### Stage 5 - BM25/RRF candidate validation

Build or load BM25 index and run read-only retrieval validation using the same settings as the benchmark winner:

- `dense_plus_bm25`
- RRF `k=60`
- BM25 `k1=1.5`
- BM25 `b=0.75`

### Stage 6 - Decision gate

Compare candidate quality, latency, memory, and operational risk. Only after a separate approval should any API integration or alias switch be considered.

## 11. Exact Stage 1 smoke commands

These commands are the intended Stage 1 commands **after** the guarded builder exists. Do not substitute `scripts/ingest_batch.py`.

Dry-run corpus/build plan:

```powershell
docker compose exec -T api python scripts/build_usoud_bge_m3_candidate.py `
  --input /app/batches/year_2026_20260708_124949.json `
  --limit-docs 25 `
  --artifact-dir /app/artifacts/nalus_update/usoud_bge_m3_full_20260708/smoke `
  --sqlite-out /app/artifacts/nalus_update/usoud_bge_m3_full_20260708/smoke/nalus_us_chunks.sqlite `
  --collection nalus_us_bge_m3_smoke_20260708 `
  --model-code bge_m3 `
  --retrieval-mode dense_plus_bm25 `
  --rrf-k 60 `
  --bm25-k1 1.5 `
  --bm25-b 0.75 `
  --no-alias `
  --dry-run
```

Smoke execution:

```powershell
docker compose exec -T api python scripts/build_usoud_bge_m3_candidate.py `
  --input /app/batches/year_2026_20260708_124949.json `
  --limit-docs 25 `
  --artifact-dir /app/artifacts/nalus_update/usoud_bge_m3_full_20260708/smoke `
  --sqlite-out /app/artifacts/nalus_update/usoud_bge_m3_full_20260708/smoke/nalus_us_chunks.sqlite `
  --collection nalus_us_bge_m3_smoke_20260708 `
  --model-code bge_m3 `
  --retrieval-mode dense_plus_bm25 `
  --rrf-k 60 `
  --bm25-k1 1.5 `
  --bm25-b 0.75 `
  --no-alias `
  --execute
```

Smoke read-only validation:

```powershell
docker compose exec -T api python scripts/validate_usoud_bge_m3_candidate.py `
  --sqlite /app/artifacts/nalus_update/usoud_bge_m3_full_20260708/smoke/nalus_us_chunks.sqlite `
  --collection nalus_us_bge_m3_smoke_20260708 `
  --model-code bge_m3 `
  --retrieval-mode dense_plus_bm25 `
  --top-k 5 `
  --read-only
```

The validation script is also not present yet. If it is not implemented, use the smoke builder's own read-only validation mode instead. The key requirement is that validation must query only `nalus_us_bge_m3_smoke_20260708`.

## 12. Final recommendation

Full ingest is **not safe to start now**.

The old `scripts/ingest_batch.py` is not suitable for the requested BGE-M3 full-corpus candidate. It is an old dense-only ingest helper, defaults to mpnet, creates new collections with a 10-dimensional mock-vector size, has no BGE-M3 model selection, has no BM25/RRF support, mutates the manifest, and is not a guarded full-corpus candidate builder.

The right next step is to implement a small guarded US/NALUS candidate builder and run Stage 1 smoke only. If Stage 1 passes, proceed to pilot. Only after pilot proves throughput, memory, disk, and resume behavior should a full `nalus_us_bge_m3_full_20260708` build be considered.

Keep `nalus_live` and `nalus_stable_20260326` untouched throughout.
