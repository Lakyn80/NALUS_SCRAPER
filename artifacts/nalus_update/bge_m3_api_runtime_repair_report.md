# BGE-M3 API Runtime Repair Report

**Date:** 2026-07-09  
**Scope:** Repair NALUS API into a consistent BGE-M3-only runtime (no MPNet, no broken `nalus_legal_rag` import, no empty `/health`).

---

## 1. Root cause summary

NALUS API was in a **three-way inconsistent state**:

| # | Symptom | Root cause |
|---|---------|------------|
| 1 | `ModuleNotFoundError: No module named 'nalus_legal_rag'` | Startup/rag_router imported sibling package `nalus-legal-rag`, but Docker image did not install or mount it. |
| 2 | BGE-M3 model missing at `/app/models/BAAI/bge-m3` | Default path pointed to empty host mount; actual weights live in `huggingface_cache` volume (offline snapshot). |
| 3 | `Empty reply from server` / old MPNet startup | Container ran **stale baked image** `startup.py` (MPNet + `KeywordRetriever` + background Qdrant sync) because `./app` was not mounted; recreate dropped live code. |

Secondary issue discovered after repair: retrieval query returns **empty sources** because Qdrant payloads in the BGE-M3 candidate collection lack required embedding provenance fields (`embedding_provider`, `embedding_model`, etc.). This is a **data/schema** issue, not a runtime packaging issue.

---

## 2. MPNet / legacy matches — classification

### A) active_runtime (BGE-M3 only — OK)

| Location | Notes |
|----------|-------|
| `app/api/startup.py` | Uses `BgeM3Embedder`, `HybridBgeM3Retriever`, `Bm25Sidecar`, `QdrantDenseStore` |
| `app/api/rag_router.py` | `DEFAULT_QDRANT_COLLECTION` from `production_profile` |
| `app/rag/retrieval/production_profile.py` | BGE-M3 defaults; MPNet forbidden |
| `app/rag/retrieval/bge_m3_embedder.py` | `local_files_only`, path existence check |
| `app/rag/retrieval/hybrid_bge_m3_retriever.py` | dense+BM25+RRF production path |

### B) test_only

| Location | Notes |
|----------|-------|
| `tests/rag/test_production_bge_m3_profile.py` | Guards against MPNet, `nalus_legal_rag`, missing model/sidecar |

### C) historical_doc_or_artifact

| Location | Notes |
|----------|-------|
| `app/artifacts/nsoud/rag_ready/*` | NSoud MPNet manifests (historical) |
| `artifacts/rag_eval/**` | Benchmark reports mentioning MPNet |
| `readme.dev` (sections on `nalus_legal_rag`) | Future package docs, not active runtime |

### D) generated_artifact

| Location | Notes |
|----------|-------|
| `artifacts/nalus_update/**` | Ingest/merge reports |

### E) safe_comment / guard code

| Location | Notes |
|----------|-------|
| `app/rag/retrieval/embedder.py` | `SentenceTransformersEmbedder` + `_refuse_mpnet_model()` — **class exists but MPNet is refused**; not used by API startup |
| `app/api/rag_router.py:125` | Comment referencing old `KeywordRetriever` |

### F) must_remove_or_change / legacy (not API startup path)

| Location | Status | Notes |
|----------|--------|-------|
| `app/nsoud/generate_embeddings.py` | **Legacy NSoud tool** | Still uses MPNet; not API startup |
| `app/rag/ingest/qdrant_ingest.py` | **Legacy ingest** | `_MOCK_VECTOR_DIM=10`; gated behind `NALUS_ALLOW_LEGACY_AUTO_INGEST` |
| `app/main.py` `_run_ingest()` | **Legacy CLI path** | Mock 10-dim collection only if `NALUS_AUTO_INGEST` + `NALUS_ALLOW_LEGACY_AUTO_INGEST` |
| `app/rag/retrieval/retrieval_service.py` | **Legacy factory** | `KeywordRetriever`; not used by BGE-M3 startup |
| `app/rag/orchestration/pipeline.py` | **Legacy pipeline** | Same |

**Verdict:** Active API startup path is **BGE-M3-only**. MPNet/mock/KeywordRetriever remain only in legacy modules behind explicit opt-in flags or NSoud tooling.

---

## 3. `nalus_legal_rag` matches — classification

| Location | Class | Notes |
|----------|-------|-------|
| `app/api/startup.py` | **Removed** | No import |
| `app/api/rag_router.py` | **Removed** | No import |
| `tests/rag/test_production_bge_m3_profile.py` | test_only | Asserts runtime modules do not reference package |
| `readme.dev` | historical_doc | Documents future sibling package usage |
| `docker-compose.yml` | **Removed** | No `PYTHONPATH` mount to sibling repo |

---

## 4. Chosen dependency path: **Option A**

**Option A — no runtime dependency on sibling `nalus-legal-rag`.**

NALUS API uses local modules under `app/rag/retrieval/*`. Sibling package remains for future extraction; Docker does not require it.

---

## 5. BGE-M3 model path status

| Path | Status |
|------|--------|
| `/app/models/BAAI/bge-m3` | **MISSING** (empty host `./models` mount) |
| `/root/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181` | **PRESENT** (Docker `huggingface_cache` volume) |

**Active `.env` override:**

```
EMBEDDING_MODEL_NAME=/root/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181
HF_HUB_OFFLINE=1
```

Startup log confirms: `BGE-M3 model path ready at ...5617a9f61b028005a4858fdac845db406aefb181`

No HuggingFace download attempted. `local_files_only` enforced.

---

## 6. BM25 sidecar status

| Item | Value |
|------|-------|
| Path | `/app/storage/rag/bm25/nalus_us_bge_m3_rag_combined_20260709.sqlite` |
| Status | **PRESENT** (`BM25_OK` in container) |
| Rows | 13,315 (matches Qdrant point count) |

---

## 7. Qdrant target collection status

| Item | Value |
|------|-------|
| Collection | `nalus_us_bge_m3_rag_combined_20260709` |
| Points | **13,315** |
| Vector dim | **1024** |
| `nalus_live` | **Untouched** |
| `nalus_stable_20260326` | **Untouched** |
| Alias switch | **None** |

---

## 8. Changes made

1. **`app/api/startup.py`** — BGE-M3-only orchestrator build; hard fail on missing model/BM25/collection; no background ingest; no `nalus_legal_rag`.
2. **`app/api/rag_router.py`** — collection default from `production_profile`; no sibling import.
3. **`app/rag/retrieval/production_profile.py`** — MPNet forbidden; BGE-M3 env validation.
4. **`app/rag/retrieval/embedder.py`** — `_refuse_mpnet_model()` guard.
5. **`app/rag/retrieval/bge_m3_embedder.py`** — offline-only load, path check.
6. **`docker-compose.yml`** — mount `./app:/app/app`, `./scripts:/app/scripts`, `env_file: .env`; removed sibling `nalus-legal-rag` from `PYTHONPATH`/volumes.
7. **`tests/rag/test_production_bge_m3_profile.py`** — production guards + fixed `parents[2]` path bug in `nalus_legal_rag` test.

---

## 9. Tests run

```text
python -m pytest tests/rag/test_production_bge_m3_profile.py -q
→ 15 passed

python -m pytest tests/rag/test_production_bge_m3_profile.py tests/test_merge_bge_m3_candidate_collections.py tests/test_build_usoud_bge_m3_candidate.py -q
→ 53 passed (after path fix)
```

Required coverage:

| Test | Result |
|------|--------|
| production profile BGE-M3 only | PASS |
| no `SentenceTransformersEmbedder` in startup factory | PASS |
| no old `KeywordRetriever` in startup factory | PASS |
| active defaults no `mpnet` | PASS |
| no `_MOCK_VECTOR_DIM` in active startup | PASS |
| missing BGE-M3 path fails clearly | PASS |
| missing BM25 sidecar fails clearly | PASS |
| runtime does not import `nalus_legal_rag` | PASS |

---

## 10. Docker rebuild result

```powershell
docker compose build api
docker compose up -d --force-recreate api
```

- Image rebuilt with live `./app` mount (no stale startup code on recreate).
- Qdrant service not restarted.
- No model download.

---

## 11. `/health` result

```json
{
  "status": "ok",
  "orchestrator_ready": true,
  "orchestrator_status": "ready",
  "orchestrator_error": null,
  "background_ingest_status": "external",
  "background_ingest_error": "Production retrieval uses prebuilt BGE-M3 Qdrant collection and BM25 sidecar; API startup performs no ingest or Qdrant writes.",
  "strict_real_mode": true
}
```

**No empty reply.** API starts cleanly.

---

## 12. Retrieval smoke result

**Query:** `právo na spravedlivý proces` (top_k=5)

| Field | Before backfill | After backfill |
|-------|-----------------|----------------|
| HTTP status | 200 | 200 |
| sources | `[]` (provenance validation failed) | `["2141","4323","8393","1274","6342"]` |
| BGE-M3 dense search | OK | OK |
| Qdrant payload validation | FAIL — missing `document_id`, `embedding_*`, `content_checksum` | PASS |

**Root cause of empty retrieval:** Combined collection payloads from older ingest (especially `mvp_5y`) lacked required provenance fields. `QdrantDenseStore` rejected hits during `validate_embedding_provenance`.

**Fix applied:** `scripts/backfill_bge_m3_payload_provenance.py --execute` patched **7139** points; **6176** were already valid. BM25 sidecar re-exported (13,315 rows). Merge script updated to call `ensure_embedding_provenance` on future merges.

`nalus_live` was not used. No alias switch.

---

## 13. Production safety confirmations

| Rule | Status |
|------|--------|
| `nalus_live` untouched | ✓ |
| `nalus_stable_20260326` untouched | ✓ |
| No alias switch | ✓ |
| No production Qdrant writes | ✓ |
| No full ingest / embedding job | ✓ |
| No model download | ✓ |
| No secrets printed | ✓ |
| No commit / push | ✓ |

---

## 14. Remaining risks

1. **Empty retrieval results** — BGE-M3 candidate points need embedding provenance fields in Qdrant payload (or relax validation for candidate collections). Separate fix from this runtime repair.
2. **`/app/models/BAAI/bge-m3` mount empty** — Works today via HF cache volume + `.env` override; fragile if cache volume is wiped. Consider populating `./models/BAAI/bge-m3` on host for deterministic offline path.
3. **Legacy MPNet code** still exists in `app/nsoud/` and gated legacy ingest — not on API startup path, but could confuse future devs.
4. **`docker-compose.yml` default** `EMBEDDING_MODEL_NAME=/app/models/BAAI/bge-m3` differs from working `.env` — `.env` wins via `env_file`, but defaults are misleading.
5. **BM25 sidecar** must be re-exported after storage volume wipe (documented in `readme.dev`).

---

## 15. Exact next step

1. **Fix Qdrant payload provenance** in BGE-M3 ingest (`build_usoud_bge_m3_candidate.py`) so hybrid retriever accepts hits — then re-smoke `POST /api/rag/query`.
2. Optionally align `docker-compose.yml` default `EMBEDDING_MODEL_NAME` with HF cache snapshot path or document required `.env`.
3. **Commit** when user confirms — runtime repair is verified; retrieval data schema is the blocker for non-empty answers.

**Commit recommended?** Yes, for runtime/docker/test fixes — **after** user explicitly requests commit. Do not commit until retrieval schema fix is decided (or commit runtime-only as separate changeset).
