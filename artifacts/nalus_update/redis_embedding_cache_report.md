# Redis Embedding Cache Report

**Date:** 2026-07-09  
**Scope:** Optional Redis-backed BGE-M3 embedding cache for query and chunk vectors in NALUS-SCRAPER local runtime.

---

## 1. What was added

| File | Purpose |
|------|---------|
| `app/rag/retrieval/embedding_cache.py` | Cache protocol, Redis/in-memory/null backends, key helpers, query/chunk cache orchestration |
| `app/rag/retrieval/cached_bge_m3_embedder.py` | `CachedBgeM3Embedder` wrapper around local `BgeM3Embedder` |
| `tests/rag/test_embedding_cache.py` | Unit tests (fake Redis, in-memory, no real Redis) |

**Integrations:**

| Path | Wiring |
|------|--------|
| Query retrieval | `app/api/startup.py` → `build_cached_bge_m3_embedder()` → `QdrantDenseStore` → `embed_query()` |
| Chunk ingest/build | `scripts/build_usoud_bge_m3_candidate.py` → `_encode_chunks()` → `embed_texts_with_cache()` |
| Health | `app/api/main.py` `/health` exposes `embedding_cache_enabled`, `embedding_cache_backend`, `embedding_cache_error` |

**Not changed:** Qdrant (primary dense store), BM25 sidecar, RRF fusion, hybrid retriever logic, `nalus-legal-rag` (not imported).

---

## 2. Cache key format

Prefix from `EMBEDDING_CACHE_KEY_PREFIX` (default `nalus:embedding`).

**Query:**

```text
nalus:embedding:{profile_name}:query:{sha256(normalized_query)}
```

**Chunk:**

```text
nalus:embedding:{profile_name}:chunk:{content_checksum}
```

Normalization: strip + collapse whitespace (`normalize_text_for_embedding_cache`).

Chunk checksum: uses payload `content_checksum` when present, otherwise SHA-256 of normalized text.

---

## 3. Payload format (JSON in Redis)

```json
{
  "profile_name": "nalus_bge_m3_dense_bm25_rrf_v1",
  "embedding_model": "BAAI/bge-m3",
  "embedding_dim": 1024,
  "source": "query|chunk",
  "checksum": "...",
  "vector": [0.0, ...],
  "created_at": "2026-07-09T12:00:00+00:00",
  "schema_version": 1
}
```

Rules enforced on get/set:

- `embedding_dim` must be **1024**
- profile/model must match active config
- corrupted JSON raises `RetrievalConfigurationError`
- no MPNet/hash/mock fallback

---

## 4. Configuration

| Variable | Default | Notes |
|----------|---------|-------|
| `REDIS_URL` | falls back to `RAG_QUERY_CACHE_URL` | Docker default: `redis://redis:6379/1` (DB 1, separate from query cache DB 0) |
| `EMBEDDING_CACHE_ENABLED` | `false` | Must be explicitly enabled |
| `EMBEDDING_CACHE_KEY_PREFIX` | `nalus:embedding` | |
| `QUERY_EMBEDDING_CACHE_TTL_SECONDS` | `604800` | `0` = no expiry |
| `CHUNK_EMBEDDING_CACHE_TTL_SECONDS` | `0` | `0` = no expiry (persistent chunk vectors) |
| `EMBEDDING_CACHE_FAIL_OPEN_ON_REDIS_ERROR` | `false` | If `true`, disable cache when Redis unavailable |

**Enable in `.env`:**

```env
EMBEDDING_CACHE_ENABLED=1
REDIS_URL=redis://redis:6379/1
```

**Disable:**

```env
EMBEDDING_CACHE_ENABLED=0
```

---

## 5. Tests

```text
python -m pytest tests/rag/test_embedding_cache.py tests/rag/test_production_bge_m3_profile.py -q
→ 29 passed
```

Coverage includes: deterministic keys, normalization, wrong-dim rejection, corrupted payload, fake Redis client, cache hit/miss for query and chunk, disabled path, Redis unavailable when enabled.

**Real Redis in tests:** No — `MagicMock` / `InMemoryEmbeddingCache` only.

---

## 6. Production safety

| Check | Result |
|-------|--------|
| Qdrant touched | No (tests only; no ingest run) |
| `nalus_live` touched | No |
| `nalus_stable_20260326` touched | No |
| Alias switch | No |
| Model download | No |
| `nalus-legal-rag` imported | No |
| MPNet/hash/mock in active runtime | No |

---

## 7. Docker smoke

With `EMBEDDING_CACHE_ENABLED=0` (default):

- API starts normally
- `/health` includes `embedding_cache_enabled: false`

With cache enabled and Redis up:

- `/health` shows `embedding_cache_enabled: true`, `embedding_cache_backend: redis`
- Repeated identical queries reuse cached query vectors (no extra `model.encode` for cache hits)

---

## 8. Recommended next step

1. Plan ÚS + NSoud (150) Q&A evaluation benchmark over combined RAG candidate.
2. Enable `EMBEDDING_CACHE_ENABLED=1` during long ingest resumes to avoid re-embedding duplicate chunks.
3. Optional: expose cache hit/miss metrics in `/health` or structured logs.
