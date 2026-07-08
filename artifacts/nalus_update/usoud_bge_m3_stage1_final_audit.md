# Ustavni soud / NALUS - BGE-M3 Stage 1 Final Audit

Status: **FINAL AUDIT BEFORE STAGE 2** - no Stage 2 run, no full ingest, no commit.

Generated: 2026-07-08

## 1. Current Qdrant state

Read-only check from Docker against `http://qdrant:6333`.

### Aliases

| Alias | Target collection |
|---|---|
| `nalus_live` | `nalus_stable_20260326` |

### Relevant collection counts

| Collection | Point count |
|---|---:|
| `nalus_stable_20260326` | 784,812 |
| `nalus_live` | 784,812 |
| `nalus` | 770,776 |
| `nalus_us_bge_m3_smoke_20260708` | 445 |

### Other collections currently present

| Collection | Point count |
|---|---:|
| `nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1` | 1,862 |
| `nalus_client_lf__multilingual_e5_base__rag_eval__nalus_client_longform_v1__63119240e1` | 1,862 |
| `nalus_client_lf__multilingual_e5_large__rag_eval__nalus_client_longform_v1__63119240e1` | 1,862 |
| `nalus_client_lf__multilingual_e5_small__rag_eval__nalus_client_longform_v1__63119240e1` | 1,862 |
| `nalus_client_lf__paraphrase_multilingual_mpnet_base_v2__rag_eval__nalus_client_longform_v1__63119240e1` | 1,862 |
| `nalus_rag_eval__bge_m3__rag_eval__nalus_nsoud_pilot_v1__63119240e1` | 1,862 |
| `nalus_rag_eval__multilingual_e5_base__rag_eval__nalus_nsoud_pilot_v1__63119240e1` | 1,862 |
| `nalus_rag_eval__multilingual_e5_large__rag_eval__nalus_nsoud_pilot_v1__63119240e1` | 1,862 |
| `nalus_rag_eval__multilingual_e5_small__rag_eval__nalus_nsoud_pilot_v1__63119240e1` | 1,862 |
| `nalus_rag_eval__paraphrase_multilingual_mpnet_base_v2__rag_eval__nalus_nsoud_pilot_v1__63119240e1` | 1,862 |
| `nsoud_chunks_section_aware_test_2025_01_03` | 1,862 |
| `nsoud_chunks_test_2025_01_03` | 1,785 |

Qdrant metadata exposed through the inspected client path did not provide collection modified timestamps. Recent writes are therefore inferred from API logs and point-count deltas, not collection metadata timestamps.

## 2. Did Stage 1 smoke touch production?

Conclusion: **No evidence that Stage 1 smoke builder modified production.**

Evidence:

- `artifacts/nalus_update/usoud_bge_m3_smoke_20260708/execute_summary.json` records:
  - `collection_name`: `nalus_us_bge_m3_smoke_20260708`
  - smoke collection before/after: `null -> 445`
  - `nalus_live` before/after: `784812 -> 784812`
  - `nalus_stable_20260326` before/after: `784812 -> 784812`
  - `aliases_changed`: `false`
- The execute command explicitly targeted:
  - `--collection-name nalus_us_bge_m3_smoke_20260708`
  - `--execute`
  - `--recreate-smoke-collection`
  - `--no-alias-update`
- The script writes to Qdrant only through:
  - `client.create_collection(collection_name=args.collection_name, ...)`
  - `client.recreate_collection(collection_name=args.collection_name, ...)`
  - `client.upsert(collection_name=args.collection_name, ...)`
- The script denylist refuses:
  - `nalus`
  - `nalus_live`
  - `nalus_stable_20260326`
  - any `nalus_stable_*` collection
- The script has no call to `update_collection_aliases`, `CreateAlias`, `DeleteAlias`, or equivalent alias mutation API.
- The script does not call `scripts/ingest_batch.py`.
- The script does not import or modify production API modules, retrieval scoring/fusion modules, or clarification gate modules.

## 3. Point-count discrepancy explanation

Previous audit count:

- `nalus_live` / `nalus_stable_20260326`: 776,424

Stage 1 smoke report count before and after smoke execute:

- `nalus_live` / `nalus_stable_20260326`: 784,812

Delta:

```text
784,812 - 776,424 = 8,388
```

This discrepancy is determinable from Docker/Qdrant evidence.

### Evidence found

The running API container environment currently has:

```text
QDRANT_COLLECTION_NAME=nalus_live
NALUS_BATCHES_DIR=/app/batches
QDRANT_URL=http://qdrant:6333
RAG_STRICT_REAL_MODE=1
```

API logs show a production background append sync writing to `nalus_live` before the Stage 1 smoke execute:

```text
2026-07-08T13:54:56 | INFO | app.api.startup | [startup] background Qdrant sync finished inserted=8388 updated=107 skipped=776317
2026-07-08T13:54:56 | INFO | app.api.main | [main] background append ingest completed
```

The logs immediately before that contain repeated Qdrant write requests to:

```text
POST/PUT http://qdrant:6333/collections/nalus_live/points
```

Because `nalus_live` is an alias to `nalus_stable_20260326`, these writes increased the target collection point count.

The inserted count in the API log exactly matches the discrepancy:

```text
inserted=8388 == 784812 - 776424
```

### What caused it

The point-count increase was caused by the existing API startup/background sync path, not by the new BGE-M3 smoke builder.

The trigger was consistent with the newly merged raw 2026 batch being visible under `/app/batches` while the running API was configured to sync into `nalus_live`. The full new 2026 batch contains 1,486 records; when parsed through the current production runtime corpus chunker it yields 12,245 chunks. The background sync log shows only 8,388 inserted points because the sync ran over the whole runtime corpus and skipped 776,317 already-present points while updating 107 changed points.

### Ruled out / not supported by evidence

| Possible cause | Finding |
|---|---|
| Stage 1 smoke builder wrote to production | Not supported. Execute summary shows production before/after unchanged, and code writes only to explicit smoke collection. |
| Alias was changed by Stage 1 | Not supported. Alias before/current remains `nalus_live -> nalus_stable_20260326`; execute summary says `aliases_changed=false`; builder has no alias mutation call. |
| Different Qdrant instance was queried | Not supported. Checks used the same Docker `api -> qdrant:6333` path; stable aliases and existing collection set are consistent. |
| Qdrant optimizer/indexing changed count | Not supported. Logs show real upserts to `nalus_live` and an exact `inserted=8388` count. |
| Previous count was approximate | Not supported by evidence. The better explanation is the logged background sync between audits. |

## 4. Stage 1 smoke result

| Item | Result |
|---|---|
| Script | `scripts/build_usoud_bge_m3_candidate.py` |
| Dry-run | PASS |
| Execute | PASS |
| Source batch | `batches/year_2026_20260708_124949.json` |
| Selected records | 20 |
| Generated chunks | 445 |
| Embedding model | `BAAI/bge-m3` |
| Vector dimension | `1024 PASS` |
| Smoke collection | `nalus_us_bge_m3_smoke_20260708` |
| Smoke point count | 445 |
| BM25 | available |
| dense_plus_bm25/RRF | available |
| Smoke queries | dense, BM25, hybrid/RRF all returned results from smoke collection |

## 5. File classification

| Path | Category | Reason | Recommended action |
|---|---|---|---|
| `scripts/build_usoud_bge_m3_candidate.py` | commit_separately | Source code for guarded Stage 1 BGE-M3 smoke builder. | Commit in Stage 1 builder commit. |
| `tests/test_build_usoud_bge_m3_candidate.py` | commit_separately | Focused safety/unit tests for collection guards, limit guard, vector-dim guard, dry-run no-write behavior. | Commit with builder. |
| `artifacts/nalus_update/usoud_bge_m3_full_rebuild_plan.md` | commit_separately | Stable planning/audit document that explains why old ingest is unsafe and defines staged rebuild plan. | Commit with Stage 1 documentation. |
| `artifacts/nalus_update/usoud_bge_m3_stage1_smoke_report.md` | commit_separately | Stable report from executed Stage 1 smoke. | Commit with Stage 1 evidence. |
| `artifacts/nalus_update/usoud_bge_m3_smoke_20260708/dry_run_summary.json` | commit_separately | Small machine-readable evidence of dry-run inputs/results. | Commit; size is small and useful for audit. |
| `artifacts/nalus_update/usoud_bge_m3_smoke_20260708/execute_summary.json` | commit_separately | Small machine-readable evidence of execute inputs/results and production before/after counts. | Commit; size is small and useful for audit. |
| `artifacts/nalus_update/usoud_bge_m3_stage1_final_audit.md` | commit_separately | This final audit explains the production count discrepancy and Stage 2 readiness. | Commit with Stage 1 evidence. |

No large generated benchmark outputs were created by this task.

## 6. Validation run

Docker-only focused test command:

```text
docker compose run --rm -T --no-deps api sh -c "python -m pip install --no-cache-dir -q -r requirements-ci.txt && python -m pytest tests/test_build_usoud_bge_m3_candidate.py -q"
```

Result:

```text
9 passed in 0.04s
```

Note: `pytest` was installed only into the temporary `docker compose run --rm` container. No production image dependency file was changed.

## 7. Safety confirmations

| Item | Status |
|---|---|
| Stage 2 started | No |
| Full ingest started | No |
| `nalus_live` touched by Stage 1 builder | No |
| `nalus_stable_20260326` touched by Stage 1 builder | No |
| Aliases updated by Stage 1 builder | No |
| API code changed | No |
| Retrieval logic changed | No |
| Clarification gate changed | No |
| `scripts/ingest_batch.py` called | No |

Important operational note: production was modified by the pre-existing API background sync path after the raw 2026 batch became visible under `/app/batches`. Future raw batch merges should account for this behavior because `QDRANT_COLLECTION_NAME=nalus_live` means API startup/background sync can write through the live alias.

## 8. Stage 2 recommendation

Stage 2 pilot is **safe to start next only after review** and only if it keeps the same safety boundary:

- write to a separate pilot collection only, for example `nalus_us_bge_m3_pilot_20260708`;
- do not update `nalus_live`;
- do not update `nalus_stable_20260326`;
- do not update aliases;
- do not restart or reconfigure production API as part of the pilot;
- keep limit bounded, e.g. 500-1000 records, not full corpus;
- record production counts before/after as the Stage 1 builder does.

Recommended commit contents:

- `scripts/build_usoud_bge_m3_candidate.py`
- `tests/test_build_usoud_bge_m3_candidate.py`
- `artifacts/nalus_update/usoud_bge_m3_full_rebuild_plan.md`
- `artifacts/nalus_update/usoud_bge_m3_stage1_smoke_report.md`
- `artifacts/nalus_update/usoud_bge_m3_stage1_final_audit.md`
- `artifacts/nalus_update/usoud_bge_m3_smoke_20260708/dry_run_summary.json`
- `artifacts/nalus_update/usoud_bge_m3_smoke_20260708/execute_summary.json`
