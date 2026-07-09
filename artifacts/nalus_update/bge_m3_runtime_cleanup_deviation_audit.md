# BGE-M3 runtime cleanup deviation audit

Date: 2026-07-09

## Original request

Puvodni smer byl vytvorit novy standalone sibling package `nalus-legal-rag` vedle `nalus-scraper`, bez zmen produkcniho runtime v `NALUS-SCRAPER`.

## What actually changed

Misto izolovaneho sibling package vznikly rozsahle zmeny primo v working tree `nalus-scraper`:

- produkcni/operacni konfigurace: `docker-compose.yml`, `.env.example`, lokalni `.env`
- API startup a router runtime: `app/api/startup.py`, `app/api/rag_router.py`, `app/main.py`
- legacy ingest guard: `scripts/ingest_batch.py`
- nove produkcni-looking retrieval moduly pod `app/rag/retrieval/*`
- zmeny embedder/retrieval service kompatibility
- dokumentace v `readme.dev`
- testy a BGE-M3 builder/provenance zmeny
- lokalni/untracked BGE-M3 artefakty a smoke/full vystupy

## Why this is a scope deviation

Tyto zmeny meni runtime a deployment chovani existujiciho `nalus-scraper` projektu pred tim, nez byl schvalen izolovany `nalus-legal-rag` package. Nejrizikovejsi odchylky jsou:

- default Qdrant collection v repo konfiguraci byla presmerovana z legacy `nalus`/`nalus_live` smerem na `nalus_bge_m3_chunks_v1`
- startup byl zmenen z legacy dense/keyword runtime na read-only BGE-M3 dense+BM25+RRF profile
- API po recreate vyzaduje final BGE-M3 kolekci a BM25 sidecar, ktere aktualne nejsou k dispozici
- `.env` je lokalne upraveny tak, ze by novy API container miril na nepostaveny target
- zmeny jsou v main working tree, ne v izolovane vetvi/package

## Git audit

Commands inspected:

- `git status --short --branch`
- `git diff --stat`
- `git diff --name-only`
- `git branch --show-current`
- `git log -1 --oneline`
- `git diff --cached --name-only`
- `git status --short --ignored -- .env`
- `git ls-files --others --exclude-standard`

Findings:

- current branch: `main`
- branch state: `main...origin/main [ahead 1]`
- last commit: `4290559 Add guarded ÚS BGE-M3 smoke builder`
- commit exists locally: yes, branch is ahead of `origin/main` by 1 commit
- current runtime cleanup changes: uncommitted
- staged files: none
- tracked modified files in status: 18
- untracked files from `git ls-files --others --exclude-standard`: 20
- untracked status entries: 15, because directories are collapsed by porcelain status
- `.env`: ignored local file exists and is not tracked by git

`git diff --stat` for tracked content changes:

```text
.env.example                               |   18 +-
.gitignore                                 |    5 +
app/api/rag_router.py                      |   61 +-
app/api/startup.py                         |  271 ++----
app/main.py                                |   14 +-
app/rag/execution/execution_service.py     |   11 +-
app/rag/retrieval/embedder.py              |    7 +-
app/rag/retrieval/models.py                |    2 +-
app/rag/retrieval/retrieval_service.py     |    7 +-
docker-compose.yml                         |   20 +-
readme.dev                                 |  107 +++
scripts/build_usoud_bge_m3_candidate.py    | 1424 ++++++++++++++++++++++++++--
scripts/ingest_batch.py                    |   17 +-
tests/test_build_usoud_bge_m3_candidate.py |  454 ++++++++-
14 files changed, 2104 insertions(+), 314 deletions(-)
```

Note: 4 clarification files are marked modified by `git status`, but `git diff` shows no content diff and porcelain v2 reports identical index/worktree object hashes. Treat them as pre-existing/unrelated working-tree noise unless a later audit finds real content changes.

## File classification

| Path | Status | Category | Recommendation | Rationale |
|---|---:|---|---|---|
| `.env` | ignored local | local_only | revert local target before any restart | Local runtime config now points at BGE-M3 final target. Not tracked, but operationally risky. |
| `.env.example` | modified | revert_candidate | revert from mainline | Changes default public config to unbuilt BGE-M3 stack. |
| `.gitignore` | modified | keep_candidate | keep only if BGE artifacts stay externalized | Useful ignores for models/BM25 sidecars, but outside original package scope. |
| `docker-compose.yml` | modified | revert_candidate | revert from mainline | Changes production API env/default collection/model/BM25 wiring. |
| `app/api/startup.py` | modified | revert_candidate | revert from mainline or isolate branch | Major runtime rewrite; requires missing BGE collection and BM25 sidecar. |
| `app/api/rag_router.py` | modified | revert_candidate | revert from mainline or isolate branch | Removes legacy fallback behavior and changes default collection. |
| `app/main.py` | modified | revert_candidate | revert from mainline | Changes auto-ingest behavior/defaults in existing app. |
| `app/rag/execution/execution_service.py` | modified | experimental_only | isolate with new retriever branch if needed | Compatibility change for new retriever interface; not needed for standalone package yet. |
| `app/rag/retrieval/embedder.py` | modified | revert_candidate | revert from mainline | Changes default embedder model to BGE-M3 in existing module. |
| `app/rag/retrieval/models.py` | modified | keep_candidate | keep only after review | Low-risk metadata/source comment change, but still outside original request. |
| `app/rag/retrieval/retrieval_service.py` | modified | keep_candidate | keep only after review | Mostly documentation/legacy labeling; not runtime-critical. |
| `app/rag/clarification/__init__.py` | modified, no content diff | keep_candidate | leave untouched; audit separately if needed | Pre-existing/unrelated status noise. |
| `app/rag/clarification/rules.py` | modified, no content diff | keep_candidate | leave untouched; audit separately if needed | Pre-existing/unrelated status noise. |
| `app/rag/clarification/service.py` | modified, no content diff | keep_candidate | leave untouched; audit separately if needed | Pre-existing/unrelated status noise. |
| `readme.dev` | modified | experimental_only | isolate or trim to audit notes | Documents BGE-M3 production-like migration in scraper repo. |
| `scripts/build_usoud_bge_m3_candidate.py` | modified | experimental_only | isolate in experiment branch/package | Useful builder/provenance work, but not original standalone package. |
| `scripts/ingest_batch.py` | modified | revert_candidate | revert from mainline | Adds guard that changes existing operational ingest path. |
| `tests/test_build_usoud_bge_m3_candidate.py` | modified | experimental_only | isolate with builder changes | Test expansion belongs with experimental builder work. |
| `tests/rag/clarification/test_legal_query_clarification.py` | modified, no content diff | keep_candidate | leave untouched; audit separately if needed | Pre-existing/unrelated status noise. |
| `app/rag/retrieval/bge_m3_embedder.py` | untracked | experimental_only | isolate branch or move to sibling package | New BGE-M3 runtime module inside scraper. |
| `app/rag/retrieval/bm25_sidecar.py` | untracked | experimental_only | isolate branch or move to sibling package | New BM25 sidecar runtime module inside scraper. |
| `app/rag/retrieval/errors.py` | untracked | experimental_only | isolate branch or move to sibling package | New production-style error module. |
| `app/rag/retrieval/hybrid_bge_m3_retriever.py` | untracked | experimental_only | isolate branch or move to sibling package | New hybrid retrieval runtime. |
| `app/rag/retrieval/production_profile.py` | untracked | experimental_only | isolate branch or move to sibling package | New production profile defaults. |
| `app/rag/retrieval/provenance.py` | untracked | experimental_only | isolate branch or move to sibling package | Useful provenance validation, but not requested in scraper. |
| `app/rag/retrieval/qdrant_dense_store.py` | untracked | experimental_only | isolate branch or move to sibling package | New Qdrant dense store. |
| `app/rag/retrieval/rrf.py` | untracked | experimental_only | isolate branch or move to sibling package | New RRF fusion helper. |
| `tests/rag/test_production_bge_m3_profile.py` | untracked | experimental_only | isolate with new retrieval modules | Tests production-like scraper runtime path. |
| `scripts/run_usoud_bge_m3_mvp_5y.ps1` | untracked | local_only | do not commit to mainline unless intentionally promoted | Local run script for experiment. |
| `artifacts/nalus_update/usoud_bge_m3_full_20260708/dry_run_summary.json` | untracked | local_only | keep as local artifact or move outside repo | Generated audit/build output. |
| `artifacts/nalus_update/usoud_bge_m3_full_20260708/execute_checkpoint.json` | untracked | local_only | keep as local artifact or move outside repo | Generated build checkpoint. |
| `artifacts/nalus_update/usoud_bge_m3_full_20260708/production_safety_snapshot_before.json` | untracked | local_only | keep as local artifact or move outside repo | Generated safety snapshot. |
| `artifacts/nalus_update/usoud_bge_m3_mvp_5y_20260708/dry_run_summary.json` | untracked | local_only | keep as local artifact or move outside repo | Generated audit/build output. |
| `artifacts/nalus_update/usoud_bge_m3_mvp_5y_20260708/execute_checkpoint.json` | untracked | local_only | keep as local artifact or move outside repo | Generated build checkpoint. |
| `artifacts/nalus_update/usoud_bge_m3_mvp_5y_20260708/production_safety_snapshot_before.json` | untracked | local_only | keep as local artifact or move outside repo | Generated safety snapshot. |
| `artifacts/nalus_update/usoud_bge_m3_pilot_20260708/dry_run_summary.json` | untracked | local_only | keep as local artifact or move outside repo | Generated pilot output. |
| `artifacts/nalus_update/usoud_bge_m3_pilot_20260708/execute_summary.json` | untracked | local_only | keep as local artifact or move outside repo | Generated pilot output. |
| `artifacts/nalus_update/usoud_bge_m3_stage2_pilot_report.md` | untracked | local_only | keep as local artifact or move outside repo | Generated report. |
| `artifacts/nalus_update/usoud_bge_m3_stage3_full_report.md` | untracked | local_only | keep as local artifact or move outside repo | Generated report. |

## Runtime and Qdrant state

Read-only checks performed:

- `docker ps`
- `docker compose ps`
- API container env inspection via `docker inspect`
- API `/health`
- Qdrant HTTP GETs from inside `nalus-scraper-api-1` to `http://qdrant:6333`
- local and container filesystem existence checks for BM25 sidecar/model path

Qdrant host note:

- `http://localhost:6333` refused connections because the `nalus-scraper-qdrant-1` service is not published on the host port.
- Qdrant is running inside compose as `nalus-scraper-qdrant-1`; read-only checks were made from `nalus-scraper-api-1` over the compose network.

Current Qdrant collections include:

- `nalus_stable_20260326`
- `nalus`
- `nalus_us_bge_m3_smoke_20260708`
- `nalus_us_bge_m3_pilot_20260708`
- `nalus_us_bge_m3_full_20260708`
- `nalus_us_bge_m3_mvp_5y_20260708`
- several `nalus_rag_eval__*` and `nalus_client_lf__*` evaluation collections

Qdrant alias state:

```json
{"aliases":[{"alias_name":"nalus_live","collection_name":"nalus_stable_20260326"}]}
```

Alias conclusion:

- `nalus_live` still points to `nalus_stable_20260326`
- production alias was not switched to `nalus_bge_m3_chunks_v1`

Target collection status:

- `nalus_bge_m3_chunks_v1`: does not exist, Qdrant returned HTTP 404
- `nalus_stable_20260326`: exists, green, vector size 768, points_count 784812
- `nalus_live`: resolves and reports same collection stats as `nalus_stable_20260326`

BM25 sidecar status:

- local path `storage/rag/bm25/nalus_bge_m3_dense_bm25_rrf_v1.sqlite`: missing
- container path `/app/storage/rag/bm25/nalus_bge_m3_dense_bm25_rrf_v1.sqlite`: missing

BGE-M3 model status:

- container path `/app/models/BAAI/bge-m3`: missing

Current API target:

- running container env: `QDRANT_COLLECTION_NAME=nalus_live`
- running container env: `QDRANT_URL=http://qdrant:6333`
- running API health: `orchestrator_ready=true`, `orchestrator_status=ready`
- changed local `.env`: `QDRANT_COLLECTION_NAME=nalus_bge_m3_chunks_v1`
- changed `docker-compose.yml` default: `nalus_bge_m3_chunks_v1`

API start safety:

- current already-running API is up because it was started with `QDRANT_COLLECTION_NAME=nalus_live`
- if the API container is recreated with the modified `.env`/compose and modified startup code, it is expected to fail safe startup because:
  - `nalus_bge_m3_chunks_v1` does not exist
  - BM25 sidecar does not exist
  - BGE-M3 model path also does not exist, though model loading is lazy and not necessarily the first startup blocker

Production data touch assessment:

- alias switch: no, alias still points to `nalus_stable_20260326`
- final BGE production collection write: no, `nalus_bge_m3_chunks_v1` does not exist
- production write endpoint evidence: Docker logs show the running API performed Qdrant `POST .../collections/nalus_live/points` during background sync at `2026-07-09T06:23:42`, then logged `inserted=0 updated=0 skipped=784812`
- conservative conclusion: production alias target was contacted through write endpoints by the running legacy startup sync, but app counters report no inserted/updated points; no alias change and no final BGE production collection change were observed

## Recommendation

Recommended decision: split into safe parts, then revert/isolate runtime changes before continuing.

Concrete recommendation:

- `revert`: production runtime/config changes in `.env.example`, `docker-compose.yml`, `app/api/startup.py`, `app/api/rag_router.py`, `app/main.py`, `scripts/ingest_batch.py`, and `app/rag/retrieval/embedder.py`
- `isolate`: new BGE-M3 retrieval modules, hybrid/BM25/RRF/provenance code, builder expansions, and production-profile tests into an experiment branch or the future `nalus-legal-rag` sibling package
- `keep_candidate`: low-risk docs/comments/metadata only after explicit review
- `local_only`: `.env` must be reset locally to a known safe target before any container recreate; generated artifacts should stay untracked or be moved outside the mainline path

## Exact next safe step

Do not continue into `nalus-legal-rag` yet.

Next safe step:

1. Create a safety patch/archive of the current working tree if preserving the experiment is desired.
2. Reset local `.env` target back to the known safe running target, e.g. `QDRANT_COLLECTION_NAME=nalus_live`, before any restart/recreate.
3. Move the BGE-M3 runtime work to an isolated branch or patch file.
4. Revert mainline runtime/config files in `nalus-scraper` to remove premature BGE-M3 startup wiring.
5. Only after the scraper runtime is safe, start a separate `nalus-legal-rag` sibling package.

No Qdrant collections were created by this audit. No ingest, alias switch, model download, smoke, rebuild, commit, push, or AI-LEGAL changes were performed by this audit.
