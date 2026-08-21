# Lex WEDOS deploy notes (dense FAST only)

## Layout on VPS (bootstrap == CD)

GitHub is the source of truth. Manual bootstrap and GitHub Actions CD must use the
**same** install path and compose file:

```text
/home/lucky/projects/apps/lex/
  docker-compose.lex.prod.yml   # synced from backend/ on each deploy
  .env.prod                     # server-only; never commit
  backend/                      # clone Lakyn80/parser-fix_API (deploy key)
  frontend/                     # clone Lakyn80/NalusFE (optional)
  data/
    qdrant/                     # Qdrant storage (Full A only)
    hf_cache/hub/models--BAAI--bge-m3/
    storage/
      rag/bm25/                 # BM25 sqlite may exist; NOT loaded when FAST=dense
      rag/archive/              # judgment_archive_v1.sqlite (document metadata)
    batches/                    # NALUS batch JSON for one-time archive build
```

Do **not** create a separate manual-only deploy path. After bootstrap, CD
(`.github/workflows/deploy-lex-wedos.yml`) updates this same installation.

## FAST retrieval profile

Compose sets:

```text
NALUS_FAST_RETRIEVAL_PROFILE=dense
NALUS_FAST_DENSE_VARIANT=current
```

Supported FAST channel values (env-only switch; no FE/API/source change):

- `dense` — BGE-M3 + Qdrant (WEDOS now)
- `bm25` — BM25 only (stronger host)
- `hybrid` — Dense + BM25 RRF (stronger host)

Within `dense`, implementation variant:

- `current` — live dense path (honors `NALUS_QDRANT_QUANTIZATION_*`)
- `v2` — classic Legal v2 plain `query_points` (pre-INT8 search policy; commit `e9fa438^`)

Rollback Dense only (API recreate required):

```text
NALUS_FAST_DENSE_VARIANT=v2
```

Legacy alias: `NALUS_LEGAL_V2_STAGE1_FAST_DENSE_ONLY=1` still forces dense when the
new selector is unset.

## Jurisprudence archive index (one-time)

Archive API reads `NALUS_JUDGMENT_ARCHIVE_SQLITE_PATH` (default under
`/app/storage/rag/archive/`). Build once on the VPS from batch JSON — do **not**
rebuild on every API restart:

```bash
cd /home/lucky/projects/apps/lex
mkdir -p data/storage/rag/archive
docker compose -f docker-compose.lex.prod.yml --env-file .env.prod run --rm \
  --entrypoint python \
  -v "$(pwd)/data/storage:/app/storage" \
  -v "$(pwd)/data/batches:/app/batches:ro" \
  api scripts/legal_v2/build_judgment_archive_index.py \
    --batches-dir /app/batches \
    --sqlite-path /app/storage/rag/archive/judgment_archive_v1.sqlite
```

Then recreate API (`docker compose ... up -d --no-deps api`) so the read-only
mount sees the new sqlite file.

## CD secrets (GitHub → WEDOS)

Set on `Lakyn80/parser-fix_API`:

- `LEX_WEDOS_SSH_HOST` / `LEX_WEDOS_SSH_USER` / `LEX_WEDOS_SSH_KEY`
- optional `LEX_WEDOS_SSH_PORT` (default 22)
- optional `LEX_FE_DEPLOY_TOKEN` (PAT with read access to `NalusFE` to rebuild FE)

`.env.prod` stays only on the server. CD may rewrite `LEX_API_IMAGE` / `LEX_FE_IMAGE`
pins; it must not upload secrets into git.

## Not started on 4 GB RAM

- FAST BM25 (in-memory multi-GB) — code ready; set `NALUS_FAST_RETRIEVAL_PROFILE=bm25|hybrid` later
- Cross-encoder / PRECISE (GPU worker later)
- ColBERT / BALANCED (PODROBNÉ)
- Full B collection
- Redis / Prometheus / Grafana

## SSH aliases (on VPS ~/.ssh/config)

```text
Host github-lex-api
  HostName ssh.github.com
  Port 443
  User git
  IdentityFile ~/.ssh/lex_api_deploy_ed25519
  IdentitiesOnly yes

Host github-lex-fe
  HostName ssh.github.com
  Port 443
  User git
  IdentityFile ~/.ssh/lex_fe_deploy_ed25519
  IdentitiesOnly yes
```
