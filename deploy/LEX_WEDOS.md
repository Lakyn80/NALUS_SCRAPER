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
    storage/                    # stub dirs (BM25 not loaded on WEDOS)
    batches/
```

Do **not** create a separate manual-only deploy path. After bootstrap, CD
(`.github/workflows/deploy-lex-wedos.yml`) updates this same installation.

## CD secrets (GitHub → WEDOS)

Set on `Lakyn80/parser-fix_API`:

- `LEX_WEDOS_SSH_HOST` / `LEX_WEDOS_SSH_USER` / `LEX_WEDOS_SSH_KEY`
- optional `LEX_WEDOS_SSH_PORT` (default 22)
- optional `LEX_FE_DEPLOY_TOKEN` (PAT with read access to `NalusFE` to rebuild FE)

`.env.prod` stays only on the server. CD may rewrite `LEX_API_IMAGE` / `LEX_FE_IMAGE`
pins; it must not upload secrets into git.

## Not started on 4 GB RAM

- FAST BM25 (in-memory multi-GB)
- Cross-encoder / PRECISE (GPU worker later)
- ColBERT / BALANCED
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
