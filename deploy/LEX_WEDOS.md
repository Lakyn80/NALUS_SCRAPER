# Lex WEDOS deploy notes (dense FAST only)

## Layout on VPS

```text
/home/lucky/projects/apps/lex/
  docker-compose.lex.prod.yml
  .env.prod
  backend/          # clone Lakyn80/parser-fix_API (optional source)
  frontend/         # clone Lakyn80/NalusFE (optional source)
  data/
    qdrant/         # Qdrant storage (Full A only)
    hf_cache/hub/models--BAAI--bge-m3/
    storage/        # empty stub dirs for BM25 path existence
    batches/
```

## Not started on 4 GB RAM

- FAST BM25 (in-memory multi-GB)
- Cross-encoder / PRECISE
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
