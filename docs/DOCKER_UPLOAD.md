# Docker upload (GHCR)

This repo has a local `Dockerfile` and a dev `docker-compose.yml`. Publish and pull images from GitHub Container Registry.

Canonical API image:

```text
ghcr.io/lakyn80/nalus-scraper-api:<tag>
```

Do **not** build images on the WEDOS VPS. Build in GitHub Actions (or locally) and `docker compose pull` on the server.

## Local push

Log in:

```powershell
echo $env:GITHUB_TOKEN | docker login ghcr.io -u lakyn80 --password-stdin
```

Use a PAT with `write:packages` (and `read:packages` for pull). Do not commit the token.

Publish API and exporter:

```powershell
.\scripts\docker_publish.ps1 -GhcrOwner lakyn80 -Tag v1.0.0 -AlsoTagLatest
```

API image only:

```powershell
.\scripts\docker_publish.ps1 -GhcrOwner lakyn80 -Tag v1.0.0 -SkipExporter
```

If you already have local compose images and do not want a rebuild:

```powershell
.\scripts\docker_publish.ps1 -GhcrOwner lakyn80 -Tag v1.0.0 -UseExistingImages -AlsoTagLatest
```

Default local source images for that fallback:

- `nalus-scraper-api:latest`
- `nalus-scraper-nalus-eval-metrics-exporter:latest`

Environment:

```powershell
$env:GHCR_OWNER="lakyn80"
$env:IMAGE_TAG="v1.0.0"
$env:DOCKER_REGISTRY="ghcr.io"
.\scripts\docker_publish.ps1 -AlsoTagLatest
```

## GitHub Actions publish

Workflow: [../.github/workflows/docker-publish.yml](../.github/workflows/docker-publish.yml)

Uses `GITHUB_TOKEN` (`packages: write`). Optional repo variable `GHCR_OWNER` (defaults to `github.repository_owner`, lowercased).

Triggers:

- manual `workflow_dispatch` (set an explicit `image_tag`; optionally also push `latest`)
- git tag `docker-v1.0.0` → image tag `1.0.0` plus `latest`

Prefer an explicit tag for WEDOS (`IMAGE_TAG=...`), not only `latest`.

## Deploy from GHCR (RAG stack)

Registry compose: [../docker-compose.registry.yml](../docker-compose.registry.yml)

```powershell
$env:GHCR_OWNER="lakyn80"
$env:IMAGE_TAG="v1.0.0"
docker compose -f docker-compose.registry.yml pull
docker compose -f docker-compose.registry.yml up -d
```

If model, storage, or batches are not in the default local folders, set:

- `NALUS_MODELS_HOST_DIR`
- `NALUS_STORAGE_HOST_DIR`
- `NALUS_BATCHES_HOST_DIR`
- `NALUS_ARTIFACTS_HOST_DIR`
- `NALUS_APP_ARTIFACTS_HOST_DIR`

## Historical court backfill (separate stack)

Do **not** use the RAG compose for NS/NSS historical downloads.

See [court_staging/WEDOS_HISTORICAL_BACKFILL.md](court_staging/WEDOS_HISTORICAL_BACKFILL.md) and [../docker-compose.court-staging.yml](../docker-compose.court-staging.yml).
