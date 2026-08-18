# WEDOS historical court backfill (NS / NSS)

Long-running HTTP → JSONL jobs on WEDOS. Separate from the RAG stack (`docker-compose.yml` / `docker-compose.registry.yml`) and from the **daily** updater (`scripts/court_staging_updater.py` + systemd timer examples).

Pipeline:

```text
GitHub Actions → ghcr.io/lakyn80/nalus-scraper-api:<tag>
→ WEDOS docker compose pull (no VPS build)
→ bind mount /opt/nalus-data/court_staging
→ NS 2020–2026 resume, then NS 1993–2019, then NSS after pilot PASS
```

Do **not** start NSS 2003–2026 until the 20-document pilot PASSes.

## Image

```text
ghcr.io/lakyn80/nalus-scraper-api:<tag>
```

Compose file: [docker-compose.court-staging.yml](../../docker-compose.court-staging.yml)

Services only:

| Service | Profile | Job |
|---|---|---|
| `ns-historical` | default | NS `--year-from 2020 --year-to 2026 --resume --delay 1.0` |
| `ns-historical-pre2020` | `ns-pre2020` | NS `--year-from 1993 --year-to 2019 --resume --delay 1.0` |
| `nss-historical` | `nss` | NSS `--year-from 2003 --year-to 2026 --resume --delay 1.5` |

There is **no** `nss-pilot` service. Probe and 20-doc pilot use `docker compose run --rm` against the same image (service `ns-historical` is only a vehicle for the image + bind mount; `--entrypoint` replaces its command so NS 2020–2026 does not start).

Bind mount:

```text
${COURT_STAGING_HOST_DIR:-/opt/nalus-data/court_staging} → /app/artifacts/court_staging
```

`restart: on-failure` — successful exit 0 leaves the container stopped. Do not use `unless-stopped` for historical jobs.

## Not this job

- Daily incremental updater: [nalus-court-staging.service.example](nalus-court-staging.service.example) / [nalus-court-staging.timer.example](nalus-court-staging.timer.example)
- RAG API, Qdrant, Redis, BM25, embeddings, GPU
- Merge into `batches/`

## Resume (already in code)

`--resume` skips months whose durable manifest entry has `status=ok`.

- NS manifest: `ns/historical/nsoud_historical_manifest.json`
- NSS manifest: `nss/historical/nssoud_historical_manifest.json`

A month that is not `ok` is re-run. The scraper loads existing `canonical_id`s from that month’s JSONL (`load_canonical_index`). `UNCHANGED` does not rewrite; `NEW` appends; `UPDATED` rewrites one id. The month file is not truncated.

Copy the **entire** `ns/historical/` directory (JSONL + manifest + partial month). Do not start from an empty host dir if local progress exists.

## Strict migration order

Do these steps in order. Do not stop the local writer until the published image is validated on WEDOS.

1. Land GHCR workflow + court-staging compose in git, then **publish** a new API image that includes `app/nsoud`, `app/nssoud`, `app/court_staging`.
2. On WEDOS: `docker login ghcr.io`, `docker compose pull`, `--help` inside the image, bind-mount write test.
3. `mkdir -p /opt/nalus-data/court_staging/{ns/historical,nss/historical,updater}`
4. **Stop** the local NS writer (find by command line, not a hardcoded PID).
5. Copy `artifacts/court_staging/ns/historical/` to WEDOS. Keep the local copy.
6. Verify file count, sizes, manifest, partial JSONL, no 0-byte month files that should have data.
7. `up -d ns-historical` with `--resume`.
8. Confirm logs skip `ok` months and load canonical ids; they must not restart 2020 from zero.
9. After 2020–2026 completes: `--profile ns-pre2020`.
10. NSS probe + 20-doc `run --rm` pilot. Only if PASS: `--profile nss`.
11. `docker rm` finished containers; **keep host data**.

JSONL is under `artifacts/` and is gitignored. Never commit it.

## Publish image (after commit; this task does not push)

GitHub Actions: workflow **Docker Publish**, `workflow_dispatch`, set `image_tag` (prefer an explicit tag, not only `latest`).

Local (optional):

```powershell
echo $env:GITHUB_TOKEN | docker login ghcr.io -u lakyn80 --password-stdin
.\scripts\docker_publish.ps1 -GhcrOwner lakyn80 -Tag v1.0.0 -AlsoTagLatest
```

WEDOS packages are typically private. Create a PAT with `read:packages` and log in on the VPS. Do not commit the token.

```bash
echo "$GHCR_TOKEN" | docker login ghcr.io -u lakyn80 --password-stdin
```

## WEDOS: pull and start NS 2020–2026

Repo checkout on the VPS must include `docker-compose.court-staging.yml`.

```bash
export GHCR_OWNER=lakyn80
export IMAGE_TAG=v1.0.0   # the tag you published
export COURT_STAGING_HOST_DIR=/opt/nalus-data/court_staging

mkdir -p /opt/nalus-data/court_staging/{ns/historical,nss/historical,updater}

cd /opt/nalus-scraper   # or the actual clone path
docker compose -f docker-compose.court-staging.yml pull
```

Validate the image (must print argparse help, not uvicorn):

```bash
docker compose -f docker-compose.court-staging.yml run --rm --no-deps --entrypoint python ns-historical \
  -m app.nsoud.run_historical_backfill --help

docker compose -f docker-compose.court-staging.yml run --rm --no-deps --entrypoint python ns-historical \
  -m app.nssoud.run_historical_backfill --help

docker compose -f docker-compose.court-staging.yml run --rm --no-deps --entrypoint python ns-historical \
  -m app.nssoud.probe_source --help
```

Bind-mount write test:

```bash
docker compose -f docker-compose.court-staging.yml run --rm --no-deps --entrypoint sh ns-historical \
  -c 'touch /app/artifacts/court_staging/.write_test && ls -la /app/artifacts/court_staging'
```

Remove the test file after it succeeds. Do not delete JSONL.

### Stop local writer (Windows, after image OK)

Do **not** use a hardcoded PID.

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'app\.nsoud\.run_historical_backfill' } |
  Select-Object ProcessId, CommandLine

# After you have confirmed the PID:
# Stop-Process -Id <pid>
```

Writer must be stopped **before** the copy so WEDOS does not resume from a torn write.

### Copy historical NS data

Source (laptop):

```text
C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper\artifacts\court_staging\ns\historical\
```

Destination (WEDOS):

```text
/opt/nalus-data/court_staging/ns/historical/
```

Copy the whole directory: monthly `nsoud_YYYY_MM.jsonl`, `nsoud_historical_manifest.json`, and any incomplete month. Example:

```powershell
# From the laptop (adjust user@host). Keep the local copy.
scp -r artifacts\court_staging\ns\historical\* user@WEDOS:/opt/nalus-data/court_staging/ns/historical/
```

Verify on WEDOS: file count, `du -sh`, manifest keys with `status=ok`, the current incomplete month JSONL is non-empty.

### Start resume

```bash
docker compose -f docker-compose.court-staging.yml up -d ns-historical
docker compose -f docker-compose.court-staging.yml logs -f ns-historical
```

Healthy resume: skipped months already `ok`; loading canonical ids for the first non-ok month; not re-fetching 2020-01 from empty.

Later waves (do not run until the previous wave is done):

```bash
docker compose -f docker-compose.court-staging.yml --profile ns-pre2020 up -d ns-historical-pre2020
```

## NSS probe and 20-document pilot (`run --rm` only)

Do **not** `up` profile `nss` yet. These overrides must include `--entrypoint python` so the `ns-historical` command is not used.

```bash
docker compose -f docker-compose.court-staging.yml run --rm --no-deps --entrypoint python ns-historical \
  -m app.nssoud.probe_source

docker compose -f docker-compose.court-staging.yml run --rm --no-deps --entrypoint python ns-historical \
  -m app.nssoud.scraper --limit 20 --max-pages 3 --delay 1.5 \
  --out artifacts/court_staging/nss/historical/pilot/pilot.jsonl
```

Pilot PASS: HTTP OK, JSONL with real decisions (about 20), valid lines, meaningful ids.

`vyhledavac.nssoud.cz` DXCFTS may return **zero** documents until POST/session wiring is right. Zero docs is **not** PASS. Do not start full NSS historical in that case.

After PASS only:

```bash
docker compose -f docker-compose.court-staging.yml --profile nss up -d nss-historical
```

## Cleanup

```bash
docker compose -f docker-compose.court-staging.yml rm -f
# keep /opt/nalus-data/court_staging
```
