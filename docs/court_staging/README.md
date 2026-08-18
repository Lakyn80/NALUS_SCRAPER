# Court staging scrape / daily updater

## Goal

Historical + daily incremental scrapes for:

- Ústavní soud (US / NALUS)
- Nejvyšší soud (NS)
- Nejvyšší správní soud (NSS)

All writes go to `artifacts/court_staging/` only while Full B / Legal v2 full-corpus builds use the frozen ÚS `batches/` snapshot.

## Hard isolation

Do **not** write to:

- `batches/` (including `manifest.json`)
- `artifacts/legal_v2/full_corpus_build_v1/eligible_document_ids.txt`
- Full B checkpoint / BM25 / Qdrant collection

Path guards live in `app/court_staging/paths.py`.

## Canonical identity

`app/court_staging/identity.py`:

1. ECLI
2. official source document id
3. spisová značka + decision_date
4. deterministic fallback (not content hash)

`content_hash` = change detection only → same `canonical_id` + new hash = **UPDATED**.

## Layout

```text
artifacts/court_staging/
  us/incremental/
  ns/historical/
  ns/incremental/
  nss/historical/
  nss/incremental/
  updater/          # watermarks + run reports
  merge_dry_run/    # report only — no physical merge
```

## CLIs

```powershell
cd C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper

# NS historical (multi-year exhaust → staging)
python -m app.nsoud.run_historical_backfill --year-from 2024 --year-to 2025 --resume --delay 1.0

# NSS probe + pilot
python -m app.nssoud.probe_source
python -m app.nssoud.scraper --limit 20 --out artifacts/court_staging/nss/historical/pilot/pilot.jsonl --max-pages 3

# NSS historical
python -m app.nssoud.run_historical_backfill --year-from 2024 --year-to 2025 --resume

# Daily unified updater (staging only)
python scripts/court_staging_updater.py --courts us,ns,nss --mode incremental --overlap-days 7
python scripts/court_staging_updater.py --mode dry-run-merge

# Windows daily schedule (no merge/index)
.\scripts\court_staging\register_daily_task.ps1 -Register
.\scripts\court_staging\register_daily_task.ps1 -RunOnce
```

## Completeness

Month `status=ok` only when every unique discovered document is fetched OK or explicitly classified (failed/skipped). See `app/court_staging/completeness.py`.

## Merge policy

- **No auto-merge** staging → `batches/`
- NS/NSS stay in staging until a future canonical ingestion layer
- ÚS may later merge into ÚS `batches/` explicitly (same schema) — not in this phase
- `--dry-run-merge` is report-only

## WEDOS historical backfill (NS/NSS)

Long-running historical jobs are **not** the daily updater. Compose: [docker-compose.court-staging.yml](../../docker-compose.court-staging.yml). Runbook: [WEDOS_HISTORICAL_BACKFILL.md](WEDOS_HISTORICAL_BACKFILL.md).

Services: `ns-historical` (2020–2026, default), `ns-historical-pre2020` (profile `ns-pre2020`), `nss-historical` (profile `nss`). NSS probe / 20-doc pilot uses `docker compose run --rm`, not a fourth service.

## Scheduler

Windows: `scripts/court_staging/register_daily_task.ps1 -Register` (daily US+NS+NSS → staging only).
Linux later: `docs/court_staging/nalus-court-staging.{service,timer}.example`.

Those timer units call `scripts/court_staging_updater.py` (incremental). Do **not** use them for historical NS/NSS backfill.

## NSS note

`vyhledavac.nssoud.cz` is a DXCFTS UI (`findform` POST + session endpoints like `/Home/MyResTRowsCont`).
Probe report is under `artifacts/court_staging/nss/historical/pilot/probe_report.json`.
The current HTTP pilot may return zero docs until search POST/session wiring is refined; historical runner + identity/completeness path is ready.
