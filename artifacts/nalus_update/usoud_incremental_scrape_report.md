# Ustavni soud / NALUS — Incremental Scrape Report

Status: **FINAL** — real run completed, validated, and merged into `batches/`.

Generated: 2026-07-08 (started ~12:27, finished ~12:49 UTC scrape time; merge
into `batches/` performed ~16:05 local)

## 1. Scraper path and exact command(s) used

- Scraper: `scripts/scrape_all_nalus.py` (unmodified, repo-tracked).
- Problem found before running: `batches/` is bind-mounted **read-only**
  inside the `api` container (`./batches:/app/batches:ro` in
  `docker-compose.yml`), and `scripts/scrape_all_nalus.py` hardcodes
  `BATCHES_DIR = PROJECT_ROOT / "batches"` / `CHECKPOINT_PATH = PROJECT_ROOT /
  "checkpoint.json"` with no env-var override (confirmed by reading the full
  script — no `os.environ` usage for output path). Verified read-only
  directly: `docker compose exec api sh -c "touch /app/batches/testwrite.tmp"`
  -> `Read-only file system`.
- Also found: `batches/manifest.json` marks year 2026 as
  `"complete": true` (from the 2026-04-03 run), and `main()` in
  `scrape_all_nalus.py` skips any year already in `completed` years
  unconditionally — even with `--year 2026` explicitly passed. So a plain
  `docker compose exec api python scripts/scrape_all_nalus.py --year 2026
  --no-ingest` would do **nothing** (year considered done) — this is the gap
  behind AGENT.md's open TODO "Pravidelny re-scrape 2026 jak pribyvaji nova
  rozhodnuti".
- Resolution used (no edits to the tracked script, no docker-compose.yml
  edits): a small **ephemeral, non-committed driver script** at
  `artifacts/nalus_incremental_batches/_driver.py` that:
  1. Loads the real `scripts/scrape_all_nalus.py` unmodified via
     `importlib.util.spec_from_file_location` (so `__file__` still resolves
     to `/app/scripts/scrape_all_nalus.py` and all `app.*` imports work
     exactly as normal).
  2. Monkeypatches only the module-level `BATCHES_DIR` and `CHECKPOINT_PATH`
     constants (after import, before calling `main()`) to point at
     `/app/artifacts/nalus_incremental_batches` — which is writable, because
     `docker-compose.yml` mounts `./artifacts:/app/artifacts` **without**
     `:ro`.
  3. That writable directory was seeded with a **copy** (not the original)
     of `batches/manifest.json` and `batches/year_2026_20260403_191318.json`,
     so the script's own dedup (`_load_existing_ids()`) still sees the 245
     already-scraped 2026 decisions and skips them, and `_completed_years()`
     still works. In the copy only, `manifest.json`'s year-2026 entry
     `"complete"` was flipped `true -> false` so the script would actually
     re-walk year 2026 instead of skipping it.
- Commands actually run:
  ```
  # dry run (native --dry-run flag, confirms only year 2026 is pending)
  docker compose exec api python /app/artifacts/nalus_incremental_batches/_driver.py \
      --year 2026 --no-ingest --dry-run

  # real run (ingestion explicitly disabled with --no-ingest to avoid the
  # known MockEmbedder(dim=10) bug in _ingest_file())
  docker compose exec api python /app/artifacts/nalus_incremental_batches/_driver.py \
      --year 2026 --no-ingest
  ```
- The real `batches/` directory and real `batches/manifest.json` and real
  `checkpoint.json` were **not** touched by the run itself. The new output
  file lands in `artifacts/nalus_incremental_batches/year_2026_<ts>.json` on
  the host (because `./artifacts` is a live bind mount), and was copied into
  `batches/` by hand afterward as a separate, explicit step (see below) —
  never overwriting an existing file, always a new filename.

## 2. Current corpus location

- `batches/*.json`, year-by-year (`year_1993_*.json` ... `year_2026_*.json`),
  plus legacy non-year batches (`batch_03_pages601_1600.json`,
  `results_rodinne_pravo_1000.json`, `all_stezovatel_chunk*.json`) from
  earlier ad-hoc full-text-query scrapes, predating the year-by-year
  approach.
- `batches/manifest.json` tracks ingestion/scrape history per file
  (`doc_count`, `complete`, `saved_at`); it is the authoritative index of
  what's been scraped, alongside per-file content.
- Total corpus size on disk: ~1.2 GB across 44 files before this run.

## 3. Previous latest decision date (before this run)

Determined by parsing `decision_date` (Czech `D. M. YYYY` format) across
**all** `batches/*.json` files (not string-sorted — parsed as real dates):

- **Latest decision_date found: 2026-03-25** (`25. 3. 2026`)
  — case `IV.ÚS 455/26 #1`, ECLI `ECLI:CZ:US:2026:4.US.455.26.1`,
  `result_id=136546`, from `batches/year_2026_20260403_191318.json`
  (245 decisions total in that file, scraped 2026-04-03).

## 4. Latest available decision date from source

- Live NALUS query for `decidedFrom=1.1.2026, decidedTo=31.12.2026`
  (verified with real outbound HTTPS requests from inside the `api`
  container — confirmed egress works: `GET
  https://nalus.usoud.cz/Search/Search.aspx` -> `200`):
  - `total_results = 1731`, `total_pages = 174` (was 245 docs / ~25 pages
    worth at last scrape).
  - Newest result currently on the site: decided **2026-06-25**
    (`III.ÚS 3315/25 #2`), published 2026-07-07.
- **Important finding from the audit** (not assumed — verified by sampling):
  decisions are not published strictly in decided-date order and publication
  can lag the decided date by weeks/months. Spot-checking pages across the
  2026 result set showed:
  - Pages 1-2 (decided 10-25 June 2026): all NEW.
  - Page ~100 (decided ~11 March 2026): **mixed** NEW/DUP on the same page —
    some March-decided cases were published after the 2026-04-03 scrape ran.
  - Pages 130-174 (decided Jan-Feb 2026): all DUP (already scraped).
  - This means a naive "only scrape decisions after date X" filter would have
    **missed** the backdated stragglers around page ~100. The approach used
    here re-walks the **entire year 2026** and relies on the scraper's own
    ID-based dedup (`ecli` / `case_reference` / `result_id`), which correctly
    captures every genuinely-new item regardless of where it falls in
    decided-date order.
  - **Cross-check on older years (to rule out a broader gap):** live
    `total_results` for 2020-2025 are all higher than the `doc_count`
    recorded in `manifest.json` for those years' single year-file (e.g. 2025:
    3966 live vs 2588 in `year_2025_*.json`). This looked alarming at first,
    but broad spot-sampling (14 pages spread across all of 2025, 8 pages
    across 2024, multiple pages across 2021/2023) found **zero** new items —
    every sampled record was already present somewhere in the corpus (in the
    legacy pre-year-by-year batch files: `batch_03_pages601_1600.json`,
    `results_rodinne_pravo_1000.json`, `all_stezovatel_chunk*.json`). The
    per-year `doc_count` in the manifest only reflects what that year's *own*
    scrape run added net-of-dedup against files that existed at the time; it
    understates true year coverage once you account for the legacy batches.
    **Conclusion: years other than 2026 are already fully covered; only 2026
    needed a real incremental re-scrape.**

## 5. Dry-run summary

- Native `--dry-run` flag (via the driver, pointed at the writable copy):
  ```
  [DRY-RUN] roky ke stazeni: [2026]
  [DRY-RUN] jiz hotove roky: []
  ```
  confirms only year 2026 is pending once its manifest entry is unmarked
  complete; this matches the audit above.
- Supplemented with a manual read-only probe (no writes, no manifest
  changes) using the same crawler functions the scraper uses
  (`fetch_page_html` + `extract_search_page`), sampling pages 1, 2, 10, 20,
  50, 80, 100, 130, 140, 145, 150, 174 of the 2026 result set against the
  full existing-ID set loaded from all of `batches/*.json` (307,677 IDs).
  Findings are summarized in section 4.
- Files that would be written: a single new file
  `batches/year_2026_<new-timestamp>.json` (never overwriting the existing
  `year_2026_20260403_191318.json`), plus a new appended entry in
  `batches/manifest.json` (append-only; existing entries untouched).
- Duplicate detection: confirmed working correctly (existing 245 2026
  decisions + a few sampled older-year IDs all resolved to `DUP` as
  expected; none were re-added).

## 6. Real run summary

- Full log: `artifacts/nalus_incremental_batches/run_output.log`.
- The run needed **three attempts** to reach page 174/174 (see the "process
  supervision issue" note at the end of this section) but the final,
  successful pass completed cleanly with no errors after it:
  ```
  [YEAR 2026] page=174/174 new=1486 dup=245
  [YEAR 2026] ulozeno 1486 rozhodnuti -> year_2026_20260708_124949.json
  [YEAR 2026] hotovo: 1486 novych, 245 duplikatu
  [INFO] Celkem novych rozhodnuti: 1486
  ```
- Output file: `year_2026_20260708_124949.json` (1486 records, verified valid
  JSON, verified UTF-8 — an earlier terminal display of the same file looked
  mojibake'd; that was a Windows console codepage rendering artifact, not
  real corruption — round-tripped correctly, e.g. `III.ÚS 3584/25 #1` /
  `STĚŽOVATEL - FO - advokát`).
- **Process-supervision issue found during this run (not a bug in the tracked
  scraper):** the orchestration used to relaunch the driver on interruption
  had a race condition — on two occasions an old still-finishing instance and
  a freshly relaunched instance both wrote to the same
  `year_2026_partial.json.tmp` path, and whichever process lost the race hit
  `FileNotFoundError` in `scripts/scrape_all_nalus.py`'s
  `_atomic_write_json()` (`os.replace(temp_path, path)` — the temp file had
  already been consumed/renamed by the other process). This happened once
  around page 24-47 and once around page 46-56. It was resolved by making the
  relaunch logic wait for/confirm the previous process had actually exited
  before starting a new one (verified via `docker top nalus-scraper-api-1`
  showing only a single `_driver.py` process at a time), after which the run
  proceeded from the checkpoint (page 57) to completion (page 174) without
  further incident. `scripts/scrape_all_nalus.py` itself was never modified.

## 7. Validation summary

| Metric | Value |
|---|---|
| Old latest decision_date (before this run) | 2026-03-25 |
| New latest decision_date (after this run) | **2026-06-25** (`III.ÚS 3315/25 #2`, `ECLI:CZ:US:2026:3.US.3315.25.2`, `result_id=137560`, published 2026-07-07) |
| New decisions discovered (site total for 2026 vs. prior local count) | 1731 (site) − 245 (prior local) = 1486 net new |
| New decisions downloaded | **1486** |
| Skipped as duplicates (re-walk of all 174 pages) | **245** (exactly the prior file's count — full agreement, no drift) |
| Failed (permanent, exhausted retries) | **0** — 0 of 1486 records have empty/missing `full_text` |
| Transient errors auto-recovered by scraper's own retry/backoff | 6 (see §9) |
| Records with duplicate/non-unique ID within the new file | 0 (1486 distinct ECLI/case_reference across 1486 records) |
| Existing records modified | **0** — `year_2026_20260403_191318.json` untouched; new data lives only in the new file |
| Final total corpus files under `batches/` | 45 (was 44) |
| Final total corpus size on disk | ~1.2 GB → ~1.22 GB (+18.9 MB) |
| New metadata fields missing vs. existing schema | None — same 18 fields present on every new record as on existing records (`result_id, case_reference, ecli, judge_rapporteur, petitioner, popular_name, decision_date, announcement_date, filing_date, publication_date, related_regulations, decision_form, importance, verdict, topics_and_keywords, detail_url, text_url, full_text`) |
| Parser/OCR/text extraction | Full text extraction **did** run (this is what the scraper always does per-result via `GetText.aspx`, same as prior runs) — this is raw scrape + text-fetch only, no chunking/embedding/ingest was invoked |

Date range of the 1486 new decisions: **2026-01-06 to 2026-06-25** — confirms
the audit finding in §4: many "new" items are backdated stragglers from
January-June that were published after the 2026-04-03 scrape, not only
recent decisions.

## 8. New decisions added

Full list of all 1486 new decisions (result_id, case_reference, ECLI,
decision_date, full text) is in `batches/year_2026_20260708_124949.json`.
Newest and oldest 8, for a quick sanity spot-check:

**Newest:**
| decision_date | case_reference | ecli | result_id |
|---|---|---|---|
| 2026-06-25 | III.ÚS 3315/25 #2 | ECLI:CZ:US:2026:3.US.3315.25.2 | 137560 |
| 2026-06-24 | Pl.ÚS 16/26 #1 | ECLI:CZ:US:2026:Pl.US.16.26.1 | 137521 |
| 2026-06-24 | Pl.ÚS 39/25 #1 | ECLI:CZ:US:2026:Pl.US.39.25.1 | 137567 |
| 2026-06-24 | Pl.ÚS 18/25 #1 | ECLI:CZ:US:2026:Pl.US.18.25.1 | 137570 |
| 2026-06-24 | II.ÚS 1027/26 #1 | ECLI:CZ:US:2026:2.US.1027.26.1 | 137572 |
| 2026-06-19 | I.ÚS 2196/25 #1 | ECLI:CZ:US:2026:1.US.2196.25.1 | 137557 |
| 2026-06-18 | III.ÚS 3503/25 #1 | ECLI:CZ:US:2026:3.US.3503.25.1 | 137537 |
| 2026-06-17 | III.ÚS 1537/26 #1 | ECLI:CZ:US:2026:3.US.1537.26.1 | 137426 |

**Oldest (backdated stragglers, published late):**
| decision_date | case_reference | ecli | result_id |
|---|---|---|---|
| 2026-01-07 | I.ÚS 714/25 #1 | ECLI:CZ:US:2026:1.US.714.25.1 | 135543 |
| 2026-01-07 | Pl.ÚS 31/25 #2 | ECLI:CZ:US:2026:Pl.US.31.25.2 | 135545 |
| 2026-01-07 | IV.ÚS 3581/25 #1 | ECLI:CZ:US:2026:4.US.3581.25.1 | 135553 |
| 2026-01-07 | Pl.ÚS 27/25 #2 | ECLI:CZ:US:2026:Pl.US.27.25.2 | 135560 |
| 2026-01-07 | II.ÚS 3578/25 #1 | ECLI:CZ:US:2026:2.US.3578.25.1 | 135612 |
| 2026-01-06 | III.ÚS 3010/25 #1 | ECLI:CZ:US:2026:3.US.3010.25.1 | 135459 |
| 2026-01-06 | II.ÚS 3657/25 #1 | ECLI:CZ:US:2026:2.US.3657.25.1 | 135468 |
| 2026-01-06 | III.ÚS 3584/25 #1 | ECLI:CZ:US:2026:3.US.3584.25.1 | 135501 |

## 9. Failures / skips with reasons

- **0 permanent failures.** All 1486 downloaded records have non-empty
  `full_text`.
- **6 transient request errors**, all auto-recovered by the scraper's
  existing retry/backoff (`REQUEST_MAX_RETRIES=3`, 1s fixed sleep) with no
  data loss:
  | URL (sz=) | Error | Outcome |
  |---|---|---|
  | `GetText.aspx?sz=3-1653-25_1` | Read timed out (30s) | Recovered on retry |
  | `GetText.aspx?sz=3-1205-26_1` | Connection aborted, then read timed out (10s) | Recovered on retry |
  | `GetText.aspx?sz=2-799-26_1` | Connection aborted | Recovered on retry |
  | `GetText.aspx?sz=1-50-26_1` | Connection aborted, then read timed out (10s) | Recovered on retry |
- **245 duplicates skipped** — expected, these are exactly the decisions
  already present in `year_2026_20260403_191318.json` (full agreement, 0
  drift, confirming stable IDs across runs).
- **2 process-supervision crashes** (not scraper bugs) during orchestration —
  see §6 for the race-condition diagnosis and fix; no data was lost or
  corrupted by these, since the scraper's own checkpoint/partial-save logic
  meant each restart resumed cleanly from the last completed page.

## 9b. Merge into the real corpus (host-side, after the container run)

Performed directly on the host (not inside the container, since `batches/` is
read-only from inside the `api` container):

1. `cp -n artifacts/nalus_incremental_batches/year_2026_20260708_124949.json
   batches/year_2026_20260708_124949.json` — plain copy to a **new** filename;
   `-n` (no-clobber) used defensively even though the target was confirmed
   not to exist beforehand.
2. Appended one new object to the `batches` array in `batches/manifest.json`
   (`file, year, doc_count, complete, saved_at, note`) — verified with `git
   diff --stat` afterward: exactly **8 lines added, 0 removed, 0 changed**.
   The pre-existing `year_2026_20260403_191318.json` entry (245 docs) is
   untouched.
3. The real `checkpoint.json` (repo root) was checked and confirmed
   unmodified by this whole task (`git status --short checkpoint.json` shows
   no change); it already read `{"year": 2027, "page": 1}` from the original
   April full-history crawl completing — that is unrelated pre-existing
   state, not something this run touched.

## 10. Qdrant / API / retrieval confirmation

- Qdrant: **read-only** queries only —
  `c.get_collections()` and `c.get_aliases()` confirmed `nalus_live` alias
  still points at `nalus_stable_20260326`, and `c.count(...)` confirmed
  776,424 points, unchanged. No `upsert`, `create_collection`, or
  `update_collection_aliases` calls were made.
- API (`app/api/*`): not touched.
- Retrieval / reranker / fusion logic (`app/rag/retrieval/`,
  `app/rag/reranker/`): not touched.
- NSoud benchmark collection
  (`nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1`)
  and `artifacts/rag_eval/`: not touched, not referenced beyond this
  acknowledgement.
- `docker-compose.yml`: **not modified** — the read-only `batches/` mount
  was left exactly as-is; the writable-redirect approach (section 1) was
  used instead.
- `scripts/scrape_all_nalus.py`: **not modified** (read and imported as-is).
- Ingestion was never triggered: `--no-ingest` was passed on every real
  invocation, and no code path that calls `_ingest_file` /
  `QdrantIngestor` / any embedder was exercised.

## 11. Recommended next step

The new decisions written to `batches/year_2026_<new-timestamp>.json` are
**raw scrape only** — they still need a separate, explicitly-requested
ingest/embedding pass (using the real 768-dim
`sentence-transformers/paraphrase-multilingual-mpnet-base-v2` embedder, via
the already-fixed `scripts/ingest_batch.py`, **not** `_ingest_file()` in
`scrape_all_nalus.py` or `_run_ingest()` in `app/main.py`, both of which
still use `MockEmbedder(dim=10)` per AGENT.md) before they are visible to
the RAG API / `nalus_live`.
