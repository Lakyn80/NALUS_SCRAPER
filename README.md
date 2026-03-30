# NALUS Scraper -> RAG Pipeline

## 1. Purpose

A fully automated pipeline for downloading and indexing all decisions of the Czech Constitutional Court (NALUS) for use in legal AI systems.

- Scrapes all decisions from NALUS (1993–present) year by year
- Extracts structured metadata and full decision text
- Ingests data into Qdrant vector database for RAG

Data source for:
- Legal AI systems
- Case law search
- Embedding + vector DB
- LLM queries over legal documents

## 2. Architecture

```text
app/
├── crawler/
│   ├── playwright_crawler.py   # POST-based HTML retrieval from NALUS search
│   ├── extractor.py            # HTML parsing into structured data
│   ├── text_fetcher.py         # Full decision text download and cleaning
│   └── paginator.py            # Results pagination
│
├── models/
│   └── search_result.py        # NalusResult / NalusSearchPage data models
│
├── services/
│   └── decision_service.py     # Enriches results with full decision text
│
├── rag/
│   └── ingest/
│       └── qdrant_ingest.py    # Chunking + idempotent Qdrant ingestion
│
├── data/
│   └── runtime_corpus.py       # Builds in-memory corpus from JSON batches
│
└── main.py                     # Pipeline orchestration

scripts/
├── scrape_all_nalus.py         # Year-by-year full corpus download with resume
└── ingest_batch.py             # Standalone Qdrant ingest for batch JSON files

batches/
├── manifest.json               # Tracks completed years and ingested batches
└── year_YYYY_*.json            # Downloaded decisions per year
```

## 3. How the pipeline works

### Step 1 — Search (NALUS)

- Submits a WebForms POST request to `/Search/Search.aspx`
- Uses `decidedFrom` / `decidedTo` date filters to scope by year
- Empty text query = all decisions in a given year

### Step 2 — Parse results

Extracted per decision:
- `case_reference` — e.g. `III.ÚS 255/26`
- `decision_date`
- `verdict`
- `ecli` — e.g. `ECLI:CZ:US:2026:3.US.255.26.1`
- `judge_rapporteur`
- `detail_url`
- `text_url` — link to `GetText.aspx`

### Step 3 — Full text download

```
https://nalus.usoud.cz/Search/GetText.aspx?sz=...
```

Stored in:
```json
"full_text": "..."
```

Timeout: 30s, retry: 3x with 2s delay, 0.5s pause between requests.

### Step 4 — Save & ingest

- JSON saved to `batches/year_YYYY_*.json`
- Automatically ingested into Qdrant after each completed year
- `manifest.json` tracks progress — safe to interrupt and resume

## 4. Key technical notes

### NALUS is not a standard web app

NALUS runs on ASP.NET WebForms and requires:
- `__VIEWSTATE`
- `__VIEWSTATEGENERATOR`
- `__EVENTVALIDATION`
- POST-back pattern

A simple `requests.get(..., params=...)` does not work.

### What NOT to use

- `klicove_slovo` — read-only UI field tied to a popup, not the actual search backend input

## 5. Scripts

### Download all of NALUS

```bash
python scripts/scrape_all_nalus.py                  # all years 1993–present
python scripts/scrape_all_nalus.py --resume         # resume from last incomplete year
python scripts/scrape_all_nalus.py --year 2020      # single year
python scripts/scrape_all_nalus.py --from-year 2010 # from year 2010 onward
python scripts/scrape_all_nalus.py --no-ingest      # save JSON only, skip Qdrant
python scripts/scrape_all_nalus.py --dry-run        # show what would be downloaded
```

### Ingest existing batches

```bash
python scripts/ingest_batch.py                      # ingest all batches/
python scripts/ingest_batch.py batches/foo.json     # ingest single file
python scripts/ingest_batch.py --url http://host:6333
```

### Single query scrape (main.py)

```bash
NALUS_QUERY="rodinné právo" NALUS_MAX_RESULTS=1000 python -m app.main
```

Environment variables:
- `NALUS_QUERY` — search query (default: `rodinné právo`)
- `NALUS_MAX_RESULTS` — max results
- `NALUS_PAGE_START` / `NALUS_PAGE_END` — page range
- `NALUS_FETCH_FULL_TEXT` — `true`/`false`
- `NALUS_AUTO_INGEST` — auto-ingest to Qdrant after scrape (default: `true`)
- `QDRANT_URL` — Qdrant URL (default: `http://localhost:6333`)
- `QDRANT_COLLECTION_NAME` — collection name (default: `nalus`)

## 6. Output format

```json
{
  "case_reference": "III.ÚS 255/26 #1",
  "decision_date": "4. 3. 2026",
  "verdict": "vyhoveno",
  "ecli": "ECLI:CZ:US:2026:3.US.255.26.1",
  "judge_rapporteur": "Krestanova Veronika",
  "text_url": "...",
  "detail_url": "...",
  "full_text": "..."
}
```

## 7. RAG pipeline

```text
NALUS
  |
Search POST (by year)
  |
HTML -> Parser -> NalusResult
  |
Full text fetch (GetText.aspx)
  |
batches/year_YYYY_*.json
  |
Chunking (runtime_corpus.py)
  |
Qdrant (idempotent upsert)
  |
Semantic search + LLM
```

## 8. Deduplication

- On startup, all existing ECLIs/case references from `batches/` are loaded into memory
- Before fetching full text, each result is checked against this set
- Duplicates are skipped — no redundant HTTP requests
- Qdrant ingestion is idempotent (deterministic UUID point IDs)

## 9. Limits

- NALUS has no public API — HTML changes can break the parser
- Rate limiting handled via delay + retry + backoff
- Some documents have inconsistent structure
- ~33,000 total decisions across 1993–2025 (~400 per year average)

## 10. Project status

| Component | Status |
|-----------|--------|
| Search (WebForms POST) | working |
| HTML parsing | working |
| Full text download | working |
| Year-by-year bulk scrape | working |
| Resume / manifest tracking | working |
| Qdrant ingest (idempotent) | working |
| JSON export | working |
| Real embeddings | pending |
