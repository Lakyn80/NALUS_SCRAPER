# AGENT.md — Stav projektu a pracovní postup

Tento dokument popisuje kompletní stav projektu k 2026-04-05 a postup který byl proveden,
aby každý navazující mohl okamžitě pokračovat.

---

## Co projekt dělá

**NALUS Scraper** stahuje rozhodnutí Ústavního soudu ČR z [nalus.usoud.cz](https://nalus.usoud.cz),
ukládá je jako JSON a ingestuje do Qdrantu jako vektorovou databázi pro RAG (Retrieval-Augmented Generation).

Výsledkem je FastAPI server s RAG pipeline — uživatel položí dotaz, systém najde relevantní
judikáty a přes LLM sestaví odpověď.

---

## Architektura

```
scripts/scrape_all_nalus.py   # stahování dat rok po roku
scripts/ingest_batch.py       # ruční ingest JSON batchů do Qdrantu
app/main.py                   # scraper entry point (single query)
app/api/main.py               # FastAPI server (RAG API)
app/rag/                      # RAG pipeline (chunker, embedder, retriever, LLM)
app/crawler/                  # Playwright crawler + text fetcher
batches/                      # stažená data (JSON, rok po roku)
docker-compose.yml            # Qdrant + Redis + API kontejnery
```

---

## Infrastruktura (Docker)

Vše běží v Dockeru. Tři kontejnery:

| Kontejner | Popis | Port |
|-----------|-------|------|
| `qdrant`  | Vektorová DB | interní `6333` (není vyexponován na localhost!) |
| `redis`   | Cache RAG dotazů | interní |
| `api`     | FastAPI RAG server | `8029` na hostu |

**Důležité:** Qdrant není dostupný na `localhost:6333` — je dostupný jen uvnitř Docker sítě
jako `http://qdrant:6333`. Při spouštění skriptů přímo v kontejneru vždy přidat `--url http://qdrant:6333`.

Kolekce v Qdrantu:
- `nalus` — **aktivní kolekce**, sem se ingestuje
- `nalus_stable_20260326` — záloha z 26.3.2026
- `nalus_live` — alias (ukazuje na `nalus`)

`.env`:
```
QDRANT_COLLECTION_NAME=nalus_live   # alias -> nalus
LLM_API_KEY=...                     # DeepSeek API klíč
```

---

## Stažená data (batches/)

Rozhodnutí jsou stažena rok po roku skriptem `scripts/scrape_all_nalus.py`.

| Roky | Stav | Počet dokumentů |
|------|------|-----------------|
| 1993–2023 | ✅ kompletní | ~66 000 |
| 2024 | ✅ kompletní | 3 514 |
| 2025 | ✅ kompletní | 2 588 |
| 2026 | ✅ (rok běží) | 245 |
| **Celkem** | | **~75 000 dokumentů** |

Dále jsou v `batches/` starší soubory:
- `results_rodinne_pravo_1000.json` — testovací batch rodinné právo
- `batch_03_pages601_1600.json` — starší obecný batch
- `all_stezovatel_chunk*.json` — fulltextové dotazy na "stěžovatel"

---

## Stav Qdrantu

K 2026-04-05 je v kolekci `nalus` **770 776 bodů** (chunků).

Embedovací model (produkce): `BAAI/bge-m3` (dim=1024). MPNet je zakázaný.

Legacy kolekce `nalus_live` / `nalus_stable_20260326` (768-dim MPNet) se nepoužívá pro nový RAG stack.

---

## Jak spustit ingest

### Ingest konkrétních souborů

```bash
# Nakopírovat aktuální skript do kontejneru (pokud byl měněn lokálně):
docker compose cp scripts/ingest_batch.py api:/app/scripts/ingest_batch.py

# Spustit ingest:
docker compose exec api bash -c "python scripts/ingest_batch.py batches/year_{1993..2023}_*.json --collection nalus --url http://qdrant:6333"
```

### Ingest všeho najednou

```bash
docker compose exec api python scripts/ingest_batch.py --collection nalus --url http://qdrant:6333
```

### Pozn.: OSError na konci (manifest) je normální

`batches/` je v Dockeru namountovaný jako read-only (`./batches:/app/batches:ro`).
Ingest data uloží správně, jen manifest.json nemůže aktualizovat. Ignoruj tuto chybu.

---

## Jak stahovat nová data

```bash
# Stáhnout rok 2026 (nebo cokoliv nového):
docker compose exec api python scripts/scrape_all_nalus.py --from-year 2026 --url http://qdrant:6333

# S resume (navázat kde skončilo):
docker compose exec api python scripts/scrape_all_nalus.py --resume --url http://qdrant:6333

# Jen konkrétní rok:
docker compose exec api python scripts/scrape_all_nalus.py --year 2026 --url http://qdrant:6333
```

Skript automaticky ingestuje každý rok po stažení (pokud nepřidáš `--no-ingest`).

---

## Problém s embedderem — OPRAVENO

**Symptom:** `Vector dimension error: expected dim: 768, got 10`

**Příčina:** `scripts/ingest_batch.py` používal `MockEmbedder(dim=10)` místo reálného
`SentenceTransformersEmbedder(dim=768)`.

**Oprava provedena** v `scripts/ingest_batch.py`:
- Importuje `SentenceTransformersEmbedder`
- Předá ho do `QdrantIngestor` jako `embedder=`

**Pozor:** Stejný bug existuje v `scripts/scrape_all_nalus.py` ve funkci `_ingest_file()`
a v `app/main.py` ve funkci `_run_ingest()` — tam se stále používá `MockEmbedder`.
Pokud se tyto cesty použijí pro nový ingest, budou posílat dim=10 vektory → selžou.

---

## Jak zkontrolovat stav Qdrantu

```bash
# Počet bodů v kolekci:
docker compose exec api python -c "from qdrant_client import QdrantClient; c = QdrantClient('http://qdrant:6333'); print(c.count(collection_name='nalus').count)"

# Seznam kolekcí:
docker compose exec api python -c "from qdrant_client import QdrantClient; c = QdrantClient('http://qdrant:6333'); cols = c.get_collections().collections; print([x.name for x in cols])"
```

---

## Health check API

```bash
curl http://localhost:8029/health
```

Vrátí stav orchestratoru, embedding modelu a ingestu.

---

## Co dál (návrhy)

- [ ] Opravit `MockEmbedder` bug v `scrape_all_nalus.py` a `app/main.py`
- [ ] Vyexponovat Qdrant port v `docker-compose.yml` pro snazší lokální práci
- [ ] Otestovat RAG dotazy přes API (`/health`, `/query`)
- [ ] Zvážit aktualizaci Qdrant image z `v1.13.6` na novější (client je v1.17.1 → varování o nekompatibilitě)
- [ ] Pravidelný re-scrape 2026 jak přibývají nová rozhodnutí
