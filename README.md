# NALUS Scraper -> RAG Pipeline

## 1. Ucel projektu

Cilem projektu je vytvorit plne automatizovany pipeline pro:

- vyhledavani rozhodnuti Ustavniho soudu v NALUS
- extrakci strukturovanych metadat
- stazeni plneho textu rozhodnuti
- ulozeni dat ve formatu vhodnem pro RAG (Retrieval-Augmented Generation)

Projekt slouzi jako datovy zdroj pro:

- pravni AI systemy
- vyhledavani judikatury
- embedding + vector DB
- LLM dotazovani nad pravnimi dokumenty

## 2. Architektura

Projekt je rozdelen do techto casti:

```text
app/
├── crawler/
│   ├── playwright_crawler.py   # ziskani HTML vysledku z NALUS search
│   ├── extractor.py            # parsovani vysledku do strukturovanych dat
│   ├── text_fetcher.py         # stazeni a cisteni plneho textu rozhodnuti
│   └── paginator.py            # nacitani dalsich stranek vysledku
│
├── models/
│   └── search_result.py        # datove modely NalusResult / NalusSearchPage
│
├── services/
│   └── decision_service.py     # enrichovani vysledku o plny text rozhodnuti
│
└── main.py                     # orchestrace cele pipeline
```

## 3. Jak pipeline funguje

### Krok 1 - Search (NALUS)

- provede se dotaz, napr. `rodinne pravo`
- pouziva se WebForms POST, ne scraping UI
- vysledkem je HTML stranky `/Search/Results.aspx`

### Krok 2 - Parsovani vysledku

Z HTML se extrahuji:

- `case_reference`, napr. `III.ÚS 255/26`
- datum rozhodnuti
- vysledek rizeni
- `ECLI`
- soudce zpravodaj
- URL na detail
- URL na plny text (`GetText.aspx`)

Vystup:

- list objektu `NalusResult`
- obaleny v `NalusSearchPage`

### Krok 3 - Stazeni plneho textu

Pro kazdy vysledek:

```text
https://nalus.usoud.cz/Search/GetText.aspx?sz=...
```

se stahne cely text rozhodnuti.

Ulozi se do:

```json
"full_text": "..."
```

### Krok 4 - Ulozeni

Vysledky se ukladaji do:

```text
results.json
```

Priklad struktury:

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

## 4. Klicove technicke poznatky

### NALUS neni klasicky web

NALUS bezi na:

- ASP.NET WebForms

Pouziva:

- `__VIEWSTATE`
- `__EVENTVALIDATION`
- POST back

Proto:

- nefunguje jednoduche `requests.get(..., params=...)` jako u bezneho vyhledavani
- UI scraping neni idealni primarni reseni

### Spravne reseni

- simulovat WebForms POST
- pouzivat spravne pole:
  - `ctl00$MainContent$text`

### Co nepouzivat

- `klicove_slovo`
  - to je readonly pole navazane na popup a neni to spravny vstup pro backend search flow

## 5. Vystup projektu

Po spusteni:

```powershell
docker run --rm nalus-scraper
```

dostanes:

- parsovane vysledky
- ulozeny JSON
- plne texty rozhodnuti

## 6. Pouziti pro RAG

Tento projekt slouzi jako ingest layer.

Dalsi krok typicky vypada takto:

1. Chunking
   - rozdelit `full_text` na casti, napr. 500-1000 tokenu
2. Embedding
   - `sentence-transformers`, OpenAI, Jina apod.
3. Ulozeni
   - FAISS, Chroma, Qdrant
4. Query
   - semantic search
   - LLM odpovedi nad relevantnimi chunky

## 7. Mozna rozsireni

1. Pagination
   - iterace pres vsechny stranky, napr. `1 -> 771`

2. Ukladani per dokument

   Misto jednoho JSON:

   ```text
   data/
   ├── 136232.txt
   ├── 136087.txt
   ```

3. Metadata DB
   - SQLite / Postgres
   - indexace podle:
     - ECLI
     - data
     - soudce

4. Filtrace
   - podle typu rozhodnuti
   - podle roku
   - podle vysledku

5. Scheduler
   - pravidelne scrapovani, napr. cron / worker

## 8. Limity

- NALUS neni API, zmeny HTML mohou rozbit parser
- rate limiting je potreba resit pres delay / retry / backoff
- nektere dokumenty maji nekonzistentni strukturu

## 9. Stav projektu

- search funguje
- parsovani funguje
- full text funguje
- Docker funguje
- JSON export funguje

## 10. Co projekt realne je

Toto neni scraper UI.

Je to:

- data ingestion layer pro pravni AI system

## 11. Finalni pipeline

```text
NALUS
  ↓
Search (POST)
  ↓
HTML
  ↓
Parser
  ↓
Structured data
  ↓
Full text fetch
  ↓
results.json
  ↓
RAG (embedding + vector DB + LLM)
```
