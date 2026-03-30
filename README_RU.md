# NALUS Scraper -> RAG Pipeline

## 1. Назначение

Полностью автоматизированный пайплайн для скачивания и индексирования всех решений Конституционного суда Чехии (NALUS) для использования в правовых AI-системах.

- Скачивает все решения из NALUS (1993–настоящее время) по годам
- Извлекает структурированные метаданные и полный текст решений
- Загружает данные в векторную базу данных Qdrant для RAG

Источник данных для:
- Правовых AI-систем
- Поиска судебной практики
- Embedding + векторная БД
- LLM-запросов над юридическими документами

## 2. Архитектура

```text
app/
├── crawler/
│   ├── playwright_crawler.py   # POST-запросы к поиску NALUS
│   ├── extractor.py            # Парсинг HTML в структурированные данные
│   ├── text_fetcher.py         # Скачивание и очистка полного текста решений
│   └── paginator.py            # Пагинация результатов
│
├── models/
│   └── search_result.py        # Модели данных NalusResult / NalusSearchPage
│
├── services/
│   └── decision_service.py     # Обогащение результатов полным текстом
│
├── rag/
│   └── ingest/
│       └── qdrant_ingest.py    # Чанкинг + идемпотентная загрузка в Qdrant
│
├── data/
│   └── runtime_corpus.py       # Построение in-memory корпуса из JSON-батчей
│
└── main.py                     # Оркестрация пайплайна

scripts/
├── scrape_all_nalus.py         # Полное скачивание корпуса по годам с возобновлением
└── ingest_batch.py             # Автономная загрузка JSON-батчей в Qdrant

batches/
├── manifest.json               # Отслеживание завершённых годов и загруженных батчей
└── year_YYYY_*.json            # Скачанные решения по годам
```

## 3. Как работает пайплайн

### Шаг 1 — Поиск (NALUS)

- Отправляет WebForms POST-запрос на `/Search/Search.aspx`
- Использует фильтры `decidedFrom` / `decidedTo` для ограничения по году
- Пустой текстовый запрос = все решения за данный год

### Шаг 2 — Парсинг результатов

Извлекается по каждому решению:
- `case_reference` — напр. `III.ÚS 255/26`
- `decision_date`
- `verdict`
- `ecli` — напр. `ECLI:CZ:US:2026:3.US.255.26.1`
- `judge_rapporteur`
- `detail_url`
- `text_url` — ссылка на `GetText.aspx`

### Шаг 3 — Скачивание полного текста

```
https://nalus.usoud.cz/Search/GetText.aspx?sz=...
```

Сохраняется в:
```json
"full_text": "..."
```

Таймаут: 30с, повторы: 3x с задержкой 2с, пауза 0.5с между запросами.

### Шаг 4 — Сохранение и загрузка

- JSON сохраняется в `batches/year_YYYY_*.json`
- После завершения каждого года автоматически загружается в Qdrant
- `manifest.json` отслеживает прогресс — можно прерывать и возобновлять

## 4. Технические особенности

### NALUS — не стандартное веб-приложение

NALUS работает на ASP.NET WebForms и требует:
- `__VIEWSTATE`
- `__VIEWSTATEGENERATOR`
- `__EVENTVALIDATION`
- POST-back паттерн

Простой `requests.get(..., params=...)` не работает.

### Что не использовать

- `klicove_slovo` — поле только для чтения, привязанное к попапу, не является реальным входом для поиска на бэкенде

## 5. Скрипты

### Скачать весь NALUS

```bash
python scripts/scrape_all_nalus.py                  # все годы 1993–настоящее время
python scripts/scrape_all_nalus.py --resume         # продолжить с последнего незавершённого года
python scripts/scrape_all_nalus.py --year 2020      # один год
python scripts/scrape_all_nalus.py --from-year 2010 # с 2010 года
python scripts/scrape_all_nalus.py --no-ingest      # только JSON, без Qdrant
python scripts/scrape_all_nalus.py --dry-run        # показать что будет скачано
```

### Загрузить существующие батчи

```bash
python scripts/ingest_batch.py                      # загрузить все батчи
python scripts/ingest_batch.py batches/foo.json     # один файл
python scripts/ingest_batch.py --url http://host:6333
```

### Поиск по запросу (main.py)

```bash
NALUS_QUERY="rodinné právo" NALUS_MAX_RESULTS=1000 python -m app.main
```

Переменные окружения:
- `NALUS_QUERY` — поисковый запрос (по умолчанию: `rodinné právo`)
- `NALUS_MAX_RESULTS` — максимум результатов
- `NALUS_PAGE_START` / `NALUS_PAGE_END` — диапазон страниц
- `NALUS_FETCH_FULL_TEXT` — `true`/`false`
- `NALUS_AUTO_INGEST` — автозагрузка в Qdrant после скрапинга (по умолчанию: `true`)
- `QDRANT_URL` — адрес Qdrant (по умолчанию: `http://localhost:6333`)
- `QDRANT_COLLECTION_NAME` — название коллекции (по умолчанию: `nalus`)

## 6. Формат данных

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

## 7. RAG пайплайн

```text
NALUS
  |
POST-поиск (по годам)
  |
HTML -> Парсер -> NalusResult
  |
Скачивание полного текста (GetText.aspx)
  |
batches/year_YYYY_*.json
  |
Чанкинг (runtime_corpus.py)
  |
Qdrant (идемпотентный upsert)
  |
Семантический поиск + LLM
```

## 8. Дедупликация

- При старте все существующие ECLI/case reference из `batches/` загружаются в память
- Перед скачиванием полного текста каждый результат проверяется по этому множеству
- Дубликаты пропускаются — лишние HTTP-запросы не выполняются
- Загрузка в Qdrant идемпотентна (детерминированные UUID идентификаторы точек)

## 9. Ограничения

- У NALUS нет публичного API — изменения HTML могут сломать парсер
- Ограничение скорости обрабатывается через задержки, повторы и backoff
- Некоторые документы имеют непоследовательную структуру
- ~33 000 решений всего за 1993–2025 (~400 решений в год в среднем)

## 10. Статус проекта

| Компонент | Статус |
|-----------|--------|
| Поиск (WebForms POST) | работает |
| Парсинг HTML | работает |
| Скачивание полного текста | работает |
| Массовое скачивание по годам | работает |
| Возобновление / manifest | работает |
| Загрузка в Qdrant (идемпотентно) | работает |
| Экспорт JSON | работает |
| Реальные эмбеддинги | в разработке |
