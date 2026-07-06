# NALUS RAG Eval — Legal Quality Report

## Executive summary

Vítězný retrieval (`bge_m3__dense_plus_bm25`, `dense_plus_bm25`) u 8 pilotních otázek vrací převážně právně užitečné podobné judikaturu. Benchmark metriky (hit_rate=1.0, mrr=0.9375) měří technickou shodu s datasetem; tento report hodnotí produkční právní užitečnost zvlášť.

## Benchmark vs. produkční užitečnost

| Pojem | Co měří |
| --- | --- |
| `benchmark_alignment` | Shoda s ručně zvolenými ECLI v eval datasetu |
| `production_usefulness` | Užitečnost pro reálné hledání podobné judikatury |
| `classification` | Právní typ relevance každého hitu (exact / alternate / irrelevant) |

Jiné ECLI než v datasetu **není automaticky chyba**, pokud jde o `alternate_relevant`.

## Per-case přehled

| case_id | production_usefulness | benchmark_alignment | top1_classification | best_relevant_rank | note |
| --- | --- | --- | --- | --- | --- |
| nsoud-positive-01 | excellent | aligned | exact_dataset_match | 1 | Shoda s benchmark datasetem i produkční relevancí. |
| nsoud-positive-02 | excellent | partially_aligned | alternate_relevant | 1 | Otázka je obecná; více ECLI může být právně relevantních. |
| nsoud-positive-03 | excellent | partially_aligned | alternate_relevant | 1 | Otázka je obecná; více ECLI může být právně relevantních. |
| nsoud-positive-04 | excellent | aligned | exact_dataset_match | 1 | Otázka je obecná; více ECLI může být právně relevantních. |
| nsoud-positive-05 | excellent | partially_aligned | alternate_relevant | 1 | Top-1 není v dataset scope, ale nalezená judikatura je tematicky blízká. |
| nsoud-positive-06 | excellent | aligned | exact_dataset_match | 1 | Shoda s benchmark datasetem i produkční relevancí. |
| nsoud-positive-07 | excellent | partially_aligned | alternate_relevant | 1 | Otázka je obecná; více ECLI může být právně relevantních. |
| nsoud-positive-08 | excellent | aligned | exact_dataset_match | 1 | Shoda s benchmark datasetem i produkční relevancí. |

## Klasifikace hitů (součty přes všechny top-k)

- `exact_dataset_match`: 12
- `alternate_relevant`: 26
- `irrelevant`: 1
- `uncertain`: 1

## Produční užitečnost (per case)

- `excellent`: 8
- `good`: 0
- `partial`: 0
- `poor`: 0

## Benchmark alignment (per case)

- `aligned`: 4
- `partially_aligned`: 4
- `misaligned`: 0

## Riziko obecných otázek

- `nsoud-positive-02` (high): dovolací důvod podle § 265b odst. 1 písm. g)
- `nsoud-positive-03` (high): dovolací důvod podle § 265b odst. 1 písm. h)
- `nsoud-positive-04` (high): dovolací důvod podle § 265b odst. 1 písm. m)
- `nsoud-positive-07` (high): náhradě nákladů dovolacího řízení

## Finální verdikt

**Ano** — BGE-M3 + BM25 hybrid je pro pilotní produkční retrieval připravený, za těchto podmínek:
- uživatel hledá podobnou judikaturu, ne jeden konkrétní ECLI z testu;
- u obecných otázek (§ dovolací důvody, náklady řízení) očekávejte více validních ECLI;
- doporučujeme doplnit LLM rerank / právní sumarizaci nad top-k chunky.
