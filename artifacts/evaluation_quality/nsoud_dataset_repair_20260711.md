# NSoud Dataset Repair — 2026-07-11

- Generated: `2026-07-12 00:45:45 Europe/Moscow`
- Candidate run: `nsoud_dataset_repaired`
- Collection: `nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1`
- Repaired sidecar: `storage/rag/bm25/nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1.provenance_repaired.sqlite`
- Qdrant access: read-only search
- Retrieval/embedding/BM25/RRF changes: none

## Executive Summary

The repair changes only four audited NSoud benchmark items. It removes one criminal/civil gold mismatch, replaces one unsupported comparative question with a narrower question that the existing gold actually answers, corrects one inflection-specific keyword, and records stronger same-document doctrine for `nsoud-qa-007`. No chunk-id constraint was invented because the current dataset schema does not support one.

The candidate is more accurate as benchmark data but does not manufacture a metric improvement. `nsoud-qa-010` remains a safe reported gap because the correct gold chunk is rank 4 and its 240-character exported snippet ends before the doctrinal passage.

## Per-Question Decisions

### `nsoud-qa-003` — `evaluator_followup_needed`

- Original/final question: `Kdy je dovolání přípustné ve věcech občanskoprávních?`
- Original/final gold: `ECLI:CZ:NS:2025:21.CDO.372.2024.1`
- Exact missing original keyword: `občanské`.
- Classification: inflection/morphological variant of source text `občanský soudní řád`.
- Applied repair: expected keywords changed to `přípustnost`, `dovolání`, `občanský`.
- Current rank 1: chunk `1214`, index `5`, expected gold.
- Same-document review: all `13` chunks; chunk `1214` is strongest and has full-chunk coverage `3/3` after repair.
- Remaining limitation: the exported snippet exposes only `2/3 = 0.6667`; the global `>= 0.67` gate was not lowered. Deterministic morphology/evidence-window handling remains an evaluator follow-up.

### `nsoud-qa-004` — `safe_gold_reannotation`

- Original question: `Jak Nejvyšší soud hodnotí zásadní právní význam v civilním dovolání?`
- Original gold: `ECLI:CZ:NS:2024:8.TDO.760.2024.1` (criminal and misaligned).
- Final question: `Jaké právní otázky mohou podle § 237 o. s. ř. založit přípustnost dovolání?`
- Final gold: `ECLI:CZ:NS:2025:33.CDO.79.2024.1`.
- Current rank 1: chunk `1000`, index `5`, final gold.
- Same-document review: all `16` chunks; chunk `1000` expressly states the § 237 criteria and covers all final keywords in the full chunk.
- Final keywords: `hmotného`, `procesního`, `ustálené rozhodovací praxe`.
- Reason: the repair preserves a civil benchmark item and removes the wrong criminal gold instead of changing the question to fit it.
- Remaining limitation: the fixed exported snippet ends before the markers, so no-LLM answer support is conservatively `partial`.

### `nsoud-qa-007` — `safe_same_document_chunk_refinement`

- Question: `Jak Nejvyšší soud posuzuje dovolací důvod podle § 265b tr. ř.?`
- Gold retained: `ECLI:CZ:NS:2025:5.TDO.1086.2024.1`.
- Current rank 1: chunk `735`, index `9`, a weaker closing summary; keyword coverage `1/3` in answer eval.
- Stronger same-document chunks:
  - `728`, index `2`: audit keyword coverage `3/3`.
  - `732`, index `6`: explains the 2022 amendment and § 265b(1)(g)/(h).
  - `733`, index `7`: explains why concrete objections must fit a statutory dovolací důvod.
- Applied repair: replaced the tautological expected answer with three doctrinal points from chunks `732–733`; gold and retrieval query remain unchanged.
- Remaining limitation: current schema has no supported expected-chunk constraint. A future within-document selection change may be evaluated separately without changing global scoring.

### `nsoud-qa-010` — `safe_question_reformulation`

- Original question: `Jaký je rozdíl mezi odmítnutím a zamítnutím dovolání?`
- Original/final gold: `ECLI:CZ:NS:2025:29.NSCR.1.2025.1`.
- Final question: `Je dovolání přípustné proti rozhodnutí, jímž odvolací soud odmítl odvolání?`
- Final answer points: objective inadmissibility under § 238(1)(e) o. s. ř.; žaloba pro zmatečnost under § 229(4) o. s. ř.
- Current gold hit: rank `4`, chunk `1644`, index `2`.
- Same-document review: all `7` chunks; chunk `1643` is operative boilerplate, while chunk `1644` explicitly supports both final answer points.
- Final keywords: `odmítl odvolání`, `objektivně přípustné`, `žaloba pro zmatečnost`.
- Remaining limitation: the exported chunk `1644` snippet ends before the doctrinal sentences, so answer eval remains `gap`, citation unavailable, and unsupported risk `1`. The result is reported rather than hidden.

## Current Top-Hit Summary

| Question | Rank 1 | Gold rank | Result |
| --- | --- | ---: | --- |
| `nsoud-qa-003` | `1214` / `ECLI:CZ:NS:2025:21.CDO.372.2024.1` | 1 | partial, citation available |
| `nsoud-qa-004` | `1000` / `ECLI:CZ:NS:2025:33.CDO.79.2024.1` | 1 | partial, citation available |
| `nsoud-qa-007` | `735` / `ECLI:CZ:NS:2025:5.TDO.1086.2024.1` | 1 | partial, citation available |
| `nsoud-qa-010` | `1014` / `ECLI:CZ:NS:2025:29.ICDO.172.2024.1` | 4 | gap, citation unavailable |

## Metrics Comparison

| Metric | `nsoud_sidecar_provenance_repaired` | `nsoud_dataset_repaired` |
| --- | ---: | ---: |
| Gold | 4 | 4 |
| Direct support | 0 | 0 |
| Partial support | 3 | 3 |
| Gap | 0 | 1 |
| Boilerplate noise | 1 | 0 |
| Citation available rate | 0.75 | 0.75 |
| Usable support rate gold | 0.75 | 0.75 |
| Unsupported risk count | 1 | 1 |
| Strict direct pass rate gold | 0.0 | 0.0 |

Retrieval candidate metrics: `pass_rate=0.9`, `source_hit@1=0.75`, `source_hit@3=0.75`, `source_hit@5=1.0`, `mean_source_constraint_match=1.0`. The one retrieval-benchmark failure is the conservatively retained `nsoud-qa-010` evidence-window limitation.

## Safety and Scope

- BGE-M3 loaded from the existing local cache; no model download occurred.
- Qdrant was queried only; no write, ingest, rebuild, alias switch, or protected collection access occurred.
- Redis was not enabled or used.
- No LLM or DeepSeek call occurred.
- Dense scoring, BM25 scoring, RRF, global `top_k`, embeddings, cache behavior, exporter queries, and Grafana queries were not changed.
- No ECLI, case number, decision date, source id, or legal conclusion was invented.

## Follow-Up

Evaluate deterministic evidence-window handling for gold chunks whose relevant doctrine falls beyond the 240-character exported snippet, starting with `nsoud-qa-003`, `004`, and `010`. Treat this as a separate evaluator/retrieval-artifact task; do not lower the global strict threshold as a shortcut.
