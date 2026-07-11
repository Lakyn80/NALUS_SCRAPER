# NSoud Strict Direct Audit

- Generated at: `2026-07-11 12:40:21 Europe/Moscow`
- Scope: audit-only; no retrieval, evaluator, dashboard, Qdrant, or sidecar behavior changed.

## Executive Summary

- `nsoud_sidecar_provenance_repaired` improved citation availability and usable support versus `nsoud_no_llm_baseline`, but `strict_direct_pass_rate_gold` stayed at `0.0`.
- No dashboard metric mapping bug was found. Prometheus labels and corpus labels are consistent with the underlying `summary.json` artifacts.
- The remaining strict-direct weakness is mostly a benchmark/evaluator alignment problem, plus one real same-document wrong-chunk case (`nsoud-qa-007`).

## Current Dashboard Interpretation

| Run | Corpus | Generated At | Gold | Strict Direct Gold | Usable Support Gold | Citation Available | Unsupported Risk Count |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `nsoud_sidecar_provenance_repaired` | `nsoud` | `2026-07-11T06:49:40Z` | 4 | 0.00 | 0.75 | 0.75 | 1 |
| `nsoud_no_llm_baseline` | `nsoud` | `2026-07-10T12:20:50Z` | 4 | 0.00 | 0.50 | 0.50 | 2 |
| `usoud_no_llm_baseline` | `usoud` | `2026-07-10T11:51:31Z` | 10 | 0.10 | 1.00 | 1.00 | 0 |
| `mixed_no_llm_baseline` | `mixed` | `2026-07-10T11:51:32Z` | 8 | 0.00 | 1.00 | 0.00 | 0 |

Key delta for NSoud candidate: `citation_available_rate_gold` moved `0.50 -> 0.75`, `usable_support_rate_gold` moved `0.50 -> 0.75`, `unsupported_answer_risk_count` moved `2 -> 1`, but `strict_direct_pass_rate_gold` remained `0.00 -> 0.00`.

## Dashboard Validation

- Export path: `app/observability/eval_metrics_exporter.py` reads `artifacts/rag_eval/legal_qa/answer_eval/*` and prefers `summary.json` over `metrics.json`.
- Metric labels: `run_name`, `corpus`.
- Relevant NSoud repaired summary timestamp is newer than the NSoud baseline (`2026-07-11T06:49:40Z` vs `2026-07-10T12:20:50Z`).
- No accidental NSoud/USoud/mixed corpus mixing was found for the named runs.
- Operational note: the exporter exposes every run directory; the dashboard is comparing named runs, not an automatic `latest` alias.

## Per-Question Diagnostic Table

| Question | Baseline | Repaired | Citation Change | Strict Direct | Root Cause |
| --- | --- | --- | --- | --- | --- |
| `nsoud-qa-003` | `partial` / citation `true` | `partial` / citation `true` / gold rank `1` | `false` | still `false` | `evaluator_too_strict_or_gold_mismatch` |
| `nsoud-qa-004` | `partial` / citation `true` | `partial` / citation `true` / gold rank `1` | `false` | still `false` | `evaluator_too_strict_or_gold_mismatch` |
| `nsoud-qa-007` | `gap` / citation `false` | `partial` / citation `true` / gold rank `1` | `true` | still `false` | `retrieval_right_document_wrong_chunk` |
| `nsoud-qa-010` | `boilerplate_noise` / citation `false` | `boilerplate_noise` / citation `false` / gold rank `9` | `false` | still `false` | `evaluator_too_strict_or_gold_mismatch` |

### nsoud-qa-003

Question: Kdy je dovolání přípustné ve věcech občanskoprávních?
Expected gold: `ECLI:CZ:NS:2025:21.CDO.372.2024.1`
Expected keywords: `přípustnost, dovolání, občanské`
Baseline: support `partial`, citation `True`, unsupported risk `False`.
Repaired: support `partial`, citation `True`, gold hit rank `1`, unsupported risk `False`.
Root cause: `evaluator_too_strict_or_gold_mismatch`
Conclusion: The repaired run already retrieves the gold document and the best same-document chunk at rank 1. Strict direct still fails because the evaluator uses exact substring keyword coverage with a >=0.67 gate; the best chunk reaches 2/3 = 0.6667, so lexical form mismatch keeps it in partial.
Recommended action: Keep retrieval unchanged; refine expected answer markers or keyword matching semantics before treating this item as a retrieval regression.
Best same-document chunk: `1214` at `chunk_index=5` with keyword coverage `0.6667` and repaired top-10 rank `1`.

| Repaired Rank | Chunk | Document | Section | Snippet |
| ---: | --- | --- | --- | --- |
| 1 | `1214` | `ECLI:CZ:NS:2025:21.CDO.372.2024.1` | `reasoning` | 6. Nejvyšší soud jako soud dovolací [§ 10a zákona č. 99/1963 Sb., občanský soudní řád, ve znění pozdějších předpisů (dále jen „o. s. ř.“)] p |
| 2 | `923` | `ECLI:CZ:NS:2025:3.TDO.1120.2024.1` | `None` | 1. 2024 do zaplacení. Se zbytkem uplatněného nároku na náhradu nemajetkové újmy byla poškozená podle § 229 odst. 2 tr. ř. odkázána na řízení |
| 3 | `46` | `ECLI:CZ:NS:2024:20.CDO.2839.2024.1` | `reasoning` | 99/1963 Sb., občanský soudní řád, ve znění účinném od 1. ledna 2022 (srov. část první čl. II bod 1 zákona č. 286/2021 Sb.), a dospěl k závěr |
| 4 | `612` | `ECLI:CZ:NS:2024:11.TDO.679.2024.1` | `None` | 10. 2021 do 31. 10. 2021. 2. Za uvedený přečin soud prvního stupně obviněnému uložil podle § 146 odst. 1 tr. zákoníku za použití § 67 odst.  |
| 5 | `1114` | `ECLI:CZ:NS:2025:4.TDO.1056.2024.1` | `None` | 3. Zároveň podle § 228 odst. 1 tr. ř. nalézací soud obviněné uložil povinnost nahradit poškozenému J. J., nemajetkovou újmu ve výši 5 000 Kč |

### nsoud-qa-004

Question: Jak Nejvyšší soud hodnotí zásadní právní význam v civilním dovolání?
Expected gold: `ECLI:CZ:NS:2024:8.TDO.760.2024.1`
Expected keywords: `zásadní, právní, význam`
Baseline: support `partial`, citation `True`, unsupported risk `False`.
Repaired: support `partial`, citation `True`, gold hit rank `1`, unsupported risk `False`.
Root cause: `evaluator_too_strict_or_gold_mismatch`
Conclusion: The benchmark question asks about civil dovolání and zásadní právní význam, but the annotated gold source is a criminal 8 Tdo case. Even though the retrieved rank-1 document matches the stored gold ECLI, the content does not cleanly answer the stated benchmark question.
Recommended action: Replace or re-annotate this item with a civil Cdo decision, or reformulate the question to the criminal doctrine actually represented by the gold source.
Best same-document chunk: `345` at `chunk_index=11` with keyword coverage `0.6667` and repaired top-10 rank `None`.

| Repaired Rank | Chunk | Document | Section | Snippet |
| ---: | --- | --- | --- | --- |
| 1 | `346` | `ECLI:CZ:NS:2024:8.TDO.760.2024.1` | `reasoning` | 21. Pro posouzení správnosti právních otázek ve smyslu uvedeného dovolacího důvodu je zásadně rozhodný skutkový stav zjištěný soudy prvního, |
| 2 | `910` | `ECLI:CZ:NS:2024:3.TDO.980.2024.1` | `None` | 45. Obviněný v dovolání výslovně uplatnil první variantu citovaného dovolacího důvodu, a to, že rozhodná skutková zjištění jsou ve zjevném ( |
| 3 | `1031` | `ECLI:CZ:NS:2024:8.TDO.1022.2024.1` | `None` | 16. V souvislosti s argumentací obviněného, která nikterak nevybočuje z prosté polemiky s hodnocením důkazů, jak bylo provedeno soudy nižšíc |
| 4 | `1814` | `ECLI:CZ:NS:2025:11.TDO.75.2025.1` | `reasoning` | 9. 2004, sp. zn. II. ÚS 279/03). 32. Nadto Nejvyšší soud i při respektování shora uvedeného interpretuje a aplikuje podmínky připuštění dovo |
| 5 | `6` | `ECLI:CZ:NS:2024:24.CDO.671.2024.1` | `reasoning` | 8. Nejvyšší soud jako soud dovolací (§ 10a zákona č. 99/1963 Sb., občanský soudní řád, ve znění pozdějších předpisů – dále jen „o. s. ř.“) p |

### nsoud-qa-007

Question: Jak Nejvyšší soud posuzuje dovolací důvod podle § 265b tr. ř.?
Expected gold: `ECLI:CZ:NS:2025:5.TDO.1086.2024.1`
Expected keywords: `265b, trestní, dovolání`
Baseline: support `gap`, citation `False`, unsupported risk `True`.
Repaired: support `partial`, citation `True`, gold hit rank `1`, unsupported risk `False`.
Root cause: `retrieval_right_document_wrong_chunk`
Conclusion: Provenance repair fixes the old false gap: the repaired run now cites the gold document at rank 1. Strict direct still fails because rank-1 chunk 735 is only the case-specific closing summary, while same-document chunks 728/732/733 carry the more doctrinal § 265b discussion and do not appear in top 10.
Recommended action: Keep the repaired provenance path; next investigate within-document chunk selection or query reformulation for this benchmark item before touching global scoring.
Best same-document chunk: `728` at `chunk_index=2` with keyword coverage `1.0000` and repaired top-10 rank `None`.

| Repaired Rank | Chunk | Document | Section | Snippet |
| ---: | --- | --- | --- | --- |
| 1 | `735` | `ECLI:CZ:NS:2025:5.TDO.1086.2024.1` | `None` | IV. Závěrečné shrnutí 9. Lze tak shrnout, že obviněný uplatnil jediný dovolací důvod uvedený v § 265b odst. 1 písm. g) tr. ř., avšak ve zněn |
| 2 | `131` | `ECLI:CZ:NS:2024:11.TDO.765.2024.1` | `reasoning` | 14. Protože platí, že dovolání lze podat jen z některého z důvodů taxativně vymezených v § 265b tr. řádu, musel dále Nejvyšší soud posoudit, |
| 3 | `991` | `ECLI:CZ:NS:2025:4.TDO.1137.2024.1` | `None` | 23. Obviněný P. D., ve svém dovolání uplatnil taktéž dovolací důvod podle ustanovení § 265b odst. 1 písm. h) tr. ř. Podle tohoto dovolacího  |
| 4 | `1789` | `ECLI:CZ:NS:2025:3.TDO.53.2025.1` | `reasoning` | 13. Státní zástupkyně proto navrhuje, aby Nejvyšší soud dovolání obviněného odmítl podle § 265i odst. 1 písm. e) tr. ř. jako zjevně neopodst |
| 5 | `1834` | `ECLI:CZ:NS:2025:11.TDO.75.2025.1` | `None` | 55. Nad rámec výše uvedeného Nejvyšší soud ve shodě s judikaturou Ústavního soudu (srov. nález Ústavního soudu ze dne 4. 5. 2006, sp. zn. I. |

### nsoud-qa-010

Question: Jaký je rozdíl mezi odmítnutím a zamítnutím dovolání?
Expected gold: `ECLI:CZ:NS:2025:29.NSCR.1.2025.1`
Expected keywords: `odmítnutí, zamítnutí, dovolání`
Baseline: support `boilerplate_noise`, citation `False`, unsupported risk `True`.
Repaired: support `boilerplate_noise`, citation `False`, gold hit rank `9`, unsupported risk `True`.
Root cause: `evaluator_too_strict_or_gold_mismatch`
Conclusion: This item is not fixed by provenance. The repaired run still retrieves mostly operative-part boilerplate, and the gold document itself does not contain a clean comparison between odmítnutí and zamítnutí. Across all chunks in the gold document, the best keyword coverage is only 1/3.
Recommended action: Reformulate or replace the benchmark item; if kept, annotate a genuinely comparative doctrinal source or a multi-source gold instead of a single odmítnutí case.
Best same-document chunk: `1642` at `chunk_index=0` with keyword coverage `0.3333` and repaired top-10 rank `None`.

| Repaired Rank | Chunk | Document | Section | Snippet |
| ---: | --- | --- | --- | --- |
| 1 | `26` | `ECLI:CZ:NS:2024:26.CDO.125.2024.1` | `operative_part` | takto: Dovolání se odmítá . |
| 2 | `884` | `ECLI:CZ:NS:2025:29.ICDO.3.2025.1` | `None` | 182/2006 Sb., o úpadku a způsobech jeho řešení (insolvenčního zákona), podle něhož nemá ve sporu o pravost, výši nebo pořadí přihlášených po |
| 3 | `497` | `ECLI:CZ:NS:2025:27.CDO.1921.2024.1` | `operative_part` | takto: Dovolání se odmítá . |
| 4 | `632` | `ECLI:CZ:NS:2024:25.CDO.3217.2023.1` | `None` | 1. 2023, č. j. 25 Co 288,289/2022-132, zastavil odvolací řízení ve vztahu mezi žalobkyní a 1. žalovanou a rozhodl o náhradě nákladů odvolací |
| 5 | `1427` | `ECLI:CZ:NS:2025:5.TDO.1071.2024.1` | `None` | IV. Posouzení důvodnosti dovolání a) Obecná východiska 15. Nejvyšší soud zjistil, že byly splněny všechny formální podmínky k podání dovolán |

## Root-Cause Breakdown

- `evaluator_too_strict_or_gold_mismatch`: 3
- `retrieval_right_document_wrong_chunk`: 1

Top patterns:
- `strict_threshold_edge_case`: 1
- `lexical_keyword_mismatch`: 1
- `question_gold_domain_mismatch`: 1
- `same_document_chunk_selection`: 1
- `operative_part_boilerplate`: 1
- `single_source_gold_too_weak`: 1

## Exact Comparison: Repaired vs Baseline

- `nsoud-qa-003`: no support-class change; provenance was already good enough, so strict direct remains blocked by lexical/threshold strictness.
- `nsoud-qa-004`: no support-class change; repaired provenance did not matter because the benchmark item itself is not cleanly aligned with its annotated gold.
- `nsoud-qa-007`: `gap -> partial`, citation `false -> true`, unsupported risk `true -> false`; repaired provenance fixed the false retrieval gap, but not the wrong-chunk/directness problem.
- `nsoud-qa-010`: still `boilerplate_noise`; repaired provenance changed the identified gold rank but did not add substantive evidence or a comparative source.

## Recommended Next Engineering Actions

- Re-annotate or replace nsoud-qa-004 and nsoud-qa-010, because their gold/question alignment is not strong enough for strict-direct benchmarking.
- For nsoud-qa-007, investigate within-document chunk selection or question reformulation before changing any global retrieval score weights.
- If strict direct remains a KPI, revisit evaluator keyword semantics for morphological variants and threshold edges, starting with nsoud-qa-003.

## Files Read

- `PROJECT_EXECUTION_PROTOCOL.md`
- `PROJECT_PROGRESS.md`
- `app/rag/eval/legal_answer_eval.py`
- `app/observability/eval_metrics_exporter.py`
- `monitoring/grafana/dashboards/legal_answer_eval_dashboard.json`
- `monitoring/prometheus/prometheus.yml`
- `artifacts/rag_eval/legal_qa/datasets/nsoud_qa_v1.jsonl`
- `artifacts/rag_eval/legal_qa/gold_source_review_20260709.md`
- `artifacts/rag_eval/legal_qa/answer_eval/nsoud_no_llm_baseline/summary.json`
- `artifacts/rag_eval/legal_qa/answer_eval/nsoud_no_llm_baseline/metrics.json`
- `artifacts/rag_eval/legal_qa/answer_eval/nsoud_no_llm_baseline/answer_eval_results.jsonl`
- `artifacts/rag_eval/legal_qa/answer_eval/nsoud_sidecar_provenance_repaired/summary.json`
- `artifacts/rag_eval/legal_qa/answer_eval/nsoud_sidecar_provenance_repaired/metrics.json`
- `artifacts/rag_eval/legal_qa/answer_eval/nsoud_sidecar_provenance_repaired/answer_eval_results.jsonl`
- `artifacts/rag_eval/legal_qa/answer_eval/nsoud_sidecar_provenance_repaired/failed_cases_report.json`
- `artifacts/rag_eval/legal_qa/runs/nsoud_full_baseline/retrieval_results.jsonl`
- `artifacts/rag_eval/legal_qa/runs/nsoud_sidecar_provenance_repaired/retrieval_results.jsonl`
- `storage/rag/bm25/nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1.sqlite`
- `storage/rag/bm25/nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1.provenance_repaired.sqlite`

## Commands Run

- `git branch --show-current`
- `git status --short`
- `rg -n "strict_direct_pass|support_level|support_keyword_coverage|boilerplate|expected_keywords|gold_hit_rank|citation_available" app/rag/eval/legal_answer_eval.py`
- `python -m pytest tests/rag/test_legal_answer_eval.py tests/rag/test_legal_answer_eval_diagnostics.py tests/observability/test_eval_metrics_exporter.py tests/test_repair_nsoud_bm25_sidecar_provenance.py -q`
- `python analysis snippets over answer_eval_results.jsonl, retrieval_results.jsonl, summary.json, and repaired/original BM25 sidecars`

## Change/Test Status

- Code changed: `false`
- Behavior changed: `false`
- Tests: `python -m pytest tests/rag/test_legal_answer_eval.py tests/rag/test_legal_answer_eval_diagnostics.py tests/observability/test_eval_metrics_exporter.py tests/test_repair_nsoud_bm25_sidecar_provenance.py -q` -> `passed` (`39` passed)
- Warnings: non-blocking `pytest-asyncio` fixture loop scope deprecation warning.
