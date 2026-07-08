# NSoud Winner Benchmark — 30-Question Audit + Corrected Retrieval Test

**Date:** 2026-07-08
**Scope:** Correction of a misdirected E2E test + corrected 30-question retrieval-only
run against the actual winning client long-form benchmark configuration.

---

## 1. What was originally wrong

The first test run sent all 30 client legal questions to the **live production FastAPI
service** (`POST /api/rag/query`, `POST /api/rag/retrieve` on `http://localhost:8029`,
the `nalus-scraper-api-1` container). That endpoint is wired (via
`app/api/startup.py` → `get_orchestrator()`) to whatever Qdrant collection
`QDRANT_COLLECTION_NAME` / the `nalus_live` alias resolves to at container startup —
**not** to the client long-form benchmark's winning configuration.

This was the wrong target for a "test the winning benchmark configuration" request,
and the run was stopped after the user flagged it.

## 2. Which collection/config the wrong test used

| Field | Value |
|---|---|
| Endpoint | `http://localhost:8029/api/rag/query`, `/api/rag/retrieve` (production API container) |
| Qdrant alias in use | `nalus_live` |
| Alias resolves to | `nalus_stable_20260326` (776 424 points) |
| Embedding model | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (dim 768) |
| Retrieval mode | Dense (Qdrant) fused with a substring `KeywordRetriever` — **not** BM25, **not** RRF |
| Corpus | Full historical **Ústavní soud** (Constitutional Court) NALUS archive (~75k decisions) |

## 3. Why that target was invalid for this evaluation

- The task is to evaluate the **client long-form benchmark's declared winner**,
  which was built and measured against a **Nejvyšší soud (NS / Supreme Court)**
  pilot corpus of **150 decisions** (981 civil + 866 criminal + 15 unlabeled chunks
  = 1862 chunks, verified directly in `artifacts/rag_eval/nalus_chunks.sqlite`,
  `source_id = 1`).
- `nalus_stable_20260326` / `nalus_live` is a completely different corpus
  (Ústavní soud, not Nejvyšší soud), a different embedding model (mpnet, not
  bge_m3), and a different fusion strategy (substring keyword match, not BM25 + RRF).
- Testing against production would have silently evaluated an unrelated system and
  produced a report that looked like it validated the benchmark winner while actually
  validating something else entirely.

## 4. Correct winning benchmark setup

Confirmed directly against the running Qdrant instance and the (unmodified) config
file `artifacts/rag_eval/client_longform_v1/configs/hybrid_bge_m3.yaml`:

| Field | Value |
|---|---|
| `winner_config_id` | `bge_m3__dense_plus_bm25` |
| Qdrant collection | `nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1` |
| Collection size | 1862 points, vector size 1024, cosine |
| Embedding model | `bge_m3` → `BAAI/bge-m3` (dim 1024) |
| Retrieval mode | `dense_plus_bm25` (RRF fusion, `rrf_k=60`, BM25 `k1=1.5`, `b=0.75`) |
| Source corpus | `source_id=1` in `artifacts/rag_eval/nalus_chunks.sqlite` → **150 distinct Nejvyšší soud (NS) decisions**, verified: 981 civil chunks, 866 criminal chunks, 15 unlabeled |
| Dataset | `artifacts/rag_eval/nalus_client_longform_eval_v1.json` (the original 8-case client long-form benchmark) |
| Documented benchmark result | hit_rate 0.875 (7/8), per `artifacts/rag_eval/client_longform_v1/failed_case_analysis.md` |

## 5. Is the winner wired into the API, or only into benchmark runner artifacts?

**Only into benchmark runner artifacts.** Confirmed by reading `app/api/rag_router.py`
and `app/api/startup.py`: the live orchestrator is built once at container startup
from `QDRANT_COLLECTION_NAME` (`nalus_live`) with the mpnet `SentenceTransformersEmbedder`.
There is no endpoint, flag, or config that can point the production API at
`nalus_client_lf__bge_m3__...`, at `bge_m3`, or at `dense_plus_bm25`/BM25 fusion.
The benchmark winner exists **only** as:
- a Qdrant collection populated by the benchmark run, and
- the `rag_eval` CLI / `SqlQdrantRagEvalBackend` (Python package
  `rag-embedding-benchmark`, mounted read-only at `/packages/rag-embedding-benchmark`
  and already `pip`-installed inside the `api` container from a previous benchmark run).

**Conclusion: the production API must not be used as a proxy for testing the winner.**
Confirmed and respected in the corrected run below.

## 6. How the corrected 30-question test was executed

- **Docker only, no local venv.** All embedding inference, BM25 indexing, and Qdrant
  search ran inside the already-running `nalus-scraper-api-1` container.
- A temporary, non-committed script,
  `.tmp/nsoud_winner_30q_probe.py` (`.tmp/` is git-ignored), was copied into the
  container (`docker compose cp`) and executed there
  (`docker compose exec api python /app/.tmp/nsoud_winner_30q_probe.py`).
- The script **imports and calls the existing, unmodified**
  `rag_eval.adapters.sql_qdrant.SqlQdrantRagEvalBackend.retrieve()` and
  `rag_eval.config.load_benchmark_config()` — the same backend class the benchmark
  runner (`run_benchmark.sh` → `rag-eval run --config hybrid_bge_m3.yaml`) uses
  internally. No retrieval, fusion, or ranking code was written, changed, or
  reimplemented; the script only supplies 30 new query strings to the existing
  `retrieve(model_code="bge_m3", collection_name="nalus_client_lf__bge_m3__...",
  retrieval_mode="dense_plus_bm25", top_k=8)` call.
- The config file `configs/hybrid_bge_m3.yaml` was read, not edited.
- No existing benchmark output (`ranking.json`, `report.md`, `winner_qa.*`,
  `combined_report.md`, `legal_quality_report.md`, `out_*/`) was touched or
  regenerated.
- No production code, retrieval logic, or clarification gate was changed.
- Nothing was committed.
- Post-processing (grouping/summarizing the JSON results into the tables below) was
  done with a local, pure-Python script that only reads already-fetched JSON — it
  performs no retrieval, embedding, or Qdrant access itself.
- **Verified zero contamination:** all 30×8 = 240 returned hits came exclusively
  from `qdrant_collection = nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1`,
  and all 240 `source_document_id` values start with `ECLI:CZ:NS:` (Nejvyšší soud).
  No hit from `nalus`, `nalus_live`, or `nalus_stable_20260326` is present anywhere
  in the corrected run.

## 7. Results — all 30 questions against the correct NSoud winner collection

Retrieval only (`top_k=8`, `dense_plus_bm25`, `bge_m3`). No answer-synthesis step was
run in this corrected pass — see §9 for why. Scores are RRF fusion scores
(`1/(rrf_k+rank)` summed across dense+BM25 rank lists, `rrf_k=60`), so they are
**small and close together by construction** (typically 0.015–0.033); rank order is
the meaningful signal, not absolute magnitude.

Legend for **relevance**: `High` = retrieved case law directly supports the doctrinal
question; `Moderate` = same general doctrinal family but not a precise match;
`Low` = off-topic or boilerplate noise; `Gap` = the 150-decision corpus structurally
has no matching case law for this question (explained per-row).

| # | Question (short) | Top ECLI (case no.) | Legal area | Relevance | Note |
|---|---|---|---|---|---|
| 1 | Obnova trestního řízení / nové důkazy | 11 Tdo 1114/2024 | criminal | Moderate | Top hit is about *ne bis in idem* / repeated conviction, not the obnova-řízení (retrial) doctrine itself; corpus likely has no dedicated obnova-řízení opinion |
| 2 | Soud se nevypořádal s argumentem | 27 Cdo 395/2024, 5 Tdo 318/2024 | civil/criminal | Moderate | Generic dovolání reasoning, not a precise "omitted argument" hit |
| 3 | Zatajení info při uzavření smlouvy, neplatnost | 29 Cdo 1793/2023, 33 Cdo 79/2024 | civil | Moderate–High | Weaker-party/consumer-protection and adhesion-contract case law is topically adjacent |
| 4 | Domovní prohlídka, nezákonné důkazy | 26 Cdo 2404/2024, 11 Tvo 22/2024 | civil/criminal | **Gap** | No hit addresses unlawful search / exclusion of evidence from a search; corpus gap |
| 5 | Odůvodnění správního rozhodnutí | mostly `takto: Dovolání se odmítá.` boilerplate | civil/criminal | **Low (noise)** | Short operative-part chunks with no substantive content dominate; also a **Gap** — no administrative-law (NSS) case exists in this NS-only corpus |
| 6 | Prominutí/navrácení lhůty (hospitalizace) | 21 Cdo 372/2024 | civil | Low–Moderate | About a shortened court-fee deadline, not excusable-neglect restitutio in integrum |
| 7 | Jediný svědek jako jediný důkaz | 11 Tdo 1127/2024 | criminal | Moderate–High | Clearly on the "sole witness testimony" evaluation theme |
| 8 | Rozpory v judikatuře | 4 Tdo 1044/2024, 4 Tdo 1100/2024 | criminal | Low–Moderate | Generic dovolání grounds, no direct "conflicting case law" doctrine hit |
| 9 | Nečinnost správního úřadu | 27 Cdo 395/2024, 29 Nd 3/2025 | civil | **Gap** | This is an administrative-court (NSS) topic; corpus contains only NS civil/criminal decisions |
| 10 | Obnova řízení — nové listinné důkazy | 23 Cdo 3170/2024 (house-defects case, unrelated) | civil/criminal | **Gap** | Same corpus gap as Q1 |
| 11 | Písařská chyba vs. důvod k odvolání | 4 Tdo 1056/2024, 8 Tdo 1022/2024 | criminal | Moderate | Plausible lexical/semantic match on clerical-error language |
| 12 | Vyloučení soudce pro poměr k věci | 11 Tvo 22/2024 (all 6 of top hits) | criminal | **High** | Single decision is specifically about judge/authority exclusion (§ 30 tr. ř.) — precise match |
| 13 | Nezákonně získaný důkaz | 4 Tdo 1044/2024, 4 Tdo 1056/2024 | criminal | High | Direct hits on "procesně nepoužitelné důkazy" doctrine |
| 14 | Správní orgán ignoroval znalecký posudek | 6 Tdo 936/2024, 6 Tdo 991/2024 | criminal | Moderate (wrong branch) | Expert-opinion-weighing doctrine present, but in criminal not administrative proceedings — **partial Gap** |
| 15 | Odvolací soud nevypořádal s námitkami | 21 Cdo 1566/2024, 3 Tdo 980/2024 | civil/criminal | Low–Moderate | Generic dovolání hits, no precise match |
| 16 | Náležitosti odůvodnění rozhodnutí | 28 Cdo 1880/2024, 29 Nd 3/2025 | civil/criminal | Moderate | On the general reasoning-adequacy theme |
| 17 | Odmítnutí znaleckého posudku jen z nesouhlasu | 21 Cdo 44/2025 | civil | Moderate–High | Plausible direct match |
| 18 | Více výkladů zákona, správný výklad | 28 Cdo 3513/2024, 11 Tdo 1114/2024 | civil/criminal | Low–Moderate | Generic, no precise interpretive-methodology hit |
| 19 | Interní metodika úřadu jako právní podklad | 6 Tdo 827/2024, 26 Cdo 125/2024 | civil/criminal | **Gap** | Administrative-law topic; corpus has no NSS case law |
| 20 | Procesní chyba i při správných skutkových zjištěních | 8 Tdo 1085/2024 | criminal | Moderate | Plausible dovolací-důvody overlap |
| 21 | Důkazy předložené před jednáním, soud je nezmínil | 4 Tdo 1056/2024, 29 Cdo 1793/2023 | civil/criminal | Low–Moderate | Same family as Q2 |
| 22 | Soud opsal ustanovení bez vztahu k případu | 8 Tdo 760/2024, 6 Tdo 936/2024 | criminal | Low–Moderate | Generic reasoning-adequacy hits |
| 23 | Zrušení rozhodnutí jen pro nedostatečné odůvodnění | 24 Cdo 3585/2024 | civil | Moderate–High | Notably higher fusion score (0.0301); plausible direct hit |
| 24 | Rozdíl skutkové zjištění vs. právní posouzení | 5 Tdo 1128/2024, 20 Cdo 3061/2024 | criminal/civil | Moderate | Corpus is built around exactly these § 265b/§237 distinctions, so conceptually adjacent |
| 25 | Rozhodnutí hlavně na jednom posudku, ostatní důkazy odporují | 4 Tdo 1044/2024, 6 Tdo 1113/2024 | criminal | Moderate | Plausible overlap with evidence-weighing doctrine |
| 26 | Správní orgán nevypořádal se všemi argumenty | 28 Cdo 2866/2024, 29 Nd 3/2025 | civil | **Gap** | Same administrative-law corpus gap as Q9/Q19 |
| 27 | Rozporné výpovědi svědků | 11 Tdo 1127/2024, 8 Tdo 1085/2024, 11 Tdo 765/2024 | criminal | **High** | Three top hits, clearly on-theme, noticeably higher scores (0.0296–0.0318) |
| 28 | Judikatura, kterou soud nezmínil | 23 Cdo 271/2024, 26 Cdo 2198/2024 | civil | High | `appeal_instruction` chunks about § 237 o. s. ř. admissibility grounds tied to departing from settled case law — precise doctrinal match |
| 29 | Rozhodnutí nevysvětluje odmítnutí důkazů (spravedlivý proces) | 4 Tdo 1044/2024, 4 Tdo 1137/2024 | criminal | High | Higher scores (0.0310–0.0315); matches the corpus's dominant "opomenuté důkazy" theme |
| 30 | Komplexní víceotázková analýza | 27 Cdo 2295/2023 (then generic mix) | civil/criminal | Low–Moderate | Expected: broad multi-issue prompts are inherently hard for single-vector dense+BM25 retrieval regardless of embedding model |

**Any result from ÚS / production alias by mistake?** No — verified programmatically:
0 of 240 hits came from any collection other than the winner collection; 0 of 240
`source_document_id` values are non-`ECLI:CZ:NS:`.

## 8. Failures / ambiguous results

- **Structural corpus gap (administrative law):** Q5, Q9, Q14, Q19, Q26 all touch
  správní-orgán / administrative-decision reasoning. The 150-decision pilot corpus is
  **civil + criminal Nejvyšší soud only** — it contains no Nejvyšší správní soud (NSS)
  case law at all. This is not a retrieval bug; it is a **dataset scope limitation**
  the winning benchmark was never built to cover. Any client question phrased around
  "správní orgán" will structurally underperform on this collection no matter how
  good the embedding/fusion choice is.
- **Structural corpus gap (obnova řízení):** Q1 and Q10 both target the "new
  evidence → retrial" doctrine; neither surfaced a clearly on-point decision, again
  most plausibly because the 150-decision sample doesn't happen to contain one.
- **Boilerplate noise:** Q5's top hits are largely one-line `takto: Dovolání se
  odmítá.` operative-part chunks — short, low-information chunks that apparently
  score deceptively well under RRF for generic/vague queries. This is a chunk-quality/
  section-type-filtering observation, not something this audit changed.
- **Low, tightly-clustered RRF scores:** across all 30 questions, scores mostly sit in
  a narrow 0.015–0.033 band. This makes it hard to build a reliable score-threshold
  cutoff (e.g., for "insufficient support" style logic) on top of this fusion — rank
  order is informative, absolute score is not. This is a characteristic of RRF with
  `rrf_k=60` on a small (150-doc) corpus, not an implementation defect.
- **No per-question ground truth for these 30 questions.** Unlike the original
  8-case client long-form benchmark (which has expected ECLI/marker labels), these
  30 questions are new ad hoc client-style questions with no expected-answer labels.
  The relevance grades above are a qualitative reviewer judgment based on reading the
  retrieved text, not a hit_rate/marker-coverage metric like the original benchmark
  used. Treat them as directional, not a formal score.

## 9. Final verdict

**Is `bge_m3__dense_plus_bm25` over the NSoud benchmark collection still valid for
long-form legal retrieval? Yes, conditionally.**

Evidence:
- On questions squarely inside the corpus's actual coverage (judge exclusion/bias —
  Q12; witness-credibility evaluation — Q7, Q27; unlawful/inadmissible evidence —
  Q13; departing-from-settled-case-law admissibility grounds — Q28; unreasoned
  rejection of evidence / right to fair trial — Q29; reasoning-adequacy — Q23), the
  winning configuration reliably surfaces topically correct Nejvyšší soud case law,
  consistent with the previously documented 0.875 hit_rate on the original 8-case
  benchmark.
- On questions outside the corpus's actual coverage — anything administrative-law
  flavored (Q5, Q9, Q14, Q19, Q26) or targeting obnova řízení (Q1, Q10) — the
  configuration cannot succeed regardless of embedding/fusion quality, because the
  underlying 150-decision sample does not contain matching case law. This is a
  **dataset-scope limitation**, not a reason to distrust `bge_m3__dense_plus_bm25`
  as a retrieval configuration.
- No configuration change is recommended from this audit. The only actionable
  follow-up is **corpus/dataset scope**: if client questions routinely touch
  administrative-law or obnova-řízení fact patterns, the 150-decision pilot corpus
  needs those decisions added before re-benchmarking — this is a dataset change, not
  a retrieval-logic change, and is explicitly out of scope for this audit.

---

## Answers to the required audit questions (A–J)

**A) What Qdrant collection was the wrong test using?**
`nalus_live` (alias) → `nalus_stable_20260326` (776 424 points).

**B) What embedding model was the wrong test using?**
`sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (dim 768).

**C) Was the wrong test hitting production API defaults?**
Yes — `POST /api/rag/query` and `/api/rag/retrieve` on the live `nalus-scraper-api-1`
container, using `get_orchestrator()`'s startup-wired live orchestrator.

**D) Was it using Ústavní soud / production archive instead of NSoud benchmark data?**
Yes — `nalus_stable_20260326` is the full historical Ústavní soud (NALUS) archive,
not the Nejvyšší soud client long-form benchmark corpus.

**E) What Qdrant collection contains the winning benchmark setup?**
`nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1`
(1862 points, 1024-dim, cosine).

**F) What embedding model does the winning setup use?**
`bge_m3` (`BAAI/bge-m3`, dim 1024).

**G) What retrieval mode does the winning setup use?**
`dense_plus_bm25` — dense (bge_m3) + BM25 (`bm25s`, `k1=1.5`, `b=0.75`), fused with
Reciprocal Rank Fusion (`rrf_k=60`).

**H) Which script or endpoint can actually query the winning benchmark setup directly?**
The `rag_eval` package's `SqlQdrantRagEvalBackend.retrieve()`
(`rag_eval/adapters/sql_qdrant.py`), driven either by the `rag-eval` CLI
(`artifacts/rag_eval/client_longform_v1/run_benchmark.sh`) against the packaged
8-case dataset, or — as done here — by directly importing and calling that same
backend class with new ad hoc queries. There is no HTTP endpoint for it.

**I) Is the winning setup wired into the API, or does it exist only inside benchmark runner artifacts?**
Only inside benchmark runner artifacts (Qdrant collection + `rag_eval` package +
YAML configs). Not reachable through `app/api/*` at all.

**J) Confirm that production API must NOT be used as a proxy for testing the winner.**
Confirmed. The corrected 30-question run in §6–§7 bypassed the production API
entirely and called the benchmark's own retrieval backend directly, inside Docker.
