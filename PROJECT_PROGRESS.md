# Project Progress

## 2026-08-13 Europe/Moscow — Task: Wire FAST / BALANCED / PRECISE profiles

- Product tiers pinned in `retrieval_profiles.py` + async Stage1 search/API:
  - **FAST** (`fast`) = A hybrid
  - **BALANCED** (`balanced`) = B + ColBERT (master-allow
    `NALUS_LEGAL_V2_COLBERT_ENABLED=1`)
  - **PRECISE** (`precise`, alias `ce7`) = B + CE-7 (master-allow
    `NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED=1`)
- `search_case_similarity_stage1` is async-first (`to_thread` for hybrid/CE;
  ColBERT via existing async backend). API endpoint awaits it.
- `retrieval_stage` provenance: `hybrid_rrf_stage_1` | `hybrid_rrf_colbert` |
  `hybrid_rrf_ce7`.
- Default request profile remains `fast`. No auto-enable of ColBERT/CE.
- Latency commit: `1eb5c20`.

## 2026-08-13 Europe/Moscow — Task: Latency/cost tier benchmark (FAST / B+ColBERT / CE)

- Runner: `scripts/legal_v2/benchmark_retrieval_latency_golden_v1.py`
  (warmup once each; 20 golden queries; CUDA synchronize on ColBERT/CE).
- Modes: FAST A; B+ColBERT (no CE); CE B (Stage1 B + CE-7). No profile activation.
- Warm wall_ms (p50 / p95 / mean):
  - FAST A: 684 / 1103 / 758
  - B+ColBERT: 612 / 903 / 654
  - CE B: 7479 / 8064 / 7264
- Ratios (p50): B+ColBERT / FAST ≈ 0.89×; B+ColBERT / CE ≈ 0.08×.
- Known quality unchanged: FAST MRR 0.607; B+ColBERT 0.625 (+0.018, Hit@5 +0.05);
  CE B MRR 0.975.
- Verdict: **KEEP_COLBERT_AS_BALANCED** (clear middle tier vs CE; not slower than FAST
  on this GPU+parallel hybrid path).
- Artifacts (not committed):
  `artifacts/legal_v2/chunking_ab_pilot_300_v1/latency_tier_v1/LATENCY_TIER_RESULTS.*`
- Next (explicit go only): product wiring of BALANCED profile, or archive decision if
  ops cost of ColBERT index outweighs latency benefit.

## 2026-08-12 Europe/Moscow — Task: Hybrid B + ColBERT golden evaluation

- Experimental Stage-1 only: B contextual dense (BGE-M3) + BM25 + ColBERT → RRF
  (no CE). Reuses `rrf_fuse` N-lists + `ColbertRetriever` (async).
- Code: `app/rag/legal_v2/retrieve/colbert_hybrid.py`,
  `scripts/legal_v2/evaluate_colbert_hybrid_golden_v1.py`,
  `tests/rag/test_legal_v2_colbert_hybrid.py`; public
  `aggregate_legal_v2_documents` helper on Legal v2 retriever.
- Depths: dense=80, BM25=80, ColBERT=80, fused=120, rrf_k=60.
- Metrics (20 golden): Hit@1=0.50, Hit@3=0.60, Hit@5=0.85, Hit@10=0.95,
  MRR=0.625, mean rank≈2.53, HN≈0.053.
- vs FAST A: Hit@1/3/10 tie; Hit@5 +0.05; MRR +0.018; better mean rank.
- Critical: `004` 5→4 (ColBERT helped); `002` still >10 with HN
  (pure ColBERT had rank 3 — fusion lost it); Hit@1 loss `020` (1→2).
- Verdict: **COLBERT HYBRID VERDICT = IMPROVES** (vs FAST A/B; still far
  below CE B). Not production-activated; FAST/CE pins unchanged.
- Artifacts (not committed):
  `artifacts/legal_v2/chunking_ab_pilot_300_v1/colbert_v1/hybrid_eval/`
- Next (explicit go only): B+ColBERT candidates → canonical CE-7.

## 2026-08-12 Europe/Moscow — Task: Pure ColBERT golden evaluation

- Runner: `scripts/legal_v2/evaluate_colbert_golden_v1.py` (async boundary;
  uses existing `ColbertRetriever` / PyLate backend; no BM25/RRF/BGE/CE).
- Corpus: Slice 4 B contextual ColBERT index
  (`legal_v2_colbert_b_contextual_300`, 4168 chunks).
- Same 20 golden queries. Metrics: Hit@1=0.30, Hit@3=0.45, Hit@5=0.45,
  Hit@10=0.55, MRR=0.371, mean rank≈2.73, HN≈0.053.
- Critical: `002`→rank 3, `004`→rank 1; `>10`: 003,005,007,008,009,012,017,018,019.
- Verdict: **COLBERT VS FAST = FAST WINS**, **COLBERT VS CE = CE WINS**.
- Decision: **PURE COLBERT = REJECTED** as standalone canonical profile.
- Artifacts (not committed):
  `artifacts/legal_v2/chunking_ab_pilot_300_v1/colbert_v1/eval/`
- FAST canonical = A; CE canonical = B contextual (unchanged).
- Hybrid B+ColBERT eval completed (see entry above).

## 2026-08-12 Europe/Moscow — Task: ColBERT backend + B contextual index

- Implemented async-first PyLate ColBERT backend under
  `app/rag/legal_v2/retrieve/colbert/` (Protocol + `PyLateColbertBackend`,
  mapping, corpus export, indexer/retriever).
- Optional deps: `requirements-colbert.txt` (isolated; do not upgrade API torch).
- Builder: `scripts/legal_v2/build_colbert_index_v1.py`
- First index: Slice 4 B contextual, **4168/4168**, FastPLAID,
  model `colbert-ir/colbertv2.0`, device `cuda`, batch 16.
- Artifacts (not committed):
  `artifacts/legal_v2/chunking_ab_pilot_300_v1/colbert_v1/`
  (`COLBERT_INDEX_READY: true`).
- FAST/CE pins unchanged. Golden ColBERT benchmark completed (see entry above).

## 2026-08-12 Europe/Moscow — Task: ColBERT retrieval foundation

- Added modular ColBERT package under `app/rag/legal_v2/retrieve/colbert/`
  (config / models / backend boundary / indexer / retriever).
- No model download, no index build, no profile activation, no benchmark.
- FAST/CE pins unchanged (A / B contextual). Profile resolver unchanged
  (`precise` remains reserved for a future late-interaction phase).
- Next (explicit go only): install/wire backend, build Slice 4 B index, eval.

## 2026-08-12 Europe/Moscow — Task: Pin FAST=A and CE=B runtime profiles

- Decision after Slice 4 FAST + canonical CE-7 A/B benchmarks:
  - **FAST canonical chunking: A** (`…chunk_ab_v8_a_current_300`)
  - **CE canonical chunking: B contextual** (`…chunk_ab_v8_b_contextual_300`)
  - CE params unchanged: `fast_ce` / `BAAI/bge-reranker-v2-m3` / candidates 30 /
    passages 7 / `diversified_stage1_evidence_v1` / pool 40 / batch 8 /
    max_length 512 / experiment `ce_bge_v2m3_p7_diverse_v1`
- Evidence:
  - FAST A/B: **A WINS** (Hit@1/10 tie; A better Hit@3/5, MRR, mean rank, HN)
  - CE A/B: **B WINS** (Hit@1=0.95, Hit@3/5/10=1.00, MRR=0.975, HN=0.000;
    query `002`: A CE `>10` → B CE `1`; no B CE regressions)
- Code: per-profile index bindings in
  `app/rag/legal_v2/retrieve/retrieval_profiles.py`; dual Stage1 runtimes in
  `case_similarity_search.py` (FAST retriever=A, CE retriever=B).
- Reporter: `scripts/legal_v2/report_chunking_ab_ce_comparison_v8.py`
- Unchanged: chunkers, indexes, golden dataset, CE hyperparameters.
- **COLBERT = NOT STARTED** (next experiment only on explicit go).

## 2026-08-12 Europe/Moscow — Task: CE A/B retrieval on Slice 4 indexes

- Goal: canonical CE-7 A/B on same Slice 4 indexes; compare vs FAST; no ColBERT.
- Config: `--profile fast_ce`, model `BAAI/bge-reranker-v2-m3`, candidates 30,
  passages 7, selector `diversified_stage1_evidence_v1`, pool 40, batch 8,
  experiment `ce_bge_v2m3_p7_diverse_v1`.
- Results (n=20): CE A Hit@1=0.90 Hit@10=0.95 MRR=0.925; CE B Hit@1=0.95
  Hit@3/5/10=1.00 MRR=0.975 HN=0.000.
- **CE A/B VERDICT: B WINS**; overall chunking flips FAST→**B** under CE-7.
- Reports: `artifacts/legal_v2/chunking_ab_pilot_300_v1/ce_ab_results/CE_AB_COMPARISON.{md,json,html}`

## 2026-08-12 Europe/Moscow — Task: FAST A/B retrieval on Slice 4 indexes

- Goal: run FAST-only case-similarity golden eval on isolated chunk_ab_v8 A/B
  indexes; produce readable MD/JSON/HTML comparison; **no CE**.
- Runner: `evaluate_case_similarity_golden_v1.py --profile fast` (canonical).
- Collections:
  - A `nalus_legal_paragraph_chunks_v2_chunk_ab_v8_a_current_300` + BM25 A
  - B `nalus_legal_paragraph_chunks_v2_chunk_ab_v8_b_contextual_300` + BM25 B
- Results (n=20, evaluable=20, retrieval_failures=0):
  - A: Hit@1=0.50 Hit@3=0.60 Hit@5=0.80 Hit@10=0.95 MRR≈0.607 HN_outrank≈0.053
  - B: Hit@1=0.50 Hit@3=0.55 Hit@5=0.75 Hit@10=0.95 MRR≈0.590 HN_outrank≈0.105
- Hit@1 transitions: gained B `017`; lost B `019`.
- **Verdict: `A WINS`** (better MRR / Hit@3 / Hit@5 / mean rank / HN rate;
  Hit@1 and Hit@10 tied). `safe_for_ce_next=true`.
- Reports: `artifacts/legal_v2/chunking_ab_pilot_300_v1/fast_ab_results/FAST_AB_COMPARISON.{md,json,html}`
- Reporter: `scripts/legal_v2/report_chunking_ab_fast_comparison_v8.py`
- Unchanged: Slice 4 indexes, golden dataset, chunkers, CE off.

## 2026-08-12 Europe/Moscow — Task: Slice 4 A/B indexes (HARD STOP before FAST/CE)

- Goal: isolated A/B BGE-M3 + BM25 indexes for `chunking_ab_pilot_300_v1` on
  parser v8; classify readiness; do **not** run FAST/CE yet.
- Chunk QA v8: `run_chunking_ab_pilot_300_chunk_qa_v8.py` + lost-paragraph
  verifier; base QA overlap/text-loss helpers tightened. Verdict
  **`CHUNK_QA_PASS_V8`** (A=6162, B=4168).
- Indexer: `build_chunking_ab_pilot_300_indexes_v8.py` (resume-by-default,
  `--side A|B|both`, `--device cpu|cuda`, experiment CUDA embedder bypass).
- GPU stack (local only): `Dockerfile.slice4-gpu`,
  `docker-compose.slice4.gpu.yml`, `requirements-slice4-gpu.txt` — dual
  networks (API + Qdrant); BM25 mount must be `nalus-scraper/storage`.
- Collections:
  - `nalus_legal_paragraph_chunks_v2_chunk_ab_v8_a_current_300` (6162)
  - `nalus_legal_paragraph_chunks_v2_chunk_ab_v8_b_contextual_300` (4168)
- Classification: **`INDEX_AB_READY`**, **`FAST_AB_SAFE_TO_START=YES`**.
- Artifacts: `artifacts/legal_v2/chunking_ab_pilot_300_v1/slice4_indexes_v8/`.
- **HARD STOP:** no FAST A/B and no CE until explicitly requested.
- Next (only on explicit go): FAST A/B retrieval benchmark on these indexes.

## 2026-08-10 Europe/Moscow — Task: SectionType parser v8 fix

- Goal: stop keyword/`HEADER` sticky contamination so same-section chunking
  boundaries are trustworthy before A/B embeddings.
- Parser `legal-decision-parser.cz-courts.v8`:
  - HEADER never inferred from body keywords (`ústavní soud` trap removed)
  - sticky body sections preserved; reasoning cues can upgrade weak zones
  - closing signatures → `instruction`, not `header`
- Re-audit on same 300 inventory (`section_type_audit_v8`):
  - header share **34.3% → 2.5%**
  - header-suspicion docs **289/300 → 1/300**
  - court_reasoning share **2.5% → 73.2%**
  - verdict **`SECTION_TYPE_OK_FOR_CHUNKING_AB`** (`block_slice4=false`)
- Remaining warning: tiny heading paragraphs (`Výrok`/`Odůvodnění`) — chunker
  attach concern, not SectionType material.
- Next: regenerate full A/B chunk QA on 300, then Slice 4 embeddings.

## 2026-08-10 Europe/Moscow — Task: SectionType audit (block Slice 4)

- Goal: before A/B embeddings, quantify whether `SectionType` labels are
  trustworthy hard boundaries for hierarchical / contextual-packed chunking.
- Inventory: frozen `chunking_ab_pilot_300_v1` (hash
  `89233b9fe9b06eda8dea00abd99a48aa54940e616aa88c00860ced4ae49c011b`).
- Runner: `scripts/legal_v2/run_section_type_audit_pilot_300.py` (no embeddings).
- Key signals on 300 docs / 16602 paragraphs:
  - `header` share **34.3%** vs `court_reasoning` only **2.5%**
  - header-suspicion docs **289/300**; header reasoning/prose flags **3713**
  - tiny structural heading candidates **282**
- Artifacts: `artifacts/legal_v2/chunking_ab_pilot_300_v1/section_type_audit/`
- **Verdict: `SECTION_TYPE_MATERIAL_REGRESSION`** → **block Slice 4**.
- Next: fix deterministic SectionType sticky/heading logic, regenerate A/B
  chunk QA, only then BGE-M3 + BM25 A/B indexes.

## 2026-08-09 Europe/Moscow — Task: report applied retrieval_stage

- Goal: `retrieval_stage` must describe the pipeline that produced the returned
  ranking (provenance), not configuration intent.
- Source of truth: `diagnostics.rerank.rerank_applied` (+ passages/document for
  CE-7 label). Helper: `build_retrieval_stage` / `RetrievalStage`.
- Labels: `hybrid_rrf_stage_1` (FAST / CE not applied);
  `hybrid_rrf_ce7` (successful 7-passage CE rerank).
- Unchanged: ranking, CE scores, Stage 1 retrieval, QuerySpec, Qdrant/BM25/RRF.

## 2026-08-09 Europe/Moscow — Task: FE retrieval profile switcher (FAST / CE-7)

- Goal: modular UI switch for Stage 1 profiles without activating CE globally.
- Backend: request field `retrieval_profile` (`fast` default, `ce7`, `precise` reserved).
  CE master-allow remains `NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED`; CE only runs when
  profile=`ce7`.
- FE (NalusFE): `RetrievalProfileSwitcher` + registry `retrievalProfiles.ts`;
  wired through `/api/retrieval/documents` → Stage 1 search.
- Stage1 compose sets CE master-allow ON so CE-7 button works; default UI stays FAST.
- Unchanged: Stage 1 knobs, golden labels, CE-10 not introduced.

## 2026-08-09 Europe/Moscow — Task: CE-7 diversified passage coverage experiment

- Goal: controlled CE passage-coverage test vs CE-3 (`3 → 7` passages) using
  deterministic selector `diversified_stage1_evidence_v1` (RRF/dense/BM25 +
  diversity), not “first 7 chunks”.
- Frozen: CE model/revision/`max_length`/batch policy, candidates=30,
  aggregation=`max`, Stage 1 / QuerySpec / BM25 / RRF / golden labels; FE OFF;
  **CE-10 not run**.
- Additive Stage-1 `chunk_evidence` provenance for selectors; CE OFF order
  regression preserved.
- Artifacts: diagnostics `.../20260808T213610Z_diagnostics_004_016`;
  full run `ce_bge_v2m3_p7_diverse_v1/20260808T214850Z`;
  rank-diff `.../20260808T214850Z_vs_fast_ce3`.
- Metrics vs FAST / CE-3: Hit@1 0.60/0.75/**0.75**; Hit@10 1.0/0.9/**1.0**;
  MRR ~0.701/~0.806/**0.8375**; HN 0.0/~0.053/**0.0**.
- 004: CE-3 out of TOP10 → CE-7 rank 3; 016: CE-3 out + HN → CE-7 rank 1.
- **Verdict: `PASSAGE_COVERAGE_FIX_CONFIRMED`**. Recommendation:
  `STOP_AT_7_AND_CONTINUE_CE_ARCHITECTURE` (do not auto-run CE-10).
- CE remains default OFF; next architecture work is not “more passages”.

## 2026-08-08 Europe/Moscow — Task: Cross-Encoder rerank experiment (OFF by default)

- Goal: additive local CE rerank above frozen Stage 1 shortlist (`FAST+CE`),
  measure ranking benefit without changing Stage 1 knobs/corpus.
- Added `app/rag/legal_v2/rerank/` (provider Protocol, ST CrossEncoder,
  passage select, max aggregation, service).
- Flag `NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED=0`; CE OFF path keeps Stage 1 order.
- Eval: `evaluate_case_similarity_golden_v1.py --profile fast|fast_ce` +
  `export_ce_confusable_review_v1.py` (UNREVIEWED labels only).
- Model: `BAAI/bge-reranker-v2-m3`; download gated by `CE_ALLOW_DOWNLOAD`.
- Untuned A/B (CPU): FAST `20260808T190653Z` vs CE `20260808T190938Z`.
  - FAST: Hit@1=0.60 Hit@10=1.0 MRR≈0.701 HN=0.0
  - CE: Hit@1=0.75 Hit@10=0.9 MRR≈0.806 HN≈0.053
  - gained Hit@1: 003,006,009,013,018; lost Hit@1 (+ Hit@10): 004,016
- **Verdict: `CE_REGRESSION`** (Hit@10 and HN gate failed despite Hit@1/MRR gains).
  Keep CE OFF; next: expand reviewed confusable set / try alternate CE or ColBERT.
- Unchanged: Qdrant/BM25/BGE-M3 embeddings, RRF, golden labels, FE default.

## 2026-08-08 Europe/Moscow — Task: Long-input SearchBrief preprocessing (OFF by default)

- Goal: modular pre-retrieval layer for long pasted legal text without changing
  validated Stage 1 retrieval.
- Added `app/rag/legal_v2/query_input/` (classifier, normalizer, extractive
  provider, Precise LLM stub, QueryInputService).
- Integration: Stage 1 search → `QueryInputService.prepare()` → QuerySpec.
- Flag `NALUS_LEGAL_V2_LONG_INPUT_ENABLED=0` (compose explicitly off).
- FE SearchBar uses textarea for multi-line paste (no second UI).
- Docs: `docs/architecture/LONG_INPUT_SEARCH_BRIEF_V1.md`.
- Unchanged: BGE-M3/BM25/RRF knobs, Qdrant corpus, golden benchmark, no LLM calls.

## 2026-08-07 Europe/Moscow — Task: Stage 1 warmup on API start

- Goal: avoid multi-minute cold first FE search after API restart.
- `NALUS_LEGAL_V2_STAGE1_WARMUP_ON_START=1` (stage1 compose): background
  lifespan task loads BGE-M3 + BM25 (`warmup` encode + BM25 search).
- `/case-similarity/ready` reports `model_loaded` / `bm25_loaded` /
  `warmup_status`; when warmup is required, `ready=true` only after warm.
- Unchanged: retrieval knobs, corpus, ranking; flag default remains off outside
  stage1 overlay.

## 2026-08-07 Europe/Moscow — Task: Fix Stage 1 full-document FE 404

- Symptom: FE search returned Stage 1 passages, but „Celý rozsudek“ showed
  „Dokument nebyl nalezen v indexovaném korpusu“ (HTTP 404).
- Cause: search used `NALUS_LEGAL_V2_QDRANT_COLLECTION` (pilot_600); full-document
  endpoint used legacy `QDRANT_COLLECTION_NAME` (`nalus_bge_m3_chunks_v1`).
- Fix: `resolve_full_document_collection_name()` prefers Stage 1 / legal_v2
  collection when those flags are on; stage1 compose sets both
  `QDRANT_COLLECTION_NAME` and `NALUS_FULL_DOCUMENT_QDRANT_COLLECTION` to pilot_600;
  match also on `canonical_document_id`.
- Unchanged: retrieval ranking, corpus rebuild, ColBERT/CE.

## 2026-08-07 Europe/Moscow — Task: Pilot 600 judgment inventory + Stage 1 eval seed

- Goal: full list of indexed pilot judgments (~622 ECLI) with topic description
  and 2 natural search questions each; prepare Stage 1 HTTP eval.
- Generator: `scripts/legal_v2/build_pilot_600_judgment_inventory.py`
  (offline heuristics, no paid LLM).
- Outputs:
  - `artifacts/legal_v2/pilot_600_judgment_inventory/pilot_600_judgment_inventory.json`
  - `artifacts/legal_v2/pilot_600_judgment_inventory/pilot_600_judgment_inventory.md`
  - `artifacts/legal_v2/pilot_600_judgment_inventory/pilot_600_search_queries.jsonl` (1244 rows)
- Quality: 622 docs, 0 missing case numbers; ~96 still weakly tagged; q1=topic,
  q2=topic+case cue. Spot-check II.ÚS 859/23 #2 q2 retrieves expected ECLI @1.
- Eval runner: `scripts/legal_v2/evaluate_pilot_600_inventory_stage1.py`
  (Hit@K / MRR against Stage 1 API). Full 1244-query run left for local PowerShell.
- Unchanged: Qdrant/BM25 corpus, Stage 1 retrieval knobs, no ColBERT/CE.

## 2026-08-06 Europe/Moscow — Task: Stage 1 result window up to 50

- Goal: keep first viewport at TOP 10, allow loading more candidates up to 50.
- Backend: `NALUS_LEGAL_V2_MAX_RESULT_LIMIT=50`, `CANDIDATE_DOCUMENTS=50`.
- Frontend: Stage 1 requests `limit=50`; ResultList shows 10 then “Načíst další”.
- Unchanged: retrieval model/BM25/RRF knobs aside from aggregation pool size;
  no ColBERT/CE; no corpus rebuild.

## 2026-08-06 Europe/Moscow — Task: Deploy Stage 1 case-similarity API + FE

- Goal:
  Expose validated Stage 1 retrieval (QuerySpec + BGE-M3 + BM25 + RRF + ECLI
  aggregation) through FastAPI + existing NalusFE for live pilot testing.
- Assets audited (read-only):
  collection `nalus_legal_paragraph_chunks_v2_pilot_600` = 14448 chunks,
  622 unique ECLI via `document_id`; BM25 `nalus_legal_paragraph_bm25_v2_pilot_600`
  = 14448 rows, perfect ECLI overlap; no doc-* primary IDs.
- API: `POST /api/rag/legal-v2/case-similarity/search` + readiness GET;
  process-singleton BGE-M3/BM25; no ColBERT/CE; no paid LLM on Stage 1 path.
- Docker: `docker-compose.stage1.local.yml` binds worktree code + parent
  storage/models to existing `nalus-scraper-qdrant-1` (not empty worktree qdrant).
- Frontend (NalusFE): v2 mode → Stage 1 endpoint; search UI at `/vyhledavani`.
- Smoke: queries A–E HTTP 200, identity OK, warm ~0.5–0.9s after first load.
- Unchanged: golden, ECLI map, retrieval knobs, no corpus rebuild, no push.
- Next: ColBERT / CE rerank; corpus-wide chunk QA; larger relevance benchmark.

## 2026-08-06 Europe/Moscow — Task: Case-similarity rank-diff audit (Hit@1)

- Goal:
  Explain unchanged Hit@1=0.60 after QuerySpec fix where `004` reached primary
  rank 1 (`20260805T234409Z` → `20260806T092207Z`).
- Evaluator semantics (from code):
  Hit@K/MRR use `best_positive_rank = min(primary, best accepted alternative)`;
  accepted alternatives count for Hit@1; ranks stored within TOP 10 only;
  HN-blocked rows remain in Hit@K denominator.
- Independent recompute matched both stored reports (12/20 Hit@1 both runs).
- Hit@1 arithmetic:
  gained=`nalus-cs-pilot-004`, lost=`nalus-cs-pilot-013` → 12+1-1=12.
- Verdict: `OFFSET_RANK1_REGRESSION`.
- Material degradations also: `003` (3→9), `009` (2→6), `013` (1→2);
  all remain Hit@10.
- Tooling: `scripts/legal_v2/compare_case_similarity_runs.py` + tests.
- Unchanged: golden, QuerySpec production logic, indexes, retrieval knobs.
- Next:
  Optional focused diagnosis of `013` (and optionally `003`/`009`) rank drift
  without changing retrieval parameters in the same task.

## 2026-08-06 Europe/Moscow — Task: Preserve negation in QuerySpec

- Goal:
  General query-understanding fix so explicitly negated requested case types
  are not reintroduced as positive expansions/hard constraints; extract
  procedural-defect signals; prioritize procedural issue over background.
- Root cause:
  `build_query_spec_v2` matched `domestic_custody` broadly and expanded
  `úprava styku…` / `opatrovnické řízení` while `_extract_negations` only set
  generic `negation_present` and never bound “Nehledám … spor o péči”.
- Changes:
  Scoped semantic negation; procedural-defect concepts; concept priority /
  focus demotion; contradiction-safe expansion filter; CONTRACTS invariants;
  focused tests (cases A–F + pilot-004 query load without production ID use).
- Single-query diagnostic (`diagnose_nalus-cs-pilot-004_after_queryspec.json`):
  expected ECLI dense doc rank **5**, BM25 **8**, RRF **4**, aggregated **1**.
- Full baseline (`20260806T092207Z`):
  evaluable=`20`, retrieval_failures=`0`;
  Hit@1=`0.60`, Hit@10=`1.0` (was 0.95), MRR≈`0.70`, HN outrank=`0.0`;
  `nalus-cs-pilot-004` primary_rank=`1`; no prior TOP-10 miss regressed.
- Unchanged: golden JSONL, ECLI map, Step 4A, retrieval knobs, no push.
- Next recommended task:
  Optional deeper ranking work for Hit@1 on 001/002/003/008/009; then
  chunking A/B/C/D / ColBERT / CE per master plan — not required for this fix.

## 2026-08-06 Europe/Moscow — Task: Diagnose case-similarity miss 004

- Goal:
  Stage-trace `nalus-cs-pilot-004` without changing config or golden.
- Expected: `ECLI:CZ:US:2025:1.US.3575.25.1`
- Result (`artifacts/.../diagnose_nalus-cs-pilot-004.json`):
  - QuerySpec keeps OSPOD/children, no-lawyer, reasoning defects, formal
    rejection in the retrieval query text; also HARD-expands
    `péče o nezletilé dítě` / soft `odmítnuto`, plus expansions
    `úprava styku…` / `opatrovnické řízení` (custody-merits bias risk).
  - Dense TOP 80: **absent**
  - BM25: present weakly (doc rank **35**, best chunk rank **78**)
  - RRF TOP 120: **absent** (BM25-only weak hit truncated)
  - Aggregation: never sees expected ECLI
- Drop point:
  `present_in_bm25_only_but_dropped_by_rrf` — primary issue is **dense miss**
  (+ BM25 too weak to carry RRF alone), not aggregation.
- Next recommended task:
  Inspect why dense misses 004 (chunk text / query expansions), still without
  golden edits; only then consider query-processing or retrieval knobs.

## 2026-08-06 Europe/Moscow — Task: Fix bm25_index_id provenance + real baseline

- Goal:
  Unblock case-similarity scoring after golden ECLI upsert stamped a mismatched
  `bm25_index_id` on 560 Qdrant payloads (`…chunks_v2_pilot_600_bm25` vs
  `…bm25_v2_pilot_600`), causing 19/20 `retrieval_error`s and a misleading
  Hit@1=1.0 over a single evaluable query.
- Fixes:
  - `scripts/legal_v2/repair_pilot_600_bm25_index_id.py` (set_payload + SQLite)
  - index helper defaults now map pilot_600 collection → existing BM25 id
  - evaluator console prints `evaluable` / `retrieval_failures`
- Repair result: Qdrant 14448/14448 on target `bm25_index_id`.
- Real baseline (`20260805T234409Z`):
  evaluable=`20`, retrieval_failures=`0`;
  Hit@1=`0.6`, Hit@10=`0.95`, MRR≈`0.70`, HN outrank=`0.0`;
  only miss outside top-10: `nalus-cs-pilot-004`.
  Report: `artifacts/legal_v2/case_similarity_golden_v1_baseline/20260805T234409Z/`.
- Next recommended task:
  Inspect misses (esp. 004, and ranks >1 for 001/002/008); optional formal
  PASS verdicts in manual_review; expand golden — not ColBERT/CE yet.

## 2026-08-06 Europe/Moscow — Task: Case-similarity untuned baseline (live)

- Goal:
  Run the first real scored case-similarity baseline against
  `nalus_legal_paragraph_chunks_v2_pilot_600` after ECLI upsert.
- Fixes:
  Evaluator import syntax; clearer `qdrant_client` missing error; Docker runner
  `scripts/legal_v2/evaluate_case_similarity_golden_v1.ps1`. Golden BM25 rows
  merged into full pilot sidecar (`14448` chunks).
- Result (`20260805T231907Z`):
  `primary_present=20`, `primary_missing=0`; Hit@1=`1.0`, Hit@10=`1.0`,
  MRR=`1.0`, HN outrank rate=`0.0` (1 HN-blocked row excluded from HN denom).
  Artifacts under `artifacts/legal_v2/case_similarity_golden_v1_baseline/20260805T231907Z/`.
- Next recommended task:
  Human review of per-query ranked lists / supporting evidence; then expand
  case-similarity or retrieval golden size — do not treat n=20 as production
  chunking/reranker winner.

## 2026-08-06 Europe/Moscow — Task: Resolve remaining case-similarity ECLIs

- Goal:
  Clear the five previously blocked 2026 ÚS identity gaps without fabricating
  identifiers.
- Method:
  Official NALUS GetText URLs (`sz=…`) plus decision dates from the same pages;
  mapping rule validated against five already-verified pilot ÚS ECLIs from
  NALUS batch metadata.
- Result:
  All 22 pilot-referenced judgments now `verified` (0 blocked).
  Examples: `IV.ÚS 650/26` → `ECLI:CZ:US:2026:4.US.650.26.1`.
  Additive upsert into pilot_600 completed earlier the same night (22 docs /
  624 chunks; collection audit: no missing golden primaries).
- Next:
  Superseded by the live baseline entry above.

## 2026-08-05 Europe/Moscow — Task: ECLI as canonical decision identity

- Goal:
  Make verified ECLI the permanent canonical identity for Legal v2 judicial
  decisions across benchmark → corpus → chunks → Qdrant → retrieval → evaluation.
- Identity contract:
  `document_id == canonical_document_id == ecli`; `source_document_id` / `doc-*`
  remain secondary traceability only.
- Mapping artifact:
  `benchmarks/legal_v2/case_similarity_document_identity_v1.json`
  — 22 unique judgments referenced by the pilot; all verified via batch /
  Justice metadata or official NALUS GetText URLs (see 2026-08-06 follow-up).
- Golden schema / builder / validator / evaluator updated to carry and match ECLI.
- Shared helper: `app/rag/legal_v2/identity.py` (`validate_decision_identity`).
- Live collection audit (`nalus_legal_paragraph_chunks_v2_pilot_600`):
  13824 chunks / 600 judgments; nearly all already use ECLI as `document_id`;
  **0/15** verified golden primary ECLIs are present in that collection
  (5 primaries remain identity-blocked). No destructive reindex performed.
- Validation:
  Builder/validator OK; deterministic rebuild OK; Step 4A unchanged;
  focused tests `63 passed` (+ identity suite); `git diff --check` clean.
- Next recommended task:
  Additive upsert of the 15 verified golden ECLIs (plus HN ECLIs) into the
  pilot collection under literal ECLI payloads; resolve the 5 blocked 2026 ÚS
  ECLIs from authoritative NALUS/export metadata before indexing them.

## 2026-08-05 Europe/Moscow — Task: Case-similarity evaluator + HN blocker schema

- Goal:
  Checkpoint the golden pilot, add explicit hard-negative evaluability fields,
  and implement the first Legal v2 hybrid document-level evaluation runner.
- Phase A commit:
  `8dddea6` — `feat(legal-v2): add case similarity golden v1 pilot`
- Phase B results:
  - Schema fields `hard_negative_evaluable` / `hard_negative_blocker`;
    `nalus-cs-pilot-007` blocked as `insufficient_same_domain_corpus`.
  - Runner: `scripts/legal_v2/evaluate_case_similarity_golden_v1.py`
    (`LegalV2HybridRetriever`, offline, no LLM).
  - Metrics module + focused tests.
  - **Real scored baseline blocked:** target collection
    `nalus_legal_paragraph_chunks_v2_pilot_600` contains 600 indexed docs with
    ECLI IDs; **0/20** golden `doc-*` primaries are present (ID/corpus mismatch).
    Compatibility audit written under
    `artifacts/legal_v2/case_similarity_golden_v1_baseline/<run_id>/`.
- Next recommended task:
  Index the 20 reviewed case-similarity judgments (and supplemental criminal HN
  sources) into a legal_v2 collection with stable document-ID mapping, then rerun
  the untuned baseline once.

## 2026-08-05 Europe/Moscow — Task: Case-similarity pilot correction (003 / 007 / 016)

- Goal:
  Targeted correction of three case-similarity pilot rows only; no clarification
  gate; no Step 4A / provider / runtime changes.
- Results:
  - `nalus-cs-pilot-003`: client paraphrase of causation conclusion; primary
    `doc-d513b3e81616439a` unchanged; longest supporting-block token overlap
    reduced to 2; forbidden eight-token legal phrase removed.
  - `nalus-cs-pilot-007`: **CORPUS BLOCKER** — no honest same-domain hard
    negatives found in reviewed pool + local raw_sources + NSoud dumps for
    lawyer discipline / former-client conflict / insolvency representation.
    Weak cross-domain HNs retained only for schema `min=1`; not claimed strong.
  - `nalus-cs-pilot-016`: hard negatives replaced with supplemental criminal
    appeals `doc-4fbdc1db957f44e7` (`6 To 41/2024`) and `doc-68c126d146c84fa1`
    (`6 To 42/2024`) loaded from `court_format_study/raw_sources` via
    `load_case_similarity_corpus()` (outside the 20 reviewed primaries).
- Validation:
  Builder/validator/export OK; deterministic rebuild byte-identical; Step 4A
  no-diff; focused tests `31 passed` (case-similarity + Step 4A); no commit.
- Next recommended task:
  Expand local corpus with lawyer-discipline / conflict-of-interest peers so
  `007` can receive honest hard negatives; then human audit of remaining rows.

## 2026-08-05 Europe/Moscow — Task: Case-similarity retrieval golden v1 pilot (20 docs)

- Goal:
  Create a source-grounded **document-level** case-similarity benchmark pilot:
  realistic user case descriptions → most similar whole judgment → supporting
  passages → hard negatives. Primary product workflow for NALUS similarity search.
- Deliverables:
  - `benchmarks/legal_v2/case_similarity_golden_v1_pilot.jsonl` (20 development rows)
  - `app/rag/legal_v2/benchmark/case_similarity_golden.py`
  - `app/rag/legal_v2/benchmark/corpus.py` (`load_reviewed_pool_corpus`)
  - `docs/architecture/CASE_SIMILARITY_RETRIEVAL_GOLDEN_V1.md`
  - builder / validator / manual-review export scripts
  - `tests/rag/test_legal_v2_case_similarity_golden_v1_pilot.py`
- Explicit non-goals:
  No Step 4A changes; no parser/Qdrant/BM25/provider/runtime retrieval; no commit.
- Next recommended task:
  Human audit of all 20 case-similarity rows (`PASS`/`FIX`/`REJECT`), then expand
  or begin document-level retrieval metrics against frozen judgments.

## 2026-08-05 Europe/Moscow — Task: Step 4A retrieval-golden v1 pilot (30 queries)

- Goal:
  Create an evidence-first, block-grounded retrieval-golden **pilot** (29 positive + 1 corpus-negative) from development archetype documents only, without claiming the final 100–150 benchmark is complete.
- Deliverables:
  - `benchmarks/legal_v2/retrieval_golden_v1_pilot.jsonl`
  - `app/rag/legal_v2/benchmark/retrieval_golden.py` (+ corpus loader)
  - `docs/architecture/RETRIEVAL_GOLDEN_V1.md`
  - `scripts/legal_v2/build_retrieval_golden_v1_pilot.py`
  - `scripts/legal_v2/validate_retrieval_golden_v1_pilot.py`
  - `tests/rag/test_legal_v2_retrieval_golden_v1_pilot.py`
- Explicit non-goals:
  No parser/Qdrant/BM25/provider changes; pilot must not select a production chunking winner.
- Next recommended task:
  Expand toward 100–150 retrieval queries with validation/locked_holdout splits; only then evaluate chunking A/B/C/D for a winner.

## 2026-08-05 Europe/Moscow — Task: Phase 2 canonical block/chunk schema

- Goal:
  Lock the master-plan document → block → child → parent data contract without cutting over production indexing or changing parser v7 rules.
- Deliverables:
  - `docs/architecture/CANONICAL_BLOCK_CHUNK_SCHEMA_V1.md` (field contract + legacy alias map)
  - `app/rag/legal_v2/schema/canonical_v1.py` (typed models, stable IDs, checksums, reconstruction validators)
  - `app/rag/legal_v2/schema/map_from_legal_v2.py` (bridge from `LegalDocumentStructure` / hierarchical chunks)
  - `tests/rag/test_legal_v2_canonical_schema_v1.py`
  - `scripts/legal_v2/export_canonical_schema_pilot.py` (offline 1–3 doc pilot under gitignored `artifacts/legal_v2/canonical_schema_pilot/`)
- Explicit non-goals:
  No parser v8, no full-corpus Qdrant/BM25 upsert, no chunking A/B winner, no 100–150 retrieval golden annotations in this task.
- Next recommended task:
  Phase 3 chunking A/B/C/D and/or Phase 4 retrieval golden skeleton — one major variable at a time. Fill `pending_external` archetype holdouts when new unseen documents are available.

## 2026-08-05 Europe/Moscow — Task: ACCEPT_V7 + Phase 1 archetypes + docs pointer cleanup

- Goal:
  Close parser v7 as the accepted structural baseline with known limitations, create the Phase 1 archetype manifest from the 20 review documents, and make `docs/architecture/NALUS_LEGAL_RAG_MASTER_PLAN.md` the single controlling plan pointer.
- Decision:
  `ACCEPT_V7_WITH_KNOWN_LIMITATIONS`. Do not open parser v8 for non-blocking label noise. Documented limitation: `KNOWN-PARSER-001` (doc 08 closing date without `dne` classified as `heading`).
- Deliverables:
  - `docs/architecture/PARSER_V7_BASELINE_DECISION.md`
  - `docs/architecture/parser_benchmark/archetypes_v1.json` (+ local mirror under `artifacts/legal_v2/parser_benchmark/`)
  - `docs/retrieval-enterprise/NALUS_SYSTEM_BUILD_PLAN.md` reduced to a pointer
  - updated `docs/retrieval-enterprise/README.md` and `LEGAL_DECISION_PARSER_V7.md`
- Coverage gaps:
  Four holdout slots remain `pending_external` because the current design set has only 20 documents (target 21 slots / 7×3 roles).
- Next recommended task:
  Superseded by the Phase 2 canonical schema entry above.

## 2026-08-05 Europe/Moscow — Task: Adopt controlling NALUS system build plan

- Goal:
  Capture the post-v7 end-to-end strategy as a binding plan: benchmark-first sequencing, dual parser/retrieval goldens, parent–child chunking, hybrid RRF, and runtime FAST/BALANCED/PRECISE profiles (ColBERT / cross-encoder cascade), not separate Git product branches.
- Deliverable:
  Initially drafted as `docs/retrieval-enterprise/NALUS_SYSTEM_BUILD_PLAN.md`; superseded by committed `docs/architecture/NALUS_LEGAL_RAG_MASTER_PLAN.md` (`1141cc0`). The short file is now only a pointer.
- Decision:
  Parser v7 stays frozen as structural baseline. Next work is archetypes + span-level retrieval golden, then one-layer experiments only.
- Next recommended task:
  Superseded by the ACCEPT_V7 / archetypes entry above.

## 2026-08-05 Europe/Moscow — Task: Czech courts parser v7 structural corrections

- Goal:
  Implement `legal-decision-parser.cz-courts.v7` to correct confirmed v6 generalization regressions across the 17 non-golden review documents via court-profile/state rules, while preserving v6 artifacts and exact goldens `05`/`11`/`16`.
- Worktree:
  `nalus-scraper-parser-fix` on `fix/legal-paragraph-parser`. Starting HEAD `d0eeee58c76b29e856ac4f905daddb596bf66034`. One commit: `fix(parser): add Czech courts parser v7 structural corrections`. No push.
- Parser corrections:
  Constitutional compact Roman(+decimal) subheadings are independent headings; Prague opening formulas merge through `Výrok` including participant numbers as prose; Olomouc civil operative/reasoning uses expected-number progression with date/list isolation; criminal Olomouc golden remains exact.
- Audit/export:
  Baseline/compare under `artifacts/legal_v2/parser_v7_audit/` reports 163 changed line classes and 22 changed boundaries vs the v6 snapshot baseline, golden_status pass, conservation/duplication/ordering/exceptions 0. Generated ignored `parser_v7_remaining_17_full.json/.md`. Historical v6 exports remain byte-identical.
- UI/API:
  Active labels/views use parser v7 (`Changed by parser v7`, `Full corpus v7 review`, TARGETED REGRESSION PASS). Historical v6 change/corpus views and `/exports/parser_v6_*` remain available.
- Manual-review safety:
  Decision store SHA-256 unchanged `F98CD519CCF28310706F70B0D65F2F15FDFC28CC530304CD4FF79890219A28FB`; history unchanged `5E0E86E5A2210800A514341E6A7A87210EBC2EC7504D379BA6DB2542EB82FACD`. No automatic approvals; assisted batches 0. Stale/conflict export categories are consistent (stale=2, conflicts=0).
- Next recommended task:
  Superseded: v7 accepted with known limitations; continue from master-plan Phase 1/2.

## 2026-08-05 Europe/Moscow — Task: Commit parser v6 + full corpus review export

- Goal:
  Commit the completed Czech courts parser v6 and validation UX, then produce a complete offline JSON/Markdown review export for the 17 non-golden review documents and expose it in the local HTML review application.
- Worktree:
  `nalus-scraper-parser-fix` on `fix/legal-paragraph-parser`. Starting HEAD was `14c1e300c46872640ebebfb84cf6e8d6686dec7b`. First commit: `feat(parser): add Czech courts parser v6 and validation UX`. Second commit: `feat(parser-review): add full corpus v6 audit export`. No push.
- Parser/corpus:
  Profile remained `legal-decision-parser.cz-courts.v6`. All 20 review documents were reparsed into a temporary snapshot, validated, then promoted into the real parser-derived review snapshot. Golden documents 05/11/16 remained exact GOLDEN PASS. Audit: conservation 0, duplication 0, ordering 0, parser exceptions 0.
- Export:
  Added `scripts/legal_v2/export_parser_v6_full_review.py` and `scripts/legal_v2/parser_review/full_export.py`. Generated ignored artifacts `artifacts/legal_v2/parser_v6_full_review/parser_v6_remaining_17_full.json` and `.md` with complete lines/boundaries/blocks for the 17 remaining documents. Statuses remain truthful automatic labels only (`PARSER_VALIDATED` / `PARSER_CHANGED_NEEDS_REVIEW`); no manual approvals were created.
- HTML/API:
  Added `Full corpus v6 review` view, JSON/Markdown download routes under `/exports/`, and `Copy document review` for complete per-document Markdown. Existing Lines, Boundaries, Changed by parser v6, Problems, Progress, and Assisted Review views remain available.
- Manual-review safety:
  Decision store stayed SHA-256 `F98CD519CCF28310706F70B0D65F2F15FDFC28CC530304CD4FF79890219A28FB`; history stayed `5E0E86E5A2210800A514341E6A7A87210EBC2EC7504D379BA6DB2542EB82FACD`. Document 2 remains 13/13 lines and 12/12 boundaries. Assisted batches applied: 0.
- Next recommended task:
  Human review of the 17-document export queue, starting with changed headings, numbered paragraphs, nested lists/tables, and possible overmerge/undersplit candidates, without treating automatic parser validation as manual approval.

## 2026-08-04 Europe/Moscow - Task: Constitutional Court parser v5 from completed review document 2

- Goal:
  Convert the completed manual review of visual parser-review document 2 (`doc-b73cac9b3dfc8a42`, source `1-3299-24_1`) into bounded deterministic Constitutional Court parser rules, then reparse the 10 Constitutional Court design documents and expose the changed review queue.
- Scope:
  Parser/profile, parser-review snapshot, audit artifacts, frontend review UI/API, and focused tests in `fix/legal-paragraph-parser`. No Qdrant, BM25, embeddings, providers/models, Docker, Redis, Celery, commits, pushes, or High Court parser generalization.
- Implementation:
  Parser profile is now `legal-decision-parser.cz-courts.v5`. `app/rag/legal_v2/ingest/parser.py` adds a court-scoped Constitutional Court line profile for NALUS headers, case/date metadata, state identifier, decision type + court title merge, decision formula ending `takto:`, operative text, reasoning heading/body, `Poučení:`, Brno closing date, and signature name + judicial role merge. `scripts/legal_v2/parser_review/snapshot.py` maps those parser-derived structures to review line classes without letting inline citations override the primary class.
- Snapshot and migration:
  Rebuilt the visual parser-review snapshot with v5 parser-derived data. High Court Prague and High Court Olomouc parser-derived snapshot subsets remained byte-identical by subset checksum. Existing document-2 manual decisions were preserved by appending 25 parser-profile migration revisions with the same manual values and `interface=parser_profile_migration`; no decisions were created for other documents and no assisted batches were applied.
- Audit result:
  Local artifacts under `artifacts/legal_v2/constitutional_parser_v5/` show 10 Constitutional Court documents parsed with exceptions 0, conservation failures 0, duplication failures 0, ordering failures 0, suspicious overmerges 0, suspicious undersplits 0. Document 2 Current Parser now has 11 blocks with ranges `1-1`, `2-2`, `3-3`, `4-5`, `6-6`, `7-7`, `8-8`, `9-9`, `10-10`, `11-11`, `12-13`. Corpus delta is 252 old blocks to 300 new blocks, 68 changed boundaries, and 148 changed line classes.
- Frontend:
  Added a read-only `Changed by parser v5` view backed by `/api/parser-v5/changes`, while existing Lines, Boundaries, Assisted Review, and Progress views remain available.
- Documentation:
  Added `docs/retrieval-enterprise/LEGAL_DECISION_PARSER_V5.md`.
- Known limitations:
  The remaining nine Constitutional Court design documents are not manually approved; their v5 deltas are a review queue, not correctness proof. Three pre-existing non-document-2 manual test decisions remain stale against parser profile v5 by design.
- Next recommended task:
  Review the `Changed by parser v5` queue, starting with the 68 boundary changes, then manually validate additional Constitutional Court documents before any broader indexing or retrieval rollout.

## 2026-08-04 Europe/Moscow - Task: Czech court format study and parser v4

- Goal:
  Build a reproducible raw/faithful-source format study for 10 Constitutional Court, 5 High Court Prague, and 5 High Court Olomouc design decisions plus a non-overlapping 10/5/5 holdout; generalize the Legal v2 parser only where the study provides structural evidence.
- Scope:
  Parser-format study in `fix/legal-paragraph-parser`. No `main` worktree edit, retrieval-core edit, Qdrant/BM25 access, index rebuild, embeddings, provider/model call, Docker, frontend, API, QuerySpec, verifier, or retrieval ranking change.
- Implementation:
  Added `scripts/legal_v2/court_format_study.py` to acquire official NALUS and Justice Open Data sources into ignored artifacts, create sample manifests, line and boundary annotations, taxonomy, evidence matrix, design/holdout validation, and acceptance reports. Parser profile is now `legal-decision-parser.cz-courts.v4`.
  `app/rag/legal_v2/ingest/parser.py` now recognizes standalone Roman section markers and bounded whole-line Czech court headings observed in the study while preserving the v3 invariant that numbered paragraphs cannot be overridden by keyword substring heading detection.
- Study result:
  Latest ignored artifacts under `artifacts/legal_v2/court_format_study/` recorded candidate population 93, design 20/20 with 10/5/5 court split, holdout 20/20 with 10/5/5 court split, no design/holdout overlap, design result pass, holdout result pass, parser exceptions 0, conservation failures 0, duplicate-text failures 0, ordering failures 0, orphan `sp. zn.` 0, orphan `č. j.` 0.
- Documentation:
  Added `docs/retrieval-enterprise/CZECH_COURT_FORMAT_STUDY.md` and linked it from the v3 parser note. Raw downloaded decisions remain ignored and are not committed.
- Known limitations:
  This is a bounded parser-format study for three court families, not universal Czech-court support and not retrieval-quality proof. Existing v2/v3 indexes and manifests were not modified.
- Next recommended task:
  Review the v4 study artifacts and, if approved, run a separate controlled task to rebuild an isolated v4 pilot index and measure chunk-boundary/retrieval impact.

## 2026-08-04 Europe/Moscow - Task: Legal paragraph parser v3 multiline numbered paragraphs

- Goal:
  Fix the Legal v2 parser so layout-wrapped numbered legal paragraphs remain one paragraph when continuation lines contain `sp. zn.`, `č. j.`, court names, or heading-like words such as `řízení`, `nález`, `odůvodnění`, and `posouzení`.
- Scope:
  Parser-only change in the dedicated `fix/legal-paragraph-parser` worktree. No `main` worktree change, retrieval-core worktree change, Qdrant/BM25 access, index rebuild, embeddings, provider calls, Docker, frontend, QuerySpec, verifier, or retrieval ranking change.
- Implementation:
  `app/rag/legal_v2/ingest/parser.py` now classifies line boundaries with explicit precedence: blank boundary, new numbered paragraph, verified genuine heading, active-numbered continuation, then prose. Numbered legal paragraph starts (`[N]`, `N.`, `N)`) cannot be headings merely because they contain heading keywords. Genuine whole-line headings remain preserved.
  Parser profile/version is now `legal-paragraph-parser.v3`; existing v2 indexes/manifests remain historical and were not modified.
- Tests and audit:
  Added regression coverage for confirmed paragraph 28 and paragraph 43 shapes, heading keyword false positives, `sp. zn.`/`č. j.` continuations, genuine headings, consecutive numbered paragraphs, heading/paragraph boundaries, text conservation, ordering, determinism, and parser profile version.
  Added read-only audit runner `scripts/legal_v2/audit_parser_fix.py`; generated parser-fix audit artifacts are local under ignored `artifacts/legal_v2/parser_fix/` and are not committed.
- Documentation:
  Added `docs/retrieval-enterprise/LEGAL_PARAGRAPH_PARSER_V3.md` documenting the root cause, corrected precedence, v3 profile, audit command, and future v3 index identifiers.
- Known limitations:
  This does not create `nalus_legal_paragraph_chunks_v3_pilot_600` or `nalus_legal_paragraph_bm25_v3_pilot_600`, and it does not prove retrieval-quality improvement. A later task must rebuild and validate an isolated v3 pilot before any benchmark or rollout claim.
- Next recommended task:
  Review the parser-fix audit and chunk-boundary delta, then use a separate controlled prompt to cherry-pick the single parser-fix commit into `main` and any approved downstream worktree.

## 2026-08-03 Europe/Moscow - Task: Retrieval enterprise architecture document set

- Created controlling architecture docs under `docs/retrieval-enterprise/` for the next retrieval modernization track:
  - `README.md`
  - `SYSTEM_ARCHITECTURE.md`
  - `IMPLEMENTATION_ROADMAP.md`
  - `PACKAGE_BOUNDARIES.md`
  - `CONTRACTS.md`
  - `DATA_AND_INDEX_LIFECYCLE.md`
  - `EVALUATION_PROTOCOL.md`
  - `SECURITY_AND_OPERATIONS.md`
  - `MIGRATION_AND_ROLLBACK.md`
  - `adr/0001-enterprise-retrieval-governance.md`
  - `adr/0002-phase-gated-additive-rollout.md`
- Purpose:
  Establish one shared enterprise specification before any new ColBERT, package-boundary, ingestion, profile, or Legal v2 integration work. Future implementation prompts must read and comply with this document set and accepted ADRs.
- Scope:
  Documentation only. No runtime code, frontend, Qdrant collection, BM25 sidecar, model/provider, benchmark gold data, package dependency, Docker, or environment setting was changed.
- Current baseline captured:
  Legal v2 remains the current isolated endpoint/pipeline with QuerySpec, BGE-M3 dense retrieval, BM25 sidecar, RRF, document aggregation, evidence selection, semantic verifier, and deterministic gate. Several top-level `app/rag/legal_v2/*.py` files are compatibility shims over the newer `ingest/`, `query/`, `retrieve/`, and `verify/` subpackages; future work should target the real subpackages.
- Next recommended task:
  Prompt 0 only: audit and definitive architecture validation against the real repository, without implementation, downloads, provider calls, new collections, commits, or pushes.

## 2026-08-03 Europe/Moscow - Task: FE related-candidate display for Legal v2 no-verified results

- Problem:
  User-facing FE search at `http://localhost:3017/vyhledavani` showed "no results" for `Matka unesla dítě z česka do Ruska`, even though backend retrieval found relevant pilot candidates. Backend log showed the request reached `POST /api/rag/search-v2` and completed with `status=no_verified_results`, `interpretation=ok`, `verified=0`, `related=5`, collection `nalus_legal_paragraph_chunks_v2_pilot_600`.
- Root cause:
  This was not a FE network failure and not an empty index. The Legal v2 gate correctly refused to promote `related_only` to verified after the regression fix, but the FE v2 mapper still consumed only `verified_documents`. `related_documents` returned by the backend were dropped, so the UI entered `NoResultsState`.
- FE change:
  In `NalusFE`, `LegalV2SearchResponse` now accepts optional `related_documents`. `mapLegalV2SearchResponse()` keeps verified documents first; only when `verified_documents` is empty does it map `related_documents` into UI results with `matchKind="related"` and `relevanceClassification`.
  `ResultList` labels all-related result sets as `Související rozhodnutí`, and `ResultCard` shows `Související kandidát` plus text that no fully verified match was found. This does not turn `related_only` into `verified_match`.
- Runtime verification:
  Rebuilt/recreated only the `NalusFE` frontend container. FE BFF `POST http://localhost:3017/api/retrieval/documents` for `Matka unesla dítě z česka do Ruska` returned `retrievalMode=v2`, `resultCount=5`. First result was `ECLI:CZ:US:2023:2.US.859.23.2`, `matchKind=related`, `relevanceClassification=related_only`.
  Backend log for the same run remained truthful: `status=no_verified_results`, `verified=0`, `related=5`.
- Validation:
  `npm run typecheck` and `npm run lint` in `NalusFE/frontend` passed. Docker frontend build passed using cached `npm ci` layer and Next production build.
- Safety:
  Backend Stage A, verifier gate, embeddings, BGE-M3 cache, pilot Qdrant, BM25, production resources, aliases, provider/model, secrets, and committed defaults were not changed. FE local v2 mode remains an explicit local runtime configuration.
- Remaining product work:
  Backend recall is still conservative for abduction-style queries: results are now visible as related candidates, not verified authorities. To make them verified, continue backend evidence/verifier tuning without re-promoting `related_only`.

## 2026-08-03 Europe/Moscow - Task: REGRESSION FIX — related_only must not verify

- Bug: soft-gate mistakenly promoted `related_only` → `verified_match` / FE results for
  „matka unesla dítě z Česka do Ruska“ (5 related_only docs counted as verified).
- Fix:
  1. `deterministic_verification_gate`: if classification=`related_only` → always `NOT_PROVEN`
     (even when provider decision is verified_match / hard constraints proven).
  2. Pipeline belt-and-suspenders: never append `related_only` into `verified_documents`.
  3. New response field `related_documents` (top related_only/partial, always returned;
     separate from verified). `rejected_documents` still debug-only.
- Live recheck: `status=no_verified_results`, verified=0, related=5
  (incl. `2.US.859.23.2`, `2.US.3057.25.1`, …). Log: `verified=0 related=5`.
- Unit tests: verifier/query_spec/e2e suite green; added
  `test_related_only_never_promoted_to_verified_match_by_gate`.
- Kept from prior soft-slot work: origin/destination/actor/event as SOFT for retrieval;
  legal_concept hard + lexical assist — but related_only can no longer pass the verify gate.
- FE: still maps only `verified_documents` → correctly shows NoResults for this query until
  optional related UI is wired.
- Next: optional FE `related_results` section; FA smoke on non-abduction queries; commit when asked.

## 2026-08-03 Europe/Moscow - Task: Soften abduction hard-gate (judgment-finder recall)

- Problem: lay query „Matka unesla dítě z česka do Ruska“ retrieved gold docs but gate required proving origin/destination/parent/child/event/relation as HARD → empty FE.
- Fix (code):
  1. `query/query_spec.py`: structural fact slots (origin/destination/actor/object/event/relation) → **SOFT**; location canonicalization (`česka`→Česká republika, `ruska`→Ruská federace); `unesla` action extract; legal_concept value = label only; `demote_structural_fact_slot_constraints` after build/merge; missing polarity defaults to **SOFT**.
  2. Interpreter prompt: soft for surface slots; hard reserved for dispositive legal requirements; merge also unions soft constraints.
  3. `verify/verifier.py`: lexical prove assist for `legal_concept:*` from supplied evidence windows; gate accepts `related_only`/`partial_match` once all remaining hard constraints are court_finding PROVEN (confidence ≥0.45 for those classes).
- Live `debug=true` after fix: **status=verified_match, verified=5, rejected=3** (~105s), including gold `ECLI:CZ:US:2023:2.US.859.23.2`.
- Unit tests: `test_legal_v2_query_spec` + `test_legal_v2_verifier` + `test_legal_v2_end_to_end` → **64 passed**.
- Risk: wider recall may raise FA on unrelated family-law docs labeled related_only — re-run 16-smoke / uq_001–003 / post-fix subset before committing policy as final.
- Next: FE check same query; optional focused smoke; commit when asked.

## 2026-08-03 Europe/Moscow - Task: COMPLETED — CZ→RU abduction empty-result root cause

- Query: **"Matka unesla dítě z česka do Ruska"**
- Verdict: **backend verifier fail-closed** — not FE bug, not empty index, not zero candidates.
- Evidence (`debug=true` live `search-v2`, ~72s):
  - Retrieval OK: dense 80 + bm25 80 → fused 120 → **40 candidate documents** on pilot_600.
  - QuerySpec: `origin=česka`, `destination=Ruska` (hard) + LLM entity/event hard constraints; log merge `origin,destination,hard_constraints`.
  - Verifier: **verified=0**, **rejected=8** (all `related_only` / `insufficient_evidence`); **0 hard constraints proven** on any candidate
    (missing: `hc_entity_parent|child|event_abduction|relation_abduction|loc_origin|loc_destination` + origin/destination constraint hashes).
  - Top rejected includes gold uq_001 docs: `2.US.859.23.2` (#1), `2.US.1626.22.1`.
  - Why earlier `rejected=0`: pipeline returns `rejected_documents=[]` / empty diagnostics unless `debug=True` (`app/rag/legal_v2/pipeline.py`) → FE only sees empty verified → NoResultsState.
- Corpus: `2.US.859.23.2` is closest hit (CZ↔Russia custody/jurisdiction, 12 rus* paragraphs; not a clean Hague „únos z ČR do Ruska“ narrative). Still fully rejected.
- Same class as post-fix eval: `uq_001` / `uq_002` / `uq_003` all `no_verified_results` with rejected_count=8.
- Artifacts:
  - `.../thinking_ab_test/fe_query_child_abduction_cz_ru_debug_20260803.json`
  - `.../thinking_ab_test/fe_query_child_abduction_cz_ru_root_cause_20260803.{json,md}`
- Next recommended task:
  Fix international-child-removal recall — why hard constraints stay not_proven on gold `2.US.859.23.2` (evidence windows / entity IDs / location genitive `česka`/`ruska` normalization). Optional: expose non-debug rejected/candidate counts to FE; do not treat this empty UI as connectivity failure.

## 2026-08-03 Europe/Moscow - Task: Live FE no-results diagnosis + child-abduction query handoff

- Product context (unchanged):
  AI judgment finder (not Q&A): query → verified court judgments.
  Pilot index `nalus_legal_paragraph_chunks_v2_pilot_600` + BM25 sidecar `nalus_legal_paragraph_bm25_v2_pilot_600`.
  Hard constraints: do not change Stage A / BGE-M3 / pilot Qdrant-BM25; no `rag2` rename; cost-sensitive; commit/push only when asked.
- Git audit (this handoff write):
  Branch `main`, HEAD `01b1e8f81ccb83071c1e7b21de0535be6a56ba03`.
  Dirty tree (pre-existing / in-progress, not overwritten): modified `.env.example`, `PROJECT_PROGRESS.md`, `app/api/rag_router.py`, interpreter/verifier packages, `reviewed_benchmark_v2.json`, verifier tests; large untracked `artifacts/` + dump/audit scripts. Do **not** commit `.env` secrets.
- Already recorded earlier today / 2026-08-02 (not re-done here):
  empty_message_content hardening (thinking retries default 3 + timeout bump; QuerySpec attempts default 3; `.env.example` knobs; unit tests).
  Benchmark corrections: `uq_028` and `uq_031` hard-negatives → strongly_relevant in `reviewed_benchmark_v2.json`.
  `hybrid_eval_59_post_fix.*`: 59/59, FA 0, FR 0 metric, cost ~$0.241; verified_match 15 / no_verified_results 44 (conservative fail-closed / recall).
- Manual content-fit audit (15 verified_match queries from post-fix eval):
  `artifacts/legal_v2/pilot_600_20260731/universal_quality/thinking_ab_test/manual_content_fit_audit_20260803.md` (+ `.json`).
  Headline: no verified doc is a hard-negative; most content fits the question; weaker/partial on `uq_016` / `uq_033` / `uq_044`.
- Full judgment dumps (BM25 sqlite, no LLM):
  `scripts/legal_v2/dump_eval_full_documents_from_qdrant.py` →
  `.../thinking_ab_test/document_reviews/*_full_documents.md` + `INDEX_from_bm25_dump.md`.
- Live FE + API enablement (local only; supersedes the brief enable note below on timeouts):
  Backend `.env`: `NALUS_LEGAL_V2_SEARCH_ENABLED=1`, pilot collection/BM25 paths, thinking hybrid + empty retries, `EMBEDDING_MODEL_NAME` = HF snapshot `5617a9f61b028005a4858fdac845db406aefb181`.
  Caution: `.env` was accidentally truncated mid-task and rewritten; API key preserved — do not log secrets.
  API container force-recreated; pilot verified (enabled, 13824 points, BM25 exists).
  Frontend repo `C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\NalusFE`:
  - `.env` + `frontend/.env`: `NALUS_RETRIEVAL_MODE=v2`, `NEXT_PUBLIC_NALUS_RETRIEVAL_MODE=v2`, API `http://host.docker.internal:8029`
  - Docker FE on http://localhost:3017 (`/vyhledavani`)
  - V2 path: `documentSearchServer.ts` → `POST /api/rag/search-v2`; maps only `verified_documents` via `mapLegalV2SearchResponse`
  - Timeout bumped 180s → **420s** (thinking often exceeds 3 min)
  - Default filters `"all"`
- Live smoke (nepřípustnost):
  Query: „kdy Ústavní soud odmítne ústavní stížnost jako nepřípustnou“
  Success ~215s: `verified_match`, 4 ECLIs initially; later re-runs often 3 verified docs
  (e.g. `1.US.2639.24.1`, `3.US.931.21.1`, `2.US.1321.25.1`, `3.US.2419.20.1` / variants).
- FE “no results” diagnosis (important):
  User saw FE “nenašli dostatečně relevantní rozhodnutí” though judgments exist for some queries.
  Findings:
  1. FE Docker logs almost empty (Next Ready only).
  2. API logs show FE/host `POST /api/rag/search-v2` → **200 OK** — FE reached API.
  3. FE `NoResultsState` = successful response with `results.length === 0` i.e. empty `verified_documents` (or mapped empty). Not “FE never called API”.
  4. Trace `api.legal_v2.search.done` existed but was DEBUG-only (`trace_event` gated by DEBUG) — hard to diagnose.
  5. Added **INFO** log in `app/api/rag_router.py`:
     `[api] legal_v2 search done status=… interpretation=… verified=… rejected=… collection=…`
  6. FE route logging + `maxDuration=420` edits in NalusFE `frontend/src/app/api/retrieval/documents/route.ts` —
     **production FE Docker image may not include these until rebuild** (`docker compose build/up` in NalusFE).
  7. Reproduce (post-recreate):
     - nepřípustnost via API: `verified_match`, verified=3 (~261s)
     - nepřípustnost via FE BFF `http://localhost:3017/api/retrieval/documents`: **resultCount=3**, retrievalMode=v2 (~105s)
     - So FE path itself works for that query.
  8. Historical FE-era clue: QuerySpec merge `fields=origin,destination,hard_constraints` (child-abduction-style) + concurrent `empty_message_content` warnings → often `no_verified_results`.
  9. Likely causes when UI shows empty: (A) different query / origin-destination QuerySpec fail-closed recall; (B) empty_message under load; (C) less likely filter wipe (defaults all); abort shows cancel/error path, not NoResults.
- Exact user query they care about — **"Matka unesla dítě z česka do Ruska"** (reproduced 2026-08-03):
  - API: `status=no_verified_results`, `interpretation=ok`, **verified=0**, **rejected=0**, ~81s
  - FE BFF: **resultCount=0**, retrievalMode=v2, ~71s
  - API log: `legal_v2.query_spec_merged fields=origin,destination,hard_constraints` then `legal_v2 search done status=no_verified_results … verified=0`
  - Conclusion: FE is correct for this query under current gate; empty UI is backend fail-closed / recall (or no candidates surfaced into rejected list), NOT a FE wiring bug for this string.
  - Slim artifact: `artifacts/legal_v2/pilot_600_20260731/universal_quality/thinking_ab_test/fe_query_child_abduction_cz_ru_20260803.json`
    (present; fields: status, elapsed_s, verified/rejected counts/ids, interpretation_status; QuerySpec payload null / incomplete — do not treat as full diagnostic dump).
- Ports / rollback:
  - API: http://localhost:8029 (`/api/rag/search-v2`)
  - FE: http://localhost:3017
  - Rollback: `NALUS_LEGAL_V2_SEARCH_ENABLED=0`; FE `NALUS_RETRIEVAL_MODE=legacy`
- Known limitations:
  Conservative gate yields many `no_verified_results` (44/59 in post-fix eval). Empty UI for abduction-style queries can be correct fail-closed behavior. Rejected candidates are not surfaced in FE. FE Docker may lag source edits until rebuild.
- Next recommended task:
  Diagnose why "Matka unesla dítě z česka do Ruska" returns verified=0/rejected=0 (candidates? QuerySpec too hard? pilot corpus coverage? verifier?).
  Optionally rebuild NalusFE for route INFO + maxDuration; product UX to surface rejected/diagnostics when verified empty; commit empty-retry / benchmark / progress / FE timeout only when user asks; never commit `.env` secrets.

## 2026-08-03 Europe/Moscow - Task: Enable local FE+API Legal v2 pilot for live testing

- Backend `.env`: `NALUS_LEGAL_V2_SEARCH_ENABLED=1` against pilot Qdrant/BM25
  `nalus_legal_paragraph_chunks_v2_pilot_600` / `nalus_legal_paragraph_bm25_v2_pilot_600` (13824 points).
- Thinking hybrid flags kept on; embedding path set to offline HF BGE-M3 snapshot.
- API container recreated; FE (`NalusFE`) set `NALUS_RETRIEVAL_MODE=v2` + public notice, rebuilt, running on `:3017`.
- Note: `search-v2` can exceed 180s on multi-candidate thinking path — FE timeout is 180s; expect long waits / occasional client timeout.
- Next: manual FE testing at http://localhost:3017/vyhledavani against pilot corpus.

## 2026-08-03 Europe/Moscow - Task: 59-eval post-fix completed + empty retries

- `hybrid_eval_59_post_fix.*`: **59/59** (resume kept 16 / reran 43). Stop none. Cost **~$0.241**.
- Quality vs prior `hybrid_eval_59_nonholdout`: FA **0 vs 14**; FR **0 vs 0**; verified_match **15 vs 26** (stricter gate / fewer approvals).
- Smoke gate **failed** (exit 2): structural `verifier_document_id_mismatch` on `uq_031` fast path (plus one tolerated network empty on `uq_058`); not an FA regression.
- Production hardening (working tree, uncommitted):
  - Thinking verifier: up to **3** extra retries on `empty_message_content` + timeout bump; fast path **1**.
  - QuerySpec default attempts **3** (`NALUS_LEGAL_V2_QUERYSPEC_MAX_PROVIDER_ATTEMPTS`).
  - Unit tests empty-content fast+thinking: **passed**.
- Next: optional mismatch hygiene; manual recall review of `no_verified_results`; commit empty-retry + `uq_031` benchmark correction when asked.
- Note: `artifacts/.../thinking_ab_test/PAUSED_59_POST_FIX.md` (now completion note).

## 2026-08-02 Europe/Moscow - Task: PAUSED — 16-smoke done, 59-eval mid-run

- Safe stop at user request (leaving session).
- 16-smoke `hybrid_smoke_16_post_uq028_fix.*`: **gate passed**, FA **0**, FR 0, interp 0, cost ~$0.102.
  - Live FA=1 on `uq_031` was benchmark error (`4.US.2338.25.1` fair-trial nález) → relabeled strongly_relevant → FA recomputed to 0.
- 59-eval `hybrid_eval_59_post_fix.*`: **paused at 16/59**, FA 0 so far, cost ~$0.101; fingerprint present for `--resume-json`.
- Pause note: `artifacts/legal_v2/pilot_600_20260731/universal_quality/thinking_ab_test/PAUSED_59_POST_FIX.md`
- Next step when back:
  1. Resume 59 with `--resume-json .../hybrid_eval_59_post_fix.json` (same budget flags).
  2. Compare FA/cost to prior `hybrid_eval_59_nonholdout`.
  3. Harden production empty_message_content retries (thinking must not give up after one empty 200).
  4. Commit `uq_031` benchmark correction if still uncommitted.

## 2026-08-02 Europe/Moscow - Task: uq_028 benchmark correction + incomplete-hard demote

- Goal:
  Stop treating content-correct `3.US.2419.20.1` as FA; unblock false rejection of clearly relevant refusals (e.g. `1.US.2639.24.1`) without further gate tightening against the query.
- Benchmark (`reviewed_benchmark_v2.json` item `uq_028`):
  Moved `ECLI:CZ:US:2020:3.US.2419.20.1` hard_negative → strongly_relevant (reason: §§9–15 subsidiarity / §43(1)(e)).
  Broadened strong: `3.US.931.21.1`, `1.US.2639.24.1`; material: `3.US.2242.20.1`, `2.US.262.24.2`; related_only: `3.US.2302.20.1`.
  Remaining HN for query: `2.US.3645.25.1` only.
  Correction note: `artifacts/.../thinking_ab_test/uq_028_benchmark_correction_20260802.md`.
- Runtime:
  Compact expand demotes exact/strong when hard PROVEN+`court_finding` coverage is incomplete (not only empty) → enables thinking escalation.
  Pipeline escalate safety net for exact/strong with ≤2 missing hard concepts.
  Smoke schema gate tolerates transient `empty_message_content` / timeout / network fail-closed (candidate still rejected).
- Live canary `hybrid_canary_uq028_after_benchmark_fix.*`:
  FA **0**, FR **0**, cost ~$0.010, verified **4** including gold `1.US.2639.24.1`, `3.US.931.21.1`, `3.US.2419.20.1` (+ `2.US.1321.25.1`).
  Smoke gate **passed** after transient-empty accounting.
- Unit tests: header-holding + incomplete-hard demote + verifier suite **40 passed**.
- Next step:
  Capped 16-smoke under budget; then review `uq_031`/`uq_037` separately (do not reverse this uq_028 correction).

## 2026-08-02 Europe/Moscow - Task: HEADER holding source repair + classification honesty

- Goal:
  Fix uq_028-style false negatives where relevant judgments had holding text mislabeled as `header`/`metadata`, so compact expand wiped all hard `PROVEN` and the gate rejected them — while also stopping exact/strong from advertising `verified_match` with zero holding-backed proof.
- Root cause:
  1. Indexed/retrieved paragraphs for some US refusals are almost all `section_type=header` → `source_of_claim=metadata` → expand/gate require `court_finding` → every hard PROVEN wiped (`proven=[]`, `source=unknown`).
  2. Compact expand still mapped exact/strong → `verified_match` even after that wipe (lying classification vs empty proof).
- Fix (no Stage A / index rebuild):
  `effective_source_of_claim` + `looks_like_court_holding_text` in `evidence/selection.py`: upgrade mislabeled header/unknown text that looks like operative holding to `court_finding`; soften HEADER ranking penalty for those paragraphs.
  Compact expand demotes exact/strong without any hard PROVEN+`court_finding` evidence to `insufficient_evidence` / `ambiguous` (`positive_classification_without_holding_proven_constraints`).
- Validation:
  `tests/rag/test_legal_v2_header_holding_source.py` + verifier suite: **39 passed**; compileall on touched packages; `git diff --check` clean on touched files.
- Paid provider calls made: **none** in this task.
- Next step:
  Re-run cheap `uq_028` canary with `--dump-full-documents` (expect rank 2/6 to survive if holdings still retrieved); then capped 16-smoke; only then capped 59.

## 2026-08-02 Europe/Moscow - Task: DeepSeek eval budget guard (no paid calls)

- Goal:
  Add strict provider-call and USD-cost budget accounting to Legal Retrieval v2 hybrid evaluation, without changing retrieval quality or Stage A/index.
- Implementation:
  Pricing table `deepseek_v4_2026_07_31` in `app/rag/llm/deepseek_pricing.py`.
  Thread-safe `EvalBudgetTracker` with pre-call reservation in `app/rag/llm/deepseek_eval_budget.py` (+ `app/rag/legal_v2/eval_budget.py` re-export).
  `DeepSeekTextLLM` captures usage into `last_meta` and settles/reserves when a tracker is bound; no prompts/keys/reasoning logged.
  Hybrid smoke CLI: `--max-cost-usd`, `--max-provider-calls`, `--max-queryspec-calls`, `--max-fast-verifier-calls`, `--max-thinking-fallback-calls`.
  Resume requires matching evaluation fingerprint (benchmark, policy, model, pricing table, budget config, index identity).
  Budget stop reasons are not treated as retrieval-quality failures; partial artifacts remain resumable.
- Validation (non-paid only):
  compileall; `tests/rag/test_deepseek_eval_budget.py` 21 passed; focused provider/verifier suites green; ruff clean on touched files; mypy on `app/rag/legal_v2` + `scripts/legal_v2`; `git diff --check` clean on new budget files.
- Paid provider calls made: **none**.
- Next step:
  One-query budget canary, then capped 16-smoke with explicit USD/call limits.

## 2026-08-02 Europe/Moscow - Task: Holding-quality gate + thinking promotion delta

- Goal:
  Cut remaining smoke false approvals (`uq_028`, `uq_031`, `uq_037`) without Stage A / index changes and without loading benchmark `excluded_concepts` into runtime.
- Code changes:
  Gate requires hard `PROVEN` with non-empty `court_finding` evidence; rejects explicit `holding_supports_query=False` / `legal_issue_match=False`.
  Compact expand + semantic validate fail-closed / downgrade non-holding `PROVEN`.
  Thinking promotion to `verified_match` only with PROVEN delta + new evidence IDs + `court_finding`.
  Pipeline document diagnostics: ECLI, rank, fast constraint snapshot, constraint status summary, promotion rejection codes.
  Smoke artifact adds per-candidate `candidate_documents` (benchmark labels eval-only) and `--query-ids` for capped retests.
- Validation:
  Unit tests for gate/promotion/logging shape added in `tests/rag/test_legal_v2_verifier.py` (run focused pytest before live spend).
- Live retest (cost-capped, user-run):
  Only `uq_028,uq_031,uq_037` via `--query-ids`; gate target FA 0/3 with ECLI/evidence logged. Do **not** run 16-smoke or 59 until that 3-query gate passes.
- Next step:
  Focused unit tests → capped 3-query live retest → new 16-smoke → only then capped 59.

## 2026-08-02 Europe/Moscow - Task: Offline forensics of 3 smoke false approvals

- Goal:
  Complete detailed root-cause cards for smoke FA cases `uq_028`, `uq_031`, `uq_037` with **zero** live LLM / DeepSeek calls.
- Method:
  Joined `hybrid_smoke_16_quality_fix.json`, pre-fix `hybrid_eval_59_nonholdout.json`, and `reviewed_benchmark_v2.json`.
- Results:
  All three hard-negative sets are also `related_only`. Path split: `uq_028`/`uq_031` FA from **fast** exact/strong; `uq_037` FA from **thinking fallback** (fast had 0 verified). `uq_031` has `verified_count==1==FA`, so the sole verified doc is a hard-negative (one of two listed ECLIs). Smoke artifact still lacks verified document IDs / evidence refs — exact ECLI for 028/037 cannot be named offline.
- Artifacts:
  `artifacts/legal_v2/pilot_600_20260731/universal_quality/thinking_ab_test/false_approval_offline_forensics_3.md` and `.json`.
- Next step:
  Fix thinking promotion policy (037), excluded-concept/fair-trial negative gate (031), exact_match overconfidence (028); optionally extend smoke logging for document IDs. No Stage A change.

## 2026-08-01 Europe/Moscow - Task: Legal v2 quality fixes + in-place modularity

- Goal:
  Fix QuerySpec `hard_constraints_lost` and verifier false-approval root causes first, then split fat `legal_v2` modules into subpackages with compatibility re-export shims. No Stage A / embedding / pilot index changes; no `rag2` rename.
- Phase 1A QuerySpec:
  Tolerant constraint parse (default category/polarity; stable `hashlib` constraint ids).
  Prompt schema includes hard `polarity` + allowed categories; intent hint no longer uses `clarification`.
  Deterministic merge of origin/destination/hard constraints/negations/entity roles before preservation validation (`query_interpreter_merged:...`).
- Phase 1B Verifier:
  `VERIFIED_MATCH` only for `exact_match`/`strong_match`; partial/related → `AMBIGUOUS`.
  Aliases: strongly/materially → `PARTIAL_MATCH`; `not_relevant` → `INSUFFICIENT_EVIDENCE`.
  Gate also requires provider `VERIFIED_MATCH`, rejects explicit `jurisdiction_match=False`, confidence `<0.6`, and non-empty `contradictory_facts`.
  Compact evidence: max 2/constraint and 12 total, hard-first concepts, evidence text limit 700.
  Hallucinated compact concept IDs are dropped instead of fail-closed; unknown evidence IDs still fail closed.
- Phase 2 modularity:
  Packages `verify/`, `query/`, `interpret/`, `retrieve/`, `evidence/`, `ingest/` with thin shims on original import paths (`verifier.py`, `query_spec.py`, `interpreter.py`, `retriever.py`, `parser.py`, …). Behavior-preserving move.
- Validation:
  `python -m compileall app/rag/legal_v2` + focused legal_v2 pytest: **94 passed**.
  16-query hybrid smoke `hybrid_smoke_16_quality_fix.*`: gate **passed** after resume of 2 schema/network flakes (kept 14). QuerySpec schema 100%, interpretation_failures **0**, false approvals **3** (prior smoke had 6). Fast/thinking schema + evidence-ID success true. Retrieval/prompt-injection/wrong-index 0.
- Artifacts:
  `artifacts/legal_v2/pilot_600_20260731/universal_quality/thinking_ab_test/hybrid_smoke_16_quality_fix.json` / `.md`.
- Constraints preserved:
  Stage A unchanged; pilot Qdrant/BM25 immutable; no commit/push; API container still has stale `LLM_API_KEY` (host `.env` suffix differs) — pass `-e LLM_API_KEY=...` or recreate api for live runs; key rotation still required before push.
- Next step:
  Optional full 59-eval under the tightened gate to measure FA/FN vs prior `hybrid_eval_59_*`; then quality review of remaining false approvals before enablement.

## 2026-08-01 Europe/Moscow - Task: Complete 59-query hybrid non-holdout evaluation

- Goal:
  Finish the full available non-holdout hybrid thinking evaluation after mid-run DeepSeek DNS outages blocked the first attempt at 51/59.
- Resilience changes:
  QuerySpec and thinking-fallback escalation treat `network_error` as retryable/escalatable.
  Full eval (`query-limit > 16`) uses `LLM_RETRY=2`, disables QuerySpec early-stop, supports `--resume-json` with checkpoint writes, and keeps 16-smoke early-stop behavior.
- Result:
  `hybrid_eval_59_nonholdout.json` completed **59/59** (resume kept 39, reran 20). Stop reason none. Strict smoke gate **failed** on quality debt, not infra abort.
  Status mix: verified_match 26, unverifiable_query 15, no_verified_results 9, query_interpretation_error 9.
  Interpretation failures are all `hard_constraints_lost` (not network). False approvals 14. Retrieval errors / prompt-injection / wrong-index 0. Final transient network failures 0.
  Latency: p50 ~113s, p95 ~270s. Fast verifier calls 211, thinking fallback 69.
- Artifacts:
  `artifacts/legal_v2/pilot_600_20260731/universal_quality/thinking_ab_test/hybrid_eval_59_nonholdout.*`,
  `hybrid_eval_59_report.md` / `.json`,
  `hybrid_eval_59_nonholdout.partial51.*` (pre-resume backup).
- Constraints preserved:
  Stage A unchanged; pilot immutable; no embedding rebuild; no commit/push; key rotation still required before push.
- Next step:
  Investigate QuerySpec `hard_constraints_lost` (especially ambiguous cases) and review the 14 false-approval rows before any enablement.

## 2026-08-01 Europe/Moscow - Task: Fix fast-verifier fail-closed and re-gate hybrid smoke

- Goal:
  Unblock the hybrid 16-query smoke gate after thinking A/B policy selection, then run the full available non-holdout semantic evaluation.
- Root causes fixed:
  1. Compact payload expansion invented `PROVEN`/`CONTRADICTED` without matching constraint evidence or with restricted-only sources (`party_claim`/`cited_case`), which later fail-closed.
  2. Fast-verifier provider timeouts were not escalated to thinking fallback because all `failed_closed` outcomes were excluded.
- Code changes:
  `app/rag/legal_v2/verifier.py` downgrades invalid compact expansions to `NOT_PROVEN` instead of emitting terminal statuses that fail closed; prompt clarifies concept/evidence alignment and restricted sources.
  `app/rag/legal_v2/pipeline.py` escalates recoverable fail-closed reasons (`timeout`, empty content, invalid JSON) to one thinking attempt.
- Validation:
  Compact expansion + escalation unit tests passed.
  16-query hybrid smoke re-run passed (`hybrid_smoke_16_rerun2.json` / promoted `hybrid_smoke_16.json`): QuerySpec/fast/thinking schema 100%, evidence-ID 100%, retrieval errors 0, prompt-injection 0. False approvals remain 6 (quality debt, not schema gate).
- Full evaluation:
  Benchmark has 59 diagnostic+tuning rows (not 64). Full non-holdout hybrid eval started as `hybrid_eval_59_nonholdout.*`.
- Constraints preserved:
  Stage A unchanged; pilot Qdrant/BM25 immutable; no commit/push; historical key rotation still required.
- Next step:
  Collect 59-query eval results; then quality review of remaining false approvals before any push.

## 2026-08-01 Europe/Moscow - Task: Fair thinking vs non-thinking Legal v2 A/B and hybrid policy

- Goal:
  Replace the insufficient 30-second QuerySpec diagnostic with a fair thinking-versus-non-thinking quality evaluation at a 120-second ceiling, then select a quality-first hybrid production policy without modifying Stage A or rebuilding embeddings.
- Starting audit:
  Branch `main`, HEAD `e0396d4ef08d9525c05d2fac8110698435f30aa1`. Large pre-existing dirty Legal v2 / artifact worktree was preserved and not restored or overwritten.
- Why 30 seconds was insufficient:
  Thinking QuerySpec completed with final `message.content` in about 9–26 seconds under a fair 120-second ceiling. A 30-second cutoff incorrectly treated slow-but-valid legal thinking as failure.
- A/B design:
  4 diagnostic/tuning intents (actors, countries/jurisdictions, procedural, ambiguous). QuerySpec 8 calls and verifier 12 calls, identical prompts/schemas/evidence/timeouts within each comparison, explicit `thinking` enabled/disabled only. QuerySpec scored via production `interpret_query_spec_v2`.
- QuerySpec result:
  Thinking schema success 3/4, non-thinking 2/4, timeouts 0. Thinking uniquely preserved mother/child roles and succeeded on the clarification case. Selected: thinking enabled, timeout 120s, max_tokens 8000, max 2 provider attempts.
- Verifier result:
  Non-thinking 6/6 schema success at ~2.3s average with compact evidence-ID payload retained. Thinking 5/6 with final content within 120s, not materially better on false approvals. Selected: fast non-thinking verifier at 30s, thinking fallback only for difficult classifications, max 2 candidates/query.
- Structural gate:
  2/2 thinking QuerySpec, 2/2 fast verifier, 2/2 thinking fallback, 0 timeouts, gate passed.
- 16-query hybrid smoke:
  Second run exercised real retrieval/verification after BM25 `Path` coercion and a shared-retriever lock: 16/16 QuerySpec schema success, 73 fast verifier calls, 15 thinking-fallback calls, 0 retrieval errors, 0 timeouts, 0 prompt-injection, thinking-fallback schema 100%. Smoke gate did **not** pass because fast verifier had fail-closed schema/evidence-ID failures (`verifier_evidence_required_for_terminal_status`, `verifier_restricted_source_claim_used_as_proof`) and 6 hard-negative false approvals. Full 64-query evaluation remains blocked by this smoke gate.
- Immutability:
  Pilot Qdrant 13824 / BM25 13824 / checksum `85ceb99dfc9bbf682d59628d6efdb861b61ce96dac3e9946583f7eb4f7de816f` unchanged; production resources untouched; no GPU/CUDA/downloads/Redis required for this task.
- Security:
  `.env.example` has one `LLM_API_KEY=your-api-key-here`. Historical key-like value remains in seven commits. Key rotation required; safe to push before rotation: no. No commit/push performed.
- Artifacts:
  `artifacts/legal_v2/pilot_600_20260731/universal_quality/thinking_ab_test/` (`current_state.md`, `case_selection.*`, `ab_results.*`, `quality_review.md`, `mode_policy.json`, `structural_gate.*`, `hybrid_smoke_16.*`).
- Next step:
  Reduce fast-verifier fail-closed rate on the compact evidence-ID path (empty/terminal evidence and restricted-source proof), re-run the 16-query hybrid smoke to gate pass, then run the full 64-query semantic evaluation under the selected hybrid policy. Do not push before key rotation.

## 2026-07-31 Europe/Moscow - Task: Forensic audit previous five-year build expectation

- Goal:
  Read-only audit of why the earlier BGE-M3 indexing was believed to take about three hours while current Legal Retrieval v2 estimates about `91.5h`; no build, tests, Qdrant/BM25 mutation, aliases, downloads, GPU/CUDA, frontend, or runtime source edits.
- Result:
  Claim classification: `PARTIALLY_SUPPORTED`. The approximately three-hour evidence applies to `mvp_recent_3h`, a `600`-document newest-first slice producing `4,980` chunks in about `2h41m`, not to a completed five-year build. The old five-year candidate `mvp_5y` stopped in progress at `650` indexed documents and `8,335` inserted points out of `18,062` selected records and `155,414` expected chunks.
- Root cause:
  The old three-hour slice was conflated with a full five-year build. Current Legal Retrieval v2 has a larger six-year scope, higher chunk density, paragraph-aware payloads, richer BM25 metadata, checkpointing, and validation; old vectors are not directly reusable because chunk boundaries, payload schema, and fingerprints differ.
- Artifacts:
  `artifacts/evaluation_quality/previous_five_year_build_audit_20260731.md` and `artifacts/evaluation_quality/previous_five_year_build_audit_20260731.json`.
- Recommendation:
  `E. RECONSTRUCT_EXPECTATION`; rebaseline expected CPU build time from document count, chunk density, and measured per-phase throughput before any full build.

## 2026-07-31 Europe/Moscow - Task: Gate six-year Legal Retrieval v2 CPU build

- Goal:
  Prepare and gate a CPU-only Legal Retrieval v2 build for the exact decision-date window `2020-07-31` through `2026-07-31`, without GPU/CUDA, model/package downloads, frontend changes, `search-v2` enablement, or production Qdrant collection changes.
- Starting audit:
  Branch `main`, HEAD `91bd906d4383cb81ed9f1b13fd99687b021599b0`. Pre-existing tracked changes were already present in `PROJECT_PROGRESS.md`, `docs/LEGAL_RETRIEVAL_V2.md`, `app/rag/legal_v2/index_builder.py`, `scripts/legal_v2/build_index.py`, and `tests/rag/test_legal_v2_end_to_end.py`, plus generated/untracked artifacts under `artifacts/**`. They were not reverted.
- What changed:
  Added shared Legal v2 decision-date parsing/filtering for ISO and Czech date formats.
  Added `--decision-date-from` and `--decision-date-to` to `scripts/legal_v2/build_index.py` and `scripts/legal_v2/source_inventory.py`.
  Added source-selection/date-filter stats to the Legal v2 build manifest.
  Added batch checkpointing through `legal_v2_execute_checkpoint.json`, with guarded `--resume` and intentional `--stop-after-document-batches` for safe stop/resume testing.
  Updated `docs/LEGAL_RETRIEVAL_V2.md` to document the six-year window, inventory command, date-filtered build command, and checkpoint resume rules.
- Source range inventory:
  Command: `docker compose exec -T api python scripts/legal_v2/source_inventory.py --decision-date-from 2020-07-31 --decision-date-to 2026-07-31 --json-output /app/artifacts/legal_v2/source_inventory_20260731_6y.json --markdown-output /app/artifacts/legal_v2/source_inventory_20260731_6y.md`.
  Total discovered source documents: `103,638`.
  Exact in-range documents: `21,776` total, comprising `21,626` constitutional and `150` supreme documents.
  Out-of-range documents: `81,862`.
  Missing/invalid decision dates in discovered complete documents: `0`.
- Runtime constraints verified:
  Only `api` and `qdrant` containers were started. `docker compose ps` showed no Redis, exporter, Prometheus, or Grafana.
  API container uses CPU-only PyTorch: `torch 2.6.0+cpu`, `cuda_available=False`, `cuda_device_count=0`.
  Offline flags remained set: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `EMBEDDING_DEVICE=cpu`.
  Existing BGE-M3 snapshots were present in the Docker HuggingFace cache. No model/package download was triggered.
  `NALUS_LEGAL_V2_SEARCH_ENABLED=0` remained disabled.
- 100-document checkpointed CPU test:
  First command intentionally stopped after one 50-document batch with exit code `75`, checkpoint present, Qdrant points `904`, BM25 rows `904`, elapsed `760.3s`.
  Resume command completed the same 100-document request with `--resume`, exit code `0`, elapsed `751.4s`.
  Final 100-document manifest: `100` indexed documents, `1,810` chunks, `30` Qdrant upsert batches, `1,810` Qdrant points, validation status `pass`, Qdrant write `pass`, BM25 write `pass`.
  Checkpoint was cleared after successful completion.
  Post-run reconciliation: Qdrant points `1,810`, BM25 rows `1,810`, match `true`.
- Safety state:
  Pre/post safety snapshots are under `artifacts/legal_v2/build_6y_20260731/safety_prebuild/` and `artifacts/legal_v2/build_6y_20260731/safety_post_100/`.
  `nalus_live` alias remained pointed at `nalus_stable_20260326`.
  Production collections `nalus` and `nalus_stable_20260326` retained their pre/post point counts.
  Only isolated `nalus_legal_paragraph_chunks_v2` and isolated `nalus_legal_paragraph_bm25_v2.sqlite` changed.
- Feasibility result:
  The 100-document gate passed functionally but failed practical CPU feasibility.
  Measured throughput was about `100 docs / 1511.7s`, `1810 chunks / 1511.7s`, roughly `238 docs/hour` or `1.2 chunks/second`.
  Extrapolated full six-year scope is about `21,776` documents and about `394k` chunks at the observed chunk density, yielding about `91.5 hours` of CPU wall time before validation overhead. A 500-document CPU test would likely take a little over `2 hours`.
  Because the 100-document gate already revealed impractical CPU runtime, the 500-document test and full six-year build were not started.
- Resource estimate:
  API memory after the run was about `1.9 GiB`; Qdrant memory about `3.5 GiB`.
  C: free space was about `160.9 GB`.
  The v2 BM25 sidecar was `15.1 MB` for `1,810` chunks; linear estimate for full six-year BM25 is roughly `3.3 GB`.
  Qdrant collection disk usage was about `622 MB` for the small v2 test collection, while existing 8k-13k BGE-M3 collections were about `656 MB`; because fixed overhead dominates small collections, realistic full six-year Qdrant disk should be treated as a multi-GB range rather than a reliable linear estimate.
- Validation:
  `python -m pytest -q tests\rag\test_legal_v2_end_to_end.py tests\rag\test_legal_v2_source_inventory.py` -> `13 passed`, one non-blocking Starlette/httpx warning.
  `ruff check app\rag\legal_v2\index_builder.py app\rag\legal_v2\sources.py app\rag\legal_v2\source_inventory.py scripts\legal_v2\build_index.py scripts\legal_v2\source_inventory.py tests\rag\test_legal_v2_end_to_end.py tests\rag\test_legal_v2_source_inventory.py --no-cache` -> passed.
  `python -m mypy app/rag/legal_v2/index_builder.py app/rag/legal_v2/sources.py app/rag/legal_v2/source_inventory.py scripts/legal_v2/build_index.py scripts/legal_v2/source_inventory.py` -> passed.
- Current status:
  Full six-year build was not started. `search-v2` remains disabled. Frontend unchanged. Production Qdrant unchanged.
  Isolated v2 test index currently contains the completed 100-document / 1,810-chunk test state.
- Next recommended task:
  Do not start the full six-year CPU build on this runtime. Either accept a multi-day CPU build window, provide precomputed embeddings/index artifacts, or use an explicitly approved faster indexing runtime. If another CPU-only gate is required, run the 500-document test as a separate long-running benchmark, not as an automatic precursor to full build.

## 2026-07-31 Europe/Moscow - Task: Pause Legal Retrieval v2 rollout safely

- Goal:
  Stop the current Legal Retrieval v2 rollout work safely because the user needs to shut down and clean the PC now. Do not continue benchmarking, indexing, optimization, frontend work, provider testing, commits, or pushes.
- Timestamp:
  `2026-07-31T16:32:06+03:00`.
- Current process state:
  No `build_index.py` process was running when inspected. `docker compose top api` showed only the Uvicorn API process. No SIGINT or SIGKILL was required.
  The last bounded builder command had already completed: restoring the 20-document smoke index with `--limit 20 --batch-size 64 --document-batch-size 5`.
- CPU/runtime policy recorded:
  Future work is CPU-only. GPU, CUDA, NVIDIA runtime use, package downloads, and model downloads are prohibited unless the user explicitly changes this policy later.
  Current API container confirms CPU-only PyTorch: `torch 2.6.0+cpu`, `cuda_available=False`, `cuda_device_count=0`.
- Current isolated v2 state:
  Qdrant collection `nalus_legal_paragraph_chunks_v2`.
  BM25 identifier `nalus_legal_paragraph_bm25_v2`.
  Latest manifest: `artifacts/legal_v2/smoke_index_20260730/index_build_restore_20260731/legal_v2_build_manifest.json`.
  Latest manifest result: 20 source documents, 20 indexed documents, 384 chunks, 8 Qdrant upsert batches, 384 Qdrant upsert points, vector dimension 1024, validation status `pass`.
  Qdrant/BM25 verification: Qdrant point IDs `384`, BM25 row IDs `384`, ID mismatch `0`.
  20-document smoke restoration status: completed.
  Partial batch detected: no.
  Checkpoint available: no separate checkpoint; the completed manifest is available.
- Protected resources:
  Production Qdrant collections unchanged.
  Aliases unchanged.
  Production BM25 sidecars unchanged.
  Redis unchanged.
  Frontend unchanged by this pause task.
  `NALUS_LEGAL_V2_SEARCH_ENABLED` remains disabled by default and was not enabled locally.
- Build scope decision:
  Do not start the complete 103,638-document historical build again.
  The intended next final index scope is only the rolling five-year date window `2021-07-31` through `2026-07-31`.
  Before the next build, verify actual source metadata and report documents inside the range, outside the range, missing/invalid decision dates, incomplete sources inside the range, and duplicate source identifiers inside the range.
  Do not silently include documents without a valid decision date.
  The current builder does not expose a safe date-range filter; implementing and testing that filter is the first requirement for the next session.
- Exact safe resume command:
  Do not resume a build directly. First implement and verify the five-year date-range filter. The safe smoke restore command, if the 20-document smoke index must be recreated, is:
  `docker compose exec -T api python scripts/legal_v2/build_index.py --parser-quality-artifact /app/artifacts/legal_v2/parser_quality_gate_20260730/parser_quality_gate.json --gate-decision /app/artifacts/legal_v2/parser_quality_gate_20260730/gate_decision.json --limit 20 --qdrant-url http://qdrant:6333 --output-dir /app/artifacts/legal_v2/smoke_index_20260730/index_build_restore_20260731 --overwrite-bm25 --recreate-v2-collection --batch-size 64 --document-batch-size 5`.
- Exact next task:
  Add a safe Legal v2 source date inventory and builder date-range filter for `2021-07-31` through `2026-07-31`, prove that missing/invalid dates are excluded or reported, then run only a bounded CPU feasibility smoke before any five-year index build.
- Current tracked changes:
  `PROJECT_PROGRESS.md`, `docs/LEGAL_RETRIEVAL_V2.md`, `app/rag/legal_v2/index_builder.py`, `scripts/legal_v2/build_index.py`, `tests/rag/test_legal_v2_end_to_end.py`.
- Current untracked/generated changes:
  Existing generated reports under `artifacts/evaluation_quality/*.json` and `*.md`; generated Legal v2 outputs under `artifacts/legal_v2/`; existing answer-eval outputs under `artifacts/rag_eval/legal_qa/answer_eval/{mixed_document_gold_default,nsoud_document_gold_default,usoud_document_gold_default}/`; existing `artifacts/rag_eval/legal_v2_seed_comparison_20260723/`.
- Lightweight verification:
  Qdrant/BM25 consistency check -> `qdrant_points=384`, `bm25_rows=384`, `id_mismatch=0`.
  Final `git status --short`, `git diff --check`, and `docker compose ps` are recorded in the final response for this pause task.

## 2026-07-31 Europe/Moscow - Task: Attempt full Legal Retrieval v2 local rollout

- Goal:
  Execute the requested end-to-end Legal Retrieval v2 local rollout: full isolated v2 index, reviewed relevance benchmark, local backend enablement, explicit frontend v2 mode, validation, commit, and push, stopping only if an acceptance gate or genuine blocker prevents safe continuation.
- Starting audit:
  Backend branch `main`, HEAD `91bd906d4383cb81ed9f1b13fd99687b021599b0`, `origin/main` matched. Required backend governance and feature docs were read: `AGENTS.md`, `PROJECT_EXECUTION_PROTOCOL.md`, `PROJECT_PROGRESS.md`, `README.md`, and `docs/LEGAL_RETRIEVAL_V2.md`.
  Backend dirty work before editing was classified as generated local `artifacts/**` only.
  Frontend branch `main`, HEAD `9da811aca1d4f086d31f7e02e180777be296b043`. Frontend dirty work was classified as an existing uncommitted MVP retrieval/frontend/Docker integration in `../.env.example`, `../docker-compose.yml`, `frontend/.env.example`, `Dockerfile`, `README.md`, `package*.json`, `src/app/**`, `src/components/**`, `src/data/**`, `src/lib/**`, and `src/types/**`. It was not overwritten.
- Runtime readiness:
  `docker compose ps` showed backend API, Qdrant, Redis, exporter, Prometheus, and Grafana running.
  C: free space was about 152.6 GB, enough for an attempted isolated build.
  `docker compose exec -T api python -c "import qdrant_client; print(qdrant_client.__version__)"` cannot print `__version__` because this installed module does not expose it; package metadata reports `qdrant-client 1.13.3`.
  BGE-M3 was available from the existing Docker cache at `/root/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181`.
  Safe DeepSeek direct and provider smoke checks passed without printing the API key.
  Docker has an `nvidia` runtime installed, but the current API container uses CPU-only PyTorch: `torch 2.6.0+cpu`, `cuda_available=False`, `cuda_device_count=0`.
- Step:
  Add bounded Legal v2 full-index builder processing.
  Goal: prevent full-corpus indexing from building all vectors in memory and disappearing without a manifest.
  Files inspected: `app/rag/legal_v2/index_builder.py`, `scripts/legal_v2/build_index.py`, `tests/rag/test_legal_v2_end_to_end.py`, `docs/LEGAL_RETRIEVAL_V2.md`.
  Files changed: `app/rag/legal_v2/index_builder.py`, `scripts/legal_v2/build_index.py`, `tests/rag/test_legal_v2_end_to_end.py`, `docs/LEGAL_RETRIEVAL_V2.md`.
  Behavior changed: the isolated v2 builder now processes document batches, embeds/upserts in bounded batches, writes BM25 incrementally, emits progress events, and records `batch_size`, `document_batch_size`, Qdrant batch count, and upsert point count in the manifest. The CLI adds `--batch-size` and `--document-batch-size`.
  Tests added/updated: regression test proving the builder does not pass all payloads to the embedder at once.
  Verification command: `python -m pytest -q tests\rag\test_legal_v2_end_to_end.py::test_index_builder_writes_only_v2_identities tests\rag\test_legal_v2_end_to_end.py::test_index_builder_embeds_and_upserts_in_configured_batches tests\rag\test_legal_v2_qa_gate.py`; `ruff check app\rag\legal_v2\index_builder.py scripts\legal_v2\build_index.py tests\rag\test_legal_v2_end_to_end.py --no-cache`; `python -m mypy app/rag/legal_v2/index_builder.py scripts/legal_v2/build_index.py`.
  Result: pytest `13 passed` with one non-blocking Starlette/httpx deprecation warning; Ruff passed; Mypy passed.
  Risk: isolated to the Legal v2 builder; no model, embedding dimension, scoring, aliases, Redis behavior, provider configuration, endpoint default, frontend behavior, or legacy retrieval changed.
  Next step: run Docker builder smoke, then decide whether full build is feasible.
- Step:
  Attempt and reconcile full isolated v2 index build.
  Goal: build `nalus_legal_paragraph_chunks_v2` and `nalus_legal_paragraph_bm25_v2` from the complete available corpus.
  Files inspected: `artifacts/legal_v2/index_build_full_20260731/source_inventory.json`, pre/post safety snapshots, Docker/Qdrant state.
  Files changed: generated local artifacts under `artifacts/legal_v2/index_build_full_20260731/` and restored smoke output under `artifacts/legal_v2/smoke_index_20260730/index_build_restore_20260731/`; isolated v2 Qdrant collection and v2 BM25 sidecar were rewritten only for smoke restore.
  Behavior changed: no committed runtime behavior changed.
  Tests added/updated: none beyond builder tests.
  Verification command: `docker compose exec -T api python scripts/legal_v2/source_inventory.py ...`; `docker compose exec -T api python scripts/legal_v2/smoke_safety_snapshot.py --phase prebuild ...`; full build command with `--overwrite-bm25 --recreate-v2-collection --batch-size 64`; reconciliation with Qdrant/BM25 counts; bounded Docker smoke with `--limit 5`; smoke restore with `--limit 20`.
  Result: source inventory found 103,638 discovered source documents, 55 missing complete text records, 502 duplicate source-document identifiers, 0 unreadable files, and 0 unsupported formats. The first full build attempt returned from the command without a manifest and without changing the v2 collection or sidecar; reconciliation showed the previous smoke state still at 384 Qdrant points and 384 BM25 rows. The new stream builder Docker smoke passed for 5 documents / 74 chunks, but took about 88 seconds of builder duration. Restoring the 20-document smoke index passed with 20 documents / 384 chunks, Qdrant points 384, BM25 rows 384, vector dimension 1024.
  Risk: full corpus indexing in the current CPU-only Docker API runtime is not practically executable today. The 20-document smoke state was restored; no production collection, alias, production BM25, Redis, provider, or frontend behavior was changed.
  Next step: do not proceed to benchmark/frontend rollout until a GPU-enabled or otherwise accelerated API indexing runtime is available, or a precomputed full v2 embedding/index artifact is provided.
- Blocker:
  Full-corpus Legal Retrieval v2 local rollout is blocked by the current Docker API runtime being CPU-only for BGE-M3 embeddings. Measured real Docker build throughput was 20 reviewed smoke documents / 384 chunks in about 505 seconds after stream-builder changes. The expected full index is on the order of 1.5-2.0 million chunks from 103,583 complete indexed source documents, which extrapolates to many days on the current CPU-only runtime. Continuing to Phase 3-7 would violate the prompt because the required full isolated index and relevance gate have not passed.
- Current status:
  Full v2 index was not built. Local v2 collection and BM25 sidecar are restored to the prior isolated smoke scale: 384 Qdrant points and 384 BM25 rows. `NALUS_LEGAL_V2_SEARCH_ENABLED` remains disabled by default. Frontend v2 mode was not connected. No commit or push was made because acceptance criteria did not pass.
- Next recommended task:
  Do not use GPU/CUDA or the NVIDIA runtime. Add and verify a safe five-year date-range filter for `2021-07-31` through `2026-07-31`, then evaluate CPU feasibility on that reduced scope before any resumed indexing.

## 2026-07-31 Europe/Moscow - Task: Complete live Legal Retrieval v2 API endpoint wiring

- Goal:
  Complete, validate, and document the existing uncommitted live `POST /api/rag/search-v2` implementation in `app/api/rag_router.py` without enabling Legal Retrieval v2 by default and without touching frontend, Qdrant data, BM25 data, Redis, embeddings, scoring, aliases, or production retrieval endpoints.
- Starting audit:
  Branch `main`, HEAD `54dac350dced5a1198f2ae0103cf1945bb036620`, `origin/main` matched HEAD.
  Required governance and feature docs were read: `AGENTS.md`, `PROJECT_EXECUTION_PROTOCOL.md`, `PROJECT_PROGRESS.md`, `README.md`, `docs/LEGAL_RETRIEVAL_V2.md`, `docs/DOCUMENT_LEVEL_RETRIEVAL.md`, and `docs/CONSTRAINT_AWARE_RETRIEVAL.md`.
  Previous audit artifacts were read: `artifacts/evaluation_quality/rag_router_dirty_diff_audit_20260731.md` and `.json`.
  Dirty worktree was classified before editing: intended tracked work was the existing `app/api/rag_router.py` live-route diff; generated/untracked local output remained under `artifacts/**`; no staged files, merge, rebase, or cherry-pick was in progress.
- Scope:
  In scope: `app/api/rag_router.py`, route registration cleanup in `app/api_app.py`, removal of the superseded guard router, focused API tests in `tests/api/test_rag_api.py`, `docs/LEGAL_RETRIEVAL_V2.md`, and this progress entry.
  Out of scope: frontend, `/api/rag/retrieve`, `/api/rag/retrieve-documents`, `/api/rag/retrieve-verified`, `/api/rag/query`, full-document reconstruction, Qdrant writes/rebuilds, BM25 rebuilds, aliases, Redis, provider credentials, embedding model/dimension, scoring formulas, full v2 index build, and unrelated generated artifacts.
- Step:
  Consolidate `search-v2` route registration.
  Goal: make exactly one registered `POST /api/rag/search-v2` route and avoid route-order shadowing.
  Files inspected: `app/api/rag_router.py`, `app/api_app.py`, `app/api/legal_v2_guard_router.py`, audit artifacts.
  Files changed: `app/api_app.py`, deleted `app/api/legal_v2_guard_router.py`.
  Behavior changed: the disabled guard is consolidated into the live route handler in `rag_router.py`; the duplicate guard router is no longer registered.
  Tests added/updated: route-count endpoint test.
  Verification command: `python -m pytest -q tests\api\test_rag_api.py::TestLegalV2SearchEndpoint`.
  Result: `20 passed`, one non-blocking Starlette/httpx deprecation warning.
  Risk: low; public disabled behavior remains controlled and enabled behavior now reaches the live handler only when feature-flagged.
  Next step: keep the route disabled by default until a separate relevance benchmark and rollout decision.
- Step:
  Add Legal v2 runtime dependency injection and cache controls.
  Goal: preserve production lazy initialization while allowing deterministic tests without Qdrant, BM25, BGE-M3, DeepSeek, or secrets.
  Files inspected: `app/rag/legal_v2/pipeline.py`, `app/rag/legal_v2/retriever.py`, `app/rag/legal_v2/interpreter.py`, `app/rag/legal_v2/verifier.py`.
  Files changed: `app/api/rag_router.py`, `tests/api/test_rag_api.py`.
  Behavior changed: `get_legal_v2_runtime_provider()` returns a callable; disabled requests never call the runtime factory; enabled requests use the factory and cached runtime guarded by a lock; `reset_legal_v2_runtime_for_tests()` clears cache for tests.
  Tests added/updated: disabled no-init, enabled fake runtime success, lazy cache reuse/reset, missing runtime config, missing Qdrant/BM25 dependency, provider timeout/invalid output, verifier failure, zero-result, limits, and evidence provenance tests.
  Verification command: `python -m pytest -q tests\api\test_rag_api.py::TestLegalV2SearchEndpoint`.
  Result: `20 passed`, one non-blocking Starlette/httpx deprecation warning.
  Risk: medium; enabled path now wires live v2 runtime when explicitly enabled, but default remains off and no legacy fallback is introduced.
  Next step: run broader API and Legal v2 regression suites before any commit.
- Step:
  Fix typing and privacy-safe response mapping.
  Goal: remove the dirty-induced `qdrant_client` mypy failure and prevent raw provider/error/path/text leakage from API responses and logs.
  Files inspected: `app/api/rag_router.py`, `app/rag/legal_v2/pipeline.py`, `app/rag/legal_v2/verifier.py`.
  Files changed: `app/api/rag_router.py`, `tests/api/test_rag_api.py`.
  Behavior changed: `qdrant_client` is loaded through `importlib.import_module()` only during enabled runtime construction; narrow Protocol annotations resolve older dependency-return mypy mismatches; exception logging records exception type only; provider/index/diagnostics/metadata payloads are bounded and redacted.
  Tests added/updated: log-capture and response assertions prove fake raw queries, secrets, raw provider bodies, local paths, and paragraph text metadata are not exposed.
  Verification command: `python -m mypy app/api/rag_router.py`; `ruff check app\api\rag_router.py tests\api\test_rag_api.py --no-cache`.
  Result: mypy passed with no issues; ruff passed.
  Risk: low; response detail becomes safer while preserving typed document/evidence contract.
  Next step: run full required validation and validator.
- Current status:
  `POST /api/rag/search-v2` is technically wired and deterministic-testable. `NALUS_LEGAL_V2_SEARCH_ENABLED=0` remains the default. Frontend is not connected. The 20-document smoke index remains only an isolated wiring/smoke asset, not a production relevance benchmark.
- Final validation:
  `python -m pytest -q tests\api\test_rag_api.py` -> `64 passed`, one non-blocking Starlette/httpx deprecation warning.
  Exact PowerShell command `python -m pytest -q tests/rag/test_legal_v2_*.py` -> failed before test collection because the wildcard was not expanded and pytest reported `file or directory not found`.
  Explicit Legal v2 suite `python -m pytest -q tests\rag\test_legal_v2_end_to_end.py tests\rag\test_legal_v2_evaluation.py tests\rag\test_legal_v2_parser_chunking.py tests\rag\test_legal_v2_qa_gate.py tests\rag\test_legal_v2_query_spec.py tests\rag\test_legal_v2_source_inventory.py tests\rag\test_legal_v2_verifier.py` -> `41 passed`, one non-blocking Starlette/httpx deprecation warning.
  `python -m pytest -q tests\rag\test_document_retrieval.py tests\rag\test_full_document_retrieval.py` -> `19 passed`.
  `python -m pytest -q tests\rag\test_constraint_pipeline.py tests\rag\test_constraint_verification.py` -> `8 passed`.
  `python -m compileall app scripts tests` -> passed.
  `ruff check app/api/rag_router.py app/rag/legal_v2 scripts/legal_v2 tests/api/test_rag_api.py tests/rag/test_legal_v2_*.py --no-cache` -> passed.
  `python -m mypy app/api/rag_router.py app/rag/legal_v2` -> passed, `20` source files checked.
  `git diff --check` -> passed with CRLF normalization warnings only.
  `docker compose ps` -> API, Qdrant, Redis, exporter, Prometheus, and Grafana were running.
  Isolated TestClient smoke with temporary `NALUS_LEGAL_V2_SEARCH_ENABLED` override and fake runtime -> disabled `404`, enabled zero-result `200`; no real Qdrant, BM25, BGE-M3, DeepSeek, Redis, or provider credentials used.
  Validator without allowlist `python scripts\validate_nalus_task.py --task-name "Complete live Legal Retrieval v2 API endpoint wiring" --mode implementation --write-report artifacts\evaluation_quality\legal_v2_live_api_endpoint_validator_20260731.md --write-json artifacts\evaluation_quality\legal_v2_live_api_endpoint_validator_20260731.json` -> `FAIL` only for conservative task-scope DeepSeek references and generated/unrelated artifact warnings.
  Validator with explicit documented allowlist for intentional `search-v2` wiring terms `deepseek_call`, `top_k_change`, `rrf_change`, `bm25_change`, `dense_change`, and `logger_change` -> `WARN`, `0` failures. Remaining findings are pre-existing/generated untracked artifact directories: `artifacts/legal_v2/`, `mixed_document_gold_default/`, `nsoud_document_gold_default/`, `usoud_document_gold_default/`, and `legal_v2_seed_comparison_20260723/`.
- Known limitations:
  The endpoint is not production-ready until a reviewed full-corpus v2 index and relevance benchmark approve quality. External DeepSeek live smoke was not run in this task.
- Next recommended task:
  Build and manually review a Legal Retrieval v2 relevance benchmark against the isolated smoke/full v2 index before any frontend connection or default enablement decision.

## 2026-07-31 Europe/Moscow - Task: Fix Legal Retrieval v2 disabled endpoint CI guard

- Goal:
  Fix the CI failure where `POST /api/rag/search-v2` returned FastAPI's generic `Not Found` instead of an explicit disabled Legal Retrieval v2 response when `NALUS_LEGAL_V2_SEARCH_ENABLED` is unset.
- Starting audit:
  Branch `main`, HEAD `85d4c78348cbaedad7b440475f70b0afbc07ec2b`.
  Required governance files read: `AGENTS.md`, `PROJECT_EXECUTION_PROTOCOL.md`, `PROJECT_PROGRESS.md`, and `docs/LEGAL_RETRIEVAL_V2.md`.
  Dirty worktree classified before editing. Pre-existing modified `app/api/rag_router.py` contains a larger Legal v2 runtime route diff and remains risky/outside this minimal CI guard task. Generated `artifacts/**` remain local outputs and were not touched.
- Scope:
  In scope: API app registration and a disabled-by-default fallback guard for `/api/rag/search-v2`.
  Out of scope: full Legal v2 runtime execution, DeepSeek provider wiring, Qdrant/BM25 retrieval behavior, model loading, aliases, Redis, frontend behavior, and the existing dirty `app/api/rag_router.py` implementation.
- What changed:
  Added `app/api/legal_v2_guard_router.py`, a small fallback router that returns `404` with an explicit disabled message while `NALUS_LEGAL_V2_SEARCH_ENABLED` is unset.
  Registered the fallback router after the main RAG router in `app/api_app.py`, so a future full `/api/rag/search-v2` implementation in `app/api/rag_router.py` takes precedence and is not shadowed.
  If the feature flag is enabled but no full runtime route is registered, the fallback returns an explicit `503` instead of silently calling providers or changing retrieval behavior.
- Tests run:
  `python -m pytest -q tests\rag\test_legal_v2_end_to_end.py::test_search_v2_endpoint_disabled_by_default` -> `1 passed`, one non-blocking Starlette/httpx deprecation warning.
  Isolated fallback-router `TestClient` smoke -> `404` with detail containing `disabled`.
  `python -m pytest -q tests\rag\test_legal_v2_end_to_end.py tests\api\test_rag_api.py` -> `52 passed`, one non-blocking Starlette/httpx deprecation warning.
  `python -m compileall app tests scripts` -> passed.
  `ruff check app\api\legal_v2_guard_router.py app\api_app.py tests\rag\test_legal_v2_end_to_end.py --no-cache` -> passed.
- Behavior preserved:
  No Qdrant write, no BM25 write, no model download, no DeepSeek call, no Redis behavior change, no production alias change, no retrieval scoring/ranking/top_k/RRF change, and no frontend behavior change.
- Known limitations:
  This task intentionally fixes only the disabled-default endpoint contract. It does not validate or approve the larger dirty `app/api/rag_router.py` Legal v2 runtime route.
- Next recommended task:
  Review the separate `app/api/rag_router.py` Legal v2 runtime diff as its own controlled task before enabling or committing live `search-v2` behavior.

## 2026-07-31 Europe/Moscow - Task: Push Legal Retrieval v2 QA gate and smoke-index commit

- Goal:
  Record the safe commit and push outcome for the intentional Legal Retrieval v2 parser QA gate and isolated smoke-index implementation.
- Governance:
  Required governance files were read before Git operations: `AGENTS.md`, `PROJECT_EXECUTION_PROTOCOL.md`, `PROJECT_PROGRESS.md`, and `docs/LEGAL_RETRIEVAL_V2.md`.
  Branch was `main`; pre-push local HEAD was `79e35ee799cd48699fecd12129494c081e46d692`.
  Dirty files were classified before staging and push.
- Commit pushed:
  `79e35ee799cd48699fecd12129494c081e46d692` - `Complete isolated Legal Retrieval v2 smoke indexing`.
  Previous remote HEAD was `302eba8053b4ea11c725cb5df88623ec5818f965`.
  Remote divergence before push was `0 1` for `origin/main...HEAD`, so local `main` was one commit ahead and remote was not ahead.
  Push command `git push origin main` succeeded: `302eba8..79e35ee main -> main`.
- Scope preserved:
  Committed only intended Legal v2 source, scripts, tests, docs, and this progress file from the QA gate / smoke-index task.
  `app/api/rag_router.py` remained modified but unstaged and uncommitted.
  Generated artifacts under `artifacts/`, isolated Qdrant/BM25 runtime data, caches, logs, and secrets were not committed.
  No force push, rebase, reset, clean, or broad staging command was used.
- Verification:
  Local `HEAD` and `origin/main` matched after push at `79e35ee799cd48699fecd12129494c081e46d692`.
  Post-push `git status --short` still showed only the preserved unrelated `app/api/rag_router.py` modification and untracked generated artifact directories.
- Known limitations:
  The NALUS validator still reports a blocking `deepseek_call` in the pre-existing excluded `app/api/rag_router.py` dirty diff, plus generated-artifact warnings. These were not part of the pushed commit.
- Next recommended task:
  Review and either intentionally finish or discard the separate `app/api/rag_router.py` work before any task that changes runtime RAG API behavior.

## 2026-07-31 Europe/Moscow - Task: Legal Retrieval v2 initial QA gate and isolated smoke index

- Goal:
  Define and enforce the initial Legal Retrieval v2 parser QA gate policy, re-evaluate the completed 30-document manual QA review, and build/validate only a small isolated v2 smoke index if the gate passes.
- Starting audit:
  Branch `main`, HEAD `302eba8053b4ea11c725cb5df88623ec5818f965`.
  Required governance files read: `AGENTS.md`, `PROJECT_EXECUTION_PROTOCOL.md`, `PROJECT_PROGRESS.md`, and `docs/LEGAL_RETRIEVAL_V2.md`.
  Dirty worktree classified before continuing. Pre-existing unrelated `app/api/rag_router.py` remains out of scope and was not edited. Generated artifacts under `artifacts/` and the isolated v2 BM25 sidecar are local outputs and must not be committed.
- Scope:
  In scope: Legal v2 QA gate policy/docs, deterministic gate evaluator, source-risk handling, reviewed-document smoke selection, isolated v2 collection `nalus_legal_paragraph_chunks_v2`, isolated BM25 sidecar `nalus_legal_paragraph_bm25_v2`, smoke validation scripts, focused tests, and this progress entry.
  Out of scope: production aliases, existing production Qdrant collections, existing production BM25 sidecars, frontend, Redis behavior, provider configuration, retrieval scoring/ranking, embeddings model/dimension, and full v2 index build.
- What changed:
  Added policy version `legal_v2_initial_index_qa_v1` to `docs/LEGAL_RETRIEVAL_V2.md`.
  Added `app/rag/legal_v2/qa_gate.py` and evaluator CLI support for a deterministic `pass` / `blocked` / `invalid` parser QA gate.
  Added source-risk policy for incomplete sources and duplicate source identifiers; incomplete or conflicting duplicate sources are not silently merged or marked complete.
  Added smoke-build safety inputs to `scripts/legal_v2/build_index.py`: explicit parser-quality artifact selection, gate-decision requirement, and document-id based source discovery.
  Added parent-window provenance into indexed child payloads: `parent_window_id`, parent paragraph IDs, parent child chunk IDs, checksum, token count, and truncation flag.
  Added `scripts/legal_v2/smoke_safety_snapshot.py` and `scripts/legal_v2/validate_smoke_index.py` for pre/post safety snapshots, Qdrant/BM25 integrity checks, provenance checks, and bounded smoke query reporting.
  Added focused QA gate and build-gate tests.
- Gate decision:
  `python scripts\legal_v2\evaluate_parser_quality_gate.py --parser-quality artifacts\legal_v2\parser_quality_gate_20260730\parser_quality_gate.json --manual-review-summary artifacts\legal_v2\parser_quality_gate_20260730\manual_review_summary.json --parse-audit artifacts\legal_v2\parse_audit_full_20260730\legal_v2_parse_audit.json --source-inventory artifacts\legal_v2\source_inventory_20260730.json --output-dir artifacts\legal_v2\parser_quality_gate_20260730` -> `decision=pass smoke_index_permitted=True`.
  Gate inputs: 30 samples, 30 reviewed, 30 approved, 0 rejected, 0 needs_review, full parse audit `pass`, 0 reconstruction failures, 0 boundary violations, 0 duplicate IDs, 0 cross-document mixing, 0 unresolved blocking defects.
  Source inventory risks were reported, not hidden: 55 records missing complete text and 502 duplicate source-document identifiers. Reviewed samples indexed by the smoke build had 0 incomplete-source records and 0 unresolved conflicting duplicate-source records.
- Isolated smoke index:
  Final build command: `docker compose exec -T api python scripts/legal_v2/build_index.py --parser-quality-artifact /app/artifacts/legal_v2/parser_quality_gate_20260730/parser_quality_gate.json --gate-decision /app/artifacts/legal_v2/parser_quality_gate_20260730/gate_decision.json --limit 20 --qdrant-url http://qdrant:6333 --output-dir /app/artifacts/legal_v2/smoke_index_20260730/index_build --overwrite-bm25 --recreate-v2-collection`.
  The command wrapper timed out after 40 minutes, so the state-changing operation was reconciled before reporting success. Reconciliation showed Qdrant count 384 and manifest `validation_status: pass`, `qdrant_write_status: pass`, `bm25_write_status: pass`.
  Result: pass by reconciliation; 20 source documents indexed, 384 child chunks, Qdrant collection `nalus_legal_paragraph_chunks_v2`, BM25 sidecar `/app/storage/rag/bm25/nalus_legal_paragraph_bm25_v2.sqlite`, embedding model `BAAI/bge-m3`, dimension 1024.
  The Docker build manifest records `git_commit: unknown` and `dirty: true` because `/app` in the container was not a Git worktree; host repository HEAD was recorded separately in this progress entry.
- Smoke validation:
  `docker compose exec -T api python scripts/legal_v2/smoke_safety_snapshot.py --phase postbuild --qdrant-url http://qdrant:6333 --output-dir /app/artifacts/legal_v2/smoke_index_20260730` -> postbuild snapshot written.
  `docker compose exec -T api python scripts/legal_v2/validate_smoke_index.py --qdrant-url http://qdrant:6333 --build-manifest /app/artifacts/legal_v2/smoke_index_20260730/index_build/legal_v2_build_manifest.json --prebuild-snapshot /app/artifacts/legal_v2/smoke_index_20260730/prebuild_snapshot.json --postbuild-snapshot /app/artifacts/legal_v2/smoke_index_20260730/postbuild_snapshot.json --output-dir /app/artifacts/legal_v2/smoke_index_20260730` -> `pass`.
  Validation result: Qdrant points 384, BM25 rows 384, vector dimension 1024, duplicate chunk IDs `false`, missing provenance 0, missing document IDs 0, cross-document mixing `false`, Qdrant/BM25 ID mismatch 0, Qdrant/BM25 text fingerprint mismatch 0, production changes 0.
  Positive smoke queries with expected candidates passed. English/no-candidate probes were explicitly reported as `no_candidates` under the current fail-closed hybrid retriever requirement that both dense and BM25 return candidates.
- Tests and checks:
  `python -m pytest -q tests\rag\test_legal_v2_qa_gate.py` -> `11 passed`.
  `python -m pytest -q tests\rag\test_legal_v2_end_to_end.py` -> `8 passed`, one non-blocking Starlette/httpx deprecation warning.
  `python -m pytest -q tests\rag\test_legal_v2_end_to_end.py tests\rag\test_legal_v2_qa_gate.py` -> `19 passed`, one non-blocking Starlette/httpx deprecation warning.
  `python -m pytest -q tests/rag/test_legal_v2_*.py` -> failed before running tests because PowerShell did not expand the wildcard and pytest reported `file or directory not found`.
  Explicit Legal v2 suite `python -m pytest -q tests\rag\test_legal_v2_end_to_end.py tests\rag\test_legal_v2_evaluation.py tests\rag\test_legal_v2_parser_chunking.py tests\rag\test_legal_v2_qa_gate.py tests\rag\test_legal_v2_query_spec.py tests\rag\test_legal_v2_source_inventory.py tests\rag\test_legal_v2_verifier.py` -> `41 passed`, one non-blocking Starlette/httpx deprecation warning.
  `ruff check app\rag\legal_v2 scripts\legal_v2 tests\rag\test_legal_v2_end_to_end.py tests\rag\test_legal_v2_qa_gate.py --no-cache` -> passed.
  `mypy app\rag\legal_v2 scripts\legal_v2\build_index.py scripts\legal_v2\parser_quality_gate.py scripts\legal_v2\evaluate_parser_quality_gate.py scripts\legal_v2\smoke_safety_snapshot.py scripts\legal_v2\validate_smoke_index.py` -> passed.
  `python -m compileall app scripts tests` -> passed.
  `git diff --check` -> passed; Git reported only CRLF normalization warnings.
- Runtime/infra behavior:
  `docker compose ps` showed `api`, `qdrant`, `redis`, `prometheus`, `grafana`, and `nalus-eval-metrics-exporter` running.
  Qdrant was modified only for isolated collection `nalus_legal_paragraph_chunks_v2`.
  BM25 was modified only for isolated sidecar `storage/rag/bm25/nalus_legal_paragraph_bm25_v2.sqlite`.
  No production alias changed; `nalus_live` remained mapped to `nalus_stable_20260326`.
  No model was downloaded; Docker loaded BGE-M3 from the existing local HuggingFace cache path.
  No Redis, provider, frontend, production retrieval scoring, embedding model, embedding dimension, or fallback behavior was changed.
- Validator:
  `python scripts\validate_nalus_task.py --task-name "Legal Retrieval v2 initial QA gate and isolated smoke index" --mode implementation --write-report artifacts\evaluation_quality\legal_v2_initial_qa_gate_smoke_validator_20260730.md --write-json artifacts\evaluation_quality\legal_v2_initial_qa_gate_smoke_validator_20260730.json` -> `FAIL`, 19 findings.
  The single failure is `deepseek_call` in pre-existing unrelated `app/api/rag_router.py`, which remained out of scope and was not edited.
  Warnings include existing generated artifact directories plus intentional Legal v2 smoke/index validation terms (`bm25_change`, `dense_change`, `top_k_change`) in scripts/tests that validate the isolated v2 Qdrant/BM25 smoke index; no production retrieval behavior or scoring was changed.
- Known limitations:
  This is a small isolated smoke index, not the full Legal Retrieval v2 index.
  The v2 search endpoint remains disabled by default.
  Smoke queries validate candidate retrieval and provenance only; they do not validate final legal answer quality or semantic verifier output.
  The current hybrid retriever fails closed when dense or BM25 has no candidates, so English/no-candidate probes are recorded as `no_candidates` rather than treated as positive retrieval matches.
- Next recommended task:
  Build a reviewed Legal Retrieval v2 retrieval benchmark/gold set against the isolated smoke index before any full v2 index build or `search-v2` activation decision.

## 2026-07-31 Europe/Moscow - Task: Commit and push consolidated Legal Retrieval v2 QA work

- Goal:
  Commit and push the intentional source, test, and documentation changes from the documentation consolidation, Legal Retrieval v2 Phase 1 readiness, source inventory, parser QA review, and parent-window hard-limit fix.
- Starting audit:
  Branch `main`, starting HEAD `736c2ab9041fd02219f7c40f65a9f0dbe8ed0193`.
  Existing dirty work was classified before staging. The unrelated modified `app/api/rag_router.py` remains out of scope and was not staged.
- Commit scope:
  Intended docs: `AGENTS.md`, `PROJECT_EXECUTION_PROTOCOL.md`, `PROJECT_PROGRESS.md`, `docs/LEGAL_RETRIEVAL_V2.md`, deleted legacy `AGENT.md`, deleted legacy `readme.dev`.
  Intended source: `app/project_validation/file_classifier.py`, `app/rag/legal_v2/chunking.py`, `app/rag/legal_v2/verifier.py`, `app/rag/legal_v2/source_inventory.py`, `scripts/legal_v2/parser_quality_gate.py`, `scripts/legal_v2/source_inventory.py`.
  Intended tests: `tests/rag/test_legal_v2_parser_chunking.py`, `tests/rag/test_legal_v2_source_inventory.py`.
  Excluded from commit: `app/api/rag_router.py`, generated `artifacts/`, generated Qdrant/BM25/model/cache/runtime data, and unrelated evaluation outputs.
- Validation before commit:
  Focused Legal v2 pytest, compileall, ruff, mypy, git diff checks, and NALUS validator were run during the underlying tasks. The final commit command sequence repeats the required git staging checks.
- Validator state:
  NALUS validator still fails in the dirty worktree because of the pre-existing unrelated `app/api/rag_router.py` diff and untracked generated artifacts. No validator finding is introduced by staging generated runtime/index data because those files are not staged.
- Push:
  Requested by the user on 2026-07-31. Commit hash and push result are reported in the final task report.
- Next recommended task:
  Define the Legal Retrieval v2 parser QA approval threshold in `docs/LEGAL_RETRIEVAL_V2.md`, then rerun the validator and decide whether isolated smoke index build is permitted.

## 2026-07-30 Europe/Moscow - Task: Legal Retrieval v2 manual parser QA review

- Goal:
  Perform evidence-backed manual QA review of the 30 representative Legal Retrieval v2 parser samples, update the review artifacts, and determine whether the quality gate permits an isolated smoke index build.
- Starting audit:
  Branch `main`, HEAD `736c2ab9041fd02219f7c40f65a9f0dbe8ed0193`.
  Pre-existing dirty files were preserved, including documentation consolidation changes, prior Legal v2 source-inventory additions, generated artifacts, and the unrelated modified `app/api/rag_router.py`.
- Review result:
  Reviewed all 30 regenerated representative samples against source text, normalized text, parsed paragraphs, child chunks, parent windows, reconstruction, parser diagnostics, source-completeness evidence, and duplicate-source evidence.
  Manual status summary: approved `30`, rejected `0`, needs_review `0`; reviewed `27` Ústavní soud samples and `3` Nejvyšší soud samples.
  Sample coverage includes short judgments `3`, long judgments `4`, damaged formatting `28`, citation-heavy samples `30`, and punctuation-heavy/truncated-parent-window samples `7`.
  No reviewed sample belonged to the source-inventory missing-complete-text group or duplicate-source-ID group. No cross-document mixing was detected.
- Defect found and fixed:
  Manual QA found a reproducible chunking defect: parent evidence windows could exceed `parent_hard_max_tokens` when anchored to a long source paragraph.
  Fixed `app/rag/legal_v2/chunking.py` so over-limit parent windows fall back to the anchor child chunk text and are marked `truncated=True`.
  Added a focused regression test in `tests/rag/test_legal_v2_parser_chunking.py`.
- Full parse audit rerun:
  Reran `docker compose exec -T api python scripts/legal_v2/audit_corpus.py --output-dir /app/artifacts/legal_v2/parse_audit_full_20260730`.
  Result: pass; 103,638 documents parsed, 0 failed, 0 reconstruction failures, 0 boundary violations, 0 overlong chunks, 0 duplicate IDs.
- Artifacts updated:
  Updated `artifacts/legal_v2/parser_quality_gate_20260730/parser_quality_gate.json`.
  Updated `artifacts/legal_v2/parser_quality_gate_20260730/parser_quality_gate.md`.
  Created `artifacts/legal_v2/parser_quality_gate_20260730/manual_review_summary.json`.
  Created `artifacts/legal_v2/parser_quality_gate_20260730/manual_review_summary.md`.
- Quality gate:
  Status remains `BLOCKED` for index-build purposes.
  Reason: all 30 samples are approved, but `docs/LEGAL_RETRIEVAL_V2.md` does not define the approval threshold required by the task. The threshold cannot be verified without inventing policy.
  No Qdrant index, BM25 index, production retrieval, API route, frontend, aliases, collections, embeddings, DeepSeek behavior, Redis behavior, or provider configuration was changed.
- Validation:
  Focused chunking regression: `python -m pytest -q tests\rag\test_legal_v2_parser_chunking.py` -> `5 passed`.
  Full required validation is recorded in the task final report.
- Next recommended task:
  Define the Legal Retrieval v2 parser QA approval threshold and rejected/approved sample policy in `docs/LEGAL_RETRIEVAL_V2.md`, then rerun the NALUS validator and decide whether the isolated smoke index build is permitted.

## 2026-07-30 Europe/Moscow - Task: Legal Retrieval v2 Phase 1 runtime/index readiness

- Goal:
  Complete Legal Retrieval v2 Phase 1 by verifying dependencies, validating the full parser corpus, creating evidence-backed parser QA, and building/running the isolated v2 index only if parser readiness gates pass.
- Starting audit:
  Branch `main`, HEAD `736c2ab9041fd02219f7c40f65a9f0dbe8ed0193`.
  Pre-existing dirty files before this task included the documentation consolidation changes (`AGENTS.md`, `PROJECT_EXECUTION_PROTOCOL.md`, `PROJECT_PROGRESS.md`, `app/project_validation/file_classifier.py`, deleted `AGENT.md`, deleted `readme.dev`), pre-existing modified `app/api/rag_router.py`, and untracked generated artifacts under `artifacts/evaluation_quality/`, `artifacts/legal_v2/`, and `artifacts/rag_eval/`.
- Dependency/runtime decision:
  Local Python 3.12 does not have `qdrant_client`.
  Docker `api` runtime is running and has `qdrant-client` version `1.13.3` by package metadata. The module does not expose `qdrant_client.__version__`.
  `requirements.txt` and `requirements-ci.txt` already declare `qdrant-client`; no dependency declaration change or image rebuild was required.
- Baseline validation before indexing:
  `python -m pytest -q tests\rag\test_legal_v2_parser_chunking.py tests\rag\test_legal_v2_query_spec.py tests\rag\test_legal_v2_verifier.py tests\rag\test_legal_v2_evaluation.py tests\rag\test_legal_v2_end_to_end.py` -> `25 passed`, one Starlette/httpx deprecation warning.
  Initial focused ruff found an unused `os` import in `app/rag/legal_v2/verifier.py`; removed it.
  `ruff check app\rag\legal_v2 scripts\legal_v2 tests\rag\test_legal_v2_*.py --no-cache` -> passed after fixes.
  `mypy app\rag\legal_v2` -> passed.
- Source inventory:
  Added `app/rag/legal_v2/source_inventory.py`, `scripts/legal_v2/source_inventory.py`, and `tests/rag/test_legal_v2_source_inventory.py`.
  Ran `docker compose exec -T api python scripts/legal_v2/source_inventory.py --json-output /app/artifacts/legal_v2/source_inventory_20260730.json --markdown-output /app/artifacts/legal_v2/source_inventory_20260730.md`.
  Result: 103,638 discovered documents across 45 files; constitutional 103,488, supreme 150; 0 missing stable IDs, 55 missing complete text records, 502 duplicate source-document identifiers, 0 unreadable files, 0 unsupported formats.
- Full parse audit:
  First full audit wrote `artifacts/legal_v2/parse_audit_full_20260730/` and failed closed on 2 documents with `boundary_overlong_chunk`.
  Root cause was `_split_by_tokens()` splitting by whitespace while audit validation counts regex word tokens.
  Fixed `app/rag/legal_v2/chunking.py` so hard splitting uses the same token counter as validation and added a punctuation-heavy regression test.
  Rerun `docker compose exec -T api python scripts/legal_v2/audit_corpus.py --output-dir /app/artifacts/legal_v2/parse_audit_full_20260730` -> pass.
  Final audit result: 103,638 documents parsed, 1,311,123 paragraphs, 1,043,986 child chunks, 1,043,986 parent windows, 0 failed documents, 0 reconstruction failures, 0 offset failures, 0 boundary violations, 0 overlong chunks, 0 duplicate IDs.
- Parser QA:
  Updated `scripts/legal_v2/parser_quality_gate.py` to create a bounded representative sample and explicit review fields instead of selecting only the first N documents.
  Ran `docker compose exec -T api python scripts/legal_v2/parser_quality_gate.py --limit 30 --output-dir /app/artifacts/legal_v2/parser_quality_gate_20260730`.
  Result: 30 reviewed sample artifacts, 27 constitutional and 3 supreme; category coverage includes short/long judgments, numbered paragraphs, damaged formatting, citations, long factual sections, long legal reasoning, boilerplate, recent decisions, older decisions, Ústavní soud, and Nejvyšší soud.
  Review status summary: approved `0`, rejected `0`, needs_review `30`.
- Blocker:
  Index build did not run. `docs/LEGAL_RETRIEVAL_V2.md` is authoritative and says the v2 index should be built only after parser audit and quality review pass. The full parse audit passed, but parser QA remains `needs_review` with no approved review manifest. Proceeding to smoke or full Qdrant/BM25 index build would violate the repository gate.
- Behavior preserved:
  No production retrieval endpoint, frontend route, BGE-M3 model, embedding dimension, dense scoring, BM25 formula, RRF formula, Redis behavior, provider default, Qdrant alias, production collection, or production BM25 sidecar was changed.
  `NALUS_LEGAL_V2_SEARCH_ENABLED` remains disabled by default.
- Generated artifacts:
  `artifacts/legal_v2/source_inventory_20260730.*`
  `artifacts/legal_v2/parse_audit_full_20260730/`
  `artifacts/legal_v2/parser_quality_gate_smoke_20260730/`
  `artifacts/legal_v2/parser_quality_gate_20260730/`
  `artifacts/evaluation_quality/legal_v2_phase1_completion_20260730.*`
- Known limitations:
  Phase 1 indexing is blocked until parser QA is explicitly reviewed and enough samples are approved or rejected with concrete reasons.
  Local Python still lacks `qdrant_client`; Docker `api` remains the canonical runtime for future indexing unless local dependencies are installed through project-managed dependency setup.
- Final validation:
  `python -m pytest -q tests\rag\test_legal_v2_parser_chunking.py tests\rag\test_legal_v2_query_spec.py tests\rag\test_legal_v2_verifier.py tests\rag\test_legal_v2_evaluation.py tests\rag\test_legal_v2_end_to_end.py tests\rag\test_legal_v2_source_inventory.py` -> `27 passed`, one Starlette/httpx deprecation warning.
  `python -m compileall app scripts tests` -> passed.
  `ruff check app\rag\legal_v2 scripts\legal_v2 tests\rag\test_legal_v2_*.py --no-cache` -> passed.
  `mypy app\rag\legal_v2` -> passed.
  `git diff --check` -> passed.
  `python scripts\validate_nalus_task.py --task-name "Legal Retrieval v2 Phase 1 runtime/index readiness" --mode implementation --write-report artifacts\evaluation_quality\legal_v2_phase1_validator_20260730.md --write-json artifacts\evaluation_quality\legal_v2_phase1_validator_20260730.json` -> `FAIL`, 13 findings. The only failure was `deepseek_call` in pre-existing dirty `app/api/rag_router.py`; warnings were for pre-existing unknown dirty files/directories and risk terms in the same pre-existing `app/api/rag_router.py` diff.
- Next recommended task:
  Perform human parser QA review of `artifacts/legal_v2/parser_quality_gate_20260730/`, create/update `artifacts/legal_v2/parser_quality_manifest.json` with justified statuses, then rerun the parser QA gate and only proceed to isolated smoke index if the quality review passes.

## 2026-07-30 Europe/Moscow - Task: Consolidate root agent and handoff documentation

- Goal:
  Remove duplicated root handoff documents so future work has one rule source and one progress source.
- Starting audit:
  Branch `main`, HEAD `736c2ab9041fd02219f7c40f65a9f0dbe8ed0193`.
  Pre-existing changes were present before this documentation cleanup: modified `app/api/rag_router.py`, untracked `artifacts/evaluation_quality/citizenship_query_retrieval_audit_20260712.*`, untracked `artifacts/legal_v2/`, and untracked generated evaluation output directories under `artifacts/rag_eval/`.
- Findings:
  `AGENTS.md` is the newest and authoritative agent rules file.
  `PROJECT_PROGRESS.md` is the newest and authoritative chronological handoff log.
  `readme.dev` was an older mixed developer-note and handoff file last updated around 2026-07-12; its current-state content is superseded by `PROJECT_PROGRESS.md`, `README.md`, and focused files under `docs/`.
  `AGENT.md` was stale project state from 2026-04-05 and conflicted with the current BGE-M3, document retrieval, observability, and Legal Retrieval v2 state.
- What changed:
  Added a canonical documentation policy to `AGENTS.md`.
  Updated `PROJECT_EXECUTION_PROTOCOL.md` so mandatory task startup reads `PROJECT_PROGRESS.md` and `AGENTS.md`, not stale `readme.dev`.
  Updated `app/project_validation/file_classifier.py` so `AGENTS.md` is classified as documentation and deleted legacy `readme.dev` is no longer listed as a canonical docs file.
  Removed duplicate root handoff files `AGENT.md` and `readme.dev`.
  Kept `PROJECT_PROGRESS.md` as the single place to append every completed task/handoff entry.
- Current latest substantive implementation state to build on:
  The latest implementation entry before this cleanup is `2026-07-23 Europe/Moscow - Task: Legal Retrieval v2 end-to-end pipeline`.
  That entry says the next technical step is to install/verify runtime Qdrant dependencies, run the full parse audit and manual parser QA review, then build the isolated v2 index `nalus_legal_paragraph_chunks_v2` / `nalus_legal_paragraph_bm25_v2`.
- Tests:
  `python -m pytest tests/test_nalus_task_validator.py -q` -> `11 passed`.
  `git diff --check` -> passed.
- Behavior preserved:
  No API, retrieval, embedding, Qdrant, BM25, RRF, Redis, Docker, frontend, generated artifact, or runtime behavior was changed. Validator classification now treats `AGENTS.md` as documentation and no longer normalizes `readme.dev` as an active root docs file.
- Next recommended task:
  Continue from the 2026-07-23 Legal Retrieval v2 end-to-end entry unless the user explicitly switches priority.

## 2026-07-23 Europe/Moscow - Task: Legal Retrieval v2 end-to-end pipeline

- Goal:
  Add the first isolated Legal Retrieval v2 end-to-end implementation: source adapters, parse audit, parser quality artifact, versioned v2 index builder, BGE-M3 + BM25 + RRF retriever, DeepSeek-backed QuerySpec interpreter, paragraph evidence selection, DeepSeek-backed semantic verifier, deterministic fail-closed gate, disabled API endpoint, metrics, scripts, tests, and documentation.
- Starting audit:
  Branch `main`, HEAD `cdfac0af582036afabe0b127636b8943b219f524`.
  Pre-existing untracked generated artifacts were present under `artifacts/evaluation_quality/citizenship_query_retrieval_audit_20260712.*`, `artifacts/rag_eval/legal_qa/answer_eval/{mixed_document_gold_default,nsoud_document_gold_default,usoud_document_gold_default}/`, and `artifacts/rag_eval/legal_v2_seed_comparison_20260723/`.
- What changed:
  Added source adapters and source discovery for NALUS/Constitutional Court, Supreme Court, and generic fallback.
  Added a parse-only audit runner and parser QA gate artifacts under `scripts/legal_v2/`.
  Added an isolated v2 index builder that writes only `nalus_legal_paragraph_chunks_v2` and `nalus_legal_paragraph_bm25_v2`.
  Added provider-backed DeepSeek QuerySpec and semantic verifier wrappers with deterministic fake providers for tests.
  Added a v2 hybrid retriever using BGE-M3 dense retrieval, BM25, and RRF over isolated v2 names.
  Added paragraph-aware evidence selection and stricter verifier evidence-window validation.
  Added `POST /api/rag/search-v2`, disabled by default through `NALUS_LEGAL_V2_SEARCH_ENABLED=0`.
  Added low-cardinality legal v2 metrics and bounded trace/log events.
  Updated `.env.example`, `docker-compose.yml`, and `docs/LEGAL_RETRIEVAL_V2.md`.
- Real-corpus parse audit:
  Ran `python scripts\legal_v2\audit_corpus.py --limit 200 --output-dir artifacts\legal_v2\parse_audit_20260723`.
  Result: `PASS`, 200 documents, 2,988 paragraphs, 2,602 child chunks, 2,602 parent windows, 523 boilerplate paragraphs, 1,154 citation blocks, 0 reconstruction failures, 0 offset failures, 0 boundary violations, 0 overlong chunks, 0 duplicate IDs.
- Parser QA artifact:
  Ran `python scripts\legal_v2\parser_quality_gate.py --limit 12 --output-dir artifacts\legal_v2\parser_quality_gate_20260723`.
  Result artifact defaults reviewed documents to `needs_review`.
- Benchmark:
  Ran `python scripts\legal_v2\run_benchmark.py --output-dir artifacts\legal_v2\benchmark_20260723`.
  It wrote a `blocked` report because no live v2 index has been built in this environment.
- Blocked build/live smoke:
  `python scripts\legal_v2\build_index.py --limit 5 --output-dir artifacts\legal_v2\index_build_20260723 --overwrite-bm25 --recreate-v2-collection` was blocked before any write because `qdrant_client` is not installed in the local Python environment.
  `python scripts\legal_v2\live_smoke.py --output-dir artifacts\legal_v2\live_smoke_20260723` was blocked because DeepSeek credentials are not configured.
- Tests:
  `python -m pytest -q tests\rag\test_legal_v2_parser_chunking.py tests\rag\test_legal_v2_query_spec.py tests\rag\test_legal_v2_verifier.py tests\rag\test_legal_v2_evaluation.py tests\rag\test_legal_v2_end_to_end.py` -> `23 passed`, one Starlette/httpx deprecation warning.
- Behavior preserved:
  Existing production endpoints, frontend behavior, production Qdrant collection, production BM25 sidecar, retrieval profile, cache behavior, and feature flags remain unchanged. No commit or push.
- Known limitations:
  The full v2 index was not built locally because `qdrant_client` is missing. Parser QA is generated but not manually approved. Live DeepSeek smoke is blocked until credentials and the v2 index exist.
- Next recommended step:
  Install the runtime Qdrant dependency in this environment, run the full parse audit and manual parser QA review, then execute `scripts/legal_v2/build_index.py` against the isolated v2 collection.

## 2026-07-23 Europe/Moscow - Task: Universal Verified Legal Retrieval v2 foundation

- Goal:
  Add disabled-by-default runtime and evaluation foundations for Universal Verified Legal Retrieval v2: paragraph-aware legal document structure, deterministic parsing, hierarchical child chunks, parent evidence windows, versioned indexing contract, universal QuerySpec v2, final semantic verifier interface, deterministic fail-closed gate, diagnostics, tests, hard-negative fixtures, and an offline comparison report writer.
- Scope:
  Additive backend/evaluation code only. No production frontend switch, no active production retrieval profile change, no current Qdrant collection or BM25 sidecar overwrite, no external or paid LLM provider calls, no commit, and no push.
- Starting audit:
  Branch `main`, HEAD `017c1957935cf1ab71a7eedaa479122a284ffcfb`.
  Pre-existing untracked generated artifacts were present under `artifacts/evaluation_quality/citizenship_query_retrieval_audit_20260712.*` and `artifacts/rag_eval/legal_qa/answer_eval/{mixed_document_gold_default,nsoud_document_gold_default,usoud_document_gold_default}/`.
  Recent HEAD history started with `017c195 feat(observability): add correlation context and structured logging`, followed by `333ce2f Add observability engineering guardrails`, `9b29ad5 Add Docker registry publishing support`, `2e45e98 Add verified and full document retrieval APIs`, `069d7ca Harden production retrieval candidate selection`, `aa63a3a Add offline document retrieval benchmark`, `33b1711 Add document-level retrieval pipeline`, and `71755ca Enable evidence windows for document-gold evaluation`.
- What changed:
  Added `app/rag/legal_v2/models.py` with stable paragraph/chunk IDs, section enum, paragraph metadata provenance, document reconstruction, and parsing diagnostics.
  Added `app/rag/legal_v2/parser.py` with deterministic line-ending normalization, numbered-paragraph detection, heading and section transitions, damaged-format fallback segmentation, boilerplate/citation classification, source offsets, source order, and diagnostics.
  Added `app/rag/legal_v2/chunking.py` with paragraph-aware child chunks, sentence-aware splitting for overlong paragraphs, complete paragraph/sentence overlap, no incompatible section crossing, parent evidence windows, deterministic IDs, source spans, paragraph text maps, and reconstruction.
  Added `app/rag/legal_v2/indexing.py` with disabled v2 indexing contract and proposed collection/profile `nalus_legal_paragraph_chunks_v2` plus BM25 sidecar id `nalus_legal_paragraph_bm25_v2`.
  Added `app/rag/legal_v2/query_spec.py` with a universal typed QuerySpec v2 contract preserving `original_query`, `normalized_query`, `structured_query`, and entity-preserving `retrieval_queries`.
  Added `app/rag/legal_v2/verifier.py` with provider-agnostic structured verifier interface, deterministic fake verifier, strict output validation, evidence paragraph validation, and deterministic gate.
  Added `app/rag/legal_v2/diagnostics.py` with bounded runtime diagnostic payloads and explicit Prometheus label-safety flags.
  Added `app/rag/legal_v2/evaluation.py` with offline comparison metrics and JSON/Markdown report writer for pass/failure/blocked/exception states.
  Added focused tests under `tests/rag/test_legal_v2_*.py` and hard-negative fixture `tests/fixtures/legal_v2_hard_negatives.jsonl`.
  Generated seed offline comparison artifact under `artifacts/rag_eval/legal_v2_seed_comparison_20260723/`.
- Tests and validation run:
  `python -m compileall app tests` -> passed.
  `python -m pytest -q tests\rag\test_legal_v2_parser_chunking.py tests\rag\test_legal_v2_query_spec.py tests\rag\test_legal_v2_verifier.py tests\rag\test_legal_v2_evaluation.py` -> `17 passed`.
  `python -m pytest -q tests\rag\test_full_document_retrieval.py tests\rag\test_document_retrieval.py tests\rag\test_constraint_pipeline.py tests\rag\test_constraint_verification.py tests\rag\test_production_bge_m3_profile.py` -> `45 passed`.
  `ruff check app\rag\legal_v2 tests\rag\test_legal_v2_parser_chunking.py tests\rag\test_legal_v2_query_spec.py tests\rag\test_legal_v2_verifier.py tests\rag\test_legal_v2_evaluation.py --no-cache` -> passed.
  `mypy app\rag\legal_v2` -> passed with no issues in 9 source files.
  `git diff --check` -> passed.
- Offline seed comparison:
  Tiny deterministic seed only; no production readiness claimed.
  `current_production_chunks`: candidate_recall `1.0`, exact_precision `0.25`, hard_negative_false_positives `1`, verified_document_precision `0.333`.
  `paragraph_child_chunks`: candidate_recall `1.0`, exact_precision `0.5`, hard_negative_false_positives `0`, verified_document_precision `0.0`.
  `paragraph_child_parent_windows`: candidate_recall `1.0`, exact_precision `0.5`, hard_negative_false_positives `0`, verified_document_precision `1.0`.
- Behavior preserved:
  Existing production retrieval config remains on `nalus_bge_m3_dense_bm25_rrf_v1` / `nalus_bge_m3_chunks_v1`. No API route, frontend, active collection alias, embedding model, Qdrant data, BM25 scoring, RRF behavior, Redis/cache behavior, or provider configuration was changed.
- Known limitations:
  The parser and QuerySpec extraction are conservative deterministic first-pass helpers, not a complete legal NLP system.
  The LLM verifier is an interface plus deterministic fake provider; no paid/external provider is enabled.
  The seed comparison is intentionally small and synthetic, useful for contract validation only.
- Next recommended task:
  Add an offline v2 index builder that reads reviewed legal documents, writes only the new `nalus_legal_paragraph_chunks_v2` payload format, and runs the comparison runner on a curated gold dataset before any production activation discussion.

## 2026-07-22 Europe/Moscow - Task: Implement Observability Phase A runtime foundation

- Goal:
  Implement executable Phase A observability primitives: canonical correlation context, request ID generation, structured JSON-compatible logging, centralized redaction, FastAPI middleware, context cleanup, and tests.
- Scope:
  Runtime observability only. No OpenTelemetry, Loki, Tempo, Grafana service, alerting, audit hash chain, failure injection, retrieval ranking, query rewriting, API response contract, or business logic changes were intended.
- What changed:
  Added `app/core/context.py` with contextvars-backed `correlation_id`, `request_id`, `operation_id`, `workflow_id`, `job_id`, and `task_id`, secure generated IDs, inbound ID validation, explicit binding/reset helpers for future workers, and cleanup.
  Added `app/core/redaction.py` with recursive redaction for mappings, nested structures, lists, tuples, Pydantic models, dataclasses, headers, exceptions, structured extras, string secret patterns, and stable idempotency-key fingerprinting.
  Extended `app/core/logging.py` with automatic context enrichment through a LogRecordFactory, JSON log formatting controlled by `LOG_FORMAT=json` or `LOG_JSON=1`, structured fields, redacted extras, and duplicate-handler protection.
  Updated `app/core/tracing.py` so trace payloads are redacted while preserving existing trace formatting compatibility.
  Added `app/api/middleware.py` and installed it in `app/api/main.py` and `app/api_app.py`. The middleware accepts valid `X-Correlation-ID`, replaces invalid or missing values, always generates `X-Request-ID`, returns both response headers, logs request start/completion/failure events, and clears context in `finally`.
  Added deterministic Phase A tests for context validation/binding/cleanup, middleware response headers and leakage protection, concurrent request isolation, structured log enrichment, JSON log validity, redaction coverage, idempotency key safety, and duplicate handler protection.
  Made `tests/api/test_main_startup.py` timing deterministic under the new middleware and annotated `app/api/rag_router.py` query-cache state for the touched `app/api/main.py` mypy check.
  Updated `docs/OBSERVABILITY_CONTRACT.md` with concise Phase A runtime usage.
- Tests and validation run:
  `python -m pytest -q tests/test_observability_context.py tests/test_redaction.py tests/test_structured_logging.py tests/api/test_observability_middleware.py` -> `34 passed`, one Starlette/httpx deprecation warning.
  `python -m pytest -q tests/test_core_tracing.py tests/api/test_main_startup.py tests/api/test_rag_api.py` -> `70 passed`, one Starlette/httpx deprecation warning.
  `python -m compileall app tests` -> passed.
  `ruff check app/core/context.py app/core/redaction.py app/core/logging.py app/core/tracing.py app/api/middleware.py app/api/main.py app/api_app.py tests/test_observability_context.py tests/test_redaction.py tests/test_structured_logging.py tests/api/test_observability_middleware.py tests/api/test_main_startup.py --no-cache` -> passed.
  `mypy app/core/context.py app/core/redaction.py app/core/logging.py app/core/tracing.py app/api/middleware.py app/api_app.py` -> passed.
  `mypy app/api/main.py` -> passed after annotating query-cache state.
- Runtime smoke:
  Inline `TestClient(app.api_app.app)` smoke hit `/docs` twice. First request used inbound `X-Correlation-ID=smoke-12345678` and authorization/cookie header secrets; response returned the same correlation ID and an `X-Request-ID`. Second request generated a different correlation ID and a fresh request ID. Captured middleware events were `http.request.started` and `http.request.completed` for both requests. Middleware logs did not contain the header secret, and `get_context()` was fully empty after both requests.
- Blocked broader validation:
  `ruff check app tests --no-cache` currently fails on unrelated pre-existing lint findings across older modules/tests.
  `mypy app` currently fails on unrelated pre-existing missing stubs and type issues across older modules; the new Phase A modules pass narrow mypy.
- Known limitations:
  Phase A does not add distributed tracing, collector/exporter backends, central log aggregation, alert rules, audit chains, failure injection, or diagnostics endpoints.
  Middleware logs safe path/route only and intentionally does not log request bodies, raw query strings, authorization headers, cookies, or request payloads.
- Next recommended task:
  Start Phase B by designing an OpenTelemetry-compatible tracing plan that maps the existing lightweight `trace_event` call sites to bounded spans without adding a duplicate monitoring stack.

## 2026-07-22 Europe/Moscow - Task: Set production observability and reliability guardrails

- Goal:
  Adapt the supplied universal production observability, reliability, audit, and failure-detection prompt into durable repository rules before continuing with implementation work.
- Scope:
  Documentation and agent guardrails only. No runtime code, API behavior, retrieval ranking, Qdrant data, Docker topology, Prometheus scrape config, Grafana dashboards, generated reports, or feature flags were changed.
- What changed:
  Added root `AGENTS.md` with project variables, critical workflows, critical invariants, mandatory pre-task audit steps, observability implementation policy, validation expectations, and git policy.
  Added `docs/OBSERVABILITY_CONTRACT.md` with the project-specific observability contract, current stack inventory, correlation/logging/tracing/metrics/reporting/audit/failure-injection requirements, and a phased implementation backlog.
  Added `.cursor/rules/observability-reliability.mdc` so Cursor also picks up the same guardrails without duplicating the full contract.
  Added `docs/runbooks/observability-incidents.md` with initial operator guidance for duplicate side effects, reconciliation, verification failures, audit integrity failures, authentication/authorization spikes, provider-cost spikes, queue backlog, trace exporter outage, and log backend outage.
- Starting state:
  Branch `main`, HEAD `aa63a3a05b81e8555a3c84da351d6f1ac2faa8e3`.
  The worktree already had pre-existing modified and untracked files before this setup.
  Existing stack inspection found classic text logging in `app/core/logging.py`, lightweight trace events in `app/core/tracing.py`, Prometheus/Grafana under `monitoring/`, and existing observability tests/exporters under `app/observability/` and `tests/observability/`.
- Tests run:
  Documentation-only setup; no application tests were run.
  `git diff --check` should be run before committing this together with any future implementation phase.
- Known limitations:
  This setup does not implement structured JSON logging, OpenTelemetry, audit hash chains, report validators, alert rules, controlled failure injection, or live observability runners. Those remain phased implementation work and must be added with focused tests.
- Next recommended task:
  Start Phase A: canonical correlation context, structured logging, and centralized redaction, reusing `app/core/logging.py` and existing API boundaries.

## 2026-07-13 11:20 Europe/Moscow — Task: Add disabled constraint-aware verified document retrieval

- Goal:
  Add an additive backend retrieval path that can interpret structured legal query constraints, verify candidate documents against bounded full-document evidence, and reject partial/contradictory matches without changing the stable MVP chunk-level retrieval flow.
- Scope:
  Backend retrieval/config/API/observability/tests/docs only. The frontend, BGE-M3 embeddings, BM25, RRF, Qdrant collection/data, ingestion, Redis/cache behavior, query rewrite, answer generation, and existing `/api/rag/retrieve` and `/api/rag/query` response contracts were not changed.
- What changed:
  Added typed constraint models, validated config, deterministic structured-query interpretation, deterministic constraint verification, and an additive pipeline in `app/rag/retrieval/`.
  Added `POST /api/rag/retrieve-verified`, disabled by default through `NALUS_CONSTRAINT_RETRIEVAL_ENABLED=0`.
  The new endpoint groups candidate chunks by canonical document id, reconstructs bounded full-document text using the existing read-only full-document store, verifies hard constraints, and returns only verified unique documents.
  Added strict behavior for hard constraints: mismatch or not-proven excludes a document in strict mode, and no hidden threshold lowering or unrelated fallback is applied.
  Added Prometheus metrics through the existing metrics stack with bounded labels only: endpoint, status, decision status, constraint category, verification status, and method.
  Added config examples to `.env.example` and Docker environment defaults to keep the feature disabled unless explicitly enabled.
  Added `docs/CONSTRAINT_AWARE_RETRIEVAL.md` and a manually reviewable seed dataset fixture for future evaluation work.
- Why it changed:
  Previous failures showed that lexical/chunk retrieval can return partial matches such as citizenship mentions without the requested nationality relation, or child-abduction country mentions without the requested destination/actor relation. The new module verifies material constraints at document level before returning results.
- Files changed:
  `app/rag/retrieval/constraint_models.py`
  `app/rag/retrieval/constraint_config.py`
  `app/rag/retrieval/structured_query.py`
  `app/rag/retrieval/constraint_verification.py`
  `app/rag/retrieval/constraint_pipeline.py`
  `app/observability/constraint_retrieval_metrics.py`
  `app/api/rag_router.py`
  `.env.example`
  `docker-compose.yml`
  `docs/CONSTRAINT_AWARE_RETRIEVAL.md`
  `tests/rag/test_structured_query.py`
  `tests/rag/test_constraint_verification.py`
  `tests/rag/test_constraint_pipeline.py`
  `tests/observability/test_constraint_retrieval_metrics.py`
  `tests/api/test_rag_api.py`
  `tests/fixtures/constraint_retrieval_seed_dataset.jsonl`
- Tests run:
  `python -m pytest tests/rag/test_structured_query.py tests/rag/test_constraint_verification.py tests/rag/test_constraint_pipeline.py tests/observability/test_constraint_retrieval_metrics.py -q` -> initial `1 failed, 11 passed`; fixed parent-role detection for forms such as `matkou`.
  `python -m pytest tests/rag/test_structured_query.py tests/rag/test_constraint_verification.py tests/rag/test_constraint_pipeline.py tests/observability/test_constraint_retrieval_metrics.py -q` -> `12 passed`.
  `python -m pytest tests/api/test_rag_api.py -q` -> `44 passed`.
  `python -m pytest tests/rag/test_document_retrieval.py tests/rag/test_full_document_retrieval.py -q` -> `19 passed`.
  `python -m pytest tests/api/test_rag_api.py tests/rag/test_structured_query.py tests/rag/test_constraint_verification.py tests/rag/test_constraint_pipeline.py tests/rag/test_document_retrieval.py tests/rag/test_full_document_retrieval.py tests/observability/test_constraint_retrieval_metrics.py -q` -> `75 passed`.
  `python -m compileall app\rag\retrieval\constraint_config.py app\rag\retrieval\constraint_models.py app\rag\retrieval\structured_query.py app\rag\retrieval\constraint_verification.py app\rag\retrieval\constraint_pipeline.py app\observability\constraint_retrieval_metrics.py app\api\rag_router.py` -> passed.
- Smoke result:
  Runtime Docker smoke was not run in this task. The endpoint is disabled by default, and focused API tests verify disabled behavior, successful verified retrieval when explicitly enabled, empty verified result without fallback, and provider failure as 503.
- Behavior preserved:
  Existing `/api/rag/retrieve` remains chunk-level and backward compatible. Existing `/api/rag/query` remains unchanged. Document-level ranking endpoint `/api/rag/retrieve-documents` remains separately gated. No embedding, ranking, Qdrant, BM25, RRF, Redis/cache, model-provider, or frontend behavior was changed.
- Known limitations:
  The first rollout is deterministic and conservative. It does not use an LLM verifier and does not claim absolute legal relevance. The seed dataset is for manual review and is not a gold benchmark. Full-document verification depends on reconstructable same-document chunks.
- Next recommended task:
  Run the disabled endpoint in an isolated environment on manually reviewed citizenship and child-abduction queries, then build a curated gold dataset before considering frontend exposure or production activation.

## 2026-07-13 09:45 Europe/Moscow — Task: Fix court filters and full-judgment result presentation for MVP search

- Goal:
  Fix broken Ústavní soud / Nejvyšší soud filters, stop presenting repeated chunk hits as separate decisions, and make full judgments accessible directly from each frontend result card while keeping MVP search on the original stable chunk-level retrieval path.
- Scope:
  Backend metadata/source filter recognition, frontend chunk-result grouping by document id, frontend inline full-judgment loading, focused tests, runtime smoke, and documentation only. Retrieval ranking, embeddings, BM25/RRF scoring, Qdrant data, Redis/cache behavior, and the disabled additive document-level ranking endpoint were not changed.
- What changed:
  Updated `app/api/rag_router.py` so court/source filtering recognizes `source`, `court`, `court_name`, `document_id`, `source_document_id`, `case_reference`, `reference`, and ECLI prefixes such as `ECLI:CZ:US` / `ECLI:CZ:NS`.
  Updated retrieved chunk response projection to infer `court_name` from metadata/ECLI when explicit court metadata is missing.
  Added endpoint regression tests for `usoud / nalus`, ECLI-only ÚS results, `nsoud`, and ECLI-only NS results.
  Updated NalusFE chunk mapping to group chunk-level results by canonical `documentId`, merge supporting passages, preserve best score, and fill court/ECLI from document identity when metadata is incomplete.
  Added inline full-judgment loading in each result card through the existing read-only `GET /api/retrieval/documents/{document_id}` proxy.
  Changed the results heading from "Nalezená relevantní rozhodnutí" to "Nalezená rozhodnutí" and clarified that ordering is technical relevance, not a legal-relevance guarantee.
  Updated NalusFE README.
- Why it changed:
  The attached frontend output showed repeated chunk hits from the same decisions, missing court/ECLI labels, broken court filters returning zero results, and result cards showing only passages while full judgment text required opening a separate detail page.
- Tests run:
  Backend: `python -m pytest tests/api/test_rag_api.py tests/rag/test_full_document_retrieval.py tests/rag/test_document_retrieval.py -q` -> `59 passed`.
  Frontend: `npm run typecheck` -> passed.
  Frontend: `npm run lint` -> passed.
  Frontend Docker build during `docker compose up -d --build frontend` -> passed.
- Smoke result:
  Backend direct `POST /api/rag/retrieve` for `udělení českého občanstvi ruskému občanu`: no source filter -> 50 chunks; `sources=["constitutional"]` -> 50 chunks; `sources=["supreme"]` -> 0 chunks in the current ÚS/NALUS collection.
  Frontend proxy `POST /api/retrieval/documents`: `court=all` -> 19 unique documents; `court=usoud` -> 22 unique documents; `court=nsoud` -> 0 results.
  Frontend `/vyhledavani?q=udělení českého občanstvi ruskému občanu` returned HTTP 200, rendered `Nalezená rozhodnutí`, included `Zobrazit celý rozsudek zde`, and included the known document `ECLI:CZ:US:2023:3.US.3469.22.1`.
  Frontend full-document proxy `GET /api/retrieval/documents/ECLI%3ACZ%3AUS%3A2023%3A3.US.3469.22.1` returned HTTP 200, `full_text_availability_status=available`, `chunk_count=12`, and full text length `15734`.
  Backend `POST /api/rag/retrieve-documents` returned HTTP 404, confirming the unfinished document-level ranking endpoint remains disabled.
- Behavior preserved:
  Search still uses the original MVP `/api/rag/retrieve` flow. BGE-M3 embeddings, BM25, RRF, top-k request size, Qdrant collection/data, Redis/cache behavior, query rewrite behavior, answer generation, and document-level ranking feature flag default were not changed.
- Known limitations:
  The remaining irrelevant/weak results are a retrieval-quality issue in the original chunk-level ranking, not a frontend rendering bug. This task reduces duplicate clutter and fixes filters/metadata but does not add calibrated legal relevance filtering.
  Current runtime collection appears to be ÚS/NALUS-focused for this query; `court=nsoud` correctly returns no results under the current source filter.
- Next recommended task:
  Add a separate relevance-calibration task for the citizenship/Russian-citizen query: reviewed gold set, query expansion/synonyms, safe thresholding, duplicate-aware document ranking, and explicit rollout criteria before changing production ranking.

## 2026-07-13 03:40 Europe/Moscow — Task: Show full judgments in frontend while keeping MVP chunk-level search stable

- Goal:
  Keep the unfinished additive document-level ranking module disabled for MVP search, but allow the frontend to display full judgments instead of only citations/passages when a user opens a result detail.
- Scope:
  Added a read-only document-by-id reconstruction endpoint, frontend full-document proxy/detail rendering, focused tests, and documentation. Search ranking remains the stable chunk-level `/api/rag/retrieve` path.
- What changed:
  Added `app/rag/retrieval/full_document.py` with typed read-only Qdrant full-document reconstruction from same-document chunks ordered by `chunk_index`, document id validation, bounded chunk count, metadata normalization, explicit availability status, and diagnostics.
  Added `GET /api/rag/documents/{document_id}` to `app/api/rag_router.py`.
  Kept `POST /api/rag/retrieve-documents` disabled by default; smoke verified it returns HTTP 404 with `NALUS_DOCUMENT_RETRIEVAL_ENABLED=0`.
  Updated API logging for `/search`, `/retrieve`, and `/query` to log query length instead of raw query text.
  Added backend tests for full-document reconstruction and endpoint success/error paths.
  Updated NalusFE types/parsing and added `GET /api/retrieval/documents/{id}` as a Next.js proxy to the backend full-document endpoint.
  Updated NalusFE search mapping so result detail links use the canonical `documentId`, not `documentId#chunkId`.
  Updated the result detail tab and `/rozhodnuti/[id]` page to render full judgment text from the backend full-document endpoint.
  Increased NalusFE search proxy timeout to 60 seconds to tolerate cold start of the existing backend retrieval stack.
  Updated `docs/DOCUMENT_LEVEL_RETRIEVAL.md` and NalusFE `frontend/README.md`.
- Why it changed:
  Stable MVP search must not switch to the unfinished document-level ranking module, but the UI still needs to show complete judgments. The safe boundary is chunk-level retrieval for search plus read-only full-document reconstruction by already-known document id.
- Files changed:
  `app/rag/retrieval/full_document.py`
  `app/api/rag_router.py`
  `tests/rag/test_full_document_retrieval.py`
  `tests/api/test_rag_api.py`
  `docs/DOCUMENT_LEVEL_RETRIEVAL.md`
  NalusFE `frontend/src/types/retrieval.ts`
  NalusFE `frontend/src/lib/api/responseValidation.ts`
  NalusFE `frontend/src/lib/api/fullDocumentServer.ts`
  NalusFE `frontend/src/app/api/retrieval/documents/[id]/route.ts`
  NalusFE `frontend/src/lib/api/documentSearchServer.ts`
  NalusFE `frontend/src/lib/api/judgmentMapping.ts`
  NalusFE `frontend/src/components/ResultDetailTabs.tsx`
  NalusFE `frontend/src/app/rozhodnuti/[id]/page.tsx`
  NalusFE `frontend/README.md`
- Tests run:
  Backend: `python -m pytest tests/rag/test_full_document_retrieval.py tests/api/test_rag_api.py tests/rag/test_document_retrieval.py -q` -> `57 passed`.
  Frontend: `npm run typecheck` -> passed.
  Frontend: `npm run lint` -> passed.
  Frontend Docker build during `docker compose up -d --build frontend` -> passed.
- Smoke result:
  Backend `/health` returned `status=ok` and `orchestrator_ready=true`.
  Backend `GET /api/rag/documents/ECLI%3ACZ%3AUS%3A2026%3A3.US.446.26.1` returned HTTP 200, `full_text_availability_status=available`, `chunk_count=16`, and `full_text` length `21768`.
  Frontend proxy `GET /api/retrieval/documents/ECLI%3ACZ%3AUS%3A2026%3A3.US.446.26.1` returned the same full text length and chunk count.
  Frontend detail page `/rozhodnuti/ECLI%3ACZ%3AUS%3A2026%3A3.US.446.26.1` returned HTTP 200 and rendered full judgment content.
  Frontend search proxy `POST /api/retrieval/documents` returned HTTP 200 with 50 stable chunk-level results after backend restart.
- Behavior preserved:
  Search retrieval ranking, top-k, BGE-M3 embeddings, BM25 scoring, RRF, Qdrant collection/data, Redis/cache behavior, query rewrite behavior, answer generation, and the disabled document-level ranking endpoint default were not changed.
- Known limitations:
  Full-document reconstruction depends on same-document chunks having reliable document identifiers and preferably contiguous `chunk_index` values. If indexes are missing or duplicated, the endpoint returns `partial` with diagnostics instead of hiding the issue.
  Uvicorn access logs include the request path, so document ids in the URL can appear in access logs. No full document text is logged and the updated application search logs do not include raw queries.
- Next recommended task:
  Add a small frontend component test or Playwright smoke for opening "Celý dokument" from a search result once the frontend test harness is introduced.

## 2026-07-13 03:35 Europe/Moscow — Task: Restore stable MVP chunk-level retrieval flow

- Goal:
  Disable runtime use of the unfinished additive document-level retrieval module and return the NALUS MVP/frontend path to the stable chunk-level retrieval endpoint while keeping the new module available for separate tuning.
- Scope:
  Runtime/configuration and frontend proxy routing only. The document-level module code, BGE-M3 embeddings, Qdrant collections/data, BM25 scoring, RRF fusion, Redis/cache behavior, ingestion, thresholds, and retrieval algorithms were not changed.
- What changed:
  Set the Docker default `NALUS_DOCUMENT_RETRIEVAL_ENABLED` back to `0`.
  Kept document-level limit variables documented/configured as disabled-by-default knobs, so the module can be enabled later through an explicit rollout task.
  Updated `docs/DOCUMENT_LEVEL_RETRIEVAL.md` to state that MVP runtime should use the stable chunk-level flow while document-level retrieval is tuned separately.
  Updated NalusFE server-side proxy to call `POST /api/rag/retrieve` with `top_k=50` instead of `POST /api/rag/retrieve-documents`.
  Added frontend parsing/mapping for the stable chunk-level backend response `{ "results": [...] }`.
  Updated NalusFE README to document the stable MVP backend flow.
- Why it changed:
  The forensic audit showed that the document-level endpoint was already active in MVP runtime although the module still lacks calibrated relevance policy, metadata normalization, and full-document endpoint support. With threshold `0.0`, unrelated documents could appear in the top 50 aggregated results.
- Tests run:
  Backend: `python -m pytest tests/rag/test_document_retrieval.py tests/api/test_rag_api.py -q` -> `44 passed`.
  Frontend: `npm run typecheck` -> passed.
  Frontend: `npm run lint` -> passed.
  Frontend Docker build during `docker compose up -d --build frontend` -> passed.
- Smoke result:
  Recreated backend API container and verified `NALUS_DOCUMENT_RETRIEVAL_ENABLED=0`.
  Direct backend `POST /api/rag/retrieve-documents` returned HTTP 404 as expected.
  Direct backend `POST /api/rag/retrieve` returned 50 stable chunk-level results for the citizenship query.
  Rebuilt/recreated the NalusFE Docker container and verified `POST /api/retrieval/documents` returned 50 frontend results mapped from chunk-level backend results, with no document-level diagnostics payload.
- Known limitations:
  The first backend request after container restart can exceed the current frontend timeout because the existing retrieval path may cold-load BGE-M3 and attempt the configured query rewrite provider. A warm request succeeded. This task did not change model loading, query rewrite, or timeout policy.
  The frontend route name remains `/api/retrieval/documents` for UI compatibility, but its backend source is now chunk-level MVP retrieval.
- Next recommended task:
  Continue tuning document-level retrieval behind the disabled feature flag: relevance policy, metadata normalization, full-document endpoint, benchmark calibration, and explicit rollout criteria.

## 2026-07-13 03:10 Europe/Moscow — Task: Citizenship query retrieval forensic audit

- Goal:
  Perform a read-only forensic audit of the current NALUS document retrieval quality, metadata normalization, query rewrite behavior, ranking diagnostics, and full-document availability for the query `najdi rozsudek ústavního soudu o udělování českého občanství ruským občanům`.
- Scope:
  Audit only. Retrieval algorithms, BGE-M3 embeddings, BM25, RRF, Qdrant collections/data, aliases, ingestion, Redis, thresholds, backend API behavior, and frontend files were not changed.
- What changed:
  Created `artifacts/evaluation_quality/citizenship_query_retrieval_audit_20260712.md`.
  Created `artifacts/evaluation_quality/citizenship_query_retrieval_audit_20260712.json`.
  Updated this progress file with the audit result.
- Findings:
  Direct backend reproduction returned 50 final document-level results from 200 RRF candidate chunks and 142 unique documents before the final `max_returned_documents=50` cap.
  Query rewrite is wired in runtime through `QueryRewriteService`, but the configured DeepSeek provider returned 401/empty output during reproduction, so the effective retrieval query remained the original query.
  The configured document relevance threshold is `0.0`; therefore unrelated documents are not filtered by relevance threshold and can remain in the top 50 if their RRF/document score is high enough.
  Deterministic content classification found 7 potentially relevant citizenship-related results and 43 clearly irrelevant results; no result was classified as clearly relevant to the narrower Russian-citizen citizenship query.
  `ECLI:CZ:US:2026:3.US.446.26.1` was returned because BM25 ranked same-document chunks at rank 21; the matching text is about municipal referendum/spatial planning and contains weak lexical overlap such as `občanům`, not citizenship granting.
  `ECLI:CZ:US:2026:4.US.893.26.1` was returned because dense retrieval ranked the opening chunk at rank 27; the document concerns parental responsibility/care of a minor child, not citizenship granting.
  Final document IDs were unique and no evidence of cross-judgment chunk mixing was found in final aggregation.
- Metadata audit:
  The observed `Neuvedený soud` / `ECLI: neuvedeno` defect for `ECLI:CZ:US:2026:3.US.446.26.1` originates in backend response metadata availability/projection: the best returned chunks expose `document_id`, `source_document_id`, and `decision_date`, but not `court`, `ecli`, or `case_reference`.
  The Next.js proxy did not corrupt the data. The frontend mapper deterministically falls back to `Neuvedený soud`, undefined ECLI, and `document_id` as case reference when backend metadata is missing.
- Full-text availability:
  For both unrelated examples and inspected final documents, complete judgment text is available as same-document Qdrant chunks mirrored in the BM25 sidecar when chunk indexes are contiguous.
  `ECLI:CZ:US:2026:3.US.446.26.1` has 16 Qdrant/BM25 chunks, indexes 0-15, no missing/duplicate indexes, and deterministic reconstruction is possible.
  `ECLI:CZ:US:2026:4.US.893.26.1` has 10 Qdrant/BM25 chunks, indexes 0-9, no missing/duplicate indexes, and deterministic reconstruction is possible.
- Tests run:
  `python -m pytest tests/rag/test_document_retrieval.py tests/api/test_rag_api.py -q` -> `44 passed`.
  Repeated `pytest-asyncio` loop-scope deprecation warning is non-blocking and unrelated.
- Smoke result:
  `docker compose ps` showed backend, Qdrant, Redis, Prometheus, Grafana, and the eval metrics exporter running.
  Direct backend `POST /api/rag/retrieve-documents` reproduced 50 results.
  Read-only frontend proxy `POST /api/retrieval/documents` confirmed the metadata fallback behavior for the two known unrelated examples.
- Behavior preserved:
  No retrieval/ranking change, no threshold change, no Qdrant write, no Redis/cache behavior change, no embedding change, no model download, no frontend file modification, and no new fallback was introduced.
- Known limitations:
  The classification is deterministic and human-auditable but conservative. It does not use an LLM and does not assert absolute legal relevance beyond observable document content and query-term/topic evidence.
  The audit generated runtime artifacts and intentionally did not commit them unless a later repository policy task requests that.
- Next recommended task:
  Implement a separate read-only full-document endpoint with canonical metadata normalization, ordered same-document chunk reconstruction, explicit full-text availability status, and regression tests before exposing document detail deep links.

## 2026-07-13 02:04 Europe/Moscow — Task: NalusFE document retrieval frontend integration

- Goal:
  Connect the existing NalusFE Next.js search interface to the additive document-level FastAPI retrieval endpoint without changing retrieval ranking, embeddings, Qdrant data, BM25, RRF, Redis, ingestion, or answer generation.
- What changed:
  Added Docker environment wiring so the existing `POST /api/rag/retrieve-documents` endpoint is enabled for the local integrated runtime.
  Updated `.env.example` to show `NALUS_DOCUMENT_RETRIEVAL_ENABLED=1` for the frontend integration path.
  Clarified `docs/DOCUMENT_LEVEL_RETRIEVAL.md` so the code default remains disabled while the integrated Docker runtime can explicitly enable the endpoint.
- Why it changed:
  The frontend integration uses a server-side Next.js proxy and must call the real document-level endpoint. The running backend container needs the existing feature flag enabled for end-to-end smoke tests and local presentation use.
- Files changed:
  `docker-compose.yml`
  `.env.example`
  `docs/DOCUMENT_LEVEL_RETRIEVAL.md`
  `PROJECT_PROGRESS.md`
- Tests run:
  NalusFE frontend: `npm run lint`, `npm run typecheck`, `npm run build`, and `npm audit --audit-level=moderate` all passed after adding a safe PostCSS override for the latest Next.js transitive dependency tree.
  Backend: `python -m pytest tests/rag/test_document_retrieval.py tests/api/test_rag_api.py -q` -> `44 passed`.
  Compose config validation passed for both NalusFE and nalus-scraper.
- Smoke result:
  Backend `/health` returned `status=ok` with `orchestrator_ready=true`.
  Backend `POST /api/rag/retrieve-documents` returned 50 real unique document-level results for a Czech legal query.
  NalusFE dev server and Docker-served frontend `POST /api/retrieval/documents` both returned 50 real backend-backed results with first document `ECLI:CZ:US:2026:2.US.98.26.1`.
  Docker-served `GET /vyhledavani?q=...` returned HTTP 200 and rendered backend-backed search content, not mock data.
  Invalid frontend API input checks returned controlled HTTP 400 errors for empty query and invalid filter values.
  The smoke run showed the existing backend query-rewrite path attempted the configured text LLM and fell back to the original query after a provider 401; the frontend did not call answer-generation or chat endpoints.
- Behavior preserved:
  Retrieval ranking, document scoring, BGE-M3 embeddings, BM25, RRF, Qdrant collections/data, Redis/cache behavior, ingestion, LLM/DeepSeek calls, and existing `/api/rag/retrieve` and `/api/rag/query` behavior were not changed.
- Known limitations:
  The backend still does not expose a separate document-detail-by-id endpoint; the frontend can render document details only from search results returned by `retrieve-documents`.
  The document aggregation module is no-LLM, but the endpoint obtains candidate chunks through the existing orchestrator retrieval path, including optional query rewrite when the backend is configured with a real text LLM provider.
- Next recommended task:
  Add a dedicated read-only document detail endpoint if the product needs stable deep links to individual judgments independent of a search response.

## 2026-07-10 15:25 Europe/Moscow — Task: NSoud provenance checker + conservative single gold annotation

- Goal:
  Build a read-only NSoud provenance checker for pending legal QA items, then apply only the single conservative NSoud gold annotation that passed the check.
- What changed:
  Added `scripts/check_nsoud_gold_provenance.py` and `tests/test_check_nsoud_gold_provenance.py`.
  Added `artifacts/rag_eval/legal_qa/nsoud_provenance_check_20260710.md`.
  Added `artifacts/rag_eval/legal_qa/annotations/nsoud_provenance_candidates_20260710.jsonl`.
  Updated `scripts/apply_gold_source_annotations.py` to annotate only `nsoud-qa-007`.
  Regenerated `artifacts/rag_eval/legal_qa/datasets/nsoud_qa_v1.jsonl`.
  Refreshed `artifacts/rag_eval/legal_qa/answer_eval/nsoud_no_llm_baseline/*`.
  Updated `artifacts/rag_eval/legal_qa/gold_source_review_20260709.md`.
  Updated `artifacts/rag_eval/legal_qa/answer_eval_report_20260709.md`.
  Added `PROJECT_EXECUTION_PROTOCOL.md` as the local execution protocol for this repo.
- Why it changed:
  Provenance extraction was no longer the blocker for NSoud pending questions. The checker was needed to separate technical provenance availability from true legal relevance. Only `nsoud-qa-007` met the conservative bar for direct gold annotation.
- Files changed:
  `PROJECT_EXECUTION_PROTOCOL.md`
  `PROJECT_PROGRESS.md`
  `scripts/check_nsoud_gold_provenance.py`
  `tests/test_check_nsoud_gold_provenance.py`
  `scripts/apply_gold_source_annotations.py`
  `artifacts/rag_eval/legal_qa/gold_source_review_20260709.md`
  `artifacts/rag_eval/legal_qa/datasets/nsoud_qa_v1.jsonl`
  `artifacts/rag_eval/legal_qa/answer_eval/nsoud_no_llm_baseline/*`
  `artifacts/rag_eval/legal_qa/answer_eval_report_20260709.md`
  `artifacts/rag_eval/legal_qa/nsoud_provenance_check_20260710.md`
  `artifacts/rag_eval/legal_qa/annotations/nsoud_provenance_candidates_20260710.jsonl`
- Tests run:
  `python -m pytest tests/test_check_nsoud_gold_provenance.py tests/rag/test_legal_qa_benchmark.py tests/rag/test_legal_answer_eval.py -q`
- Smoke result:
  Read-only Qdrant lookup succeeded via `docker compose exec -T api`.
  NSoud no-LLM answer eval rerun completed and refreshed `summary.json`.
- Known limitations:
  `nsoud-qa-007` increased NSoud gold coverage from `3/10` to `4/10`, but answer-support quality for that item still evaluates as `gap`.
  The remaining pending NSoud items still need manual relevance review before any further annotation.
  Existing uncommitted generated ÚS/mixed answer eval artifacts remain in the worktree and were not part of this task.
- Next recommended task:
  Review `nsoud-qa-001`, `002`, `005`, `006`, `008`, and `009` manually against `nsoud_provenance_check_20260710.md` and decide whether any should stay pending, be reformulated, or be rejected as benchmark questions.

## 2026-07-10 20:30 Europe/Moscow — Task: Legal answer eval metric semantics repair after failed-case diagnostics

- Goal:
  Repair the interpretation of offline legal answer-eval metrics so that reports clearly separate real failures, missing-gold non-evaluable items, usable partial support, corpus-only routing support, and true retrieval misses.
- What changed:
  Updated `app/rag/eval/legal_answer_eval.py` with explicit total/gold/missing-gold/evaluable fields, gold retrieval miss metrics, unsupported-risk rate, citation-available rate over gold, and corpus-routing support rate.
  Reworked failed-case categorization to use `not_evaluable_missing_gold`, `invalid_gold_annotation`, `true_retrieval_miss`, `usable_partial_support`, `weak_partial_support`, `unsupported_boilerplate_or_gap`, `corpus_only_no_document_citation_expected`, and `metric_denominator_warning`.
  Added conservative final-status logic (`PASS` / `WARN` / `FAIL` / `FAIL_WITH_REAL_NSOUD_RISK`) driven by real failure categories instead of strict-rate thresholds alone.
  Added dedicated `nsoud-qa-007` diagnostics with expected source, retrieved top-k ids, and conservative conclusion.
  Updated the Prometheus summary compatibility path in `app/observability/eval_metrics_exporter.py`.
  Regenerated `artifacts/evaluation_quality/*` and refreshed `artifacts/rag_eval/legal_qa/answer_eval_report_20260709.md`.
- Why it changed:
  The previous diagnostic outputs overstated failure severity by treating missing gold and corpus-only mixed cases as ordinary failures. The new semantics make the reports usable for decision-making without hiding the real NSoud risks.
- Files changed:
  `app/rag/eval/legal_answer_eval.py`
  `app/observability/eval_metrics_exporter.py`
  `scripts/run_legal_answer_eval.py`
  `scripts/generate_legal_answer_eval_diagnostics.py`
  `tests/rag/test_legal_answer_eval.py`
  `tests/rag/test_legal_answer_eval_diagnostics.py`
  `tests/observability/test_eval_metrics_exporter.py`
  `artifacts/evaluation_quality/*`
  `artifacts/rag_eval/legal_qa/answer_eval_report_20260709.md`
- Tests run:
  `python -m pytest tests/rag/test_legal_answer_eval.py -q`
  `python -m pytest tests/rag/test_legal_answer_eval_diagnostics.py -q`
  `python -m pytest tests/observability/test_eval_metrics_exporter.py -q`
- Smoke result:
  `python scripts/generate_legal_answer_eval_diagnostics.py --runs-dir artifacts/rag_eval/legal_qa/answer_eval --output-dir artifacts/evaluation_quality` completed successfully and produced updated JSON/Markdown diagnostics.
- Known limitations:
  The worktree still contains pre-existing dirty offline answer-eval artifacts under `artifacts/rag_eval/legal_qa/answer_eval/usoud_no_llm_baseline/*` and `mixed_no_llm_baseline/*`.
  No new commit was created in this task.
  `nsoud-qa-007` remains a conservative true retrieval miss in the current frozen baseline.
- Next recommended task:
  Review the NSoud criminal-dovolani benchmark questions around § 265b tr. ř., especially `nsoud-qa-007` and `nsoud-qa-010`, and decide whether the next action is query reformulation, gold refinement, or a separate retrieval-quality investigation.

## 2026-07-10 21:10 Europe/Moscow — Task: Read-only NSoud retrieval risk investigation for `nsoud-qa-007` and `nsoud-qa-010`

- Goal:
  Verify whether the post-diagnostics NSoud risk cases are true retrieval misses, provenance/export artifacts, or benchmark-design issues, without changing retrieval logic or retrieval data.
- What changed:
  Added `artifacts/evaluation_quality/nsoud_retrieval_risk_investigation_20260710.md`.
  Added `artifacts/evaluation_quality/nsoud_retrieval_risk_investigation_20260710.json`.
- Why it changed:
  The repaired diagnostics still flagged `FAIL_WITH_REAL_NSOUD_RISK`, but `nsoud-qa-007` and `nsoud-qa-010` needed direct read-only verification against Qdrant, BM25 sidecar contents, and current top-50 retrieval behavior.
- Files changed:
  `PROJECT_PROGRESS.md`
  `artifacts/evaluation_quality/nsoud_retrieval_risk_investigation_20260710.md`
  `artifacts/evaluation_quality/nsoud_retrieval_risk_investigation_20260710.json`
- Tests run:
  `python -m pytest tests/rag/test_legal_answer_eval.py tests/rag/test_legal_answer_eval_diagnostics.py -q`
- Smoke result:
  Read-only Qdrant inspection via `docker compose exec -T api` confirmed that `nsoud-qa-007` expected source `ECLI:CZ:NS:2025:5.TDO.1086.2024.1` is present in the collection and already corresponds to frozen baseline chunk `735`.
  BM25 sidecar inspection confirmed `1862/1862` rows have blank `document_id` and `source_document_id`, which explains provenance loss in BM25-backed frozen hits.
- Known limitations:
  No code or retrieval data was changed in this task, so the existing diagnostics artifacts remain unchanged until a future provenance/export fix or benchmark-item reformulation is executed.
  `nsoud-qa-010` remains a benchmark-quality risk because the current expected source is mostly operative `Dovolání se odmítá` boilerplate and does not cleanly support the doctrinal distinction in the question.
- Next recommended task:
  Remove `nsoud-qa-007` from the “true retrieval miss” bucket by fixing provenance/export visibility for BM25-backed NSoud hits, then reformulate or replace `nsoud-qa-010` before using it as a hard retrieval-quality signal.

## 2026-07-11 09:50 Europe/Moscow — Task: NSoud BM25 sidecar provenance repair without scoring changes

- Goal:
  Repair the NSoud BM25 sidecar so BM25 and hybrid retrieval artifacts expose correct provenance metadata, while preserving BM25 scoring, dense scoring, and RRF behavior.
- What changed:
  Updated `scripts/build_bm25_sidecar_from_qdrant.py` to flatten and export richer provenance fields from Qdrant payloads.
  Updated `app/rag/retrieval/bm25_sidecar.py` so BM25 retrieval results hydrate provenance metadata from explicit sidecar columns.
  Added `scripts/repair_nsoud_bm25_sidecar_provenance.py` with `--dry-run` and `--execute` modes and strict `chunk_id`-based mapping to read-only Qdrant payloads.
  Added `tests/test_repair_nsoud_bm25_sidecar_provenance.py`.
  Wrote candidate repaired sidecar `storage/rag/bm25/nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1.provenance_repaired.sqlite`.
  Created candidate run `artifacts/rag_eval/legal_qa/runs/nsoud_sidecar_provenance_repaired/` and candidate answer eval `artifacts/rag_eval/legal_qa/answer_eval/nsoud_sidecar_provenance_repaired/`.
  Added repair reports `artifacts/evaluation_quality/nsoud_bm25_sidecar_provenance_repair_20260710.md` and `.json`.
- Why it changed:
  The original NSoud sidecar had blank provenance in `1862/1862` rows, which made frozen BM25-backed hits lose usable `document_id` and `source_document_id` metadata even though the corresponding Qdrant points already had correct provenance.
- Files changed:
  `PROJECT_PROGRESS.md`
  `app/rag/retrieval/bm25_sidecar.py`
  `scripts/build_bm25_sidecar_from_qdrant.py`
  `scripts/repair_nsoud_bm25_sidecar_provenance.py`
  `tests/test_repair_nsoud_bm25_sidecar_provenance.py`
  `artifacts/evaluation_quality/nsoud_bm25_sidecar_provenance_repair_20260710.md`
  `artifacts/evaluation_quality/nsoud_bm25_sidecar_provenance_repair_20260710.json`
- Tests run:
  `python -m pytest tests/test_repair_nsoud_bm25_sidecar_provenance.py -q`
  `python -m pytest tests/rag/test_production_bge_m3_profile.py tests/test_merge_bge_m3_candidate_collections.py tests/rag/test_legal_qa_benchmark.py -q`
  `python -m pytest tests/rag/test_legal_answer_eval.py tests/rag/test_legal_answer_eval_diagnostics.py -q`
- Smoke result:
  `docker compose exec -T api python scripts/repair_nsoud_bm25_sidecar_provenance.py ... --dry-run` confirmed `1862/1862` deterministic matches and zero text mismatches.
  `docker compose exec -T api python scripts/repair_nsoud_bm25_sidecar_provenance.py ... --execute` produced a repaired candidate sidecar with `0` blank `document_id`, `source_document_id`, `ecli`, `case_number`, and `source`.
  Candidate retrieval benchmark kept `hit@1=0.700`, `hit@5=1.000`, `pass_rate=1.000`, while `nsoud-qa-007` now exposes rank-1 ECLI metadata directly from the retrieval artifact.
- Known limitations:
  `court` and `spisova_znacka` remain blank where they are absent in Qdrant payloads; the repair does not invent fields.
  `nsoud-qa-010` remains a real answer-support / boilerplate benchmark risk and still drives the candidate-only diagnostic final status to `FAIL_WITH_REAL_NSOUD_RISK`.
  Existing dirty generated ÚS/mixed answer-eval artifacts in the worktree remain unrelated and untouched.
- Next recommended task:
  Use the repaired sidecar/export path as the NSoud benchmark candidate, then either update the diagnostics status wording to distinguish answer-support risk from retrieval-miss risk more explicitly, or reformulate `nsoud-qa-010` before treating NSoud as fully green.

## 2026-07-11 12:40 Europe/Moscow — Task: NSoud strict direct pass audit

- Goal:
  Explain why `nsoud_sidecar_provenance_repaired` still has `strict_direct_pass_rate_gold=0.0` after provenance repair, and verify that the Grafana/Prometheus metrics path is reading the intended artifacts.
- What changed:
  Added `artifacts/evaluation_quality/nsoud_strict_direct_audit_20260711_124021.md`.
  Added `artifacts/evaluation_quality/nsoud_strict_direct_audit_20260711_124021.json`.
- Why it changed:
  The repaired NSoud run improved citation availability and reduced unsupported answer risk, but the dashboard still showed weak strict-direct performance. A per-question audit was needed to separate benchmark/gold misalignment, same-document wrong-chunk retrieval, and any possible dashboard mapping issue.
- Files changed:
  `PROJECT_PROGRESS.md`
  `artifacts/evaluation_quality/nsoud_strict_direct_audit_20260711_124021.md`
  `artifacts/evaluation_quality/nsoud_strict_direct_audit_20260711_124021.json`
- Tests run:
  `python -m pytest tests/rag/test_legal_answer_eval.py tests/rag/test_legal_answer_eval_diagnostics.py tests/observability/test_eval_metrics_exporter.py tests/test_repair_nsoud_bm25_sidecar_provenance.py -q`
- Smoke result:
  Read-only inspection confirmed the dashboard exporter is reading per-run `summary.json` files from `artifacts/rag_eval/legal_qa/answer_eval/*` with labels `(run_name, corpus)`.
  No dashboard query/label bug was needed to explain the NSoud strict-direct weakness.
- Known limitations:
  The audit is intentionally read-only; no retrieval logic, evaluator behavior, or benchmark source data was changed in this task.
  `nsoud-qa-004` and `nsoud-qa-010` still look like benchmark/gold alignment risks rather than clean retrieval regressions.
  `nsoud-qa-007` still needs a focused same-document chunk-selection follow-up before it can become a strict-direct pass.
- Next recommended task:
  Re-annotate or replace `nsoud-qa-004` and `nsoud-qa-010`, then run a narrowly scoped follow-up on `nsoud-qa-007` to test whether a better same-document chunk can be surfaced without changing global BM25/dense/RRF scoring.

## 2026-07-11 13:10 Europe/Moscow — Task: NALUS Production Task Validator

- Goal:
  Add a reusable deterministic validator for NALUS production tasks that checks dirty-file scope, risky diffs, documentation/test expectations, and task-safety signals before commit or final reporting.
- What changed:
  Added `app/project_validation/` with git-state parsing, file classification, diff scanning, reporting, and orchestration modules.
  Added CLI entrypoint `scripts/validate_nalus_task.py`.
  Added `tests/test_nalus_task_validator.py`.
  Added `docs/NALUS_TASK_VALIDATOR.md`.
- Why it changed:
  The repo needed a project-specific equivalent of the Memorial/Eternal World task validator so future NALUS tasks can detect accidental baseline-artifact staging, risky retrieval/Qdrant/model changes, missing progress updates, and missing tests before commits.
- Files changed:
  `PROJECT_PROGRESS.md`
  `app/project_validation/__init__.py`
  `app/project_validation/schemas.py`
  `app/project_validation/git_status.py`
  `app/project_validation/file_classifier.py`
  `app/project_validation/diff_scanner.py`
  `app/project_validation/report.py`
  `app/project_validation/validator.py`
  `scripts/validate_nalus_task.py`
  `tests/test_nalus_task_validator.py`
  `docs/NALUS_TASK_VALIDATOR.md`
- Tests run:
  `python -m pytest tests/test_nalus_task_validator.py -q`
  `python -m pytest tests/rag/test_legal_answer_eval.py tests/rag/test_legal_answer_eval_diagnostics.py tests/observability/test_eval_metrics_exporter.py tests/test_repair_nsoud_bm25_sidecar_provenance.py -q`
  `python scripts/validate_nalus_task.py --task-name "NALUS Production Task Validator" --mode implementation --expected-branch main --no-write`
- Known limitations:
  The validator is intentionally heuristic and diff-based; it does not understand semantic intent beyond configured patterns.
  Risk detection is intentionally conservative and currently scans changed source/test diffs, not full repository history.
  Generated validation reports are optional runtime artifacts and are not committed by default.
- Next recommended task:
  Run the validator before future NALUS commits and extend allowlists/risk rules only when an intentional change type repeatedly appears in real workflow.

## 2026-07-12 00:36 Europe/Moscow — Task: Refresh ÚS and Mixed no-LLM canonical answer-eval baselines

- Goal:
  Persist the intentionally regenerated canonical ÚS and Mixed no-LLM answer-eval artifacts so a clean checkout and exporter restart preserve the current verified monitoring values.
- What changed:
  Refreshed the canonical `usoud_no_llm_baseline` artifacts to represent `10/20` gold questions with `1` direct and `9` partial support results.
  Refreshed the canonical `mixed_no_llm_baseline` artifacts to represent `8/10` corpus-only gold questions with successful corpus routing.
  Persisted the generated diagnostics files emitted alongside both canonical runs.
- Why it changed:
  Gold annotation coverage was expanded after the prior canonical artifacts were committed. Persisting the regenerated outputs prevents Grafana and Prometheus values from reverting after checkout or restart.
- Expected metrics:
  ÚS: `gold=10`, `usable_support_rate_gold=1.0`, `citation_available_rate=1.0`, `unsupported_answer_risk_count=0`.
  Mixed: `gold=8`, `corpus_only_count=8`, `usable_support_rate_gold=1.0`, `corpus_routing_support_rate=1.0`, `citation_available_rate=0.0`, `unsupported_answer_risk_count=0`.
- Exporter/Grafana verification:
  Restarted `nalus-eval-metrics-exporter` and confirmed the expected `legal_answer_eval_gold`, `legal_answer_eval_usable_support_rate_gold`, and `legal_answer_eval_citation_available_rate` series for both named runs at `http://localhost:9108/metrics`.
  The exporter uses `legal_answer_eval_citation_available_rate`; no Grafana query change was required.
- Files changed:
  `PROJECT_PROGRESS.md`
  `artifacts/rag_eval/legal_qa/answer_eval/usoud_no_llm_baseline/*`
  `artifacts/rag_eval/legal_qa/answer_eval/mixed_no_llm_baseline/*`
- Tests run:
  `python -m pytest tests/rag/test_legal_answer_eval.py tests/observability/test_eval_metrics_exporter.py -q` -> `32 passed` with one non-blocking `pytest-asyncio` deprecation warning.
- Behavior preserved:
  Retrieval, BGE-M3, embedding dimensions/provider, dense scoring, BM25 scoring, RRF, global `top_k`, Qdrant collections/aliases/data, Redis/cache behavior, model loading, and LLM/DeepSeek behavior were not changed.
- Known limitations:
  Mixed citation availability remains `0.0` by design because all eight Mixed gold items are corpus-only and do not require document citations.
- Next recommended task:
  Complete the evidence-backed NSoud QA dataset/gold repair and regenerate an isolated `nsoud_dataset_repaired` candidate without changing retrieval scoring.

## 2026-07-12 00:45 Europe/Moscow — Task: NSoud QA dataset and gold repair

- Goal:
  Conservatively repair the four NSoud benchmark/gold issues identified by the strict-direct audit, regenerate an isolated retrieval/no-LLM candidate, and verify monitoring compatibility without changing retrieval scoring.
- Original issues and decisions:
  `nsoud-qa-003`: `evaluator_followup_needed` — corrected the inflection-specific expected keyword `občanské` to source form `občanský`; retained question and ECLI.
  `nsoud-qa-004`: `safe_gold_reannotation` — replaced the mismatched criminal `8 Tdo` gold with civil rank-1 `ECLI:CZ:NS:2025:33.CDO.79.2024.1` and reformulated the item to the § 237 o. s. ř. criteria explicitly supported by chunk `1000`.
  `nsoud-qa-007`: `safe_same_document_chunk_refinement` — retained the verified ECLI and query; replaced the tautological answer point with doctrine from same-document chunks `732–733`, while recording weaker rank-1 closing-summary chunk `735`.
  `nsoud-qa-010`: `safe_question_reformulation` — removed the unsupported odmítnutí-versus-zamítnutí comparison and asked the narrower admissibility question directly supported by existing-gold chunk `1644`.
- Dataset/gold changes:
  Updated only `nsoud-qa-003`, `004`, `007`, and `010` in `nsoud_qa_v1.jsonl`.
  Updated the reproducible NSoud ECLI map in `scripts/apply_gold_source_annotations.py` and the human gold review table.
  Added idempotence, evidence-alignment, unchanged-item, and no-invented-provenance regression coverage in `tests/test_nsoud_dataset_repair.py`.
- Candidate artifacts:
  Retrieval: `artifacts/rag_eval/legal_qa/runs/nsoud_dataset_repaired/` using the existing repaired sidecar and read-only Qdrant search.
  Answer eval/diagnostics: `artifacts/rag_eval/legal_qa/answer_eval/nsoud_dataset_repaired/` with `--no-llm --require-citations`.
  Repair audit: `artifacts/evaluation_quality/nsoud_dataset_repair_20260711.md` and `.json`.
- Metrics before (`nsoud_sidecar_provenance_repaired`):
  `gold=4`, `direct=0`, `partial=3`, `gap=0`, `boilerplate_noise=1`, `citation_available_rate=0.75`, `usable_support_rate_gold=0.75`, `unsupported_answer_risk_count=1`, `strict_direct_pass_rate_gold=0.0`.
- Metrics after (`nsoud_dataset_repaired`):
  `gold=4`, `direct=0`, `partial=3`, `gap=1`, `boilerplate_noise=0`, `citation_available_rate=0.75`, `usable_support_rate_gold=0.75`, `unsupported_answer_risk_count=1`, `strict_direct_pass_rate_gold=0.0`.
  Retrieval candidate: `pass_rate=0.9`, `source_hit@1=0.75`, `source_hit@3=0.75`, `source_hit@5=1.0`, `mean_source_constraint_match=1.0`.
- Monitoring verification:
  Restarted `nalus-eval-metrics-exporter`; all requested `legal_answer_eval_*` metrics for `run_name="nsoud_dataset_repaired"` were exposed with actual values.
  Prometheus query for `legal_answer_eval_gold{run_name="nsoud_dataset_repaired"}` returned `4`; metric names remain Grafana-compatible and no dashboard query changed.
- Files changed:
  `PROJECT_PROGRESS.md`
  `artifacts/rag_eval/legal_qa/datasets/nsoud_qa_v1.jsonl`
  `artifacts/rag_eval/legal_qa/gold_source_review_20260709.md`
  `scripts/apply_gold_source_annotations.py`
  `tests/test_nsoud_dataset_repair.py`
  `artifacts/evaluation_quality/nsoud_dataset_repair_20260711.md`
  `artifacts/evaluation_quality/nsoud_dataset_repair_20260711.json`
  `artifacts/rag_eval/legal_qa/runs/nsoud_dataset_repaired/*`
  `artifacts/rag_eval/legal_qa/answer_eval/nsoud_dataset_repaired/*`
- Tests run:
  `python -m pytest tests/rag/test_legal_answer_eval.py -q` -> `22 passed`.
  `python -m pytest tests/rag/test_legal_answer_eval_diagnostics.py -q` -> `2 passed`.
  `python -m pytest tests/rag/test_legal_qa_benchmark.py -q` -> `19 passed`.
  `python -m pytest tests/observability/test_eval_metrics_exporter.py -q` -> `10 passed`.
  `python -m pytest tests/test_repair_nsoud_bm25_sidecar_provenance.py -q` -> `5 passed`.
  `python -m pytest tests/test_nalus_task_validator.py -q` -> `9 passed`.
  `python -m pytest tests/test_nsoud_dataset_repair.py -q` -> `3 passed`.
  Repeated `pytest-asyncio` default-loop-scope deprecation warning is non-blocking and unrelated to this task.
- Runtime/infra safety:
  Qdrant access was read-only search; no ingest, collection rebuild, write, or alias switch occurred.
  BGE-M3 loaded from the existing local cache; no model download occurred.
  Redis was not enabled or used; no LLM or DeepSeek call occurred.
  Dense scoring, BM25 scoring, RRF, global `top_k`, embeddings, cache behavior, and fallback behavior were unchanged.
- Validator result:
  `python scripts/validate_nalus_task.py --task-name "NSoud QA dataset repair" --mode eval_change --expected-branch main --no-write` -> understood `WARN` with exactly two `unknown_dirty_file` findings.
  Both warnings are intentional classifier limitations for the explicitly allowed task files `artifacts/rag_eval/legal_qa/datasets/nsoud_qa_v1.jsonl` and `artifacts/rag_eval/legal_qa/gold_source_review_20260709.md`; documentation/test checks passed and all safety summaries remained `no`.
- Known limitations:
  `nsoud-qa-010` remains an honest unsupported risk: the correct doctrinal gold chunk is rank 4, but its fixed 240-character exported snippet ends before the supporting sentences.
  `nsoud-qa-003` remains at exported-snippet coverage `2/3 = 0.6667`, below the unchanged `>= 0.67` strict gate.
- Next recommended task:
  Add and test deterministic evidence-window handling for gold chunks whose relevant doctrine lies beyond the exported snippet, without lowering the strict threshold or changing global retrieval scoring.

## 2026-07-12 — Task: Shared Grafana — Add Eternal World to NALUS Grafana

- Goal:
  Use the existing Grafana on `http://localhost:3002` as one UI for NALUS and Eternal World while retaining two independent Prometheus instances and TSDBs.
- Architecture:
  Preserved NALUS datasource `Prometheus` / UID `prometheus` / internal URL `http://prometheus:9090` as the only default datasource.
  Added `Eternal World Prometheus` / UID `eternal-world-prometheus`, with URL supplied through `ETERNAL_WORLD_PROMETHEUS_URL` and local Docker default `http://host.docker.internal:9090`.
  NALUS Prometheus remains on host port `9091`; Eternal World Prometheus remains on `9090`.
  Separated dashboard provider paths into `/var/lib/grafana/dashboards/nalus` and `/var/lib/grafana/dashboards/eternal-world` to prevent overlapping scans and duplicate UIDs.
- Dashboard source-of-truth:
  Eternal World dashboard files are mounted read-only from the sibling Eternal World repository. No dashboard JSON copy is maintained in NALUS.
  Provider folders are `NALUS` and `Eternal World`.
- Configuration:
  Added environment overrides for the Eternal World Prometheus URL and dashboard directory.
  Added `host.docker.internal:host-gateway` for portable local host routing where Docker supports `host-gateway`.
  Bind mounts use `create_host_path: false`, so a missing sibling checkout fails explicitly.
- Validator support:
  Added an explicit `infra_config` classification for Compose, monitoring provisioning, and `.env.example` files.
  Fixed `--allow-risk infra_or_dependency_change` so an explicitly authorized infrastructure task can pass without weakening Qdrant/model/retrieval safety rules.
- Tests and validation:
  `docker compose config --quiet` passed.
  `python -m json.tool monitoring/grafana/dashboards/legal_answer_eval_dashboard.json` passed.
  `python -m pytest tests/test_nalus_task_validator.py tests/observability/test_shared_grafana_provisioning.py tests/observability/test_eval_metrics_exporter.py -q` -> `25 passed` with the existing non-blocking `pytest-asyncio` warning.
  Task validator in implementation mode returned `PASS` with zero findings after explicitly authorizing the requested Compose infrastructure change and the unchanged Redis context line in `.env.example`.
  Shared provisioning tests verify datasource preservation, unique datasource UIDs/default, non-overlapping provider paths, read-only mounts, and the unchanged NALUS dashboard UID bindings.
- Runtime smoke:
  Recreated only `grafana`; Grafana `11.4.0` became healthy on `3002`.
  Datasource health returned `OK` for both `prometheus` and `eternal-world-prometheus`.
  NALUS dashboard loaded in folder `NALUS`; Eternal World dashboard loaded in folder `Eternal World` with UID `eternal-world-fa-chat`.
  Grafana proxy isolation check returned NALUS `legal_answer_eval_gold` only through UID `prometheus`, and Eternal World `fa_chat_requests_total` only through UID `eternal-world-prometheus`.
  Shared Grafana provisioning logs contained no blocking datasource, dashboard, duplicate UID, or permission error.
- Behavior preserved:
  NALUS application metrics, Prometheus scrape config, exporter, retrieval, BGE-M3, BM25, RRF, Qdrant, Redis, API behavior, and production aliases were not changed.
  Eternal World application metrics and Prometheus storage were not changed.
- Known limitations:
  The local default relies on the host gateway. Linux/server deployments must override `ETERNAL_WORLD_PROMETHEUS_URL` with an address reachable from the Grafana container.
  Shared Grafana currently remains owned by the NALUS Compose stack; a dedicated observability repository is deferred until more projects require integration.
- Next recommended task:
  Move shared Grafana into a dedicated observability-stack repository only when more projects need to be added.

## 2026-07-12 17:59 Europe/Moscow — Task: Deterministic same-document evidence windows for legal answer evaluation

- Goal:
  Allow deterministic no-LLM legal answer evaluation to inspect a bounded same-document evidence window for a verified gold hit, without changing retrieval ranking, evaluator thresholds, model behavior, Qdrant state, or LLM behavior.
- Architecture:
  Added `app/rag/eval/evidence_window.py` as the typed evidence-window layer. The evaluator validates `source_document_id`, `document_id`, `ecli`, and `chunk_index`, loads same-document adjacent chunks, orders by `chunk_index`, enforces chunk and character bounds, preserves provenance diagnostics, and reports failures explicitly. The existing evaluator behavior remains the default unless `--evidence-window` is passed.
- What changed:
  Updated `app/rag/eval/legal_answer_eval.py` so enabled evidence windows evaluate keyword support against combined evidence text while source/citation matching still depends on verified document provenance.
  Updated `scripts/run_legal_answer_eval.py` with explicit evidence-window CLI options and an explicit local sidecar path option.
  Updated `scripts/generate_legal_answer_eval_diagnostics.py` so diagnostics replay the evidence-window configuration recorded in `metrics.json`.
  Added `tests/rag/test_legal_evidence_window.py` with focused unit/integration coverage for ordering, bounds, same-document enforcement, diagnostics, summary counters, and NSoud-style cases.
  Created candidate answer-eval artifacts under `artifacts/rag_eval/legal_qa/answer_eval/nsoud_evidence_window_candidate/`.
  Added `artifacts/evaluation_quality/nsoud_evidence_window_evaluation_20260712.md` and `.json`.
- Configuration:
  `--evidence-window --evidence-neighbors-before 1 --evidence-neighbors-after 1 --evidence-max-chunks 3 --evidence-max-characters 6000 --evidence-sidecar storage/rag/bm25/nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1.provenance_repaired.sqlite`.
- Evidence source:
  The candidate used the repaired local NSoud BM25 sidecar in read-only SQLite mode. Qdrant lookup was not needed and Qdrant was not contacted for this candidate evaluation.
- Candidate metrics:
  Before (`nsoud_dataset_repaired`): `gold=4`, `direct=0`, `partial=3`, `gap=1`, `boilerplate_noise=0`, `usable_support_rate_gold=0.75`, `citation_available_rate=0.75`, `unsupported_answer_risk_count=1`, `strict_direct_pass_rate_gold=0.0`.
  After (`nsoud_evidence_window_candidate`): `gold=4`, `direct=3`, `partial=1`, `gap=0`, `boilerplate_noise=0`, `usable_support_rate_gold=1.0`, `citation_available_rate=1.0`, `unsupported_answer_risk_count=0`, `strict_direct_pass_rate_gold=0.75`, `evidence_window_used_count=4`, `evidence_window_failed_count=0`, `evidence_window_truncated_count=1`, `same_document_neighbor_count=8`.
- `nsoud-qa-010` result:
  Anchor chunk `1644` remained rank `4`; chunks `1643`, `1644`, and `1645` were included from the same document. Combined evidence length was `3952`. The relevant doctrine became visible, support changed from `gap` to `partial`, citation became available, and unsupported risk cleared. This confirms exported snippet truncation rather than retrieval ranking as the issue.
- `nsoud-qa-003` result:
  Original keyword coverage was `2/3 = 0.6667`; evidence-window coverage became `3/3 = 1.0`, and the item became `direct`. The strict threshold and morphology rules were not changed. The evidence window for this item was truncated at the configured `6000` characters and reports that truncation explicitly.
- Tests run:
  `python -m pytest tests/rag/test_legal_evidence_window.py -q` -> `20 passed`.
  `python -m pytest tests/rag/test_legal_answer_eval.py -q` -> `22 passed`.
  `python -m pytest tests/rag/test_legal_answer_eval_diagnostics.py -q` -> `2 passed`.
  `python -m pytest tests/observability/test_eval_metrics_exporter.py -q` -> `10 passed`.
  `python -m pytest tests/test_nalus_task_validator.py -q` -> `11 passed`.
  `python -m pytest tests/rag/test_legal_evidence_window.py tests/rag/test_legal_answer_eval.py tests/rag/test_legal_answer_eval_diagnostics.py tests/observability/test_eval_metrics_exporter.py tests/test_nalus_task_validator.py -q` -> `65 passed`.
  Repeated `pytest-asyncio` loop-scope deprecation warning is non-blocking and unrelated to this task.
- Monitoring verification:
  Restarted `nalus-eval-metrics-exporter`. `curl.exe -s http://localhost:9108/metrics | Select-String 'run_name="nsoud_evidence_window_candidate"'` exposed the expected existing bounded metrics for the new run: `gold=4`, `direct=3`, `partial=1`, `gap=0`, `unsupported=0`, `strict_direct_pass_rate_gold=0.75`, `usable_support_rate_gold=1.0`, and `citation_available_rate=1.0`.
- Validator:
  Exact validator command without allowlist returned `WARN` for two intentional `bm25_change` findings because the evidence-window evaluator reads the local BM25 sidecar as an evidence source. The follow-up validator run with `--allow-risk bm25_change` returned `PASS` with zero findings. No BM25 scoring changed.
- Behavior preserved:
  Retrieval ranking, retrieved hit order, global `top_k`, dense scoring, BM25 scoring, RRF, BGE-M3, embedding dimensions, Qdrant collections/aliases/data, Redis/cache behavior, Grafana queries, strict-direct threshold, and LLM/DeepSeek behavior were not changed.
- Known limitations:
  Evidence windows improve evaluator visibility only and do not change retrieval ranking. `nsoud-qa-010` remains `partial` because the verified gold hit is rank `4`, and the strict-direct definition still requires rank `1`.
- Next recommended task:
  Validate evidence-window mode across ÚS and Mixed before deciding whether it should become the default no-LLM answer-eval behavior.

## 2026-07-12 22:23 Europe/Moscow — Task: Cross-corpus evidence-window validation

- Goal:
  Validate deterministic evidence-window evaluation across ÚS and Mixed corpora before considering any default behavior change, while keeping evidence windows opt-in.
- What changed:
  Extended `app/rag/eval/evidence_window.py` so the read-only BM25 sidecar evidence loader supports both known sidecar schemas: NSoud with explicit `ecli` and ÚS without `ecli` but with `document_id` / `source_document_id`.
  Fixed `evidence_window_failed_count` so corpus-only skips (`provenance_valid=None`) are not counted as failed evidence windows.
  Added focused regression tests for sidecars without `ecli` and Mixed corpus-only skip behavior.
  Created `usoud_evidence_window_candidate` and `mixed_evidence_window_candidate` answer-eval artifact directories.
  Added `artifacts/evaluation_quality/cross_corpus_evidence_window_validation_20260712.md` and `.json`.
- Candidate runs:
  ÚS: `artifacts/rag_eval/legal_qa/answer_eval/usoud_evidence_window_candidate/`.
  Mixed: `artifacts/rag_eval/legal_qa/answer_eval/mixed_evidence_window_candidate/`.
- Evidence sources:
  ÚS used `storage/rag/bm25/nalus_us_bge_m3_rag_combined_20260709.sqlite` in SQLite read-only mode.
  Mixed used no document evidence source because all gold items are corpus-only and evidence windows are skipped by design.
- ÚS before/after:
  Baseline `usoud_no_llm_baseline`: `gold=10`, `direct=1`, `partial=9`, `gap=0`, `boilerplate=0`, `usable_support_rate_gold=1.0`, `citation_available_rate=1.0`, `unsupported_answer_risk_count=0`, `strict_direct_pass_rate_gold=0.1`.
  Candidate `usoud_evidence_window_candidate`: `gold=10`, `direct=7`, `partial=3`, `gap=0`, `boilerplate=0`, `usable_support_rate_gold=1.0`, `citation_available_rate=1.0`, `unsupported_answer_risk_count=0`, `strict_direct_pass_rate_gold=0.7`, `evidence_window_used_count=10`, `evidence_window_failed_count=0`, `evidence_window_truncated_count=0`, `same_document_neighbor_count=20`.
- Mixed before/after:
  Baseline `mixed_no_llm_baseline`: `gold=8`, `corpus_only_count=8`, `usable_support_rate_gold=1.0`, `citation_available_rate=0.0`, `corpus_routing_support_rate=1.0`, `unsupported_answer_risk_count=0`.
  Candidate `mixed_evidence_window_candidate`: `gold=8`, `corpus_only_count=8`, `usable_support_rate_gold=1.0`, `citation_available_rate=0.0`, `corpus_routing_support_rate=1.0`, `unsupported_answer_risk_count=0`, `evidence_window_used_count=0`, `evidence_window_failed_count=0`, `evidence_window_truncated_count=0`.
- NSoud reference:
  `nsoud_evidence_window_candidate` remains green as the reference document-gold candidate: `gold=4`, `direct=3`, `partial=1`, `gap=0`, `usable_support_rate_gold=1.0`, `citation_available_rate=1.0`, `unsupported_answer_risk_count=0`, `strict_direct_pass_rate_gold=0.75`.
- Safety verification:
  ÚS per-row validation found no cross-document mismatch, no invalid evidence windows, and no fabricated citations.
  Mixed per-row validation found no valid or failed document evidence windows, no corpus-only citation, and no corpus-only row with evidence-window chunks.
- Tests run:
  `python -m pytest tests/rag/test_legal_evidence_window.py -q` -> `22 passed`.
  `python -m pytest tests/rag/test_legal_answer_eval.py -q` -> `22 passed`.
  `python -m pytest tests/rag/test_legal_answer_eval_diagnostics.py -q` -> `2 passed`.
  `python -m pytest tests/observability/test_eval_metrics_exporter.py -q` -> `10 passed`.
  `python -m pytest tests/test_nalus_task_validator.py -q` -> `11 passed`.
  `python -m pytest tests/rag/test_legal_evidence_window.py tests/rag/test_legal_answer_eval.py tests/rag/test_legal_answer_eval_diagnostics.py tests/observability/test_eval_metrics_exporter.py tests/test_nalus_task_validator.py -q` -> `67 passed`.
  Repeated `pytest-asyncio` loop-scope deprecation warning is non-blocking and unrelated.
- Monitoring verification:
  Restarted `nalus-eval-metrics-exporter`. The new `usoud_evidence_window_candidate` and `mixed_evidence_window_candidate` runs were visible at `http://localhost:9108/metrics` through existing `legal_answer_eval_*` metrics and bounded labels `(run_name, corpus)`. No Grafana query changed.
- Validator:
  Exact validator command without allowlist returned `WARN` for the intentional `bm25_change` sidecar-read diff. The validator run with `--allow-risk bm25_change` returned `PASS` with zero findings. No BM25 scoring changed.
- Default-mode recommendation:
  Keep evidence windows opt-in for now. Future default activation is recommended only for document-gold no-LLM answer evaluation, not globally and not for Mixed corpus-only routing evaluation.
- Known limitations:
  This validates offline no-LLM answer-eval artifacts only. It does not change or validate live generation behavior.
- Next recommended task:
  Prepare a separate default-policy task that enables evidence windows only for document-gold no-LLM evaluation, with corpus-only skip behavior explicitly documented and tested.

## 2026-07-12 23:32 Europe/Moscow — Task: Document-gold evidence-window default policy

- Goal:
  Make deterministic same-document evidence windows the default only for offline no-LLM document-gold legal answer evaluation, while keeping corpus-only routing, live runtime retrieval, LLM generation, retrieval benchmarks, model behavior, Qdrant, Redis, scoring, thresholds, and Grafana queries unchanged.
- What changed:
  Added an explicit typed evidence-window policy layer in `app/rag/eval/evidence_window.py` with `off`, `document_gold`, and `explicit_all` behavior.
  Updated `app/rag/eval/legal_answer_eval.py` so policy decisions are recorded per result with configured/effective policy, activation reason, skip reason, document-gold presence, default activation, explicit activation, and aggregate counters.
  Updated `scripts/run_legal_answer_eval.py` so the no-LLM CLI defaults to `document_gold`, preserves existing `--evidence-window`, adds `--evidence-window-policy off|document-gold|all`, adds `--no-evidence-window`, and rejects conflicting combinations.
  Updated `scripts/generate_legal_answer_eval_diagnostics.py` so diagnostics replay the recorded evidence-window policy.
  Added regression coverage for default activation, corpus-only skip, explicit off, explicit enable, LLM-mode skip, missing provenance safety, CLI conflicts, default policy mapping, counters, threshold preservation, and retrieval immutability.
  Created local candidate output directories `usoud_document_gold_default`, `nsoud_document_gold_default`, and `mixed_document_gold_default`.
  Added `artifacts/evaluation_quality/document_gold_evidence_window_policy_20260712.md` and `.json`.
- Policy behavior:
  `document_gold` activates only when `no_llm=true`, gold is available, the item is not corpus-only, and a document gold id is present. Invalid provenance still fails safely at construction time as `missing_or_invalid_provenance`; no neighboring chunks are guessed.
  Corpus-only gold is skipped with `corpus_only_gold`, citation remains unavailable by design, and the skip is not counted as an evidence-window failure.
  LLM-mode evaluation does not silently activate the document-gold default; explicit policy is required.
- Candidate runs:
  ÚS `usoud_document_gold_default`: `gold=10`, `direct=7`, `partial=3`, `usable_support_rate_gold=1.0`, `citation_available_rate_gold=1.0`, `unsupported_answer_risk_count=0`, `strict_direct_pass_rate_gold=0.7`, `evidence_window_used_count=10`, `evidence_window_failed_count=0`, `evidence_window_default_activated_count=10`.
  NSoud `nsoud_document_gold_default`: `gold=4`, `direct=3`, `partial=1`, `usable_support_rate_gold=1.0`, `citation_available_rate_gold=1.0`, `unsupported_answer_risk_count=0`, `strict_direct_pass_rate_gold=0.75`, `evidence_window_used_count=4`, `evidence_window_failed_count=0`, `evidence_window_default_activated_count=4`.
  Mixed `mixed_document_gold_default`: `gold=8`, `corpus_only_count=8`, `usable_support_rate_gold=1.0`, `citation_available_rate_gold=0.0`, `corpus_routing_support_rate=1.0`, `unsupported_answer_risk_count=0`, `evidence_window_used_count=0`, `evidence_window_failed_count=0`, `evidence_window_corpus_only_skipped_count=8`.
- Tests run:
  `python -m pytest tests/rag/test_legal_evidence_window.py -q` -> `28 passed`.
  `python -m pytest tests/rag/test_legal_answer_eval.py -q` -> `24 passed`.
  `python -m pytest tests/rag/test_legal_answer_eval_diagnostics.py -q` -> `2 passed`.
  `python -m pytest tests/observability/test_eval_metrics_exporter.py -q` -> `10 passed`.
  `python -m pytest tests/test_nalus_task_validator.py -q` -> `11 passed`.
  `python -m pytest tests/rag/test_legal_evidence_window.py tests/rag/test_legal_answer_eval.py tests/rag/test_legal_answer_eval_diagnostics.py tests/observability/test_eval_metrics_exporter.py tests/test_nalus_task_validator.py -q` -> `75 passed`.
  Repeated `pytest-asyncio` loop-scope deprecation warning is non-blocking and unrelated.
- Monitoring verification:
  Recreated only `nalus-eval-metrics-exporter`. `http://localhost:9108/metrics` exposed all three new run names through the existing `legal_answer_eval_*` bounded metrics: `usoud_document_gold_default`, `nsoud_document_gold_default`, and `mixed_document_gold_default`.
- Validator:
  Initial exact validator run returned `WARN` only because the three requested candidate run output directories were new unknown dirty files.
  Follow-up validator run with explicit `--allow-candidate-run usoud_document_gold_default --allow-candidate-run nsoud_document_gold_default --allow-candidate-run mixed_document_gold_default` returned `PASS` with zero findings.
- Behavior preserved:
  Retrieval rank/order/scores, top_k, strict thresholds, dense scoring, BM25 scoring, RRF, BGE-M3, embedding dimensions, Qdrant collections/aliases/data, Redis/cache behavior, Grafana queries, and LLM/DeepSeek behavior were not changed.
- Known limitations:
  The new policy affects offline deterministic no-LLM answer evaluation only. Candidate run directories are generated artifacts for local review and are not part of the application runtime.
- Next recommended task:
  Use the new `document_gold` policy for future offline no-LLM legal answer-eval runs, and keep live generation unchanged until a separate runtime evidence policy is explicitly designed and reviewed.

## 2026-07-13 00:18 Europe/Moscow — Task: Add document-level exhaustive retrieval pipeline

- Goal:
  Add a production-grade document-level retrieval path that returns bounded unique court decisions identified from candidate chunks, while preserving the existing chunk-level retrieval path and API compatibility.
- Scope:
  Implemented an additive module and endpoint only. Existing `/api/rag/retrieve`, `/api/rag/query`, hybrid retrieval, dense retrieval, BM25 sidecar scoring, RRF fusion, BGE-M3 embeddings, Qdrant collections, Redis/cache behavior, ingest, LLM behavior, and frontend behavior remain unchanged.
- What changed:
  Added `app/rag/retrieval/document_retrieval.py` with typed configuration, canonical document grouping, duplicate removal, deterministic document scoring, dynamic threshold filtering, best supporting passages, safe document metadata projection, and bounded diagnostics.
  Added `POST /api/rag/retrieve-documents` as an explicit additive endpoint in `app/api/rag_router.py`.
  Added disabled-by-default document retrieval configuration to `.env.example`.
  Added `docs/DOCUMENT_LEVEL_RETRIEVAL.md` describing the pipeline, config, scoring strategy, API response, safety properties, and future extension points.
  Added `tests/rag/test_document_retrieval.py` and expanded `tests/api/test_rag_api.py`.
- Configuration:
  `NALUS_DOCUMENT_RETRIEVAL_ENABLED=0` keeps the new endpoint disabled by default.
  `NALUS_DOCUMENT_MAX_CANDIDATE_CHUNKS`, `NALUS_DOCUMENT_MAX_RETURNED_DOCUMENTS`, `NALUS_DOCUMENT_MAX_SUPPORTING_CHUNKS_PER_DOCUMENT`, `NALUS_DOCUMENT_RELEVANCE_THRESHOLD`, `NALUS_DOCUMENT_SCORING_STRATEGY`, and optional `NALUS_DOCUMENT_LATENCY_BUDGET_MS` centralize document-level retrieval behavior.
- Scoring:
  The first deterministic strategy is `best_plus_average_top_chunks`, combining best chunk score with average top supporting chunk score. The strategy is explicit and can be extended without changing grouping or API contracts.
- API behavior:
  Existing `/api/rag/retrieve` response remains chunk-oriented and unchanged.
  New `/api/rag/retrieve-documents` returns `documents` and `diagnostics`. If the configured threshold filters all documents, the endpoint returns an empty `documents` list with diagnostics and does not silently lower thresholds or fall back to unrelated documents.
- Tests run:
  `python -m pytest tests/rag/test_document_retrieval.py -q` -> `10 passed`.
  `python -m pytest tests/api/test_rag_api.py -q` -> `34 passed`.
  `python -m pytest tests/rag/test_production_bge_m3_profile.py tests/rag/test_retrieval_service.py -q` -> `39 passed`.
  `python -m pytest tests/rag/test_document_retrieval.py tests/api/test_rag_api.py tests/rag/test_production_bge_m3_profile.py tests/rag/test_retrieval_service.py tests/test_nalus_task_validator.py -q` -> `94 passed`.
  Repeated `pytest-asyncio` loop-scope deprecation warning is non-blocking and unrelated.
- Validator:
  Initial validator run failed only because `PROJECT_PROGRESS.md` had not yet been updated; diff-scan warnings matched intentional runtime/API/config terms and existing generated candidate output directories from the previous task.
  Follow-up validator run with explicit allowlist for the intentional `top_k_change`, `logger_change`, `bm25_change`, `rrf_change`, `dense_change`, and existing generated candidate run directories returned `PASS` with zero findings.
- Runtime/API smoke:
  `docker compose ps` showed `api`, `qdrant`, `redis`, `prometheus`, `grafana`, and `nalus-eval-metrics-exporter` running.
  Focused API smoke `python -m pytest tests/api/test_rag_api.py::TestRawRetrieveEndpoint::test_document_retrieve_returns_unique_documents_with_diagnostics tests/api/test_rag_api.py::TestRawRetrieveEndpoint::test_existing_retrieve_response_shape_remains_backward_compatible -q` -> `2 passed`.
- Behavior preserved:
  No ingest, no Qdrant write, no embedding regeneration, no model download, no Redis enablement, no LLM/DeepSeek call, no BM25 scoring change, no RRF change, no default API behavior change, and no hidden threshold fallback.
- Known limitations:
  This first implementation groups and scores already retrieved candidates. It does not yet benchmark document-level recall against legal QA datasets and does not implement document-level reranking or follow-up retrieval.
- Next recommended task:
  Add an offline document-level retrieval benchmark that compares unique-document recall against the existing chunk-level benchmark under controlled candidate pool and threshold settings.

## 2026-07-13 01:52 Europe/Moscow — Task: Offline document-level retrieval benchmark

- Goal:
  Add a production-quality offline benchmark for the additive document-level retrieval pipeline, measuring multi-document recall and diagnostics without changing retrieval, ranking, embeddings, Qdrant, BM25, RRF, Redis, LLM behavior, APIs, or frontend behavior.
- Scope:
  Added a separate benchmark only. Existing legal QA benchmark, answer evaluation, document-level retrieval runtime endpoint, hybrid retrieval, and all production retrieval components remain unchanged.
- What changed:
  Added `app/rag/eval/document_retrieval_benchmark.py` with typed JSONL dataset support for multiple relevant documents per question, deterministic candidate/final recall metrics, precision@K, duplicate rate, zero-result rate, latency metrics, failure classification, and report writing.
  Added `scripts/run_document_retrieval_benchmark.py` as a read-only runner using the existing `build_hybrid_retriever` search function without modifying retrieval behavior.
  Extended `app/observability/eval_metrics_exporter.py` to expose document benchmark summaries through the existing Prometheus exporter and conventions. New metrics use bounded labels only: `run_name` and `corpus`.
  Added `docs/DOCUMENT_LEVEL_RETRIEVAL_BENCHMARK.md` documenting dataset format, metrics, failure categories, reports, runner usage, Prometheus label safety, and extension points.
  Added tests for dataset loading, duplicate gold normalization, candidate recall, final recall, precision, large/multiple gold sets, zero relevant documents, failure categories, report generation, runner config, and exporter metrics.
- Dataset format:
  JSONL items include `id`, `corpus`, `question`, and `relevant_document_ids`. Optional metadata includes `legal_topic` and `difficulty`.
  `relevant_document_ids` supports arbitrary counts. Duplicate identifiers are normalized and deduplicated deterministically.
- Metrics implemented:
  Chunk recall@10/20/50/100, document recall@10/20/50/100, precision@10/20/50/100, candidate pool coverage, unique document coverage, duplicate rate, zero result rate, average retrieved documents, average candidate chunks, average latency, and document aggregation latency.
- Failure diagnostics:
  `relevant_document_never_retrieved`, `relevant_document_removed_by_aggregation`, `relevant_document_removed_by_threshold`, `relevant_document_removed_by_returned_document_limit`, `duplicate_handling_issue`, `metadata_issue`, and `unknown`.
- Reports:
  Writer produces `metrics.json`, `summary.json`, `per_question.jsonl`, `per_question.csv`, and `summary.md`.
  No real benchmark output artifact was generated or committed in this task.
- Observability:
  Reused the existing Prometheus exporter. No second metrics system was added.
  Prometheus labels remain bounded to `run_name` and `corpus`; tests verify raw query text, document ids, and ECLI values are not emitted as labels.
- Tests run:
  `python -m pytest tests/rag/test_document_retrieval_benchmark.py -q` -> `9 passed`.
  `python -m pytest tests/test_run_document_retrieval_benchmark.py -q` -> `2 passed`.
  `python -m pytest tests/observability/test_eval_metrics_exporter.py -q` -> `11 passed`.
  `python -m pytest tests/rag/test_document_retrieval_benchmark.py tests/test_run_document_retrieval_benchmark.py tests/observability/test_eval_metrics_exporter.py tests/rag/test_document_retrieval.py tests/rag/test_legal_qa_benchmark.py -q` -> `51 passed`.
  Repeated `pytest-asyncio` loop-scope deprecation warning is non-blocking and unrelated.
- Validator:
  Initial validator run failed only because `PROJECT_PROGRESS.md` had not yet been updated. Diff-scan warnings were for intentional evaluation terms (`top_k`, BM25/RRF mentions in safety documentation, Redis rejection, and logger calls without raw query logging).
- Behavior preserved:
  Retrieval logic, ranking, embeddings, Qdrant collections/data, BM25 scoring, RRF fusion, Redis behavior, DeepSeek/LLM prompts, API behavior, and frontend behavior were not changed.
- Known limitations:
  This task implements and tests the framework. It does not create a curated multi-document gold dataset and does not run a real corpus benchmark artifact.
- Next recommended task:
  Build a reviewed multi-document gold dataset for ÚS/NSoud and run the new benchmark once the gold set is approved.

## 2026-07-31 23:xx Europe/Moscow — Task: Local Legal Retrieval v2 pilot 600

- Goal:
  Safely free local disk, build an isolated Legal Retrieval v2 pilot over approximately 600 deterministic six-year documents, validate integrity/retrieval, and wire explicit local frontend v2 mode without changing production defaults.
- Disk cleanup:
  Removed Docker build cache with `docker builder prune -a -f` and dangling images with `docker image prune -f`. Preserved `nalus-scraper_qdrant_storage`, `nalus-scraper_huggingface_cache`, source data, model cache, production BM25, and benchmark sidecars.
  Deleted only obsolete Qdrant collections proven unaliased and incompatible with Legal v2 reuse: `nalus_us_bge_m3_mvp_recent_3h_20260709` (4,980 points) and `nalus_us_bge_m3_mvp_5y_20260708` (8,335 points). No BM25 sidecars were deleted.
- Pilot manifest:
  Wrote `artifacts/legal_v2/pilot_600_20260731/pilot_manifest_600.json`, `pilot_manifest_600.md`, and `pilot_document_ids.txt`.
  Selected 600 complete documents in the `2020-07-31` through `2026-07-31` window: 450 Ústavní soud and 150 Nejvyšší soud. Validation found date violations 0, missing/invalid dates 0, incomplete documents 0, duplicate selected documents 0, and unresolvable selected IDs 0.
- Builder changes:
  `LegalV2BuildConfig` now accepts isolated pilot collection names under `nalus_legal_paragraph_chunks_v2_*` with a non-canonical BM25 index id, while retaining protected collection checks. CLI env config now passes Qdrant collection and BM25 index id into the builder. Payload and BM25 provenance now record the configured pilot identities.
- Pilot build:
  Built `nalus_legal_paragraph_chunks_v2_pilot_600` and `/app/storage/rag/bm25/nalus_legal_paragraph_bm25_v2_pilot_600.sqlite` using cached `BAAI/bge-m3`, dimension 1024, CPU-only, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`.
  First run with `document-batch-size=64` was stopped and discarded because a long second document batch created too large a no-checkpoint window. Only the isolated failed pilot collection/BM25/checkpoint were deleted. Successful rerun used `batch-size=128` and `document-batch-size=8`.
  Final build: 600 indexed documents, 13,824 chunks, duration 10,578,114 ms, 1.3068 chunks/s, 0 failed documents, checkpoint removed after success.
- Integrity gate:
  `artifacts/legal_v2/pilot_600_20260731/pilot_integrity_validation.json` passed. Qdrant points 13,824, BM25 rows 13,824, unique pilot documents 600/600, Qdrant/BM25 id mismatch 0, content checksum mismatch 0, date violations 0, metadata identity mismatches 0, duplicate chunk IDs 0, protected aliases/collections/BM25 changes 0.
- Retrieval/runtime gate:
  `pilot_retrieval_gate.json` passed for deterministic pilot retrieval: 14 reviewed/gold pilot QA items, recall@10 0.5714, recall@20 0.7143, smoke candidate failures 0, external LLM calls in corrected gate 0.
  Precision@5 over the single-gold pilot QA subset was 0.1000 (7 top-5 hits over 14 queries). A supplemental deterministic gate covering the requested practical query set passed: 11 queries, clarification correctness 2/2, zero-result correctness 1/1, unverified candidates returned 0, legacy fallback 0, endpoint crashes 0.
  An earlier retrieval-gate attempt accidentally allowed 2 DeepSeek calls during pipeline smoke, and an initial supplemental-gate attempt accidentally allowed 11 DeepSeek query-interpretation calls before being rerun with `NALUS_LEGAL_V2_QUERY_PROVIDER=none`. No external provider was called during indexing, and the corrected gate/supplement made 0 external LLM calls.
  Local API `/docs` and `/health` were reachable after runtime overrides. `POST /api/rag/search-v2` succeeded with pilot provenance under reduced local runtime limits (`candidate_documents=5`, `returned_verified_documents=3`) but returned `no_verified_results` for the tested child-removal query.
- Frontend:
  Added explicit `NALUS_RETRIEVAL_MODE=legacy|v2` support. Committed defaults remain `legacy`. In v2 mode, the Next.js proxy calls `/api/rag/search-v2` without automatic legacy fallback and maps verified documents/evidence only. Added visible pilot notice text and clarification message handling.
  Frontend Docker image was not rebuilt because Docker build would run `npm ci` after cache cleanup and package downloads are prohibited. Local Next dev server was briefly verified on `http://localhost:3017/vyhledavani` with the pilot notice visible; the background job was not durable after the shell ended.
- Validation:
  Backend focused pytest: 95 passed, 1 unrelated Starlette warning. `python -m compileall app scripts tests` passed. `ruff check ... --no-cache` passed. `git diff --check` reported only CRLF warnings. `python -m mypy ...` failed only on missing local `qdrant_client` import stubs for `scripts/legal_v2/live_smoke.py`.
  Frontend `npm run typecheck`, `npm run lint`, and `npm run build` passed.
- Defaults preserved:
  `NALUS_LEGAL_V2_SEARCH_ENABLED` committed default remains disabled. Frontend committed default remains legacy. `search-v2` was not made default. Production alias `nalus_live -> nalus_stable_20260326` and protected production collections/BM25 sidecars were unchanged.
- Remaining limitation:
  This is a pilot-corpus build, not full six-year coverage. Real DeepSeek endpoint verification is slow and returned no verified documents for the tested child-removal query under local pilot limits.
- Next recommended task:
  Before any full 21,776-document build, add a builder-safe per-document or smaller checkpoint strategy as the default for CPU runs, then run a controlled resumed full build into a new isolated non-production collection with the exact six-year manifest and no frontend default change.

## 2026-08-01 02:xx Europe/Moscow — Task: Legal Retrieval v2 universal pilot quality iteration

- Goal:
  Improve the existing immutable 600-document Legal Retrieval v2 pilot as a general-purpose Czech legal judgment retrieval system without rebuilding the pilot, starting the full six-year build, enabling production defaults, using Redis, or calling external providers before a deterministic quality gate passes.
- Scope:
  Runtime changes were limited to general query understanding, document aggregation, evidence-window selection, verifier output normalization, and the `search-v2` response contract. Dense similarity, BM25 scoring, RRF fusion, BGE-M3 model, embedding dimension, Qdrant collections, aliases, production BM25 sidecars, frontend defaults, and `search-v2` disabled-by-default behavior were preserved.
- Immutable pilot snapshot:
  Wrote `artifacts/legal_v2/pilot_600_20260731/universal_quality/pre_task_snapshot.json` and `.md` before tuning, and `post_task_snapshot.json` and `.md` after tuning.
  Pilot Qdrant points remained 13,824; pilot BM25 rows remained 13,824; pilot BM25 checksum remained `85ceb99dfc9bbf682d59628d6efdb861b61ce96dac3e9946583f7eb4f7de816f`; aliases, protected collection counts, and production BM25 checksums were unchanged.
- Reviewed benchmark:
  Wrote `reviewed_benchmark.json` and `.md` under `artifacts/legal_v2/pilot_600_20260731/universal_quality/`.
  The artifact contains 20 intents and 64 queries: 48 gold/evaluable queries, 8 ambiguous queries, 8 zero-result queries, 12 hard-negative pairs, and diagnostic/tuning/holdout split sizes 28/20/16.
  This is broader than the earlier 14-item pilot gate but still below the requested minimum of 36 reviewed legal intents, so it is not sufficient for a final universal-quality claim.
- Failure tracing:
  Wrote representative traces under `artifacts/legal_v2/pilot_600_20260731/universal_quality/traces/` and `failure_classification.json`/`.md`.
  Top failure classes were clarification policy for broad concept-only queries, query interpretation gaps, RRF ranking, chunk-to-document aggregation, and verifier instruction/evidence coverage.
- Changes retained:
  QuerySpec now recognizes general legal-concept synonym groups across family, citizenship, service/deadline, civil/criminal/constitutional procedure, fair-trial/evidence, property, contracts, damages, employment, public administration, jurisdiction, and extraordinary remedies.
  Hybrid retrieval still uses the same dense/BM25/RRF formulas, but document aggregation adds a small bounded evidence-coverage bonus from hard/soft constraint coverage, relation coverage, and multi-passage support.
  Evidence selection now considers both hard and soft constraints when choosing bounded windows.
  Verifier normalization now carries `classification` values (`strongly_relevant`, `materially_relevant`, `related_only`, `incidental_overlap`, `not_relevant`, `insufficient_evidence`) and safe coverage diagnostics; the deterministic final gate remains fail-closed on hard constraints.
  API `search-v2` document results now include `relevance_classification`.
  `scripts/legal_v2/live_smoke.py` uses a narrow mypy ignore for the optional `qdrant_client` runtime dependency.
  Exact diagnostic query strings were removed from `scripts/legal_v2/validate_smoke_index.py` smoke defaults.
- Deterministic metrics:
  Baseline on the reviewed set before tuning: candidate macro precision@5 0.1083, macro recall@10 0.4271, recall@20 0.6042.
  Tuning iteration 1 after retained general changes: candidate macro precision@5 0.2083, candidate macro recall@10 0.6042, candidate recall@20 0.6771, candidate MRR 0.5447.
  Verified returned-document precision@5 was 0.1958 and verified recall@10 was 0.4896. Clarification correctness was 1.0 and zero-result correctness was 1.0.
  The deterministic gate failed because recall@10/20 stayed below threshold and deterministic hard-negative verified leakage was 23. Provider calls were 0 and retrieval errors were 0.
- Regressions/gates:
  Because the deterministic universal gate did not pass, the real HTTP DeepSeek provider gate and frontend v2 validation were not run.
  No full build was started, no pilot/production index was rebuilt or modified, no model/package download was performed, and no GPU/CUDA path was used or configured.
- Validation:
  `python -m compileall app scripts tests` passed.
  `python -m mypy app/api/rag_router.py app/api_app.py app/rag/legal_v2 scripts/legal_v2` passed.
  Focused pytest for Legal v2 QuerySpec/verifier/end-to-end/API passed: 96 tests passed with one unrelated Starlette warning.
  Targeted ruff on changed runtime/test files passed.
  Broad requested ruff command failed on pre-existing unused imports/local variables in unrelated tests.
  `git diff --check` reported CRLF normalization warnings only.
  Project validator failed because of pre-existing dirty/generated artifacts, existing builder/test Qdrant write terms, and diff-scan detection of a DeepSeek reference in the changed verifier file; no DeepSeek call was made in this task.
- Remaining blockers:
  The reviewed benchmark must be expanded to at least 36 reliable legal intents from the immutable pilot or the requirement must be adjusted.
  The deterministic verifier/final gate needs a stronger general evidence relevance model; current token coverage admits related hard negatives as verified.
  Do not run the real HTTP provider gate or frontend v2 validation until the deterministic universal gate passes.
- Next recommended task:
  Expand the reviewed pilot benchmark to the requested intent count, then implement a second general verifier/evidence iteration focused on rejecting related-only hard negatives without lowering recall in family, maintenance, service/deadline, citizenship, and procedural domains.

## 2026-08-01 03:xx Europe/Moscow — Task: Legal Retrieval v2 candidate/semantic architecture correction

- Goal:
  Correct the Legal Retrieval v2 pilot architecture so deterministic retrieval is evaluated as a candidate supplier for an LLM verifier, not as a complete semantic legal judge.
- Benchmark:
  Expanded the canonical reviewed benchmark artifact to v2 under `artifacts/legal_v2/pilot_600_20260731/universal_quality/reviewed_benchmark.json`.
  The benchmark now reports 52 distinct `intent_id` values, 80 total queries, 64 gold/evaluable queries, diagnostic/tuning/holdout sizes 34/25/21, 8 ambiguous queries, 8 zero-result queries, and 94 hard-negative document pairs.
  Added review log artifacts `benchmark_review_log.json` and `.md`; new labels use immutable pilot BM25 evidence only. No documents were indexed or rebuilt.
- Architecture split:
  Added `architecture_responsibility_split.json` and `.md`.
  Stage A deterministic retrieval is responsible for query normalization, explicit constraints, legal synonym expansion, dense/BM25/RRF retrieval, document aggregation, and objective hard-constraint contradictions only.
  Stage B DeepSeek semantic verification is responsible for legal issue matching, factual/procedural similarity, holding support, related-only rejection, evidence selection, and final verified ranking.
- Candidate retrieval:
  Ran `baseline_candidate_metrics.json` over the expanded benchmark with candidate window 60 and provider calls 0.
  Baseline candidate metrics: macro precision@5 0.18125, macro recall@10 0.671875, recall@20 0.765625, MRR 0.53245, gold coverage in candidate window 0.953125, retrieval errors 0.
  Added general QuerySpec concept aliases for right to interpreter, migration/asylum/residence, enforcement, court costs, burden of proof, limitation periods, public-law sanctions, legal standing, child contact, administrative procedure, court competence, tax, validity of legal acts, and procedural default.
  Tuning iteration 1 did not improve the candidate gate: macro recall@10 stayed 0.671875, recall@20 decreased to 0.75, gold coverage@60 stayed 0.953125. Because the approved candidate gate did not pass, no HTTP/DeepSeek/frontend gate was run.
- Semantic verifier implementation:
  Extended verifier classification support to the Stage B classes `exact_match`, `strong_match`, `partial_match`, `related_only`, `contradictory`, and `insufficient_evidence`.
  DeepSeek prompt instructions now require structured semantic fields, treat candidate judgment text as untrusted evidence, and require evidence quotations from supplied windows only.
  Added validation that semantic payloads with required fields fail closed when malformed, when positive results lack evidence, or when evidence quotes are not present in supplied candidate text.
  Added prompt-injection resistance coverage where instruction-like candidate text is treated only as evidence.
  Wrote `deepseek_prompt_and_schema.json` and `.md`. HTTP/semantic/frontend artifacts were written with explicit `blocked` status because the candidate gate failed.
- Validation:
  Focused Legal v2/API pytest passed: 100 tests passed with one unrelated Starlette warning.
  `python -m mypy app/api/rag_router.py app/api_app.py app/rag/legal_v2 scripts/legal_v2` passed.
  Targeted ruff on changed Legal v2/API/test files passed.
  `python -m compileall app scripts tests` passed.
  Benchmark leakage scan over `app` and `scripts` found no exact diagnostic benchmark query/ECLI matches.
  `git diff --check` reported CRLF normalization warnings only.
- Data safety:
  Added `candidate_semantic_pre_task_snapshot.*` and `candidate_semantic_post_task_snapshot.*`, both tied to the last verified immutable snapshot. Pilot points remained 13,824 and pilot BM25 rows remained 13,824; no production Qdrant, alias, or production BM25 change was performed.
- Remaining blockers:
  Candidate retrieval gate is still false under the existing recall@10/recall@20 thresholds, despite high coverage in a wider LLM candidate window. DeepSeek HTTP validation and frontend validation remain blocked by gate order.
- Next recommended task:
  Investigate the three true candidate-window misses and the low constitutional/criminal/extraordinary-remedy recall cases with full trace comparison, then decide whether to implement a general second-stage candidate expansion into the LLM window or correct proven annotation errors before any real DeepSeek gate.

## 2026-08-01 04:xx Europe/Moscow — Task: Correct Stage A gate and run bounded DeepSeek semantic smoke

- Goal:
  Apply the corrected two-stage Legal Retrieval v2 quality gates: treat Stage A as a broad candidate supplier evaluated by coverage/recall, preserve valid benchmark/verifier work, revert only the regressive candidate-ranking effect from iteration 1, and enter real DeepSeek semantic smoke only after Stage A and local Stage B contract checks pass.
- Starting state:
  Backend branch `main`, HEAD `e0396d4ef08d9525c05d2fac8110698435f30aa1`. Frontend branch `main`, HEAD `9da811aca1d4f086d31f7e02e180777be296b043`. Both repositories had pre-existing dirty work; unrelated frontend files were inspected but not changed. No commit or push was performed.
- Change classification:
  Wrote `current_change_classification.json` and `.md` under `artifacts/legal_v2/pilot_600_20260731/universal_quality/`. Valid work was preserved: benchmark v2, architecture split, QuerySpec concepts, semantic verifier classes/schema validation, evidence quote validation, prompt-injection protection, and frontend explicit v2 mode. Generated artifacts remain local and should not be committed.
- Candidate iteration 1 rollback:
  The regressive iteration was identified as newer QuerySpec concept aliases affecting Stage A ranking. The concepts remain in structured `QuerySpec` for Stage B metadata, but only the baseline candidate-retrieval concept set is allowed to add Stage A hard constraints and retrieval-query expansions.
- Stage A rerun:
  Added eval-only `scripts/legal_v2/evaluate_stage_a_candidate_gate.py` and ran it inside the API container against immutable pilot resources `nalus_legal_paragraph_chunks_v2_pilot_600` and `nalus_legal_paragraph_bm25_v2_pilot_600`.
  Result: Stage A gate passed under the corrected criteria. Current rerun metrics were precision@5 `0.190625`, recall@10 `0.6875`, recall@20 `0.7578125`, MRR `0.51707`, coverage@60 `0.953125`, retrieval errors `0`, zero-candidate gold queries `0`, wrong index identity `0`, runtime benchmark leakage `0`, query-specific production rules `0`, and endpoint-independent retrieval crashes `0`.
  The original baseline was not reproduced exactly: precision@5 and recall@10 improved, while recall@20 and MRR were lower than the earlier artifact. This was reported in `stage_a_baseline_comparison.json` and `.md`; Stage A still qualifies only as a candidate pool for Stage B, not final ranking.
- Stage B local contract:
  Deterministic QuerySpec/verifier/API tests passed. Provider payload validation remains fail-closed for malformed output, incomplete output, fabricated evidence quotes, missing positive evidence, and prompt-injection-like candidate text.
- Bounded DeepSeek smoke:
  Added eval-only `scripts/legal_v2/evaluate_deepseek_semantic_smoke.py` and ran the intended configured DeepSeek QuerySpec provider and semantic verifier on a bounded diagnostic/tuning smoke. No prompts, raw provider responses, or secrets were printed or written.
  Across the three bounded diagnostic smoke attempts in this phase, DeepSeek made 12 provider calls total. The final smoke attempt stopped fail-closed after 4 provider calls: 2 QuerySpec calls and 2 semantic verifier calls. It processed 2 of 16 planned smoke rows, then stopped for repeated structural failures. Failure classes were `query_interpreter_invalid_json` and semantic verifier invalid JSON/schema failures. Wrong index identity remained `0`; verified hard-negative leakage was `0` before stop.
- General tuning attempts:
  Iteration 1: targeted rollback of newer QuerySpec concepts from Stage A ranking inputs while preserving them for Stage B metadata.
  Iteration 2: added safe extraction of the first balanced JSON object from provider text envelopes in both QuerySpec and verifier parsing.
  Iteration 3: tightened QuerySpec and verifier prompts to require exactly one JSON object and string enum values for semantic similarity fields.
  After these general fixes, the bounded DeepSeek smoke still failed structurally, so no further provider runs were started.
- Gates not run:
  Full 64-query semantic evaluation was not run. Real HTTP `POST /api/rag/search-v2` gate was not run. Frontend validation on port `3017` was not run. Holdout answers were not used for tuning.
- Validation:
  `python -m pytest -q tests\rag\test_legal_v2_query_spec.py` -> 9 passed.
  `python -m pytest -q tests\rag\test_legal_v2_verifier.py tests\rag\test_legal_v2_end_to_end.py tests\api\test_rag_api.py` -> 92 passed with one unrelated Starlette/httpx deprecation warning.
  `python -m pytest -q tests\rag\test_legal_v2_query_spec.py tests\rag\test_legal_v2_verifier.py` -> 25 passed.
  Targeted Ruff and Mypy for changed QuerySpec/verifier/eval runner files passed. Targeted compileall for changed files passed.
- Safety:
  No rebuild/reindex was performed. Pilot Qdrant points and BM25 rows were not modified. Production Qdrant, aliases, and production BM25 were not modified. BGE-M3 remained CPU-only from the existing cache. No model/package download, GPU, CUDA, Redis dependency, legacy fallback, commit, or push was introduced.
- Remaining blocker:
  Stage B cannot proceed until the DeepSeek provider path reliably returns valid structured JSON for both QuerySpec and semantic verification under the current model/configuration. The next step is a safe redacted provider response-shape diagnostic or provider adapter fix; do not run the 64-query semantic evaluation, HTTP gate, or frontend validation until the bounded smoke passes structurally.

## 2026-08-01 05:xx Europe/Moscow — Task: DeepSeek Legal v2 adapter structured-output diagnosis

- Goal:
  Narrow the Stage B blocker to the concrete Legal v2 QuerySpec and semantic-verifier provider calls without spending more calls on authentication, transport availability, or generic JSON-mode support.
- Security cleanup:
  Sanitized root `.env.example` so it has one `LLM_API_KEY=your-api-key-here` placeholder and no duplicate key-like value. Git history search found the removed key-like value in 7 commits including current `HEAD`, so key rotation is mandatory before treating the key as safe. The local `.env` runtime secret was not committed by this task and was not intentionally printed in reports.
- Response-shape diagnostic:
  Added `scripts/legal_v2/diagnose_deepseek_adapter.py` and wrote redacted artifacts under `artifacts/legal_v2/pilot_600_20260731/universal_quality/deepseek_adapter_fix/`. The diagnostic made exactly 2 provider calls: 1 QuerySpec and 1 semantic verifier. It recorded envelope keys, `finish_reason`, content length/hash, extraction method, token usage, truncation indicators, parse status, and schema errors without prompts, complete provider output, authorization headers, or secrets.
- Failure classification:
  QuerySpec diagnostic returned HTTP 200 with non-empty `message.content`, direct JSON parse success, and schema enum mismatch: provider intent `legal_information_retrieval` was outside local `QueryIntent`. Semantic verifier diagnostic returned HTTP 200 with `finish_reason=length`, empty `message.content`, non-empty `reasoning_content`, and no JSON object in extracted content. This distinguishes QuerySpec schema drift from verifier output truncation/empty content.
- Adapter and parser changes:
  Added shared `app/rag/legal_v2/structured_output.py` for strict JSON object extraction with diagnostics. Interpreter and verifier now use the shared extractor. DeepSeek text adapter now classifies empty `message.content`, tool-call-without-content, and refusal instead of handing empty content to JSON parsing. QuerySpec accepts known intent aliases while preserving fail-closed behavior for unknown values. QuerySpec and verifier Legal v2 structured-output providers use a 6000-token floor to avoid DeepSeek v4 reasoning consuming the whole 2400-token budget. Verifier normalization safely accepts bare evidence quote strings only after the existing supplied-evidence validation can prove the quote was in the provided evidence.
- Structural gate:
  First 2-query structural gate after the initial limit fix had QuerySpec 2/2 success but verifier 0/2 success: one timeout and one evidence-passage schema mismatch. After verifier prompt/normalization refinement, the final gate was run with `LLM_RETRY=0` to prevent a third transport attempt per operation. QuerySpec remained 2/2 structurally successful; verifier remained 0/2 because both verifier calls timed out at the configured 30-second timeout. Total counted LLM operations in the final gate were 2 QuerySpec and 2 verifier, cache hits 0, wrong index identity 0, secrets logged 0.
- Gates not run:
  The bounded 16-query semantic smoke was not run because the 4-operation structural gate did not pass. Full 64-query semantic evaluation, HTTP gate, and frontend validation were not run.
- Validation:
  Focused deterministic tests for structured output, DeepSeek text adapter, QuerySpec, and verifier passed locally. Mypy and Ruff for the touched adapter/Legal v2 files passed before final repository-wide validation.
- Safety:
  No retrieval ranking, dense scoring, BM25 scoring, RRF, embeddings, Qdrant collections, BM25 rows, aliases, production resources, Redis behavior, frontend, model downloads, GPU, CUDA, commit, or push changed in this task.
- Remaining blocker:
  Semantic verifier requests remain too slow for the configured 30-second provider timeout on real pilot evidence windows. The next step is to reduce verifier prompt/evidence payload size or introduce an explicit reviewed Legal v2 verifier timeout policy, then rerun the 2 QuerySpec + 2 verifier structural gate before any 16-query smoke.

## 2026-08-01 05:xx Europe/Moscow — Task: Compact Legal v2 semantic verifier payload

- Goal:
  Fix the remaining DeepSeek semantic-verifier timeout/missing-final-content blocker by making the verifier a short classifier for one candidate judgment instead of a long legal-analysis generator.
- Preflight:
  Backend stayed on branch `main`, HEAD `e0396d4ef08d9525c05d2fac8110698435f30aa1`. The worktree was already dirty from prior Legal v2 phases; unrelated changes were not reverted. Stage A candidate retrieval, dense retrieval, BM25, RRF, embeddings, Qdrant, BM25 sidecars, production resources, aliases, frontend, Redis, GPU/CUDA, and downloads were out of scope.
- Payload measurement:
  Wrote `payload_measurement.json` and `.md` under `artifacts/legal_v2/pilot_600_20260731/universal_quality/verifier_latency_fix/`. The original verifier prompt for representative `uq_001` was 17,191 characters with 5 evidence windows and 7,000 evidence characters; it required copied evidence text in the provider output and used the prior 6,000-token verifier floor. The corrected prompt is 6,493 characters with 3 bounded evidence windows, a 1,024-token verifier output budget, and evidence-ID output only. Token estimates are character based; no tokenizer was downloaded.
- Verifier redesign:
  The verifier prompt now supplies compact query fields, up to 12 concept IDs, and up to 3 request-local evidence IDs. The model returns `document_id`, `classification`, `confidence`, `supported_concept_ids`, `missing_concept_ids`, `contradiction_ids`, `evidence_ids`, and `reason_code` only. It must not copy evidence, produce markdown, or write a legal memorandum.
  Application validation expands compact payloads back into the existing internal verifier contract. Unknown evidence IDs, duplicate evidence IDs, unknown concept IDs, positive classifications without evidence IDs, contradiction without evidence IDs, and too-long reason codes fail closed. Evidence IDs are mapped only to supplied same-operation evidence windows, and matching evidence for a supported concept must belong to that constraint.
- Output budget and reasoning mode:
  The 6,000-token verifier floor was removed. The verifier now uses `NALUS_LEGAL_V2_VERIFIER_MAX_TOKENS` when set, otherwise a verifier-specific 1,024-token budget bounded to 256..2048. QuerySpec keeps its already-working configuration. Local code and existing diagnostics show only JSON object response-format support; no documented direct/non-reasoning parameter is available in the current adapter, so no speculative parameter was sent.
- One-case real diagnostic:
  Ran one real verifier operation on a reviewed diagnostic case with `LLM_RETRY=0`, compact evidence-ID prompt, one candidate judgment, and the 30-second timeout policy. It did not timeout; latency was about 11.3 seconds. It still failed closed with `verifier_provider_error:empty_message_content:none:unknown`, meaning DeepSeek returned HTTP 200 but no usable final `message.content`. Because the failure was not a timeout, the 60-second timeout diagnostic was not run. QuerySpec provider calls for this diagnostic were 0.
- Gates not run:
  The 2+2 structural gate was not rerun after the one-case diagnostic because Phase 11 did not produce valid final JSON. The bounded 16-query smoke, full 64-query semantic evaluation, HTTP gate, and frontend validation were not run.
- Security:
  `.env.example` remained sanitized with one placeholder `LLM_API_KEY`. The previously exposed key-like value is still present in 7 historical commits, so key rotation remains mandatory before any safe push. No old key value, prompts, complete provider outputs, complete judgment text, authorization headers, or reasoning content were intentionally logged.
- Validation:
  Focused deterministic tests for verifier compact evidence IDs, structured output, DeepSeek text adapter, and QuerySpec passed locally before runtime diagnostics. Final broader validation is recorded in the task artifacts and final report.
- Remaining blocker:
  Even after compacting the verifier prompt and reducing output budget, the current `deepseek-v4-flash` call can return empty final `message.content`. The next step is either a documented provider/model mode that emits direct final content for classification, a different configured non-reasoning model/provider, or an even smaller two-step deterministic classifier design; do not run 2+2, 16-query smoke, full 64, HTTP, or frontend gates until one-case verifier final JSON succeeds.

## 2026-08-01 06:xx Europe/Moscow — Task: DeepSeek V4 Flash non-thinking verifier mode

- Goal:
  Fix the remaining Legal Retrieval v2 semantic-verifier empty final-content failure by using the official DeepSeek V4 Flash non-thinking request parameter for the compact one-candidate verifier classification.
- Request tracing:
  Wrote redacted request-shape artifacts under `artifacts/legal_v2/pilot_600_20260731/universal_quality/verifier_non_thinking_fix/`.
  Before the fix, the compact verifier request used `model=deepseek-v4-flash`, `response_format={"type":"json_object"}`, `max_tokens=1024`, timeout 30 seconds, temperature 0, one message, no tools/stream/reasoning_effort, and omitted `thinking`.
  After the fix, the verifier sends top-level direct-HTTP `thinking={"type":"disabled"}`. SDK-style `extra_body` is not used because the project DeepSeek adapter uses direct HTTP. QuerySpec remains unchanged and still omits `thinking`.
- Runtime changes:
  Added typed `DeepSeekThinkingMode` for `DeepSeekTextLLM` with `PROVIDER_DEFAULT`, `ENABLED`, and `DISABLED`. The provider default still omits the parameter. Only `DeepSeekSemanticVerifierProvider` requests `DISABLED`.
  Added one bounded retry inside the semantic verifier only for `empty_message_content`; the repeated request keeps the same prompt, evidence windows, `thinking.disabled`, 1024 max tokens, and timeout. Repeated empty content still fails closed.
  Fixed the compact evidence-ID expansion so internally generated legacy evidence passages are exact supplied substrings even for long evidence windows. This preserves the evidence-ID design while satisfying the existing quote-grounding validator.
- One-case non-thinking diagnostic:
  The final recorded verifier diagnostic used one reviewed diagnostic query and one candidate judgment with 3 evidence windows, 4,200 evidence-character cap, `max_tokens=1024`, timeout 30 seconds, and `thinking.disabled`.
  Result: HTTP 200, latency 1,566 ms, `finish_reason=stop`, final `message.content` present with 351 characters, `reasoning_content` absent, JSON parse success, schema success, evidence-ID success, classification `partial_match`, retry used false.
  Two additional provider calls occurred while building the diagnostic artifact: one crashed after HTTP 200 due to a local diagnostic field-name bug, and one exposed the long-evidence expansion incompatibility before the fix. No prompts, complete provider outputs, reasoning content, or secrets were logged.
- 2+2 structural gate:
  Ran the bounded gate with `LLM_RETRY=0`, query limit 2, one verifier candidate per query, and verifier timeout 30 seconds.
  Result: gate did not pass. QuerySpec was 1/2 because the first QuerySpec call timed out; the second QuerySpec call succeeded. Semantic verifier was 1/1 reached and passed structurally with classification `strong_match`, final content success, schema success, evidence-ID success, timeout 0 for the reached verifier call, wrong index identity 0, secrets logged 0.
  Because the required 2 QuerySpec + 2 verifier structural success was not met, the 16-query smoke was not run.
- Gates not run:
  The 16-query smoke, full 64-query semantic evaluation, HTTP gate, and frontend validation were not run.
- Validation:
  `python -m compileall app scripts tests` passed.
  Focused pytest for DeepSeek provider, Legal v2 QuerySpec, verifier, and structured output passed: 119 tests passed.
  Targeted Ruff on changed provider/Legal v2/test files passed.
  `python -m mypy app/rag/legal_v2 scripts/legal_v2` passed.
  `git diff --check` passed with CRLF warnings only.
  Project validator failed with 41 findings, mainly generated/dirty artifacts, historical `.env.example` diff heuristics, and pre-existing builder/test Qdrant/BM25 safety terms. This is not a validator pass.
- Security:
  `.env.example` has exactly one `LLM_API_KEY=your-api-key-here` placeholder and no tracked working-tree key-like `LLM_API_KEY` hit. Git history still contains the exposed key or stable prefix in 7 commits, so key rotation remains mandatory and safe-to-push remains no until rotation is verified. No key was intentionally printed or committed by this task.
- Immutability:
  Pilot Qdrant points remained 13,824; pilot BM25 rows remained 13,824; pilot BM25 checksum remained `85ceb99dfc9bbf682d59628d6efdb861b61ce96dac3e9946583f7eb4f7de816f`. No production Qdrant, aliases, production BM25, Stage A retrieval, dense retrieval, BM25, RRF, embeddings, frontend, Redis behavior, GPU/CUDA path, model download, or package download changed.
- Remaining blocker:
  The verifier non-thinking path now works on one reviewed case, but the required 2+2 structural gate is blocked by QuerySpec reliability under the current 30-second provider timeout. Do not run the 16-query smoke, full 64, HTTP gate, frontend validation, commit, or push until a 2 QuerySpec + 2 verifier gate passes.

## 2026-08-04 20:xx Europe/Moscow — Task: Legal decision parser generalization v6

- Goal:
  Generalize the deterministic Legal v2 parser from Constitutional Court v5 to a bounded v6 profile covering corrected golden decisions for Constitutional Court, High Court Prague, and High Court Olomouc.
- Worktree:
  Work stayed in `nalus-scraper-parser-fix` on `fix/legal-paragraph-parser`, HEAD `14c1e300c46872640ebebfb84cf6e8d6686dec7b`. No commit, push, merge, stash, reset, clean, or checkout was performed.
- Parser:
  Bumped parser profile to `legal-decision-parser.cz-courts.v6`. Added court-specific deterministic line grouping for Constitutional Court title/Roman heading blocks, High Court Prague opening and nested statutory lists, and High Court Olomouc reasoning-state numbering with nested list/table rejection. Inline citations remain secondary and no golden line uses primary `citation_continuation`.
- Golden audit:
  `scripts/legal_v2/audit_parser_v6.py` generated audit artifacts under `artifacts/legal_v2/parser_v6_audit/`. Final audit status was `pass` for 20 design documents with 0 conservation failures, 0 duplication failures, 0 ordering failures, 0 parser exceptions, 0 primary citation classifications, 542 changed line classes, 66 changed boundaries, and 427 changed block ranges versus the saved v5 snapshot baseline.
- Golden fixtures:
  Constitutional Court exact 54 classes, 53 boundaries, and 46 block ranges passed. High Court Prague exact 57 classes, 56 boundaries, and 39 block ranges passed. High Court Olomouc validated the complete 698-line fixture, exact 74 top-level reasoning starts, paragraph sequence 1..74, false starts 182 and 296-301 rejected, nested/table rows recognized, and text conservation preserved.
- Review snapshot:
  A temporary v6 parser-derived snapshot rebuild passed for 20 documents, 1407 lines, and 1387 boundaries. The real parser-derived review snapshot was then rebuilt with parser profile v6 while preserving raw line order, manual decisions, and manual history.
- Manual review:
  Valid document-2 v5 decisions were migrated append-only to v6 using `parser_profile_migration`: 13 line decisions and 12 boundary decisions. Document 2 remains complete at 13/13 lines and 12/12 boundaries with 0 unresolved. No pending items were automatically approved and no Assisted Review batch was applied. The validator still reports the expected incomplete corpus state: 19 incomplete documents and 3 pre-existing stale parser-profile decisions.
- Frontend/API:
  Added `/api/parser-v6/changes` and updated the review UI to show `Changed by parser v6` with separate Changed Lines / Classes, Changed Boundaries, and Changed Blocks sections. Lines, Boundaries, Progress, and Assisted Review views remain present.
- Documentation:
  Added `docs/retrieval-enterprise/LEGAL_DECISION_PARSER_V6.md` documenting shared core behavior, court profiles, hierarchy/table handling, audit artifacts, retained v5 behavior, and known limitations.

## 2026-08-05 00:xx Europe/Moscow — Task: Parser review status and UX redesign

- Goal:
  Separate automatic parser validation status from human manual-review status in the local parser-review UI/API so passing parser v6/golden output is not presented as ambiguous `pending` or `unresolved` manual state.
- Status model:
  Added explicit parser-validation statuses `AUTO_VALIDATED_GOLDEN`, `PARSER_VALIDATED`, `PARSER_CHANGED_NEEDS_REVIEW`, `PARSER_CONFLICT`, and `PARSER_UNVALIDATED`, plus manual-review statuses `NOT_MANUALLY_REVIEWED`, `MANUALLY_ACCEPTED`, `MANUALLY_OVERRIDDEN`, `MANUAL_DECISION_STALE`, and `MANUAL_CONFLICT`. Legacy `pending` and `unresolved` remain internal/backward-compatible values only.
- API:
  Added deterministic status derivation from the v6 review manifest, golden spec/checksums, v6 audit artifacts, current parser-derived snapshot, and manual decision store. API responses for documents, lines, boundaries, boundary cards, progress, problems, and parser-v6 change queues now expose separate `parser_validation_*` and `manual_review_*` fields.
- UI:
  Updated navigation, document header, Lines, Boundaries, Changed by parser v6, Problems, and Progress views. Parser validation badges are primary; manual review controls are secondary and collapsed by default. Progress now separates parser validation from manual review.
- Data safety:
  No manual decisions or history were written. Manual store stayed `31002` bytes with SHA-256 `F98CD519CCF28310706F70B0D65F2F15FDFC28CC530304CD4FF79890219A28FB`; history stayed `98322` bytes with SHA-256 `5E0E86E5A2210800A514341E6A7A87210EBC2EC7504D379BA6DB2542EB82FACD`. Document 2 remains manually complete at 13/13 lines and 12/12 boundaries with 0 unresolved.
- Validation:
  Focused visual-review and parser tests passed: 77 tests. HTTP smoke on a temporary loopback port passed for documents, progress, lines, boundary cards, v6 changes, problems, assisted summary, and static assets. Document 11 shows `GOLDEN PASS`; line 36 shows `AUTO-VALIDATED · GOLDEN v6` and `list_or_table`; boundary L35 -> L36 shows `MERGE`; boundary L42 -> L43 shows `SPLIT`.
