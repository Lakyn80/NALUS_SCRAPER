# Legal Retrieval v2

## Long-input SearchBrief (optional pre-retrieval layer)

Disabled by default. See `docs/architecture/LONG_INPUT_SEARCH_BRIEF_V1.md`.

When `NALUS_LEGAL_V2_LONG_INPUT_ENABLED=1`, long pasted legal text is condensed
deterministically to a short `SearchBrief` before `build_query_spec_v2`. Short
queries pass through unchanged. Active provider: extractive (no LLM).

## Stage 1 case-similarity search (production-testable)

Status: **deployed for local Docker pilot testing**. Candidate retrieval only —
no ColBERT, no paid LLM on this path. Cross-Encoder rerank is an **optional
experiment** (`NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED=0` by default): when OFF,
Stage 1 ordering is unchanged; when ON, CE reranks the Stage 1 shortlist only
(`FAST_PLUS_CE_EXPERIMENT`, not a user-facing PRECISE profile yet).

### CE passage coverage experiment (CE-7)

This is a **passage coverage experiment**, not open-ended CE metric tuning.

Why 7 passages: three Stage-1 evidence passages can underrepresent long judicial
decisions. Seven passages allow two strong evidence opportunities from each major
Stage 1 retrieval channel (RRF / dense / BM25) plus one diversity/support slot,
while keeping CE inference bounded (`≤ 30 × 7 = 210` pairs/query).

Selector policy: `diversified_stage1_evidence_v1` (deterministic; no randomness;
does not read golden labels / expected ECLI). CE-3 reference selector remains
`first_n_stage1_order_v1`.

Frozen from CE-3: model `BAAI/bge-reranker-v2-m3`, candidate docs `30`,
document aggregation `max`, `max_length=512`. Only intended quality variable:
passage policy `3 × first_n` → `7 × diversified`.

Pilot golden result (`ce_bge_v2m3_p7_diverse_v1` / `20260808T214850Z`):

```text
Hit@1=0.75  Hit@10=1.0  MRR=0.8375  HN outrank=0.0  failures=0
```

Critical CE-3 misses recovered into TOP 10: `004` (rank 3), `016` (rank 1).
Classification: `PASSAGE_COVERAGE_FIX_CONFIRMED`. Recommendation:
`STOP_AT_7_AND_CONTINUE_CE_ARCHITECTURE`. **CE-10 was intentionally not run**
and requires a separate decision; current evidence does not justify it.

Env knobs (still default OFF for CE):

- `NALUS_LEGAL_V2_CE_PASSAGES_PER_DOCUMENT` (experiment used `7`)
- `NALUS_LEGAL_V2_CE_PASSAGE_SELECTOR=diversified_stage1_evidence_v1`
- `NALUS_LEGAL_V2_CE_EVIDENCE_POOL_LIMIT` (experiment used `40`)

### Pipeline

```text
user query
→ build_query_spec_v2 (deterministic)
→ original + focused retrieval queries
→ BGE-M3 dense retrieval
→ BM25 sparse retrieval
→ RRF fusion
→ chunk-to-document aggregation by ECLI
→ TOP candidate judgments (default 10 shown in UI; API can return up to 50)
→ POST /api/rag/legal-v2/case-similarity/search
→ existing NalusFE `/vyhledavani`
```

Default FE search requests `limit=50` and initially displays the first 10 with
“Načíst další výsledky”. Internal aggregation pool is `NALUS_LEGAL_V2_CANDIDATE_DOCUMENTS=50`.

### Validation (case-similarity golden v1 pilot)

```text
evaluable: 20/20
retrieval_failures: 0
Hit@10: 1.0
hard-negative outrank: 0.0
```

This validates **pilot candidate recall**, not final legal ranking precision.
Exact order inside TOP 10 is provisional until ColBERT / cross-encoder stages.
Optional request field ``retrieval_profile`` selects ``fast`` (default) or
``ce7`` (requires ``NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED=1`` master-allow).
``precise`` is reserved for a later phase.

### Live assets (do not recreate)

| Asset | Value |
| --- | --- |
| Qdrant collection | `nalus_legal_paragraph_chunks_v2_pilot_600` |
| Chunks | 14448 (Cosine, dim 1024) |
| Unique ECLI (resolved) | 622 |
| BM25 index ID | `nalus_legal_paragraph_bm25_v2_pilot_600` |
| BM25 sidecar | `/app/storage/rag/bm25/nalus_legal_paragraph_bm25_v2_pilot_600.sqlite` (14448 rows) |

Identity: every chunk resolves to a valid ECLI via `document_id` (and, where
backfilled, `ecli` / `canonical_document_id`). API responses always expose

`document_id == canonical_document_id == ecli`.

### Endpoints

- Readiness: `GET /api/rag/legal-v2/case-similarity/ready`
- Search: `POST /api/rag/legal-v2/case-similarity/search`
- Full judgment (FE „Celý rozsudek“): `GET /api/rag/documents/{ecli}`
  — when Stage 1 / Legal v2 search flags are on, reconstruction uses the same
  collection as search (`NALUS_LEGAL_V2_QDRANT_COLLECTION`), not the legacy
  `QDRANT_COLLECTION_NAME`. Optional override:
  `NALUS_FULL_DOCUMENT_QDRANT_COLLECTION`.
- Flag: `NALUS_LEGAL_V2_CASE_SIMILARITY_ENABLED=1` (also enabled when
  `NALUS_LEGAL_V2_SEARCH_ENABLED=1`)

Request:

```json
{
  "query": "Hledám rozhodnutí o úpravě styku rodiče s nezletilým dítětem.",
  "limit": 10,
  "include_debug": false
}
```

Response (abbreviated):

```json
{
  "query": "...",
  "result_count": 10,
  "retrieval_stage": "hybrid_rrf_stage_1",
  "results": [
    {
      "rank": 1,
      "document_id": "ECLI:CZ:US:...",
      "canonical_document_id": "ECLI:CZ:US:...",
      "ecli": "ECLI:CZ:US:...",
      "court": "Ústavní soud",
      "case_number": "...",
      "decision_date": "...",
      "document_type": "...",
      "score": 0.0,
      "relevant_passages": [{"text": "...", "chunk_id": "...", "section": "..."}]
    }
  ],
  "diagnostics": {
    "collection": "nalus_legal_paragraph_chunks_v2_pilot_600",
    "bm25_index_id": "nalus_legal_paragraph_bm25_v2_pilot_600",
    "total_latency_ms": 0.0
  }
}
```

`retrieval_stage` is provenance for the **returned ranking**, not configuration intent:

- `hybrid_rrf_stage_1` — Stage 1 BGE-M3 + BM25 + RRF produced the final order
  (FAST profile, or CE not applied)
- `hybrid_rrf_ce7` — Stage 1 candidates were successfully reranked by the validated
  seven-passage Cross-Encoder pipeline (`rerank_applied=true`, 7 passages/document)

CE-7 remains a validated experimental rerank profile; do not treat it as globally
production-proven.

### Local Docker startup (reuse existing Qdrant data)

Do **not** start the worktree `qdrant` service for Stage 1 — use the existing
`nalus-scraper-qdrant-1` data volume.

```powershell
cd C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper-parser-fix
docker stop nalus-scraper-api-1
docker compose -f docker-compose.yml -f docker-compose.stage1.local.yml up -d --build --no-deps api
```

Frontend (sibling repo `NalusFE`, retrieval mode `v2`):

```powershell
cd C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\NalusFE
# .env: NALUS_RETRIEVAL_MODE=v2, NALUS_API_BASE_URL=http://host.docker.internal:8029
docker compose up -d --build frontend
```

| Surface | URL |
| --- | --- |
| Frontend search | http://localhost:3017/vyhledavani |
| Backend API | http://localhost:8029 |
| OpenAPI | http://localhost:8029/docs |
| Health | http://localhost:8029/health |
| Stage 1 readiness | http://localhost:8029/api/rag/legal-v2/case-similarity/ready |

BGE-M3 and BM25 initialize once per API process. With
`NALUS_LEGAL_V2_STAGE1_WARMUP_ON_START=1` (enabled in
`docker-compose.stage1.local.yml`), the API warms them in a background task at
boot. Until warmup finishes, `GET .../case-similarity/ready` returns
`ready=false` and `status=cold|warming` (fields: `model_loaded`, `bm25_loaded`,
`warmup_status`). After warmup, first FE searches are typically under 1s on the
pilot corpus. With the flag off, runtime stays lazy on first search.

### Benchmark

```powershell
.\scripts\legal_v2\evaluate_case_similarity_golden_v1.ps1
```

---

## Verified search-v2 (DeepSeek QuerySpec + verifier)

Status: implemented as an isolated, disabled-by-default pipeline for the
**verified** path (paid DeepSeek). Distinct from Stage 1 candidate search above.

## Runtime Boundary

- New endpoint: `POST /api/rag/search-v2`.
- Feature flag: `NALUS_LEGAL_V2_SEARCH_ENABLED=0`.
- New Qdrant collection: `nalus_legal_paragraph_chunks_v2`.
- New BM25 sidecar: `storage/rag/bm25/nalus_legal_paragraph_bm25_v2.sqlite`.
- Existing `/api/rag/retrieve`, `/api/rag/query`, `/api/rag/retrieve-documents`, `/api/rag/retrieve-verified`, frontend behavior, cache behavior, and production retrieval profile are unchanged.
- Exactly one FastAPI route registers `POST /api/rag/search-v2`; the disabled
  guard behavior lives in the same handler as the live implementation.
- With `NALUS_LEGAL_V2_SEARCH_ENABLED=0`, the endpoint returns a controlled
  disabled response and does not initialize Qdrant, BM25, BGE-M3, DeepSeek
  providers, or credentials.
- With `NALUS_LEGAL_V2_SEARCH_ENABLED=1`, the endpoint initializes the Legal v2
  runtime lazily and uses only the isolated v2 Qdrant collection and v2 BM25
  sidecar. It does not fall back to legacy retrieval or pad results with
  unrelated documents.
- Stage 1 frontend path uses `/api/rag/legal-v2/case-similarity/search`, not
  `search-v2`.

## Pipeline

```text
Original query
-> DeepSeek QuerySpec v2 interpreter
-> deterministic hard-constraint preservation validation
-> BGE-M3 dense retrieval from v2 collection
-> BM25 lexical retrieval from v2 sidecar
-> RRF fusion
-> document aggregation
-> paragraph-aware evidence selection
-> DeepSeek semantic verifier
-> deterministic terminal gate
-> verified documents with paragraph evidence
```

Unit tests use deterministic fake providers and do not call DeepSeek.

## `POST /api/rag/search-v2` Contract

Request:

```json
{
  "query": "mezinárodní únos dítěte matkou do Ruska",
  "sources": ["constitutional"],
  "max_results": 10,
  "debug": false
}
```

Validation:

- `query` must be non-blank and at most 4000 characters.
- `max_results` must be between 1 and 50 and is still bounded by the runtime
  `returned_verified_documents` configuration.
- Extra request fields follow the current Pydantic policy and are ignored.

Response summary:

- `status` and `interpretation_status`;
- safe query-spec summary when available;
- `verified_documents` with document id, score, status, safe metadata, bounded
  evidence quotes, paragraph ids, section types, constraint results, verifier
  reason, and bounded verifier diagnostics;
- `rejected_documents` only when the runtime includes them, normally through
  debug behavior;
- bounded latency, provider, index, and diagnostics maps.

The API response must not expose API keys, authorization headers, raw provider
response bodies, stack traces, raw prompts, internal filesystem paths, complete
judgments, or unbounded diagnostics. Provider, verifier, retrieval, Qdrant, BM25,
or unexpected internal failures fail closed with a controlled `503` response
unless the Legal v2 pipeline returns its existing structured provider-error
result. A successful zero-result search returns an empty `verified_documents`
list with diagnostics; it does not call old retrieval.

Deterministic isolated API smoke without external providers:

```powershell
python -m pytest -q tests\api\test_rag_api.py::TestLegalV2SearchEndpoint
```

This validates API wiring against fake providers and the 20-document smoke-index
contract only. It is not a relevance benchmark and does not prove production
quality over the full corpus.

## Parser Audit

Run the parse-only audit before any embedding or index build:

```powershell
python scripts/legal_v2/audit_corpus.py --output-dir artifacts/legal_v2/parse_audit
```

The audit is read-only. It does not call DeepSeek, create embeddings, write Qdrant, or write BM25. Documents with material invariant failures are excluded from indexing.

## Parser Quality Gate

Create the human-review artifact:

```powershell
python scripts/legal_v2/parser_quality_gate.py --output-dir artifacts/legal_v2/parser_quality_gate
```

The review manifest supports `approved`, `rejected`, and `needs_review`.
The generator selects a bounded representative sample instead of blindly taking
the first N documents. The generated artifact includes explicit review fields
for beginning/end parsing, headings, numbered paragraphs, legal reasoning,
boilerplate, reconstruction, child chunks, parent windows, and cross-document
mixing. Items remain `needs_review` unless the review manifest explicitly marks
them otherwise.

Create a source inventory before a full parse audit:

```powershell
python scripts/legal_v2/source_inventory.py
```

The inventory reports discovered document counts, source files, date coverage,
missing identifiers/text, duplicate source-document identifiers, unreadable
files, and unsupported formats.

### Initial Index QA Policy

Policy version: `legal_v2_initial_index_qa_v1`.

The first isolated Legal Retrieval v2 index build may proceed only when all of
these conditions are true:

1. Full corpus parse audit status is `PASS`.
2. At least 30 representative documents were selected.
3. All selected samples were manually reviewed.
4. Manual review coverage is 100%.
5. Approval rate is 100%.
6. Rejected sample count is 0.
7. Needs-review sample count is 0.
8. Reconstruction failure count is 0.
9. Paragraph/chunk boundary violation count is 0.
10. Duplicate paragraph and chunk ID count is 0.
11. Cross-document mixing count is 0.
12. No unresolved blocking parser or chunking defect exists.
13. Beginning and ending preservation were checked.
14. Legal reasoning and operative parts were checked.
15. Source incompleteness is not hidden.
16. Duplicate source identifiers cannot cause accidental document merging.

Evaluate the deterministic gate without LLMs:

```powershell
python scripts/legal_v2/evaluate_parser_quality_gate.py `
  --output-dir artifacts/legal_v2/parser_quality_gate_20260730 `
  --parse-audit artifacts/legal_v2/parse_audit_full_20260730/legal_v2_parse_audit.json `
  --source-inventory artifacts/legal_v2/source_inventory_20260730.json
```

The gate result is `pass`, `blocked`, or `invalid`. A `pass` permits only an
isolated smoke index build; it does not activate `search-v2`, change aliases,
change frontend behavior, or permit a full production build.

Future relaxation of this strict initial policy requires a reviewed benchmark,
an explicit repository change, focused tests, and a separate approval decision.

### Source Risk Handling

Incomplete source policy:

- Do not mark an incomplete source as complete.
- Do not reconstruct missing content silently.
- Exclude incomplete documents from the initial index unless a source adapter
  provides an explicit safe partial-document policy.
- Record document IDs and exclusion reasons for excluded incomplete documents.
- Expose aggregate incomplete-source counts in the build or gate report.

Duplicate source identifier policy:

- Never merge records solely because source identifiers match.
- Classify duplicate source identifiers as byte/text-identical duplicate,
  metadata-only duplicate, or conflicting content/version.
- Identical records may be deterministically deduplicated.
- Conflicting records must receive distinct stable internal identities or be
  excluded.
- No text from conflicting records may be combined.
- Report all duplicate decisions.

Do not delete original source data during QA, smoke indexing, or source-risk
classification.

## Index Builder

Build only after parser audit and quality review pass:

```powershell
python scripts/legal_v2/build_index.py --overwrite-bm25 --recreate-v2-collection
```

The builder writes only `nalus_legal_paragraph_chunks_v2` or explicitly configured isolated `nalus_legal_paragraph_chunks_v2_*` pilot collections. Non-canonical pilot collections must use a non-canonical BM25 index id. It validates dense/BM25 chunk identity consistency and writes `legal_v2_build_manifest.json`.
The next intended final local index is not the complete historical corpus. Its
scope is the six-year decision-date window `2020-07-31` through `2026-07-31`.
The builder supports an explicit inclusive decision-date range filter. Documents
without a valid decision date are excluded and counted; they are not silently
included.

Count the exact source range before indexing:

```powershell
python scripts/legal_v2/source_inventory.py `
  --decision-date-from 2020-07-31 `
  --decision-date-to 2026-07-31 `
  --json-output artifacts/legal_v2/source_inventory_20260731_6y.json `
  --markdown-output artifacts/legal_v2/source_inventory_20260731_6y.md
```

For large indexing runs, use explicit bounded processing:

```powershell
python scripts/legal_v2/build_index.py `
  --overwrite-bm25 `
  --recreate-v2-collection `
  --decision-date-from 2020-07-31 `
  --decision-date-to 2026-07-31 `
  --batch-size 64 `
  --document-batch-size 128
```

`--batch-size` controls embedding/upsert batch size. `--document-batch-size`
controls parser/chunking batches. These controls do not change the embedding
model, vector dimension, BM25 formula, dense similarity, RRF formula, Qdrant
collection name, aliases, Redis behavior, or provider configuration.

Large builds write `legal_v2_execute_checkpoint.json` in the output directory
after every completed document batch. To verify stop/resume safely, use
`--stop-after-document-batches 1`, then rerun the same command with `--resume`
and without `--recreate-v2-collection`, `--overwrite-bm25`, or the stop flag.

For a bounded smoke index after the QA gate passes, pass an explicit reviewed
parser quality artifact and gate decision:

```powershell
python scripts/legal_v2/build_index.py `
  --parser-quality-artifact artifacts/legal_v2/parser_quality_gate_20260730/parser_quality_gate.json `
  --gate-decision artifacts/legal_v2/parser_quality_gate_20260730/gate_decision.json `
  --limit 20 `
  --output-dir artifacts/legal_v2/smoke_index_20260730 `
  --overwrite-bm25 `
  --recreate-v2-collection
```

The smoke build still writes only the isolated v2 collection and isolated v2
BM25 sidecar. It must not change aliases, production collections, production
BM25 sidecars, frontend behavior, or the disabled-by-default `search-v2`
feature flag.

## Live Smoke

Run only when the v2 index exists and DeepSeek credentials are configured:

```powershell
python scripts/legal_v2/live_smoke.py --query "únos dítěte matkou z Česka do Ruska"
```

The smoke checks the v2 collection, v2 BM25 sidecar, DeepSeek QuerySpec interpretation, hybrid retrieval, evidence selection, DeepSeek final verification, and deterministic gate. Secrets are not printed.

## Local Pilot 600

On 2026-07-31 a local Legal Retrieval v2 pilot was built from a deterministic manifest under `artifacts/legal_v2/pilot_600_20260731/`.

- Corpus label: PILOT CORPUS - APPROXIMATELY 600 DOCUMENTS.
- Decision-date window: `2020-07-31` through `2026-07-31`.
- Source range count before pilot sampling: 21,776 documents.
- Pilot selection: 600 complete documents, 450 Ústavní soud and 150 Nejvyšší soud.
- Pilot Qdrant collection: `nalus_legal_paragraph_chunks_v2_pilot_600`.
- Pilot BM25 sidecar/id: `nalus_legal_paragraph_bm25_v2_pilot_600`.
- Build output: 13,824 chunks, BGE-M3 dimension 1024, CPU-only, offline model cache.
- Build duration: 10,578,114 ms, 1.3068 chunks/s.
- Integrity gate: pass; Qdrant/BM25 ids matched, text checksums matched, date violations 0, duplicate chunk ids 0, protected production changes 0.
- Retrieval gate: pass for the deterministic pilot retrieval gate; 14 reviewed/gold pilot QA items evaluated, precision@5 0.1000, recall@10 0.5714 and recall@20 0.7143. These are pilot-corpus metrics, not full-corpus metrics.
- Practical query supplement: pass for 11 requested pilot queries with clarification correctness 2/2, zero-result correctness 1/1, unverified candidates returned 0, legacy fallback 0, and corrected-run external LLM calls 0.
- Runtime endpoint smoke: `POST /api/rag/search-v2` succeeded with local pilot runtime limits and returned pilot provenance, but the real DeepSeek endpoint smoke returned `no_verified_results` for the tested child-removal query.

## Universal Pilot Quality Iteration

On 2026-08-01 the existing 600-document pilot was evaluated for broader, cross-domain retrieval quality without rebuilding or mutating the pilot index.

- Artifacts:
  `artifacts/legal_v2/pilot_600_20260731/universal_quality/pre_task_snapshot.json`,
  `reviewed_benchmark.json`, `baseline_metrics.json`, trace JSON files,
  `failure_classification.json`, `tuning_iteration_1.json`,
  `validator_final.json`, and `post_task_snapshot.json`.
- Benchmark:
  The reviewed artifact currently has 20 legal intents and 64 queries: 48 evaluable/gold queries, 8 ambiguous queries, 8 zero-result queries, 12 hard-negative pairs, and diagnostic/tuning/holdout split sizes 28/20/16. This does not yet meet the requested 36-intent minimum for a final universal-quality claim.
- General runtime changes:
  QuerySpec gained legal-concept normalization for common Czech legal domains and lay/formal synonyms. Document aggregation adds a bounded constraint/evidence coverage bonus after unchanged dense, BM25, and RRF retrieval. Evidence selection includes hard and soft constraints. Verifier output now supports the six relevance classifications while the deterministic final gate still requires all hard constraints to be proven.
- API contract:
  `search-v2` verified/rejected document objects include `relevance_classification`. Internal raw provider payloads remain redacted by the existing safe-payload filter.
- Deterministic metrics:
  Baseline candidate macro precision@5 was 0.1083, macro recall@10 was 0.4271, and recall@20 was 0.6042.
  After the first retained general iteration, candidate macro precision@5 was 0.2083, candidate macro recall@10 was 0.6042, and candidate recall@20 was 0.6771. Verified returned-document precision@5 was 0.1958 and verified recall@10 was 0.4896. Clarification correctness and zero-result correctness were both 1.0.
- Gate status:
  The universal deterministic pilot gate failed. Blocking failures were recall below threshold and deterministic hard-negative verified leakage 23. Provider calls were 0.
- Execution limits:
  Because the deterministic gate failed, the real DeepSeek HTTP gate and frontend v2 validation were not run. No GPU/CUDA path, model download, package download, Redis requirement, production alias change, production BM25 change, pilot Qdrant write, or pilot BM25 write occurred.
- Immutability:
  Post-task comparison passed: pilot points remained 13,824, pilot BM25 rows remained 13,824, the pilot BM25 checksum was unchanged, protected production collection counts were unchanged, aliases were unchanged, and production BM25 checksums were unchanged.

## Candidate Retrieval and DeepSeek Semantic Split

The follow-up architecture separates deterministic candidate retrieval from DeepSeek semantic legal verification.

- Stage A deterministic retrieval:
  QuerySpec, legal synonym expansion, dense retrieval, BM25 retrieval, RRF fusion, document aggregation, and objective metadata contradiction checks construct the candidate window. Stage A does not attempt to prove full legal relevance through handcrafted semantic coverage.
- Stage B DeepSeek semantic verification:
  DeepSeek classifies candidates as `exact_match`, `strong_match`, `partial_match`, `related_only`, `contradictory`, or `insufficient_evidence`. Only `exact_match`, `strong_match`, and evidence-supported `partial_match` may be returned as verified authorities.
- Evidence validation:
  Semantic payloads are validated against required structured fields. Positive classifications require evidence passages, and every evidence quote must be present in the supplied evidence windows. Candidate judgment text is untrusted evidence, not instructions.
- Expanded benchmark:
  The reviewed benchmark v2 contains 52 distinct `intent_id` values and 80 queries, with diagnostic/tuning/holdout sizes 34/25/21. The added intents cover right to interpreter, migration/asylum/residence, enforcement, court costs, burden of proof, limitation periods, public-law sanctions, legal standing, child contact, administrative procedure, court competence, tax, validity of legal acts, and procedural default.
- Candidate gate result:
  Baseline candidate metrics on benchmark v2 were precision@5 0.18125, recall@10 0.671875, recall@20 0.765625, MRR 0.53245, and gold coverage in candidate window 60 of 0.953125.
  Under the corrected Stage A gate, candidate precision@5 is measured but does
  not block Stage B when coverage@60 >= 0.95, recall@20 >= 0.75, recall@10 >=
  0.65, wrong index identity is 0, runtime benchmark leakage is 0, query-specific
  production rules are 0, and endpoint-independent retrieval crashes are 0.
  The baseline therefore qualifies for Stage B as a candidate supplier, not as
  final user-facing ranking.
- Candidate iteration 1:
  A first general QuerySpec alias iteration was regressive for ranking. The
  newer legal concepts are preserved in structured `QuerySpec`, but they are no
  longer used as Stage A hard constraints or retrieval-query expansions unless
  they are in the baseline candidate-retrieval concept set. A rerun after this
  targeted rollback passed the corrected Stage A gate with precision@5
  0.190625, recall@10 0.6875, recall@20 0.7578125, MRR 0.51707, and
  coverage@60 0.953125. This rerun was not an exact bit-for-bit reproduction of
  the original baseline artifact.
- Stage B smoke:
  A bounded real DeepSeek semantic smoke was run on diagnostic/tuning queries
  only. Across the three bounded diagnostic attempts in this phase it made 12
  provider calls; the final attempt stopped fail-closed after 4 provider calls
  because the QuerySpec and
  semantic verifier responses repeatedly failed the structured JSON contract.
  No raw prompts, raw provider responses, or secrets were logged. Full 64-query
  semantic evaluation, HTTP validation, and frontend validation were not run
  because Stage B did not pass structurally.
- DeepSeek adapter diagnostic:
  A redacted two-call response-shape diagnostic found two distinct structural
  failures. QuerySpec can return syntactically valid JSON but with provider enum
  drift such as `legal_information_retrieval`; the local schema now maps known
  aliases and otherwise fails closed to `unknown`. Semantic verifier calls can
  exhaust the 2400-token Legal v2 budget in DeepSeek `reasoning_content`, leaving
  `message.content` empty with `finish_reason=length`; QuerySpec and verifier
  structured-output calls now use a 6000-token floor and the shared adapter
  classifies empty `message.content` explicitly. JSON extraction is centralized
  for pure JSON, one JSON code fence, small prose envelopes, and one
  unambiguous balanced object.
- Structural gate after adapter fixes:
  Two QuerySpec operations succeeded structurally after the output-limit fix.
  Two semantic-verifier operations still failed closed by provider timeout at
  the configured 30-second timeout, even with retry disabled for the gate run.
  Therefore the 16-query semantic smoke, full 64-query evaluation, HTTP gate,
  and frontend validation remain blocked.
- Verifier compact-classifier iteration:
  The semantic verifier input was changed from a verbose legal-analysis request
  with copied evidence quotes to a compact one-judgment classification request.
  Evidence windows now receive request-local IDs (`E1`, `E2`, ...), concepts
  receive request-local IDs (`C1`, `C2`, ...), and the provider is asked to
  return enum values plus evidence IDs instead of copying quoted passages. The
  application resolves valid evidence IDs back to immutable supplied evidence
  and rejects unknown, duplicate, cross-constraint, or ungrounded positive
  references.
  A representative verifier prompt decreased from 17,191 characters, 5 evidence
  windows, and 7,000 evidence characters to 6,493 characters, 3 bounded evidence
  windows, and a 1,024-token verifier-specific output budget. The previous
  6,000-token verifier floor was not retained.
  The follow-up non-thinking iteration corrected the previous conclusion about
  direct reasoning control. DeepSeek V4 Flash supports an OpenAI-compatible
  top-level request parameter `thinking={"type":"disabled"}`. The project uses a
  direct HTTP adapter, so the semantic verifier sends that top-level parameter;
  SDK-style `extra_body` is not used. QuerySpec keeps provider-default thinking
  behavior because its structural gate had already passed.
  A final recorded one-case verifier diagnostic with non-thinking mode, 1,024
  output tokens, 30-second timeout, and 3 evidence windows returned HTTP 200,
  `finish_reason=stop`, non-empty final `message.content`, no
  `reasoning_content`, valid JSON, valid schema, valid evidence IDs, and
  classification `partial_match`.
  The subsequent 2+2 structural gate did not pass because the first QuerySpec
  call timed out; the second QuerySpec call and the one reached semantic
  verifier call succeeded structurally. Because the required 2 QuerySpec + 2
  verifier success was not met, the 16-query smoke, full 64-query evaluation,
  HTTP gate, and frontend validation remain blocked.

### Fair thinking A/B and hybrid policy (2026-08-01)

The previous QuerySpec diagnostic used a 30-second wall-clock cutoff. That is
not an adequate failure threshold for legal thinking QuerySpec work. A fair A/B
gave both thinking and non-thinking modes the same 120-second diagnostic ceiling
and recorded actual latency without stopping successful thinking work merely
because it exceeded 30 seconds.

A/B design (diagnostic/tuning only; holdout excluded):
- 4 representative legal intents
- QuerySpec: 4 queries × 2 explicit modes = 8 provider calls
- Verifier: 6 reviewed candidates × 2 explicit modes = 12 provider calls
- Identical prompts, schemas, evidence, temperature, and timeout ceiling within
  each comparison; only thinking mode (and its justified output budget) differed
- QuerySpec scoring used production `interpret_query_spec_v2` (repair +
  preservation), not raw `QuerySpecV2.from_dict`

Selected quality-first hybrid policy:
- QuerySpec: thinking enabled, timeout 120s, max_tokens 8000, one bounded retry
  for timeout/empty/invalid structured output (max 2 provider calls)
- Fast verifier: thinking disabled, compact evidence-ID schema, max_tokens 1024,
  production timeout candidate 30s, max 3 evidence windows / 4200 characters
- Thinking verifier fallback: enabled only for difficult candidates
  (`partial_match`, `insufficient_evidence`, contradictions, sparse missing
  concepts, and related close-call cases), timeout 120s, max 2 candidates/query,
  final JSON required in `message.content`, `reasoning_content` never parsed

Structural gate after policy selection: 2/2 thinking QuerySpec, 2/2 fast
verifier, 2/2 thinking fallback, 0 timeouts, 0 evidence-ID failures.
Artifacts live under
`artifacts/legal_v2/pilot_600_20260731/universal_quality/thinking_ab_test/`.

Bounded 16-query hybrid smoke under the selected policy later passed after
compact-verifier fail-closed fixes (`hybrid_smoke_16.json`). The available
non-holdout diagnostic+tuning set has **59** rows (not 64). Full hybrid
evaluation `hybrid_eval_59_nonholdout.*` completed 59/59 after DNS-resilient
resume; that run still failed the strict quality gate on
`hard_constraints_lost` and hard-negative false approvals (infra abort was not
the cause). See `hybrid_eval_59_report.md`.

### QuerySpec merge + verifier gate (quality follow-up)

Behavior notes (Stage A / embeddings / pilot index unchanged):

- QuerySpec interpretation merges deterministic fallbacks (hard constraints,
  origin/destination, negations, mother|father|child roles) before preservation
  validation so provider JSON gaps no longer surface as `hard_constraints_lost`
  when the deterministic builder already has those facts.
- Constraint parse is tolerant: invalid/empty category → `entity`, polarity →
  `hard`; missing `constraint_id` uses a stable SHA-1 id.
- Verifier decisions: only `exact_match` / `strong_match` map to
  `VERIFIED_MATCH`; `partial_match` and weaker relevance classes map to
  `AMBIGUOUS` (pipeline still accepts only `VERIFIED_MATCH`).
- Deterministic gate additionally requires provider `VERIFIED_MATCH`, rejects
  explicit `jurisdiction_match=false`, confidence below `0.6`, and non-empty
  `contradictory_facts`.
- Compact evidence windows: up to 2 per constraint and 12 total; concepts are
  hard-first; evidence text bound defaults to 700 characters for the 1024-token
  verifier budget. Unknown compact concept IDs are dropped; unknown evidence IDs
  still fail closed.

Package layout is modular under `app/rag/legal_v2/` (`query/`, `interpret/`,
`retrieve/`, `evidence/`, `verify/`, `ingest/`) with compatibility shims on the
original module paths.

Post-fix 16-query smoke: `hybrid_smoke_16_quality_fix.*` — gate passed,
QuerySpec schema 100%, interpretation failures 0, false approvals 3.

Environment knobs:

```env
NALUS_LEGAL_V2_QUERYSPEC_TIMEOUT_SECONDS=120
NALUS_LEGAL_V2_VERIFIER_TIMEOUT_SECONDS=30
NALUS_LEGAL_V2_VERIFIER_THINKING_TIMEOUT_SECONDS=120
NALUS_LEGAL_V2_VERIFIER_THINKING_FALLBACK=1
NALUS_LEGAL_V2_VERIFIER_THINKING_FALLBACK_MAX_PER_QUERY=2
NALUS_LEGAL_V2_VERIFIER_MAX_CANDIDATES_PER_QUERY=8
```

Do not choose non-thinking QuerySpec solely because it is faster. Stage A,
embeddings, pilot Qdrant/BM25 points, production indexes, and aliases remain
immutable for this workstream.

Committed defaults remain disabled/legacy. Local pilot activation uses uncommitted runtime overrides only:

```env
NALUS_LEGAL_V2_SEARCH_ENABLED=1
NALUS_LEGAL_V2_QDRANT_COLLECTION=nalus_legal_paragraph_chunks_v2_pilot_600
NALUS_LEGAL_V2_BM25_SIDECAR_PATH=/app/storage/rag/bm25/nalus_legal_paragraph_bm25_v2_pilot_600.sqlite
NALUS_LEGAL_V2_BM25_INDEX_ID=nalus_legal_paragraph_bm25_v2_pilot_600
NALUS_LEGAL_V2_CANDIDATE_DOCUMENTS=5
NALUS_LEGAL_V2_RETURNED_VERIFIED_DOCUMENTS=3
```

Use `document-batch-size=8` for local CPU pilot/resume runs. A first attempt with `document-batch-size=64` had long unsafe intervals between checkpoints and was discarded by deleting only the isolated failed pilot collection and pilot BM25 sidecar.

DeepSeek configuration is read from the runtime environment. Docker Compose loads `.env`
and then applies service `environment` entries; `LLM_MODEL_DEEPSEEK` is now passed
through as `${LLM_MODEL_DEEPSEEK:-deepseek-v4-flash}` so `.env` is not masked by a
hard-coded compose value.

For a bounded provider diagnostic before the full Legal v2 smoke:

```powershell
python scripts/legal_v2/deepseek_smoke.py --mode direct
python scripts/legal_v2/deepseek_smoke.py --mode provider
```

These diagnostics print only safe configuration fields, request shape summaries,
status codes, provider error codes/messages, and short output previews. They do not
print the API key or full prompts.

Legal v2 QuerySpec and verifier calls use `NALUS_LEGAL_V2_LLM_MAX_TOKENS` when
set. If it is unset and generic `LLM_MAX_TOKENS` is lower than 2400, Legal v2 uses
2400 for those structured calls because DeepSeek v4 responses can otherwise spend
the 800-token budget on `reasoning_content` and truncate the final JSON.

## Rollback

Set:

```env
NALUS_LEGAL_V2_SEARCH_ENABLED=0
```

No production index, sidecar, endpoint, cache, or frontend rollback is required because v2 is isolated.

## Known Limitations

- Parser readiness still depends on manual review of the QA artifact.
- Full v2 index build requires local `qdrant_client`, Qdrant, and the offline BGE-M3 model.
- Live semantic control requires configured DeepSeek credentials.
- The v2 endpoint returns no thematic fallback when hard constraints are not proven.
