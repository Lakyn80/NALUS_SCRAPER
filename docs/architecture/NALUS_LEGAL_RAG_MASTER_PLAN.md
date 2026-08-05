# NALUS Legal RAG — Binding Implementation Plan

**Project:** NALUS / Czech legal case law  
**Current parser baseline:** `legal-decision-parser.cz-courts.v7`  
**Parser baseline commit:** `a53bf53c1904585f9bd9f81367971be2b43f3dbb`  
**Purpose of this document:** define the development order, experiments, validation gates, and production architecture so the system is not tuned blindly and every change remains measurable.

---

## 1. Primary system objective

NALUS must do more than retrieve a document containing similar words. For a legal query, it must:

1. retrieve the correct minimal legal passage;
2. provide sufficient surrounding context;
3. distinguish primary from alternatively relevant case law;
4. generate an answer only from retrievable evidence;
5. provide exact citations to the document, block, and source lines;
6. abstain from a confident answer when the evidence is insufficient;
7. satisfy client SLAs for accuracy, latency, and cost.

Target pipeline:

```text
raw judicial decisions
    ↓
deterministic structural parser
    ↓
atomic legal blocks
    ↓
parent–child chunking
    ↓
dense + sparse candidate generation
    ↓
RRF fusion
    ↓
optional ColBERT late-interaction reranking
    ↓
optional cross-encoder reranking
    ↓
deduplication and parent/context expansion
    ↓
LLM answer with claim-level citations and abstention
```

---

## 2. Binding principles

### 2.1 Parser and chunker are separate layers

The parser describes source structure:

- `heading`
- `operative_clause` or a compatible operative block
- `numbered_paragraph_start`
- `numbered_paragraph_continuation`
- `list_or_table`
- `instruction`
- `signature`
- `metadata`
- `layout_noise`

The chunker decides which parser blocks should be indexed together.

A heading may remain a separate parser block while being attached to the following legal paragraph as retrieval context. Parser boundaries must not be changed merely to improve embedding behavior.

### 2.2 The parser is not the main RAG metric

The parser must reliably preserve:

- text;
- order;
- legal units;
- hierarchy;
- stable identities;
- exact citable ranges.

It is not useful to manually perfect every line label if the difference does not affect chunk boundaries, retrieval, context assembly, or citation accuracy.

The final metric is:

```text
query → correct passage → sufficient context → correct answer → verifiable citation
```

### 2.3 Every change is evaluated against a frozen benchmark

During a single experiment, only one major variable may change:

- parser;
- chunking;
- embedding;
- sparse retriever;
- fusion;
- reranker;
- query processing;
- context assembly;
- generation prompt.

Results without a controlled baseline do not count as evidence of improvement.

### 2.4 Two client branches are runtime profiles, not Git branches

The implementation remains shared. Pipeline behavior is selected by configuration:

- `FAST`
- `BALANCED`
- `PRECISE`

Clients may see only:

- **Standard**
- **Maximum accuracy**

Internally, the system should use three profiles and optionally an automatic router.

---

## 3. Current state and the role of parser v7

According to the current report, parser v7:

- processed 20 documents;
- preserved 1,407 lines and 1,387 boundaries;
- created 629 blocks in the full snapshot;
- has zero conservation, duplication, ordering, and parser-exception failures;
- preserved the three exact goldens 05, 11, and 16;
- corrected confirmed structural regressions for the Constitutional Court, High Court Prague, and civil decisions of High Court Olomouc;
- did not modify the manual review store;
- is committed but not pushed.

This means v7 is a suitable **candidate baseline for further RAG work**, not proof that every one of the 20 documents is fully human-verified.

### Immediate gate before further development

Before continuing:

1. independently review the v7 JSON and Markdown exports;
2. confirm exact v7 SHA-256 checksums;
3. confirm the v7 export contains no new systematic errors;
4. mark commit `a53bf53...` as the parser baseline;
5. preserve v6 only as a historical audit baseline;
6. only then decide whether to push or merge.

---

## 4. Two separate golden datasets

## 4.1 Parser golden

The parser golden validates ingestion and structural rules.

Minimum archetypes:

| Court | Decision type | Structural variant |
|---|---|---|
| Constitutional Court | short order | short unnumbered reasoning |
| Constitutional Court | order | Roman sections and numbered paragraphs |
| Constitutional Court | judgment | multiple operative clauses and complex subsections |
| High Court Prague | civil/commercial | multiline opening formula |
| High Court Prague | commercial | citations, lists, and tabular items |
| High Court Olomouc | civil | Roman operative clauses and short numbered reasoning |
| High Court Olomouc | criminal | deep hierarchy, lists, and false numeric starts |

For each archetype, use three roles:

1. **development** — may be used while designing rules;
2. **regression** — automated tests after every change;
3. **locked holdout** — must not be used when implementing rules.

The target minimum is 21 representative documents: 7 archetypes × 3 roles.

### Parser metrics

- text conservation: 100%;
- duplication: 0;
- source ordering: 100%;
- block coverage: 100%;
- boundary precision/recall/F1;
- line-class accuracy only for labels that affect chunking;
- hierarchy accuracy;
- exact block match;
- citation-offset stability;
- results reported per archetype, not only as a global average.

---

## 4.2 Retrieval golden

The retrieval golden is more important for final system quality.

Recommended item schema:

```json
{
  "query_id": "nalus-rag-001",
  "query": "When may a court reject an appeal on points of law for failure to define admissibility?",
  "query_type": "legal_rule",
  "jurisdiction": "CZ",
  "court_scope": ["constitutional_court", "supreme_court"],
  "required_legal_concepts": [
    "admissibility of appeal on points of law",
    "Section 237 Code of Civil Procedure"
  ],
  "primary_relevant_spans": [
    {
      "document_id": "...",
      "start_line": 30,
      "end_line": 31
    }
  ],
  "alternative_relevant_spans": [],
  "hard_negatives": [],
  "answerable_from_corpus": true,
  "notes": ""
}
```

### Required query categories

- exact case number or ECLI;
- specific statutory provision;
- legal principle expressed in plain language;
- synonymous or non-legal phrasing;
- long client factual description;
- procedural stage;
- factually similar case;
- query requiring two or more passages;
- negative query with no answer in the corpus;
- hard-negative query with a highly similar but legally different passage;
- temporal or court restriction;
- query with alternatively relevant decisions.

### Benchmark size

- first usable version: 100–150 queries;
- before a production decision: 200–300 queries;
- recommended split:
  - 60% development/tuning;
  - 20% validation;
  - 20% locked holdout.

The locked holdout must not be used to tune `top_k`, RRF parameters, thresholds, prompts, or models.

---

## 5. Canonical data model

Each parser block must have a stable identity independent of retrieval profile.

### Document

```text
document_id
source_document_id
ecli
case_number
court
court_chamber
decision_type
decision_date
jurisdiction
language
source_url
source_checksum
parser_profile
```

### Parser block

```text
block_id
document_id
block_index
line_start
line_end
raw_text
normalized_text
primary_class
all_line_classes
section_path
heading_context
paragraph_number
hierarchy_level
parent_block_id
citations
statutes
case_references
dates
source_checksum
```

### Child retrieval chunk

```text
chunk_id
document_id
source_block_ids
line_start
line_end
chunk_text
embedding_text
section_path
heading_context
primary_paragraph_number
parent_id
token_count
chunking_profile
content_checksum
```

### Parent context

```text
parent_id
document_id
child_ids
line_start
line_end
parent_text
section_path
context_type
token_count
content_checksum
```

Stable identity must support:

- comparison of chunking variants;
- source reconstruction;
- exact citation;
- result deduplication;
- regression tracking after reindexing.

---

## 6. Chunking experiments

First create three or four variants over the same parser blocks.

### Variant A — fixed-size control baseline

- approximately 400 tokens;
- small overlap;
- used only as a control experiment;
- not the target architecture.

### Variant B — one legal paragraph

- one main numbered paragraph;
- its continuation lines;
- its nested list or table when it fits;
- heading context as metadata and prefix.

### Variant C — paragraph groups

- two or three short related paragraphs;
- respects section boundaries;
- never joins operative part with reasoning;
- never joins instruction/signature with reasoning.

### Variant D — parent–child

**Child:**

- approximately 180–500 tokens;
- atomic legal paragraph or a safe segment of it;
- heading and section prefix;
- exact source lines and block IDs.

**Parent:**

- approximately 700–1,500 tokens;
- entire logical section or surrounding paragraph group;
- used for generation, not necessarily for first-stage retrieval.

### Chunking acceptance criteria

The winner is not selected by subjective inspection. It is selected using:

- span Recall@k;
- span precision;
- MRR;
- nDCG;
- noise ratio;
- fragmentation of relevant legal units;
- number of relevant characters in context;
- number of duplicate children from the same parent;
- citation-offset stability;
- p95 latency and index size.

---

## 7. Retrieval baseline

Shared candidate-generation layer:

```text
original query
    ↓
deterministic extraction of filters and identifiers
    ↓
BGE-M3 dense retrieval
    +
BM25 or BGE-M3 sparse retrieval
    ↓
RRF
    ↓
candidate set
```

### Mandatory baseline experiments

1. BM25/sparse;
2. BGE-M3 dense;
3. dense + sparse + RRF;
4. hybrid + ColBERT;
5. hybrid + cross-encoder;
6. hybrid + ColBERT + cross-encoder.

The same benchmark, chunks, filters, and candidate limits must be used wherever the experiment allows it.

### Metadata filtering

Use hard filters only on reliable fields:

- court;
- jurisdiction;
- proceeding type;
- date;
- exact ECLI;
- exact case number.

An uncertain value extracted during query processing must not become a hard filter that removes the correct result. Use it as a boost or soft preference.

---

## 8. Runtime retrieval profiles

## 8.1 FAST

```text
hybrid dense + sparse
    ↓
RRF
    ↓
deduplication
    ↓
parent expansion
```

Use cases:

- navigation;
- exact ECLI/case number;
- autocomplete;
- low-cost tier;
- very low latency budget.

Initial experimental configuration:

```text
dense_top_k: 50
sparse_top_k: 50
rrf_candidate_limit: 50
final_limit: 8
```

## 8.2 BALANCED

```text
hybrid + RRF
    ↓
ColBERT late-interaction reranking
    ↓
deduplication
    ↓
parent expansion
```

Use cases:

- standard legal search;
- interactive UX;
- improved accuracy without full cross-encoder cost.

Initial configuration:

```text
dense_top_k: 80
sparse_top_k: 80
rrf_candidate_limit: 80
colbert_input_limit: 30
final_limit: 8
```

## 8.3 PRECISE

```text
hybrid + RRF
    ↓
ColBERT top 30–40
    ↓
cross-encoder top 10–20
    ↓
deduplication
    ↓
parent expansion
```

Use cases:

- complex legal analysis;
- long factual queries;
- multiple legal issues;
- high cost of error;
- client with a higher latency budget.

Initial configuration:

```text
dense_top_k: 100
sparse_top_k: 100
rrf_candidate_limit: 100
colbert_input_limit: 40
cross_encoder_input_limit: 15
final_limit: 6
```

These values are not production truth. The benchmark will determine the final values.

---

## 9. Index strategy

### Shared identity

All retrieval variants must use the same:

- `document_id`;
- `block_id`;
- `chunk_id`;
- parser and chunking checksums.

### Dense and sparse

Use isolated, versioned index names or Qdrant aliases:

```text
nalus_legal_child_v1_dense
nalus_legal_child_v1_sparse
```

Named vectors in one collection may be used when they simplify atomic switching and rollback.

### ColBERT

ColBERT requires token-level multivectors. Recommended options:

1. named multivector in the same logical collection; or
2. separate collection mapped through the same `chunk_id`.

The decision must be based on measured:

- RAM;
- disk usage;
- ingest duration;
- p95 latency;
- operational complexity;
- rollback behavior.

### Cross-encoder

The cross-encoder does not need its own document index. It runs online only on a limited candidate set and uses the same child texts.

---

## 10. Query processing

Query processing must preserve the original query and create only a structured supplement.

### Deterministic extraction

- ECLI;
- case number;
- statutory provision;
- statute;
- court;
- date;
- procedural stage;
- party role;
- requested remedy.

### Clarification gate

An ambiguous query must not be freely rewritten. The system:

1. decides whether the query is sufficient for retrieval;
2. asks a short clarification question when necessary;
3. stores a safely reusable clarification pattern;
4. does not invent missing facts as certainty.

### Query expansion

Query expansion is a separate experiment. It must be compared against a no-expansion baseline and must not alter the locked benchmark.

---

## 11. Deduplication and context assembly

After reranking:

1. remove near-identical child chunks;
2. limit the number of children from the same parent;
3. preserve distinct relevant lines of argument;
4. attach the parent or neighboring paragraphs;
5. remain within the context budget;
6. preserve mapping from every sentence to source lines.

### Context-selection rules

- operative clauses must not be lost;
- heading context must be attached;
- previous/next paragraph is added only when structurally necessary;
- boilerplate is preferentially removed;
- signatures, technical metadata, and layout noise are normally excluded from LLM context;
- cited statutes and case law inside the passage remain preserved.

---

## 12. Answer generation

The LLM must not replace retrieval.

### Mandatory behavior

- answer only from supplied evidence items;
- claim-level citations;
- distinguish legal rule, application, and factual conclusion;
- explicitly identify alternatively relevant case law;
- never invent ECLI, statutory provisions, or facts;
- abstain when evidence is insufficient;
- disclose corpus limitations.

### Output data model

```text
answer
claims[]
claim_id
claim_text
supporting_chunk_ids[]
supporting_line_ranges[]
confidence
unsupported_claims[]
abstention_reason
retrieval_profile
trace_id
```

---

## 13. Evaluation by layer

## 13.1 Parser

- conservation;
- duplication;
- ordering;
- boundary precision/recall/F1;
- exact block match;
- hierarchy accuracy;
- offset stability.

## 13.2 Chunking

- relevant span intact;
- fragmentation rate;
- overmerge/noise rate;
- token distribution;
- parent reconstruction;
- citation stability.

## 13.3 Retrieval

- Recall@5, @10, @20;
- MRR;
- nDCG@10;
- span recall;
- span precision;
- alternative relevance coverage;
- hard-negative rejection;
- breakdown by query type;
- breakdown by court/archetype.

## 13.4 Reranking

- delta over hybrid baseline;
- top-1 and top-5 quality;
- pair count;
- p50/p95/p99 latency;
- CPU/GPU time;
- memory;
- failure/fallback behavior.

## 13.5 Generation

- answer correctness;
- claim support;
- faithfulness;
- citation precision;
- citation recall;
- completeness;
- unsupported-claim rate;
- abstention accuracy.

## 13.6 End-to-end

- legally usable answer;
- citation opens the correct document and range;
- relevant passage is sufficiently precise;
- latency SLA;
- cost per query;
- repeat stability;
- traceability.

---

## 14. Diagnostic error taxonomy

Each failure must have one primary cause:

```text
PARSER_ERROR
CHUNK_BOUNDARY_ERROR
CHUNK_CONTEXT_ERROR
MISSING_METADATA
QUERY_PROCESSING_ERROR
CLARIFICATION_ERROR
DENSE_RETRIEVAL_MISS
SPARSE_RETRIEVAL_MISS
FUSION_ERROR
FILTER_ERROR
COLBERT_RERANK_ERROR
CROSS_ENCODER_RERANK_ERROR
DEDUPLICATION_ERROR
CONTEXT_ASSEMBLY_ERROR
GENERATOR_REASONING_ERROR
UNSUPPORTED_CLAIM
CITATION_ERROR
ABSTENTION_ERROR
BENCHMARK_GOLD_ERROR
```

The fix must be applied only to the layer that caused the failure.

---

## 15. Experimental discipline

Each experiment must record:

```text
run_id
git_commit
parser_profile
chunking_profile
embedding_model
sparse_profile
collection/index identity
fusion config
reranker model
candidate limits
filters
benchmark version
query split
hardware
latency percentiles
quality metrics
failure taxonomy
artifact checksums
```

### Prohibited practices

- changing multiple major layers at once;
- tuning on the locked holdout;
- declaring a winner from a few sample queries;
- rewriting the benchmark to match the current model;
- mixing parser correctness with retrieval correctness;
- comparing runs over different chunks without explicit labeling;
- introducing silent fallback behavior that hides failures;
- automatically treating a different but relevant decision as a failure.

---

## 16. Phased implementation roadmap

## Phase 0 — lock parser v7

**Goal:** confirm v7 as a safe structural baseline.

Tasks:

- independent review of v7 JSON/Markdown;
- confirm commits and checksums;
- inspect v7 changed items;
- decide push/merge;
- create a tag or documented baseline ID;
- archive v6 as historical baseline.

**Gate:**

- no newly confirmed systematic parser error;
- all v7 tests green;
- raw/manual data unchanged;
- reproducible export.

## Phase 1 — parser archetypes and holdout

**Goal:** prevent future overfitting to one document format.

Tasks:

- define seven archetypes;
- select development/regression/holdout documents;
- create minimal exact parser fixtures;
- create aggregate and per-archetype reports.

**Gate:**

- locked holdout exists;
- parser changes cannot pass based on one court type alone;
- conservation and identity remain 100%.

## Phase 2 — canonical block/chunk schema

**Goal:** stable data contract for all later experiments.

Tasks:

- Pydantic/typed models;
- stable IDs and checksums;
- document → block → child → parent relations;
- export and reconstruction tests;
- migration without changing raw text.

**Gate:**

- every child reconstructs to original lines;
- every parent contains only its own children;
- no duplication or loss.

## Phase 3 — chunking A/B/C/D

**Goal:** select chunking based on retrieval outcomes.

Tasks:

- fixed baseline;
- paragraph;
- paragraph groups;
- parent–child;
- same embedding/retriever for all variants;
- span-level benchmark run.

**Gate:**

- winner shows measurable improvement on validation;
- improvement is confirmed on locked holdout;
- citation stability and latency are acceptable.

## Phase 4 — retrieval benchmark 100–150

**Goal:** stop making decisions from eight smoke-test queries.

Tasks:

- create real queries;
- annotate primary/alternative spans;
- add hard negatives;
- add negative and multi-hop items;
- create development/validation/holdout split;
- create benchmark validator.

**Gate:**

- no holdout overlap;
- every item has a source span;
- benchmark errors are separated from model errors.

## Phase 5 — hybrid baseline

**Goal:** freeze a production candidate-generation baseline.

Tasks:

- BM25/sparse;
- BGE-M3 dense;
- RRF;
- metadata filters;
- deduplication;
- per-query diagnostics.

**Gate:**

- hybrid beats standalone baselines on validation;
- holdout confirms improvement;
- index identity and run metadata are reproducible.

## Phase 6 — ColBERT and cross-encoder profiles

**Goal:** establish the latency/accuracy trade-off.

Tasks:

- ColBERT benchmark;
- cross-encoder benchmark;
- cascade benchmark;
- RAM/disk/latency measurements;
- model selection on the Czech legal benchmark.

**Gate:**

- `BALANCED` has measurable benefit over `FAST`;
- `PRECISE` has measurable benefit over `BALANCED`;
- cost and p95 latency match client SLA.

## Phase 7 — query router

**Goal:** select the correct runtime profile.

Tasks:

- explicit client tier;
- latency budget;
- query complexity;
- exact-identifier detection;
- multi-issue detection;
- safe fallback to the user-selected profile.

**Gate:**

- router never reduces explicitly requested accuracy;
- decision is logged as bounded metadata;
- no raw query leakage into unsafe metrics.

## Phase 8 — context assembly and citations

**Goal:** provide the LLM with sufficient but low-noise evidence.

Tasks:

- parent expansion;
- neighboring-paragraph rules;
- deduplication;
- context budget;
- exact source mapping;
- citation renderer.

**Gate:**

- every claim maps to a chunk and source lines;
- no citation points to a nonexistent block;
- noise ratio does not worsen without recall benefit.

## Phase 9 — generation and abstention

**Goal:** legally cautious, evidence-backed answers.

Tasks:

- evidence-only prompt;
- claim extraction;
- citation verifier;
- unsupported-claim guard;
- abstention policy;
- answer benchmark.

**Gate:**

- unsupported-claim rate below the defined threshold;
- citation precision/recall meets the threshold;
- negative queries lead to abstention.

## Phase 10 — production observability and rollout

**Goal:** safe operations and measurable quality.

Tasks:

- trace ID;
- retrieval profile;
- stage latencies;
- candidate counts;
- cache/model/index identity;
- bounded error taxonomy;
- shadow deployment;
- canary rollout;
- rollback alias.

**Gate:**

- bad answers can be reproduced;
- rollback is tested;
- no sensitive raw data in metrics;
- SLA dashboards exist.

---

## 17. Initial SLA and decision table

Values must be confirmed by benchmark; these are starting operational targets.

| Profile | Primary purpose | P95 retrieval latency | Candidates | Reranking |
|---|---|---:|---:|---|
| FAST | navigation and low-cost search | lowest | 50 | none |
| BALANCED | standard legal search | medium | 80 | ColBERT |
| PRECISE | complex legal analysis | higher | 100 | ColBERT + cross-encoder |

Winner selection must always consider together:

- Recall@10;
- nDCG@10;
- span recall;
- citation precision;
- p95 latency;
- RAM/index size;
- computational cost;
- reranker pairs/query.

---

## 18. Git, versioning, and artifacts

### Recommended profiles and versions

```text
parser_profile
chunking_profile
retrieval_profile
benchmark_version
index_version
answer_policy_version
```

### Git rules

- each behavioral change gets its own commit;
- benchmark data and code remain separate from generated run artifacts;
- no push before a validation gate;
- no silent model, index, or fallback change;
- parser v6 and v7 outputs are never overwritten;
- production alias changes only after holdout PASS.

### Run artifacts

Each meaningful run creates:

```text
summary.json
per_query.jsonl
failures.md
latency.json
config.json
checksums.json
```

---

## 19. Definition of Done for every phase

A phase is complete only when:

- scope was respected;
- baseline and candidate are explicit;
- tests passed;
- holdout was not used for tuning;
- run is reproducible;
- index/model identity is recorded;
- raw sources were not modified;
- no hidden fallback exists;
- known limitations are listed;
- the next exact step is defined;
- the commit reflects the actual change;
- push/merge occurs only after explicit approval.

---

## 20. Exact next steps from the current state

### Step 1 — complete v7 review

Status: **done** as `ACCEPT_V7_WITH_KNOWN_LIMITATIONS`.

Record: `docs/architecture/PARSER_V7_BASELINE_DECISION.md`

Obtain and independently inspect:

```text
artifacts/legal_v2/parser_v7_full_review/parser_v7_remaining_17_full.json
artifacts/legal_v2/parser_v7_full_review/parser_v7_remaining_17_full.md
```

Output:

- confirmed errors;
- confirmed correct regressions;
- decision `ACCEPT_V7` or `FIX_V8`;
- no further parser tuning without concrete retrieval impact.

Current decision: `ACCEPT_V7_WITH_KNOWN_LIMITATIONS` including `KNOWN-PARSER-001`.

### Step 2 — create parser archetype manifest

Status: **initial draft created**.

Suggested / created file:

```text
docs/architecture/parser_benchmark/archetypes_v1.json
artifacts/legal_v2/parser_benchmark/archetypes_v1.json
```

It must include development/regression/holdout roles and the reason each document was selected. Four holdout slots remain `pending_external` until additional unseen documents are added.

### Step 3 — design canonical child/parent schema

Do not index the full corpus yet. First create:

- typed model;
- stable identities;
- reconstruction test;
- one small isolated pilot.

### Step 4 — create retrieval benchmark v1

First iteration target:

- 100–150 queries;
- span annotations;
- alternative relevance;
- hard negatives;
- locked holdout.

### Step 5 — run chunking experiment

Run A/B/C/D over the same benchmark and the same retriever.

### Step 6 — freeze the winning chunking profile

Only then create the production dense/sparse index.

### Step 7 — retrieval and reranking experiments

In this order:

1. BM25;
2. BGE-M3;
3. hybrid RRF;
4. hybrid + ColBERT;
5. hybrid + cross-encoder;
6. hybrid + ColBERT + cross-encoder.

### Step 8 — define client profiles

Use benchmark results to determine actual:

- `top_k`;
- models;
- p95 SLA;
- accuracy;
- cost;
- router rules.

### Step 9 — context assembly and claim-level citations

The legal RAG pipeline is not complete without this phase.

### Step 10 — generation, abstention, and production rollout

Only after the retrieval benchmark is stable.

---

## 21. Explicitly excluded for now

- no manual review of hundreds of thousands of lines without prioritization;
- parser is not treated as the final product;
- ColBERT is not added before chunking selection;
- cross-encoder is not run over the full corpus;
- chunking, embedding, and reranker are not changed together;
- models are not selected from marketing benchmarks;
- locked holdout is not used for tuning;
- client modes are not implemented as separate codebases;
- LLM is not used to hide retrieval failures;
- production index is not overwritten during experiments.

---

## 22. Final target architecture

```text
                         ┌──────────────────────────┐
                         │ Query processing         │
                         │ identifiers + soft hints │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │ BGE-M3 dense + sparse    │
                         │ candidate generation     │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │ RRF fusion               │
                         └────────────┬─────────────┘
                                      │
                 ┌────────────────────┼────────────────────┐
                 │                    │                    │
        ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────────┐
        │ FAST            │  │ BALANCED        │  │ PRECISE             │
        │ no reranker     │  │ ColBERT         │  │ ColBERT → CE        │
        └────────┬────────┘  └────────┬────────┘  └────────┬────────────┘
                 └────────────────────┼────────────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │ dedup + parent expansion │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │ evidence context         │
                         │ exact line mapping       │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │ LLM answer               │
                         │ claim citations          │
                         │ abstention               │
                         └──────────────────────────┘
```

---

## 23. Decision

NALUS will be built as a benchmark-driven legal RAG system.

The binding order is:

```text
v7 parser baseline
→ parser archetypes + locked holdout
→ canonical block/chunk schema
→ retrieval golden
→ chunking A/B test
→ hybrid baseline
→ ColBERT/cross-encoder profiles
→ context assembly
→ claim-level citations
→ generation and abstention
→ production rollout
```

Any change to this order must have a concrete technical reason and must not bypass the benchmark or locked holdout.
