# Failed Case Analysis — client_longform_v1

**Benchmark:** NALUS NSOud Client Long-Form Retrieval Eval v1  
**Winner config:** `bge_m3__dense_plus_bm25`  
**Collection:** `nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1`  
**Metrics:** hit_rate 0.875 (7/8), missing_expected_marker_count 1  

**Source artifacts inspected (no rerun):**
- `artifacts/rag_eval/client_longform_v1/winner_qa.json`
- `artifacts/rag_eval/client_longform_v1/winner_legal_eval.json`
- `artifacts/rag_eval/nalus_client_longform_eval_v1.json`

---

## 1. Failed case identification

| Field | Value |
|-------|-------|
| **case_id** | `client-longform-04` |
| **pilot_case_id** | `nsoud-positive-04` |
| **difficulty** | hard |
| **benchmark result** | FAIL — expected marker absent from all top-5 hits |
| **production_usefulness** | good (legal eval) |
| **benchmark_alignment** | partially_aligned |

**Client-style question:**

> Klient se odvolal proti rozsudku, ale odvolací soud jeho odvolání zamítl po věcném přezkoumání. Teď chce podat dovolání a tvrdí, že chyby vznikly už před soudem prvního stupně. Potřebuji najít rozhodnutí, která vysvětlují, jak se v dovolání argumentuje v situaci, kdy odvolací soud odvolání zamítl, ale dovolatel tvrdí, že už předchozí řízení mělo právní vadu. Zajímá mě hlavně vazba mezi rozhodnutím odvolacího soudu a dovolacími důvody vztahujícími se k předchozímu řízení.

All five `retrieval_hits` have `matched_markers: []` for the expected marker. No other case in the 8-case benchmark has zero marker matches across top-5.

---

## 2. Expected marker and source scope

| Field | Value |
|-------|-------|
| **Expected marker** | `dovolací důvod podle § 265b odst. 1 písm. m)` |
| **Marker aliases** | none (`aliases: []`) |
| **minimum_coverage** | 1.0 |
| **allow_partial** | false |

**Expected ECLI / source_scope (`multi_document`):**

| ECLI |
|------|
| `ECLI:CZ:NS:2024:11.TDO.765.2024.1` |
| `ECLI:CZ:NS:2024:4.TDO.1044.2024.1` |
| `ECLI:CZ:NS:2024:6.TDO.827.2024.1` |
| `ECLI:CZ:NS:2024:6.TDO.976.2024.1` |

All expected documents are **trestní** (TDO) decisions. The client question does not mention trestní řízení, § 265b, or písm. m) explicitly.

---

## 3. Top-5 retrieved hits

| Rank | document_id | Score | matched_markers | Legal summary | Classification |
|------|-------------|-------|-----------------|---------------|----------------|
| 1 | `ECLI:CZ:NS:2024:23.CDO.271.2024.1` | 0.0164 | — | **Občanskoprávní** dovolání (o. s. ř.): NS zrušil rozsudek odvolacího soudu pro neúplné právní posouzení; zkoumal vad řízení podle § 229 o. s. ř.; důvody zrušení se vztahují i na rozsudek soudu prvního stupně (§ 243e). | `alternate_relevant` |
| 2 | `ECLI:CZ:NS:2024:8.TDO.1022.2024.1` | 0.0164 | — | **Trestní** obecná teorie dovolání: § 265b tr. ř. je mimořádný opravný prostředek, NS není třetí instance; omezený rozsah přezkumu skutkových zjištění. | `alternate_relevant` |
| 3 | `ECLI:CZ:NS:2025:4.TDO.1056.2024.1` | 0.0161 | — | **Trestní** posouzení dovolání po zamítnutí odvolání: NS hodnotí dovolací důvody g) a h), nikoli m); opakování argumentace z nižších stupňů. | `alternate_relevant` |
| 4 | `ECLI:CZ:NS:2025:4.TDO.1100.2024.1` | 0.0161 | — | **Trestní** obecná teorie § 265b: restriktivní pojetí dovolacích důvodů, NS vázán uplatněnými důvody (§ 265f). | `alternate_relevant` |
| 5 | `ECLI:CZ:NS:2024:6.TDO.991.2024.1` | 0.0159 | — | **Trestní** dovolání: opakování argumentace z prvostupňového a druhostupňového řízení; posouzení důvodů g) a h), ne m). | `alternate_relevant` |

None of the retrieved chunks contain the exact substring `dovolací důvod podle § 265b odst. 1 písm. m)`. None match any expected TDO ECLI in `source_scope`.

---

## 4. Legal relevance assessment

### A) Did the system retrieve legally relevant case law?

**Partially yes.** The retrieval surfaced procedurally related decisions:

- Rank 1 addresses the same *narrative pattern* (appeal dismissed on merits → cassation arguing prior-instance defects, link between appellate and first-instance rulings), but in **civil procedure** (o. s. ř.), not trestní řízení.
- Ranks 2–5 are trestní dovolání decisions discussing § 265b generally, but focus on subsections **g)** and **h)** or general doctrine, not the specific **písm. m)** ground (legal defect in prior proceedings when appeal was dismissed).

### B) Is this a real retrieval failure?

**Partially.** The system did not surface any of the four expected TDO decisions containing the marker. The long-form client text omits the statutory anchor (§ 265b písm. m)), so hybrid retrieval ranked a civil CDO precedent first and generic trestní § 265b chunks afterward. This is weaker than ideal for a lawyer seeking trestní písm. m) authority, but not a complete miss on the legal topic.

### C) Eval marker / source_scope problem?

**Yes, significantly.** The benchmark requires an exact marker substring with no aliases. The client question is domain-ambiguous (could be read as civil or criminal dovolání) and never names § 265b or písm. m). The eval therefore penalizes retrieval that is *production-useful* but does not hit a very specific statutory phrase. Legal eval already rates this case as `good` / `partially_aligned`.

### D) Would a lawyer find the retrieved hits useful?

**Mixed.**

- Rank 1 could be **misleading** if the lawyer's matter is trestní — it applies o. s. ř., not tr. ř.
- Ranks 2–5 are useful background on trestní dovolání limits and repeat-argument doctrine, but do not directly answer the písm. m) linkage question after appeal dismissal.

### E) Recommended changes

| Option | Recommendation |
|--------|----------------|
| dataset marker | **Yes** — add aliases: `§ 265b odst. 1 písm. m)`, `písm. m) tr. ř.` |
| eval logic | Consider partial credit for `alternate_relevant` trestní § 265b hits in long-form eval |
| top_k | Investigate ranks 6–10 before changing (marker may appear deeper) |
| query formulation | Add optional rewrite extracting branch of law + statutory terms from client text |
| retrieval config | No change as first step |
| nothing | Not recommended |

---

## 5. Final verdict

| Category | Verdict |
|----------|---------|
| Benchmark marker hit | **FAIL** |
| Real retrieval failure | **Partial** — expected TDO ECLI with písm. m) not in top-5 |
| Marker/eval problem | **Yes** — strict substring + no aliases + domain ambiguity in question |
| Acceptable alternate relevant result | **Yes** — legal eval: 5× `alternate_relevant`, production_usefulness `good` |

**Overall:** This is primarily an **eval-marker / source_scope strictness** issue with a **secondary retrieval weakness** (wrong branch ranked first, specific písm. m) authority missing). It is **not** a case of irrelevant retrieval.

---

## 6. Concrete next step

### Primary recommendation: legal query clarification gate

`client-longform-04` is **not** proof that BGE-M3 / Qdrant embeddings are broken. The main issue is an **under-specified legal domain** in a long client narrative. A production system should detect this ambiguity and ask a clarifying question **before** returning potentially misleading case law.

Implemented module: `app/rag/clarification/`

**Important separation of concerns:**

| Component | Responsibility |
|-----------|----------------|
| **Qdrant (main collections)** | Court decision / case-law retrieval |
| **Redis** | Fast cache of clarification payloads and generated questions |
| **Qdrant `legal_query_clarification_patterns` (optional)** | Similarity lookup for ambiguous-query patterns only |
| **LLM** | Last resort when rules, templates, exact cache, and semantic reuse are insufficient |

Expected client-facing response for this case:

> Potřebuji upřesnit jednu věc: jde o trestní dovolání podle trestního řádu, nebo o civilní dovolání podle občanského soudního řádu? Podle toho se výrazně liší relevantní judikatura Nejvyššího soudu.

Architecture:

```text
User longform question
→ deterministic ambiguity rules (no LLM)
→ if clear: proceed_to_retrieval (Qdrant unchanged)
→ if ambiguous:
    → Redis exact cache by normalized query hash
    → semantic pattern reuse (in-memory or Qdrant clarification patterns)
    → deterministic template question
    → LLM only if template insufficient
→ optional post-retrieval domain mismatch check (CDO top-1 + TDO ranks 2–5)
→ store clarification payload in Redis
```

### Secondary (benchmark-only) options

1. Add marker aliases to `client-longform-04` in eval v1.1 (e.g. `§ 265b odst. 1 písm. m)`).
2. Inspect ranks 6–10 in raw benchmark output before changing `top_k`.

**Do not treat this single fail as a reason to change retrieval config or rerun the benchmark.**
