# Gold Annotation Follow-up Plan — 2026-07-10

Návrh dalšího postupu pro rozšíření gold anotací. Toto není provedení změn, jen schvalovací dokument.

## Aktuální stav

- Gold už existuje, ale jen částečně.
- Aktuálně je anotováno `10/40` otázek:
- ÚS: `5/20`
- NSoud: `3/10`
- Mixed: `2/10`
- Pending zůstává `30/40` položek.

Reference:
- `artifacts/rag_eval/legal_qa/gold_source_review_20260709.md`
- `artifacts/rag_eval/legal_qa/datasets/usoud_qa_v1.jsonl`
- `artifacts/rag_eval/legal_qa/datasets/nsoud_qa_v1.jsonl`
- `artifacts/rag_eval/legal_qa/datasets/mixed_qa_v1.jsonl`

## Klíčové zjištění pro prioritizaci

- ÚS pending položky mají ve frozen `usoud_full_baseline/retrieval_results.jsonl` použitelný rank-1 `document_id` v hit metadata.
- Mixed pending položky jsou z větší části vhodné pro `corpus_only` anotaci, pokud je potvrzené `corpus_hit@3=true`.
- NSoud pending položky v aktuálním frozen `nsoud_full_baseline/retrieval_results.jsonl` pro rank-1 typicky nemají použitelný `source_document_id` ani `ECLI`, takže nejsou nejlepší první kandidát na rychlé rozšíření gold.

## Doporučené pořadí

### Fáze 1: ÚS pending doplnit jako první

Důvod:
- nejnižší riziko
- frozen data už obsahují rank-1 document provenance
- rozšíření gold zde jde udělat bez změny retrieval pipeline

Pending ID:
- `usoud-qa-002`
- `usoud-qa-005`
- `usoud-qa-006`
- `usoud-qa-007`
- `usoud-qa-008`
- `usoud-qa-010`
- `usoud-qa-011`
- `usoud-qa-013`
- `usoud-qa-014`
- `usoud-qa-015`
- `usoud-qa-016`
- `usoud-qa-017`
- `usoud-qa-018`
- `usoud-qa-019`
- `usoud-qa-020`

Pravidlo:
- `source_pending=false` nastavit jen tam, kde je rank-1 stabilní a obsahově dává smysl proti otázce
- nevymýšlet `case_reference` ani `decision_date`
- gold review doplnit o každé nově schválené ID + ECLI

### Fáze 2: Mixed rozšířit jen jako corpus-only, ne document-gold

Důvod:
- cílem je rozšířit coverage bezpečně
- mixed eval už umí korektně pracovat s `corpus_only`
- není potřeba vymýšlet dokumentovou citaci tam, kde ji benchmark nemá mít

Kandidáti vhodní pro první pass:
- `mixed-qa-001`
- `mixed-qa-003`
- `mixed-qa-006`
- `mixed-qa-007`
- `mixed-qa-008`
- `mixed-qa-009`

Odložit na druhý pass:
- `mixed-qa-004`
- `mixed-qa-010`

Důvod odložení:
- u těchto dvou je `expected_target_corpus=ambiguous`
- v aktuálním frozen běhu není dobrý důvod je schválit uspěchaně

Pravidlo:
- u mixed nepřidávat document gold bez explicitního podkladu
- `source_pending=false` ano, ale `expected_source_constraints` ponechat bez falešné dokumentové citace

### Fáze 3: NSoud řešit až po ÚS a mixed

Důvod:
- je to nejméně přímočarý korpus pro další gold
- current frozen baseline pro pending otázky typicky nenese přímo použitelný `source_document_id`
- bez toho by hrozila vymyšlená nebo slabě podložená anotace

Pending ID:
- `nsoud-qa-001`
- `nsoud-qa-002`
- `nsoud-qa-005`
- `nsoud-qa-006`
- `nsoud-qa-007`
- `nsoud-qa-008`
- `nsoud-qa-009`

Než se začne anotovat:
- ověřit, zda jde ECLI spolehlivě vytáhnout z deeper hit metadata nebo z kolekce bez změny scoringu
- pokud ne, udělat samostatný maintenance krok pro lepší provenance export do eval artefaktů
- teprve potom doplnit gold review

## Co se má po schválení reálně změnit

- update `artifacts/rag_eval/legal_qa/datasets/*.jsonl`
- update `artifacts/rag_eval/legal_qa/gold_source_review_20260709.md`
- re-run answer eval pro dotčené korpusy
- refresh `summary.json` v `answer_eval/*`

## Co se nemá dělat

- nezapínat LLM
- neměnit BGE-M3 / BM25 / RRF
- nepřepínat aliasy
- nepsat do produkčních kolekcí
- nevymýšlet ECLI, spisovku ani datum rozhodnutí

## Doporučené schválení

Pokud chceš nejvyšší poměr přínos/riziko, schválit tento postup:

1. ÚS pending
2. Mixed corpus-only kandidáti
3. NSoud až po samostatném provenance checku

Tohle pořadí maximalizuje nové gold coverage bez rozbití guardrailů, které jsou v handoffu.
