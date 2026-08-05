# Legal decision parser v7

## Scope

`legal-decision-parser.cz-courts.v7` corrects confirmed generalization regressions from the v6 full-corpus review of the 17 non-golden design documents.

Parser v6 and its audit/export artifacts remain the immutable historical baseline. Exact golden documents remain only `05`, `11`, and `16`.

Production rules are court-profile and parser-state based. They must not key on document ID, source ID, review index, exact case number, party/judge names, or exact raw sentences.

## Confirmed structural corrections

### Constitutional Court compact subheadings

Compact whole-line Roman(+decimal) section headings such as `V.1 Obecná východiska`, `IV. Posouzení …`, `II. Skutkový stav …` are independent `heading` blocks with SPLIT boundaries on both sides when they are short structural captions followed by the next numbered legal paragraph.

### High Court Prague opening formula

A Prague case-opening formula beginning with `Vrchní soud v Praze` plus `rozhodl` or `jako soud odvolací` merges participant/address/representation/subject-matter lines until the whole-line heading `Výrok`. A line containing `č. j.` inside that formula remains `prose_continuation`.

### High Court Olomouc civil reasoning state

v6 incorrectly generalized the 698-line criminal golden onto shorter civil decisions. v7 dispatches civil vs criminal Olomouc structure from observed format after `Výrok` / `Odůvodnění`:

- Roman operative clauses after `Výrok` are independent `numbered_paragraph_start` blocks;
- civil reasoning uses expected top-level Arabic numbering progression;
- physical continuations remain `numbered_paragraph_continuation`;
- genuine nested bullets/lists use `list_or_table`;
- a date such as `1. července 2014` is not a top-level item or table when the expected next paragraph number is higher;
- statutory/semicolon-dense sentences do not force `list_or_table` in civil reasoning.

The criminal golden (`16`) remains exact, including rejection of false starts at lines 182 and 296–301.

## Audit and export

```text
artifacts/legal_v2/parser_v7_audit/
artifacts/legal_v2/parser_v7_full_review/parser_v7_remaining_17_full.json
artifacts/legal_v2/parser_v7_full_review/parser_v7_remaining_17_full.md
```

Historical v6 paths remain available:

```text
artifacts/legal_v2/parser_v6_audit/
artifacts/legal_v2/parser_v6_full_review/parser_v6_remaining_17_full.json
artifacts/legal_v2/parser_v6_full_review/parser_v6_remaining_17_full.md
```

Export consistency rules:

- active `parser_manual_conflicts` and `stale_manual_decisions` are disjoint categories;
- summary counts must equal candidate-list lengths;
- legitimate long opening formulas are not flagged as suspicious overmerges merely because they span many physical lines.

## UI

The local review UI shows parser v7 labels and adds:

- `Changed by parser v7`
- `Full corpus v7 review`

Historical v6 change/corpus views and `/exports/parser_v6_*` routes remain available.

Targeted regression documents may display `TARGETED REGRESSION PASS`. Exact `GOLDEN PASS` remains reserved for documents `05`, `11`, and `16`.

## Baseline acceptance

Decision: **`ACCEPT_V7_WITH_KNOWN_LIMITATIONS`**

Canonical record:

[`docs/architecture/PARSER_V7_BASELINE_DECISION.md`](../architecture/PARSER_V7_BASELINE_DECISION.md)

Known limitation:

- `KNOWN-PARSER-001` — closing location/date without `dne` may be labeled `heading` instead of `metadata`; boundaries and reconstruction remain correct; no demonstrated retrieval impact; do not open parser v8 for this label.

Parser tuning for non-blocking label noise is stopped. Next work follows
[`docs/architecture/NALUS_LEGAL_RAG_MASTER_PLAN.md`](../architecture/NALUS_LEGAL_RAG_MASTER_PLAN.md).

## Rebuild

```powershell
python scripts/legal_v2/audit_parser_v7.py --write-baseline
python scripts/legal_v2/build_visual_parser_review.py
python scripts/legal_v2/audit_parser_v7.py
python scripts/legal_v2/export_parser_v7_full_review.py --verify-determinism
```
