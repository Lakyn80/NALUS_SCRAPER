# Czech Court Format Study

Status: parser-format study and deterministic parser profile for sampled Czech
court families only. No index was rebuilt.

## Scope

The study covers:

- Constitutional Court of the Czech Republic;
- High Court in Prague;
- High Court in Olomouc.

It does not claim support for every Czech court type.

## Source Selection

The design sample uses fixed seed `20260804`.
The holdout sample uses fixed seed `20260805`.

Sources are official public records:

- Constitutional Court decisions from NALUS `https://nalus.usoud.cz`;
- High Court decisions from Justice Open Data
  `https://rozhodnuti.justice.cz/api/opendata` and final documents under
  `https://rozhodnuti.justice.cz/api/finaldoc/`.

Raw downloaded sources are stored only under ignored
`artifacts/legal_v2/court_format_study/raw_sources/` and are not committed.

## Generated Artifacts

Run:

```powershell
python scripts/legal_v2/court_format_study.py
```

The runner writes:

- `design_sample_manifest.json`
- `holdout_sample_manifest.json`
- `sampling_report.md`
- `design_line_annotations.jsonl`
- `design_boundary_annotations.jsonl`
- `design_document_summaries.jsonl`
- `format_inventory.json`
- `format_taxonomy.md`
- `rule_evidence_matrix.json`
- `baseline_design_results.json`
- `final_design_results.json`
- `holdout_results.json`
- `boundary_changes.jsonl`
- `manual_review_report.md`
- `parser_acceptance_report.json`
- `parser_acceptance_report.md`

All generated files live under ignored `artifacts/` and are not committed.

## Format Taxonomy

The design corpus uses line-level structural classes for metadata, court
identity, decision type, operative text, section headings, numbered paragraph
starts and continuations, case references, statutory references, instruction
sections, signatures, page layout material, prose, lists, and table-like rows.

Every non-empty design source line receives exactly one class. Adjacent
non-empty line pairs receive a deterministic boundary annotation.

## Parser Grammar

The parser remains a common deterministic engine. The study did not justify
three separate parser implementations. Court-specific behavior is limited to
bounded whole-line titles and metadata markers observed in the sample.

Precedence:

1. blank structural boundary;
2. new numbered legal paragraph;
3. standalone Roman section marker;
4. verified whole-line heading;
5. active numbered paragraph continuation;
6. ordinary prose.

The parser still forbids broad keyword-substring heading detection. A numbered
paragraph is not a heading merely because it contains words such as `rizeni`,
`nalez`, `oduvodneni`, or `posouzeni`.

## Parser Profile

The generalized profile is:

```text
legal-decision-parser.cz-courts.v4
```

Historical v2 and v3 index resources remain historical. This task does not
create Qdrant collections, BM25 indexes, embeddings, or retrieval benchmarks.

## Validation Summary

Latest local study artifacts recorded:

- candidate population: 93 official documents;
- design sample: 10 Constitutional Court, 5 High Court Prague, 5 High Court
  Olomouc;
- holdout sample: 10 Constitutional Court, 5 High Court Prague, 5 High Court
  Olomouc;
- design result: pass;
- holdout result: pass;
- parser exceptions: 0;
- conservation failures: 0;
- duplicate-text failures: 0;
- ordering failures: 0;
- orphan `sp. zn.` blocks: 0;
- orphan `c. j.` / `č. j.` blocks: 0.

## Known Limitations

The study uses a bounded sample and deterministic structural review artifacts.
It is not a full judicial-format census, a retrieval-quality benchmark, or a
production rollout. Future expansion should add more court families and preserve
the same design/holdout separation.
