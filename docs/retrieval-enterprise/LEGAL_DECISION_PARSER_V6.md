# Legal decision parser v6

## Scope

`legal-decision-parser.cz-courts.v6` generalizes the deterministic Czech legal-decision parser from the v5 Constitutional Court-only profile to three isolated court profiles:

- `constitutional_court`
- `high_court_prague`
- `high_court_olomouc`

The update is derived from the immutable golden package under `artifacts/legal_v2/parser_golden_inputs/` and the existing design-document raw snapshots. It does not rebuild Qdrant, BM25, embeddings, indexes, aliases, or historical parser manifests.

## Shared Core

The shared core keeps source-order preservation, text conservation, stable paragraph identities, generic numbered-paragraph behavior, Roman heading detection, heading inference, and citation/statute detection as secondary features.

Primary line class describes the full structural role of a line. Inline `č. j.`, `sp. zn.`, `§`, statute references, dates, and case citations must not override a line's primary role.

## Court Profiles

### Constitutional Court

The Constitutional profile preserves the v5 document-2 behavior and adds support for the corrected 54-line golden:

- topic lines as headings;
- three-line `NÁLEZ / Ústavního soudu / Jménem republiky` title blocks;
- Roman marker/title heading pairs;
- independent numbered reasoning paragraphs;
- `V Brně dne <textual or numeric date>` metadata;
- signature name plus role merge.

### High Court Prague

The Prague profile recognizes:

- multi-line opening/case formula blocks before `Výrok`;
- operative and reasoning headings;
- top-level numbered reasoning paragraphs;
- nested statutory lists attached to their parent paragraph.

The corrected golden requires lines 1-12 as one opening block and lines 36-42 as `list_or_table` inside paragraph 19.

### High Court Olomouc

The Olomouc profile uses explicit parser state:

- lines before `Odůvodnění` are opening/operative structure, not top-level reasoning;
- top-level reasoning numbering starts only after the reasoning heading;
- a top-level reasoning paragraph can start only when the visible number equals the expected next number and the remaining line is structurally compatible with a legal paragraph;
- nested numeric lists, lettered subitems, dash bullets, and semicolon table rows remain attached to the current parent block.

The 698-line golden validates 74 top-level reasoning paragraphs with the exact sequence 1..74 and rejects explicit false starts at lines 182 and 296-301.

## Audit Artifacts

Generated local audit artifacts live under:

```text
artifacts/legal_v2/parser_v6_audit/
```

They include golden input checksums, v5-vs-v6 comparison outputs, changed line/boundary/block queues, hierarchy/table audits, corpus acceptance summaries, and the document-2 parser-profile migration record.

## Full-corpus review export

The reusable offline exporter is:

```text
scripts/legal_v2/export_parser_v6_full_review.py
```

Default outputs (ignored artifacts):

```text
artifacts/legal_v2/parser_v6_full_review/parser_v6_remaining_17_full.json
artifacts/legal_v2/parser_v6_full_review/parser_v6_remaining_17_full.md
```

The export covers every line, boundary, and parser block for the 17 non-golden review documents, plus concise GOLDEN PASS summaries for documents 05, 11, and 16. Non-golden documents are never marked exact golden or manually approved by the exporter. The local review UI exposes a `Full corpus v6 review` view with JSON/Markdown download links and a per-document copy action.

## Known Limitations

The profile is deterministic and bounded to the observed Czech court structures. It does not claim full support for every future court format, damaged OCR layout, scanned tables without textual delimiters, or courts outside the three profiled families. Unsupported structures should be added through new golden evidence and court-profile tests rather than broad citation or date heuristics. The 17 non-golden review documents remain a review queue, not correctness proof.
