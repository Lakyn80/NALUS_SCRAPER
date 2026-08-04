# Legal decision parser v5

## Scope

`legal-decision-parser.cz-courts.v5` is a bounded Constitutional Court structural update derived from the completed manual review of visual-review document 2 (`doc-b73cac9b3dfc8a42`, source `1-3299-24_1`).

The profile keeps the v3/v4 numbered-paragraph behavior and high-court whole-line heading support. It does not generalize High Court Prague or High Court Olomouc formatting.

## Constitutional Court changes

The v5 parser recognizes court-scoped structural lines for:

- NALUS page header.
- Constitutional Court case/date line.
- `Česká republika`.
- `USNESENÍ` or `NÁLEZ` followed by `Ústavního soudu`.
- Constitutional Court decision formula beginning `Ústavní soud rozhodl` and ending `takto:`.
- Operative text after a confirmed decision formula.
- `Odůvodnění` / `Odůvodnění:`.
- Reasoning prose after the reasoning heading.
- `Poučení:` instruction line.
- `V Brně dne <date>` closing line.
- Signature name ending `v. r.` followed by a bounded judicial-role line.

Inline `sp. zn.`, `č. j.`, statutes, dates, and paragraph symbols no longer override the primary structural class for these supported Constitutional Court roles.

## Unsupported

The v5 rules are not applied to `high_court_prague` or `high_court_olomouc`. Those court families require their own completed manual evidence before profile-specific generalization.

The update does not rebuild Qdrant, BM25, embeddings, or historical v2/v3/v4 manifests.

## Audit artifacts

Generated local artifacts are under:

```text
artifacts/legal_v2/constitutional_parser_v5/
```

They include the document-2 golden result, v4-vs-v5 corpus comparison, changed boundary/class queues, acceptance summary, and document-2 parser-profile migration record.
