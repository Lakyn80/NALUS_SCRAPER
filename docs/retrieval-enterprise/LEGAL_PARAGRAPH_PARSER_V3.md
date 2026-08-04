# Legal Paragraph Parser V3

Status: superseded by `legal-decision-parser.cz-courts.v4` for the sampled
Czech-court format study. No index has been rebuilt from either profile.

## Root Cause

The affected source text is complete and correctly ordered before parser entry.
The defect was in Legal v2 line segmentation. The old line-based parser used a
broad heading heuristic while deciding whether an appended line should flush the
current paragraph candidate. Short lines containing words such as `rizeni`,
`nalez`, `oduvodneni`, or `posouzeni` could be treated as headings even when the
line was the start of a numbered legal paragraph.

The confirmed failure shape was:

```text
28. Ve veci resene v rizeni
sp. zn. IV. US 1038/25
zrusil sluzebni organ...
```

Those lines are one numbered legal paragraph. They are not a heading plus an
orphan citation block.

## Corrected Precedence

The v3 parser applies deterministic structural precedence:

1. blank-line hard boundary;
2. new numbered legal paragraph;
3. verified whole-line heading;
4. continuation of an active numbered paragraph;
5. ordinary prose block.

The key invariant is that numbered paragraph structure is decided before heading
keywords. A numbered line is never a heading merely because it contains a
heading-like word.

## Numbered Paragraphs

The supported numbered starts remain bounded to the existing corpus formats:

- `[28] Text`
- `28. Text`
- `28) Text`

The parser does not broaden numbering to arbitrary unanchored digits.

## Continuation Lines

When a numbered paragraph is active, compatible continuation lines stay attached
until a new numbered paragraph, a blank line, or a verified whole-line heading is
encountered. This preserves layout-wrapped lines including:

- `sp. zn. IV. US 1038/25`
- `c. j. ...`
- citation continuations;
- lower-case sentence continuations;
- capitalized court or authority names continuing the same sentence.

Continuation does not depend only on lower-case starts or terminal punctuation,
because Czech legal citations and abbreviations commonly end with periods.

## Genuine Headings

Genuine headings remain preserved through structural whole-line matching and
existing all-caps heading recognition. Examples covered by regression tests:

- `Vyrok`
- `Oduvodneni`
- `Posouzeni Ustavniho soudu`
- section-style uppercase headings such as `I. SKUTKOVY STAV`

The parser no longer treats a line as a heading only because a heading keyword is
present as a substring in ordinary prose.

## Text Conservation

Parser output must preserve source order and non-whitespace source content. The
parser may normalize line wrapping into spaces inside a paragraph, but it must
not drop, duplicate, reorder, or rewrite legal text or citations.

## Parser Profile

The corrected parser profile is:

```text
legal-paragraph-parser.v3
```

Existing v2 pilot indexes and manifests remain historical. They must not be
retagged as v3. A future rebuilt pilot should use isolated v3 identities such as:

```text
nalus_legal_paragraph_chunks_v3_pilot_600
nalus_legal_paragraph_bm25_v3_pilot_600
```

This task does not create those resources.

The follow-up format study is documented in:

```text
docs/retrieval-enterprise/CZECH_COURT_FORMAT_STUDY.md
```

## Audit

Run the read-only audit without Qdrant, BM25, embeddings, Docker, or provider
calls:

```powershell
python scripts/legal_v2/audit_parser_fix.py --limit 200
```

The audit writes ignored local artifacts under:

```text
artifacts/legal_v2/parser_fix/
```

Expected files:

- `parser_fix_audit.json`
- `parser_fix_audit.md`
- `parser_fix_suspicious_samples.jsonl`

The audit compares legacy line segmentation with corrected parser output and
reports numbered-heading false positives, standalone `sp. zn.`/`c. j.`
candidates, suspicious short candidates, text-conservation failures, source
ordering failures, and bounded manual samples.

## Known Limitations

This parser fix does not prove retrieval-quality improvement by itself. Chunk
counts can change once a future v3 pilot is rebuilt, but no Qdrant collection,
BM25 sidecar, embedding cache, or runtime retrieval profile is changed here.

Future work must rebuild and validate a v3 pilot in a separate task before any
benchmark or frontend claim is made.
