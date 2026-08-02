# Benchmark correction: `uq_028`

- Date: `2026-08-02`
- Query: `kdy Ústavní soud odmítne ústavní stížnost jako nepřípustnou`
- Source canary: `hybrid_canary_uq028_header_fix.json` + `document_reviews/uq_028_full_documents.md`

## Change

| Document | Before | After |
|---|---|---|
| `ECLI:CZ:US:2020:3.US.2419.20.1` | `hard_negative` (+ related_only) | `strongly_relevant` (+ materially) |
| `ECLI:CZ:US:2021:3.US.931.21.1` | unlabeled / other | `strongly_relevant` |
| `ECLI:CZ:US:2024:1.US.2639.24.1` | unlabeled / other | `strongly_relevant` |
| `ECLI:CZ:US:2020:3.US.2242.20.1` | unlabeled / other | `materially_relevant` |
| `ECLI:CZ:US:2024:2.US.262.24.2` | unlabeled / other | `materially_relevant` |
| `ECLI:CZ:US:2020:3.US.2302.20.1` | unlabeled / other | `related_only` (contrast: complaint held admissible) |
| Remaining HN | `2.US.3645.25.1`, `3.US.2419.20.1` | only `2.US.3645.25.1` |

## Why `3.US.2419.20.1` is not a hard negative

Paragraphs 9–15 directly explain subsidiarity, exhaustion of remedies, zmatečnost overlap, and refusal under § 43(1)(e). Approving it as `exact_match` was content-correct; the FA metric was benchmark error.

## Runtime follow-up (not a further gate tighten)

Rank 2 (`1.US.2639.24.1`) failed with `final_rejection_code=not_proven` because `document_type_1` was missing while fast still advertised `verified_match` / `exact_match`, so thinking escalation never ran. Compact expand now demotes incomplete hard PROVEN coverage to `insufficient_evidence` so thinking can recover.
