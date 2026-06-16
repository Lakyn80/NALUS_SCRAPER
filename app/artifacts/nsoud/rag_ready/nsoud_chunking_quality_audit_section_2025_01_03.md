# NSoud Chunking Quality Audit Section

## Why This Audit Exists

This audit exists because section-aware legal chunking must be validated directly at the chunking layer, not only indirectly through downstream retrieval or evaluation outputs. A legal RAG pipeline can appear operational while still carrying chunk boundary defects, reconstruction defects, or section metadata defects that later degrade recall, precision, and citation reliability.

## Goal

The goal of this step is 100% validation coverage across all NS batch artifacts:

- every document
- every detected legal section
- every produced chunk

The audit is exhaustive. It is not limited to long-document previews or sampled spot checks.

## Why Embedding/Search Validation Alone Is Not Enough

Embedding quality and search relevance are downstream signals. They do not prove that:

- document text can be reconstructed from chunks
- section text can be reconstructed from chunks
- section boundaries are preserved correctly
- chunk metadata is complete and internally consistent
- previous/next chunk links are valid
- no cross-section merge happened inside a single chunk

If chunking is technically wrong, later stages can inherit hidden defects even when retrieval tests still look acceptable on a small sample.

## Technical Reconstruction vs. Legal Chunk Quality

Technical reconstruction answers whether the chunk set can reproduce the original source text exactly and whether section spans remain internally consistent.

Legal chunk quality is stricter. It asks whether the produced chunks respect meaningful legal structure, especially boundaries such as:

- header
- operative part (`takto`)
- reasoning (`odůvodnění`)
- appeal instruction (`poučení`)
- signature / closing

A chunking result can pass raw text reconstruction but still be legally poor if it merges different legal sections into one chunk.

## Rules Checked By This Audit

The audit checks:

- document-level reconstruction
- section-level reconstruction
- chunk metadata completeness
- continuous document chunk index sequences
- continuous section chunk index sequences
- valid previous/next neighbor links
- duplicate `chunk_id`
- empty `chunk_text`
- missing required metadata
- weak structure status
- `needs_review`
- known section boundary violations
- overlong standalone chunks

## How To Interpret The Result

- `FAIL` means there is a hard technical or structural defect in the current chunk set.
- `WARN` means the chunk set is technically valid, but there are still non-blocking quality signals such as overlong standalone chunks or medium structure documents.
- `PASS` means all required checks passed and no warning conditions remain.

For the current NS batch, a `WARN` result is still acceptable if all reconstruction, metadata, sequence, neighbor-link, and legal boundary checks pass and only overlong chunks and/or medium-structure documents remain.

## Next Step After A Successful Audit

Once this audit confirms that the current section-aware chunk set is technically sound, the next step is to use the validated section metadata and boundary guarantees for the next legal-aware retrieval stages. That can include retrieval evaluation, search tuning, and later chunking refinements, but only after the chunk layer itself is proven reliable.
