# NSoud Chunking Readiness

- Documents input: `app\artifacts\nsoud\rag_ready\nsoud_documents_2025_01_03.parquet`
- Chunks input: `app\artifacts\nsoud\rag_ready\nsoud_chunks_2025_01_03.parquet`
- Audit input: `app\artifacts\nsoud\rag_ready\nsoud_overlong_chunks_audit_2025_01_03.csv`
- Final status: **WARN**
- RAG_READY: **true**
- Final total documents: **150**
- Final total chunks: **1785**
- Final overlong chunk count: **14**
- Final suspicious possible missed boundary count: **12**
- Paragraph preservation: **150 passed / 0 failed**
- Empty chunk count: **0**
- Duplicate chunk_id count: **0**
- Documents with zero chunks: **0**
- Unresolved boundary issue count: **0**

## Exact Changes Made To NS Boundary Detection
- Replaced generic marker scanning with isolated NS-specific boundary helpers.
- Added deterministic numbered paragraph detection for `1.` to `200.` with context checks.
- Added deterministic numbered slash detection for `1/` to `200/` with context checks.
- Added deterministic bracketed paragraph detection for `[1]` to `[200]` with context checks.
- Added deterministic parenthesized enumeration detection for `1)` to `200)` after list/sentence separators.
- Extended roman section detection from `I.` to `X.` with false-positive guards for citation patterns such as `I. ÚS`.
- Added NS section label detection for `takto:`, `Odůvodnění:`, `Poučení:`, spaced `P o u č e n í:`, and `V Brně dne`.
- Preserved section labels and markers inside the paragraph they introduce.
- Kept paragraph-preservation validation strict across all 150 documents.

## Remaining Overlong Chunks

| Chunk ID | Case Number | Reason | Audit Classification | Length | Explanation |
| --- | --- | --- | --- | ---: | --- |
| ECLI:CZ:NS:2024:20.CDO.2839.2024.1__chunk_0002 | 20 Cdo 2839/2024 | unsafe_to_split | possible missed roman section boundary | 4245 | Audit heuristics detected inline marker-like text, but the deterministic NS detector found no safe additional structural boundary beyond the current paragraph start. |
| ECLI:CZ:NS:2024:5.TDO.318.2024.1__chunk_0019 | 5 Tdo 318/2024 | unsafe_to_split | possible missed roman section boundary | 5540 | Audit heuristics detected inline marker-like text, but the deterministic NS detector found no safe additional structural boundary beyond the current paragraph start. |
| ECLI:CZ:NS:2025:20.CDO.13.2025.1__chunk_0002 | 20 Cdo 13/2025 | unsafe_to_split | possible missed roman section boundary | 5634 | Audit heuristics detected inline marker-like text, but the deterministic NS detector found no safe additional structural boundary beyond the current paragraph start. |
| ECLI:CZ:NS:2025:20.CDO.15.2025.1__chunk_0002 | 20 Cdo 15/2025 | unsafe_to_split | possible missed roman section boundary | 5634 | Audit heuristics detected inline marker-like text, but the deterministic NS detector found no safe additional structural boundary beyond the current paragraph start. |
| ECLI:CZ:NS:2024:3.TDO.650.2024.1__chunk_0006 | 3 Tdo 650/2024 | unsafe_to_split | possible missed numbered paragraph boundary | 6056 | Audit heuristics detected inline marker-like text, but the deterministic NS detector found no safe additional structural boundary beyond the current paragraph start. |
| ECLI:CZ:NS:2025:29.ICDO.131.2023.1__chunk_0005 | 29 ICdo 131/2023 | unsafe_to_split | possible missed numbered paragraph boundary | 5305 | Audit heuristics detected inline marker-like text, but the deterministic NS detector found no safe additional structural boundary beyond the current paragraph start. |
| ECLI:CZ:NS:2025:33.CDO.889.2024.1__chunk_0003 | 33 Cdo 889/2024 | real_long_paragraph | real long paragraph | 5293 | No internal marker-like pattern was detected by the audit heuristics. |
| ECLI:CZ:NS:2025:20.CDO.3450.2024.1__chunk_0004 | 20 Cdo 3450/2024 | unsafe_to_split | possible missed roman section boundary | 5376 | Audit heuristics detected inline marker-like text, but the deterministic NS detector found no safe additional structural boundary beyond the current paragraph start. |
| ECLI:CZ:NS:2025:5.TDO.1128.2024.1__chunk_0021 | 5 Tdo 1128/2024 | unsafe_to_split | possible missed numbered paragraph boundary | 4763 | Audit heuristics detected inline marker-like text, but the deterministic NS detector found no safe additional structural boundary beyond the current paragraph start. |
| ECLI:CZ:NS:2025:29.CDO.275.2024.1__chunk_0003 | 29 Cdo 275/2024 | unsafe_to_split | possible missed roman section boundary | 4161 | Audit heuristics detected inline marker-like text, but the deterministic NS detector found no safe additional structural boundary beyond the current paragraph start. |
| ECLI:CZ:NS:2025:29.NSCR.70.2024.1__chunk_0001 | 29 NSCR 70/2024 | unsafe_to_split | possible missed roman section boundary | 8154 | Audit heuristics detected inline marker-like text, but the deterministic NS detector found no safe additional structural boundary beyond the current paragraph start. |
| ECLI:CZ:NS:2025:20.CDO.30.2025.1__chunk_0002 | 20 Cdo 30/2025 | unsafe_to_split | possible missed numbered paragraph boundary | 4443 | Audit heuristics detected inline marker-like text, but the deterministic NS detector found no safe additional structural boundary beyond the current paragraph start. |
| ECLI:CZ:NS:2025:8.TDO.1119.2024.1__chunk_0002 | 8 Tdo 1119/2024 | unsafe_to_split | possible missed numbered paragraph boundary | 4749 | Audit heuristics detected inline marker-like text, but the deterministic NS detector found no safe additional structural boundary beyond the current paragraph start. |
| ECLI:CZ:NS:2025:8.TDO.1119.2024.1__chunk_0006 | 8 Tdo 1119/2024 | real_long_paragraph | real long paragraph | 5587 | No internal marker-like pattern was detected by the audit heuristics. |

## Embeddings Readiness
- Ready for embeddings: true
- Blocking issue present: false
