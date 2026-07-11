# NSoud Retrieval Risk Investigation — 2026-07-10

Read-only investigation of `nsoud-qa-007` and `nsoud-qa-010` after legal answer-eval diagnostics repair.

## Scope and Safety

- Collection inspected: `nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1`
- BM25 sidecar inspected: `storage/rag/bm25/nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1.sqlite`
- Qdrant writes: `no`
- Alias changes: `no`
- Retrieval logic changes: `no`
- Ingest/rebuild: `no`
- LLM/DeepSeek calls: `no`

## Global Finding

- The BM25 sidecar exists and has `1862` rows, but `document_id` and `source_document_id` are blank for all `1862/1862` rows.
- This means BM25-backed hits in frozen retrieval artifacts can lose document provenance even when the corresponding chunk in Qdrant has a valid `document_id`.

## nsoud-qa-007

- Question: `Jak Nejvyšší soud posuzuje dovolací důvod podle § 265b tr. ř.?`
- Expected gold source: `ECLI:CZ:NS:2025:5.TDO.1086.2024.1`
- Qdrant expected-source presence: `yes`
  Expected source exists under top-level `document_id` with `12` chunks.
- BM25 sidecar expected-source presence by `document_id` / `source_document_id` / ECLI: `no`

### Frozen Baseline Top-10

| rank | chunk_id | frozen document_id | resolved Qdrant document_id | section_type | note |
| --- | --- | --- | --- | --- | --- |
| 1 | `735` | - | `ECLI:CZ:NS:2025:5.TDO.1086.2024.1` | `reasoning` | expected source already present in frozen top-10 once Qdrant provenance is resolved |
| 2 | `131` | `ECLI:CZ:NS:2024:11.TDO.765.2024.1` | same | `reasoning` | alternate criminal dovolani decision |
| 3 | `991` | - | `ECLI:CZ:NS:2025:4.TDO.1137.2024.1` | `reasoning` | alternate criminal dovolani decision |
| 4 | `1789` | `ECLI:CZ:NS:2025:3.TDO.53.2025.1` | same | `reasoning` | alternate criminal dovolani decision |
| 5 | `1834` | - | `ECLI:CZ:NS:2025:11.TDO.75.2025.1` | `reasoning` | alternate criminal dovolani decision |
| 6 | `1728` | `ECLI:CZ:NS:2025:6.TDO.21.2025.1` | same | `reasoning` | alternate criminal dovolani decision |
| 7 | `930` | - | `ECLI:CZ:NS:2025:3.TDO.1120.2024.1` | `reasoning` | alternate criminal dovolani decision |
| 8 | `1508` | `ECLI:CZ:NS:2025:11.TDO.875.2024.1` | same | `reasoning` | alternate criminal dovolani decision |
| 9 | `343` | - | `ECLI:CZ:NS:2024:8.TDO.760.2024.1` | `reasoning` | alternate criminal dovolani decision |
| 10 | `508` | `ECLI:CZ:NS:2024:6.TDO.976.2024.1` | same | `reasoning` | alternate criminal dovolani decision |

### Read-Only Top-50 Checks

- Hybrid top-50 rank of expected source: `4`
- Dense-only top-50 rank of expected source: `13`
- BM25-only top-50 rank of expected source: `10`

### Conclusion

- Classification: `BM25_sidecar_missing_expected_source`
- This is **not** a real retrieval miss.
- The frozen baseline already contains the expected source at rank 1 as chunk `735`, but the artifact loses the source identifier because the BM25 sidecar stores blank provenance fields.
- Dense ranking alone is weaker (`rank 13`), but hybrid and BM25 still retrieve the expected source within top-10/top-4, so the main issue is provenance/export visibility, not absence from the collection.

### Recommended Action

- Regenerate the NSoud BM25 sidecar with populated provenance fields (`document_id` / `source_document_id`) or enrich diagnostics from Qdrant by `chunk_id`.
- Rerun the failed-case diagnostics after the provenance/export fix.
- No re-ingest is needed.

## nsoud-qa-010

- Question: `Jaký je rozdíl mezi odmítnutím a zamítnutím dovolání?`
- Expected gold source: `ECLI:CZ:NS:2025:29.NSCR.1.2025.1`
- Qdrant expected-source presence: `yes`
  Expected source exists under top-level `document_id` with `7` chunks.
- BM25 sidecar expected-source presence by `document_id` / `source_document_id` / ECLI: `no`

### Frozen Baseline Top-10

| rank | chunk_id | frozen document_id | resolved Qdrant document_id | section_type | note |
| --- | --- | --- | --- | --- | --- |
| 1 | `1643` | `ECLI:CZ:NS:2025:29.NSCR.1.2025.1` | same | `operative_part` | expected source, but only operative boilerplate |
| 2 | `884` | - | `ECLI:CZ:NS:2025:29.ICDO.3.2025.1` | `reasoning` | civil reasoning, not the requested legal distinction |
| 3 | `1533` | `ECLI:CZ:NS:2025:29.NSCR.70.2024.1` | same | `operative_part` | boilerplate |
| 4 | `632` | - | `ECLI:CZ:NS:2024:25.CDO.3217.2023.1` | `reasoning` | incidental `zamít` wording in another context |
| 5 | `1427` | - | `ECLI:CZ:NS:2025:5.TDO.1071.2024.1` | `reasoning` | criminal dovolani context |
| 6 | `497` | `ECLI:CZ:NS:2025:27.CDO.1921.2024.1` | same | `operative_part` | boilerplate |
| 7 | `1176` | - | `ECLI:CZ:NS:2025:11.TDO.1127.2024.1` | `reasoning` | incidental `zamítly` wording |
| 8 | `642` | `ECLI:CZ:NS:2025:22.CDO.3556.2024.1` | same | `operative_part` | boilerplate |
| 9 | `1724` | - | `ECLI:CZ:NS:2025:6.TDO.21.2025.1` | `reasoning` | criminal dovolani context |
| 10 | `26` | `ECLI:CZ:NS:2024:26.CDO.125.2024.1` | same | `operative_part` | boilerplate |

### Expected Source Content Check

- The expected document is present in Qdrant.
- Retrieved expected chunk in frozen baseline is `1643` / `operative_part`: `takto: Dovolání se odmítá.`
- Expected document reasoning chunks mention `odmítnutí` / `odmítl`, but do **not** provide the requested explicit distinction between `odmítnutí` and `zamítnutí` dovolání.

### Read-Only Top-50 Checks

- Hybrid top-50 rank of expected source: `4`
- Dense-only top-50 rank of expected source: `4`
- BM25-only top-50 rank of expected source: `not found in top-50`
- Hybrid top-50 is heavily dominated by `operative_part` boilerplate hits of the form `Dovolání se odmítá`.
- The first explicit `Dovolání ... se zamítá` operative hit appears only at `hybrid rank 48`, and it is from a different document.

### Conclusion

- Primary classification: `dataset should be reformulated`
- Secondary factors:
  `boilerplate chunk ranking too high`
  `source support genuinely weak`
- This is not a collection-miss problem.
- The benchmark question asks for a doctrinal distinction, but the current expected source is mainly an `odmítnutí` decision and the retrieval surface is dominated by short operative boilerplate chunks.

### Recommended Action

- Reformulate `nsoud-qa-010` into a narrower, source-grounded question, or replace it with a paired-source / doctrine-backed benchmark item.
- If the item stays, mark it as unsupported by the current single-document gold and do not treat it as a retrieval miss.
- No re-ingest is needed.

## Final Assessment

- `nsoud-qa-007` should be removed from the “real retrieval miss” bucket; the expected source is retrievable and already visible in the frozen baseline once chunk provenance is resolved from Qdrant.
- `nsoud-qa-010` remains a real benchmark-quality risk, but the risk is question/source design plus boilerplate ranking, not missing collection coverage.
