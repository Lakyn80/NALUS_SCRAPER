# Previous Five-Year Build Audit - 2026-07-31

Status: `AUDIT_COMPLETE`

Classification: `PARTIALLY_SUPPORTED`

The previous approximately three-hour claim is supported only for a 600-document newest-first BGE-M3 slice. It is not supported for a completed five-year build. The prior five-year candidate is evidenced as an incomplete run with an `in_progress` checkpoint.

## Evidence

- Previous script: `scripts/build_usoud_bge_m3_candidate.py`
- Previous three-hour command:
  `python scripts/build_usoud_bge_m3_candidate.py --mode full --limit 600 --collection-name nalus_us_bge_m3_mvp_recent_3h_20260709 --source-manifest batches/manifest.json --output-dir artifacts/nalus_update/usoud_bge_m3_mvp_recent_3h_20260709 --no-alias-update --ingest-slice mvp_recent_3h --decision-date-to 2026-07-09 --newest-first --embedding-batch-size 16 --full-record-batch-size 50 --execute --recreate-full-collection`
- Previous three-hour manifest: `artifacts/nalus_update/usoud_bge_m3_mvp_recent_3h_20260709/execute_summary.json`
- Previous three-hour logs: `NOT_FOUND`
- Previous three-hour Qdrant collection: `nalus_us_bge_m3_mvp_recent_3h_20260709`
- Previous three-hour Qdrant points: `4,980`
- Previous three-hour durable BM25 sidecar: `NOT_FOUND`
- Previous three-hour durable BM25 rows: `NOT_FOUND`
- Previous three-hour date range: `decision_date_to=2026-07-09`, `decision_date_from=null`, `newest_first=true`, `limit=600`
- Previous three-hour documents: `600`
- Previous three-hour chunks: `4,980`
- Previous three-hour chunks/document: `8.3`
- Previous three-hour total duration: approximately `2h41m`
- Previous three-hour chunks/second: approximately `0.513`

The five-year candidate used the same builder with the planned date window `2021-07-08` through `2026-07-08`. Its dry run selected `18,062` records and estimated `155,414` chunks, about `8.604` chunks/document. The execute checkpoint remained `in_progress` at record `650`, with `8,335` inserted points. No completed execute summary for that five-year run was found.

Relevant five-year files:

- Dry run: `artifacts/nalus_update/usoud_bge_m3_mvp_5y_20260708/dry_run_summary.json`
- Checkpoint: `artifacts/nalus_update/usoud_bge_m3_mvp_5y_20260708/execute_checkpoint.json`
- Log: `artifacts/nalus_update/usoud_bge_m3_mvp_5y_20260708/logs/mvp_5y_20260708_224835.log`
- Partial collection: `nalus_us_bge_m3_mvp_5y_20260708`
- Partial Qdrant points: `8,335`

The later combined collection `nalus_us_bge_m3_rag_combined_20260709` merged `8,335` points from the partial five-year collection and `4,980` points from the 600-document recent collection, for `13,315` total points. Its durable sidecar is `storage/rag/bm25/nalus_us_bge_m3_rag_combined_20260709.sqlite` with `13,315` rows.

## Old Versus Current

Previous model: `BAAI/bge-m3`

Previous dimension: `1024`

Previous chunking: simple text chunks, approximately 1400 characters with 35-word overlap, flat payload metadata.

Previous embedding batch: `16`

Previous Qdrant batch: `64`

Previous BM25 strategy: candidate-run BM25 validation, with durable BM25 built later from merged Qdrant points.

Previous CPU threads: `UNKNOWN`

Previous build full or incremental: the confirmed approximately three-hour run was an MVP slice, not a full five-year build. The five-year run was a full-mode candidate but incomplete.

Previous embeddings reused: no for the 600-document execute run; the combined collection reused existing points by merge.

Current model loaded once: per builder process yes; the 100-document checkpoint/resume sample loaded once per invocation, twice overall.

Current embedding batch: `64`

Current Qdrant batch: `64`

Current BM25 strategy: SQLite `bm25_chunks` append per document batch with full metadata JSON and Qdrant/BM25 identity validation.

Current chunks/second: approximately `1.197`

## Bottlenecks

- Parsing bottleneck: secondary/possible. Legal Retrieval v2 parsing and audit metadata are richer than old chunking.
- Embedding bottleneck: primary CPU bottleneck, because BGE-M3 CPU encoding scales with chunk count.
- Qdrant bottleneck: secondary. Upserts are batched and wait for completion, but available evidence does not show Qdrant as dominant.
- BM25 bottleneck: low/secondary. Legal v2 writes richer metadata, but this is not the leading cost in the sample.
- Validation/checkpoint bottleneck: secondary. Checkpoint and identity validation add safety cost and may grow with collection size.
- Docker resource bottleneck: CPU-only Docker; CUDA/GPU unavailable by policy. CPU thread settings are `UNKNOWN`.

## Root Causes

1. The roughly three-hour run was a 600-document newest-first slice, not a completed five-year build.
2. The current target is larger: `21,776` documents and about `394k` chunks versus `600` documents and `4,980` chunks for the confirmed old run.
3. Legal Retrieval v2 creates paragraph-aware hierarchical chunks with evidence metadata, about `18.1` chunks/document in the sample versus about `8.3` to `8.6` chunks/document in the old builder.

## Reuse Assessment

- Old Qdrant points reusable: no direct reuse.
- Old vectors reusable: no.
- Old parsing reusable: partially, for source inventory/date-filter/source document input evidence only.
- Reusable documents: `UNKNOWN`
- Reusable chunks: `0` direct Legal Retrieval v2 chunks.
- Reuse reason: the BGE-M3 model and vector dimension match, but chunk text, boundaries, identifiers, payload schema, paragraph metadata, and fingerprints do not.

## Recommendation

Selected recommendation: `E. RECONSTRUCT_EXPECTATION`

Expected optimized duration: `NOT_APPLICABLE`; no optimization was implemented in this audit. The old three-hour evidence does not imply that a full six-year Legal Retrieval v2 CPU build can complete in three hours.

Confidence: `HIGH` for the three-hour/five-year discrepancy; `MEDIUM` for bottleneck attribution.

Exact next implementation task: create a read-only rebaseline estimator that separates document count, chunk density, embedding throughput, Qdrant write cost, BM25 write cost, and validation cost before any full build is started.

## Audit Side Effects

- Qdrant modified: `false`
- BM25 modified: `false`
- Aliases modified: `false`
- Embedding build run: `false`
- GPU used: `false`
- CUDA used: `false`
- Downloads: `false`
- Tests run: `false`
- Runtime source code modified: `false`
- Files created:
  - `artifacts/evaluation_quality/previous_five_year_build_audit_20260731.md`
  - `artifacts/evaluation_quality/previous_five_year_build_audit_20260731.json`
- Tracked files modified by this audit:
  - `PROJECT_PROGRESS.md`
