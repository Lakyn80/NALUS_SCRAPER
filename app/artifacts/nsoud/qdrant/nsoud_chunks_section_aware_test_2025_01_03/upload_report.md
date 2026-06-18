# NSoud Qdrant Upload Report

- Status: **PASS**
- Target collection: `nsoud_chunks_section_aware_test_2025_01_03`
- Qdrant URL: `http://qdrant:6333`
- Artifact directory: `/app/app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03`
- Input path: `/app/app/artifacts/nsoud/rag_ready/nsoud_embeddings_2025_01_03.parquet`
- Input rows: **1862**
- Uploaded points: **1862**
- Final collection point count: **1862**
- Vector size: **768**
- Distance: **Cosine**
- Duplicate point_id count: **0**
- Duplicate chunk_id count: **0**
- Missing embedding count: **0**
- Empty text count: **0**
- Missing required metadata count: **0**
- Rows outside `document_section_aware`: **0**
- Old collection untouched: **yes (1785 -> 1785)**
- Manifest path: `/app/app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/upload_manifest.json`
- Report path: `/app/app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/upload_report.md`

## Preserved Payload Metadata

- `point_id`
- `chunk_id`
- `text`
- `document_id`
- `section_id`
- `section_type`
- `section_index`
- `chunk_index`
- `chunk_index_in_section`
- `total_chunks_in_document`
- `total_chunks_in_section`
- `previous_chunk_id`
- `next_chunk_id`
- `previous_section_chunk_id`
- `next_section_chunk_id`
- `structure_confidence`
- `structure_status`
- `structure_needs_review`
- `section_source`
- `chunking_strategy`

## Warnings

- None.

## Errors

- None.

## Changed Files

- `app/nsoud/upload_to_qdrant.py`
- `/app/app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/upload_manifest.json`
- `/app/app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/upload_report.md`
