# NSoud Chunk Validation

- Input: `app\artifacts\nsoud\rag_ready\nsoud_documents_2025_01_03.parquet`
- Output Parquet: `app\artifacts\nsoud\rag_ready\nsoud_chunks_2025_01_03.parquet`
- Structure report: `app\artifacts\nsoud\rag_ready\nsoud_structure_patterns_2025_01_03.md`
- Validation status: **WARN**
- Total documents: **150**
- Total chunks: **1785**

## Status
- Overlong NS paragraphs were preserved as standalone chunks.

## Chunk Metrics
- chunks per document min: 4
- chunks per document max: 47
- chunks per document avg: 11.90
- chunk_text_length min: 13
- chunk_text_length max: 8154
- chunk_text_length avg: 1442.95
- empty chunk count: 0
- duplicate chunk_id count: 0
- documents with zero chunks: 0
- overlong NS paragraph chunk count: 14
- max overlong paragraph length: 8154

## Paragraph Preservation Check
- documents passed: 150
- documents failed: 0

## Section Marker Coverage

| Marker | Document Count |
| --- | ---: |
| takto: | 133 |
| Odůvodnění: | 127 |
| I. | 126 |
| II. | 120 |
| III. | 82 |
| IV. | 64 |
| V. | 80 |
| Poučení: | 136 |
| V Brně dne | 150 |

## Documents With Zero Chunks
- none

## Source Distribution

| Value | Count |
| --- | ---: |
| nsoud | 1785 |

## Document Type Distribution

| Value | Count |
| --- | ---: |
| USNESENÍ | 1592 |
| ROZSUDEK | 193 |

## Legal Area Distribution

| Value | Count |
| --- | ---: |
| civil | 932 |
| criminal | 840 |
| <missing> | 13 |
