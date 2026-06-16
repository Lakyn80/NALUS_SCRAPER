# NSoud Chunk Validation

- Input: `app/artifacts/nsoud/rag_ready/nsoud_documents_2025_01_03.parquet`
- Output Parquet: `app/artifacts/nsoud/rag_ready/nsoud_chunks_2025_01_03.parquet`
- Output JSONL: `app/artifacts/nsoud/rag_ready/nsoud_chunks_2025_01_03.jsonl`
- Structure report: `app/artifacts/nsoud/rag_ready/nsoud_structure_patterns_2025_01_03.md`
- Validation status: **WARN**
- Total documents: **150**
- Total chunks: **1862**

## Status
- Overlong NS paragraphs were preserved as standalone chunks.

## Chunk Metrics
- chunks per document min: 4
- chunks per document max: 47
- chunks per document avg: 12.41
- chunk_text_length min: 25
- chunk_text_length max: 8155
- chunk_text_length avg: 1384.16
- empty chunk count: 0
- duplicate chunk_id count: 0
- documents with zero chunks: 0
- overlong NS paragraph chunk count: 13

## Reconstruction Validation
- paragraph preservation passed/failed: 150/0
- document reconstruction passed/failed: 150/0
- section reconstruction passed/failed: 729/0
- unresolved boundary issue count: 0

## Structure Confidence Summary
- strong structure count: 128
- medium structure count: 22
- weak structure count: 0
- needs_review count: 0

## Marker Coverage

| Marker | Document Count |
| --- | ---: |
| ROZSUDEK | 15 |
| USNESENÍ | 135 |
| STANOVISKO | 0 |
| JMÉNEM REPUBLIKY | 15 |
| Nejvyšší soud rozhodl | 108 |
| takto: | 135 |
| Odůvodnění: | 127 |
| O d ů v o d n ě n í: | 17 |
| Poučení: | 136 |
| P o u č e n í: | 14 |
| V Brně dne | 150 |
| předseda senátu | 128 |
| předsedkyně senátu | 24 |

## Documents With Zero Chunks
- none

## Source Distribution

| Value | Count |
| --- | ---: |
| nsoud | 1862 |

## Document Type Distribution

| Value | Count |
| --- | ---: |
| USNESENÍ | 1664 |
| ROZSUDEK | 198 |

## Legal Area Distribution

| Value | Count |
| --- | ---: |
| civil | 981 |
| criminal | 866 |
| <missing> | 15 |

## Section Type Distribution

| Value | Count |
| --- | ---: |
| reasoning | 1194 |
| signature | 192 |
| operative_part | 165 |
| appeal_instruction | 161 |
| header | 150 |

## NS Section Hint Distribution

| Value | Count |
| --- | ---: |
| oduvodneni | 1194 |
| closing | 192 |
| vyrok | 165 |
| pouceni | 161 |
| header | 150 |
