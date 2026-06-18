# NSoud Qdrant Smoke Test

- Status: **PASS**
- Target collection: `nsoud_chunks_section_aware_test_2025_01_03`
- Expected point count: **1862**
- Actual point count: **1862**
- Vector size: **768**
- Tests passed: **7**
- Tests failed: **0**
- Old collection count before/after: **1785 -> 1785**
- Output report path: `/app/app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/smoke_test.md`

## Selected Cases

| test_name | row_index | chunk_id | section_type | structure_status |
| --- | ---: | --- | --- | --- |
| first_row | 0 | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0000 | header | strong |
| middle_row | 931 | ECLI:CZ:NS:2025:20.NCU.199.2024.1__chunk_0001 | operative_part | strong |
| last_row | 1861 | ECLI:CZ:NS:2025:8.TDO.1119.2024.1__chunk_0009 | signature | strong |
| section_type_operative_part | 1 | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0001 | operative_part | strong |
| section_type_reasoning | 2 | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0002 | reasoning | strong |
| section_type_appeal_instruction | 18 | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0018 | appeal_instruction | strong |
| structure_status_medium | 40 | ECLI:CZ:NS:2024:11.TVO.22.2024.1__chunk_0000 | header | medium |

## Search Results

| test_name | query_chunk_id | top_hit_chunk_id | original_found_in_top_5 | top_hit_score |
| --- | --- | --- | --- | ---: |
| first_row | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0000 | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0000 | true | 1.000000 |
| middle_row | ECLI:CZ:NS:2025:20.NCU.199.2024.1__chunk_0001 | ECLI:CZ:NS:2025:20.NCU.199.2024.1__chunk_0001 | true | 1.000000 |
| last_row | ECLI:CZ:NS:2025:8.TDO.1119.2024.1__chunk_0009 | ECLI:CZ:NS:2025:8.TDO.1119.2024.1__chunk_0009 | true | 1.000000 |
| section_type_operative_part | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0001 | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0001 | true | 1.000000 |
| section_type_reasoning | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0002 | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0002 | true | 1.000000 |
| section_type_appeal_instruction | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0018 | ECLI:CZ:NS:2025:11.TDO.1127.2024.1__chunk_0018 | true | 1.000000 |
| structure_status_medium | ECLI:CZ:NS:2024:11.TVO.22.2024.1__chunk_0000 | ECLI:CZ:NS:2024:11.TVO.22.2024.1__chunk_0000 | true | 1.000000 |

## Warnings
- None.

## Errors
- None.

## Changed Files
- `app/nsoud/smoke_test_qdrant_search.py`
- `/app/app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/smoke_test.md`
