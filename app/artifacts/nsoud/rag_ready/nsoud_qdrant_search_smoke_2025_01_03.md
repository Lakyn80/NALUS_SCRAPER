# NSoud Qdrant Search Smoke Test

- Status: **PASS**
- Collection name: `nsoud_chunks_test_2025_01_03`
- Expected point count: **1785**
- Actual point count: **1785**
- Vector size: **768**
- Number of smoke tests: **6**
- Tests passed: **6**
- Tests failed: **0**

## Results

| test_name | query_chunk_id | top_hit_chunk_id | original_found_in_top_5 | score | case_number | document_type | legal_area |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| first_row | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0000 | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0000 | true | 1.000000 | 11 Tdo 679/2024 | USNESENÍ | criminal |
| middle_row | ECLI:CZ:NS:2025:20.NCU.165.2024.1__chunk_0002 | ECLI:CZ:NS:2025:20.NCU.165.2024.1__chunk_0002 | true | 1.000000 | 20 Ncu 165/2024 | ROZSUDEK | - |
| last_row | ECLI:CZ:NS:2025:8.TDO.1119.2024.1__chunk_0008 | ECLI:CZ:NS:2025:8.TDO.1119.2024.1__chunk_0008 | true | 1.000000 | 8 Tdo 1119/2024 | USNESENÍ | criminal |
| document_type_usneseni | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0000 | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0000 | true | 1.000000 | 11 Tdo 679/2024 | USNESENÍ | criminal |
| legal_area_criminal | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0000 | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0000 | true | 1.000000 | 11 Tdo 679/2024 | USNESENÍ | criminal |
| legal_area_civil | ECLI:CZ:NS:2024:20.CDO.2839.2024.1__chunk_0000 | ECLI:CZ:NS:2024:20.CDO.2839.2024.1__chunk_0000 | true | 1.000000 | 20 Cdo 2839/2024 | USNESENÍ | civil |

## Warnings
- None.

## Errors
- None.
