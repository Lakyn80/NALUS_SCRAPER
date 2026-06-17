# NSoud Document Structure Analysis

- Status: **PASS**
- Input path: `app\artifacts\nsoud\rag_ready\nsoud_documents_2025_01_03.parquet`
- Total documents: **150**

## Metadata Distribution

- Document type distribution: {"USNESENÍ": 135, "ROZSUDEK": 15}
- Legal area distribution: {"civil": 107, "criminal": 40, "": 3}
- Missing ecli count: **0**
- Missing decision_date count: **0**
- Missing publication_date count: **0**
- Missing legal_area count: **3**
- Full text length min/max/avg: **2091 / 86652 / 17181.99**

## Marker Coverage

| marker | count | pct |
| --- | ---: | ---: |
| ROZSUDEK | 15 | 10.00 |
| USNESENÍ | 135 | 90.00 |
| STANOVISKO | 0 | 0.00 |
| JMÉNEM REPUBLIKY | 15 | 10.00 |
| Nejvyšší soud rozhodl | 108 | 72.00 |
| operative_part | 135 | 90.00 |
| Odůvodnění: | 127 | 84.67 |
| O d ů v o d n ě n í: | 17 | 11.33 |
| Poučení: | 136 | 90.67 |
| P o u č e n í: | 14 | 9.33 |
| V Brně dne | 150 | 100.00 |
| předseda senátu | 128 | 85.33 |
| předsedkyně senátu | 24 | 16.00 |
| roman_sections_any | 104 | 69.33 |
| numbered_paragraphs_any | 127 | 84.67 |

## Structure Confidence Summary

- Strong structure count: **128**
- Medium structure count: **22**
- Weak structure count: **0**
- Needs review count: **0**
- Average structure confidence: **0.912**

## Needs Review

| document_id | case_number | document_type | legal_area | confidence | status | section_order |
| --- | --- | --- | --- | ---: | --- | --- |
| - | - | - | - | - | - | - |

## Examples Of Detected Section Order

| order | count | pct |
| --- | ---: | ---: |
| header > operative_part > oduvodneni > pouceni > closing/signature | 119 | 79.33 |
| header > oduvodneni > pouceni > closing/signature | 14 | 9.33 |
| header > operative_part > oduvodneni > closing/signature > pouceni | 10 | 6.67 |
| header > operative_part > pouceni > closing/signature | 6 | 4.00 |
| header > oduvodneni > closing/signature > pouceni | 1 | 0.67 |

## Most Common Marker Combinations

| combination | count | pct |
| --- | ---: | ---: |
| DTYPE + NSR + OPER + ODUV + POUC + CLOSE + ROMAN + NUM | 60 | 40.00 |
| DTYPE + NSR + OPER + ODUV + POUC + CLOSE + NUM | 24 | 16.00 |
| DTYPE + OPER + ODUV + POUC + CLOSE + ROMAN + NUM | 15 | 10.00 |
| DTYPE + NSR + ODUV + POUC + CLOSE + ROMAN + NUM | 9 | 6.00 |
| DTYPE + NSR + OPER + ODUV + POUC + CLOSE + ROMAN | 8 | 5.33 |
| DTYPE + OPER + ODUV + POUC + CLOSE | 8 | 5.33 |
| DTYPE + OPER + ODUV + POUC + CLOSE + NUM | 8 | 5.33 |
| DTYPE + OPER + POUC + CLOSE + ROMAN + NUM | 4 | 2.67 |
| DTYPE + OPER + ODUV + POUC + CLOSE + ROMAN | 4 | 2.67 |
| DTYPE + NSR + OPER + POUC + CLOSE + ROMAN + NUM | 2 | 1.33 |
| DTYPE + NSR + ODUV + POUC + CLOSE + NUM | 2 | 1.33 |
| DTYPE + ODUV + POUC + CLOSE + NUM | 2 | 1.33 |
| DTYPE + NSR + OPER + ODUV + POUC + CLOSE | 2 | 1.33 |
| DTYPE + ODUV + POUC + CLOSE + ROMAN + NUM | 1 | 0.67 |
| DTYPE + NSR + ODUV + POUC + CLOSE + ROMAN | 1 | 0.67 |

## Recommendations For NS Chunking Rules

- Prefer a first-pass split between the operative part (`výroková část`) and `Odůvodnění` before any token-length chunking.
- Preserve numbered legal paragraphs as atomic boundaries whenever possible.
- Preserve Roman numeral subdivisions (`I.` to `XX.`) inside the operative part (`výroková část`).
- Treat `Poučení` as a late-document boundary and avoid merging it into substantive reasoning chunks.
- Use `V Brně dne` as a deterministic closing/signature boundary for trimming footer-only tails.
- Support both regular and spaced marker spellings (`Odůvodnění` / `O d ů v o d n ě n í`, `Poučení` / `P o u č e n í`).