# NSoud Relevance Evaluation Dataset

- Status: **PASS**
- Documents input: `/app/app/artifacts/nsoud/rag_ready/nsoud_documents_2025_01_03.parquet`
- Chunks input: `/app/app/artifacts/nsoud/rag_ready/nsoud_chunks_2025_01_03.parquet`
- Payload preview input: `/app/app/artifacts/nsoud/rag_ready/nsoud_qdrant_payload_preview_2025_01_03.parquet`
- Generated queries input: `/app/app/artifacts/nsoud/rag_ready/nsoud_generated_eval_queries_2025_01_03.json`
- Total documents: **150**
- Total chunks: **1862**
- positive_answerable count: **14**
- negative_not_in_batch count: **5**
- underspecified count: **6**
- weak query classification count: **4**

## Positive Answerable

| query | expected_section_types | source_case_numbers | source_chunk_ids | why_answerable |
| --- | --- | --- | --- | --- |
| bezdůvodné obohacení za užívání bytu | reasoning | 26 Cdo 439/2024 | ECLI:CZ:NS:2024:26.CDO.439.2024.1__chunk_0004 | Supported by 1 source chunks across 1 documents; evidence is concentrated in sections reasoning. |
| dovolací důvod podle § 265b odst. 1 písm. g) | signature, reasoning | 3 Tdo 650/2024, 3 Tdo 980/2024 | ECLI:CZ:NS:2024:3.TDO.650.2024.1__chunk_0017, ECLI:CZ:NS:2024:3.TDO.980.2024.1__chunk_0012, ECLI:CZ:NS:2024:3.TDO.980.2024.1__chunk_0020, ECLI:CZ:NS:2024:3.TDO.980.2024.1__chunk_0022, ECLI:CZ:NS:2024:3.TDO.980.2024.1__chunk_0023 | Supported by 5 source chunks across 2 documents; evidence is concentrated in sections signature, reasoning. |
| dovolací důvod podle § 265b odst. 1 písm. h) | reasoning | 11 Tdo 765/2024, 3 Tdo 984/2024, 8 Tdo 760/2024 | ECLI:CZ:NS:2024:11.TDO.765.2024.1__chunk_0012, ECLI:CZ:NS:2024:3.TDO.984.2024.1__chunk_0009, ECLI:CZ:NS:2024:3.TDO.984.2024.1__chunk_0014, ECLI:CZ:NS:2024:3.TDO.984.2024.1__chunk_0018, ECLI:CZ:NS:2024:8.TDO.760.2024.1__chunk_0011 | The batch contains repeated explicit criminal dovolání reasoning anchored to § 265b odst. 1 písm. h). Matched 38 chunks across 20 documents. |
| dovolací důvod podle § 265b odst. 1 písm. m) | reasoning | 11 Tdo 765/2024, 4 Tdo 1044/2024, 6 Tdo 827/2024, 6 Tdo 976/2024 | ECLI:CZ:NS:2024:11.TDO.765.2024.1__chunk_0010, ECLI:CZ:NS:2024:4.TDO.1044.2024.1__chunk_0002, ECLI:CZ:NS:2024:6.TDO.827.2024.1__chunk_0008, ECLI:CZ:NS:2024:6.TDO.976.2024.1__chunk_0006, ECLI:CZ:NS:2024:6.TDO.976.2024.1__chunk_0015 | The batch contains repeated criminal dovolání reasoning explicitly discussing § 265b odst. 1 písm. m). Matched 13 chunks across 10 documents. |
| místní příslušnosti chybějí nebo je nelze zjistit | reasoning, operative_part | 20 Nd 18/2025, 20 Nd 65/2025, 21 Nd 514/2024, 22 Nd 435/2024, 29 Nd 461/2024 | ECLI:CZ:NS:2024:21.ND.514.2024.1__chunk_0002, ECLI:CZ:NS:2024:22.ND.435.2024.1__chunk_0002, ECLI:CZ:NS:2024:29.ND.461.2024.1__chunk_0001, ECLI:CZ:NS:2025:20.ND.18.2025.1__chunk_0001, ECLI:CZ:NS:2025:20.ND.65.2025.1__chunk_0002 | Supported by 5 source chunks across 5 documents; evidence is concentrated in sections reasoning, operative_part. |
| nutná obrana vzájemné napadání | reasoning | 11 Tdo 679/2024 | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0009 | Supported by 1 source chunks across 1 documents; evidence is concentrated in sections reasoning. |
| náhradě nákladů dovolacího řízení | operative_part, reasoning | 20 Cdo 2839/2024, 20 Cdo 3061/2024, 21 Cdo 2178/2024 | ECLI:CZ:NS:2024:20.CDO.2839.2024.1__chunk_0004, ECLI:CZ:NS:2024:20.CDO.3061.2024.1__chunk_0001, ECLI:CZ:NS:2024:20.CDO.3061.2024.1__chunk_0006, ECLI:CZ:NS:2024:21.CDO.2178.2024.1__chunk_0001, ECLI:CZ:NS:2024:21.CDO.2178.2024.1__chunk_0005 | Supported by 5 source chunks across 3 documents; evidence is concentrated in sections operative_part, reasoning. |
| odpovědnosti za vady jako slevy z kupní ceny | reasoning | 23 Cdo 3170/2024 | ECLI:CZ:NS:2024:23.CDO.3170.2024.1__chunk_0002 | Supported by 1 source chunks across 1 documents; evidence is concentrated in sections reasoning. |
| pověření a nařízení exekuce | operative_part, reasoning | 20 Nd 65/2025, 21 Nd 514/2024, 22 Nd 435/2024, 29 Nd 461/2024 | ECLI:CZ:NS:2024:21.ND.514.2024.1__chunk_0001, ECLI:CZ:NS:2024:21.ND.514.2024.1__chunk_0002, ECLI:CZ:NS:2024:22.ND.435.2024.1__chunk_0001, ECLI:CZ:NS:2024:29.ND.461.2024.1__chunk_0001, ECLI:CZ:NS:2025:20.ND.65.2025.1__chunk_0001 | Supported by 5 source chunks across 4 documents; evidence is concentrated in sections operative_part, reasoning. |
| právo bydlení | reasoning | 26 Cdo 439/2024 | ECLI:CZ:NS:2024:26.CDO.439.2024.1__chunk_0004, ECLI:CZ:NS:2024:26.CDO.439.2024.1__chunk_0007, ECLI:CZ:NS:2024:26.CDO.439.2024.1__chunk_0009, ECLI:CZ:NS:2024:26.CDO.439.2024.1__chunk_0010, ECLI:CZ:NS:2024:26.CDO.439.2024.1__chunk_0011 | Supported by 5 source chunks across 1 documents; evidence is concentrated in sections reasoning. |
| přípustnost dovolání podle § 237 o. s. ř. | reasoning | 3 Tdo 650/2024, 3 Tdo 984/2024, 4 Tdo 1044/2024, 6 Tdo 827/2024, 6 Tdo 936/2024 | ECLI:CZ:NS:2024:3.TDO.650.2024.1__chunk_0016, ECLI:CZ:NS:2024:3.TDO.984.2024.1__chunk_0012, ECLI:CZ:NS:2024:4.TDO.1044.2024.1__chunk_0010, ECLI:CZ:NS:2024:6.TDO.827.2024.1__chunk_0014, ECLI:CZ:NS:2024:6.TDO.936.2024.1__chunk_0007 | Supported by 5 source chunks across 5 documents; evidence is concentrated in sections reasoning. |
| trest odnětí svobody | reasoning | 3 Tdo 984/2024, 4 Tdo 1044/2024 | ECLI:CZ:NS:2024:3.TDO.984.2024.1__chunk_0005, ECLI:CZ:NS:2024:3.TDO.984.2024.1__chunk_0008, ECLI:CZ:NS:2024:3.TDO.984.2024.1__chunk_0012, ECLI:CZ:NS:2024:3.TDO.984.2024.1__chunk_0019, ECLI:CZ:NS:2024:4.TDO.1044.2024.1__chunk_0001 | Supported by 5 source chunks across 2 documents; evidence is concentrated in sections reasoning. |
| určení místní příslušnosti | header, reasoning | 21 Nd 514/2024, 22 Nd 435/2024, 29 Nd 461/2024 | ECLI:CZ:NS:2024:21.ND.514.2024.1__chunk_0000, ECLI:CZ:NS:2024:21.ND.514.2024.1__chunk_0002, ECLI:CZ:NS:2024:22.ND.435.2024.1__chunk_0000, ECLI:CZ:NS:2024:22.ND.435.2024.1__chunk_0002, ECLI:CZ:NS:2024:29.ND.461.2024.1__chunk_0000 | Supported by 5 source chunks across 3 documents; evidence is concentrated in sections header, reasoning. |
| zastavení exekuce | reasoning, header, operative_part | 20 Cdo 2831/2024, 20 Cdo 30/2025, 26 Cdo 125/2024 | ECLI:CZ:NS:2024:26.CDO.125.2024.1__chunk_0001, ECLI:CZ:NS:2024:26.CDO.125.2024.1__chunk_0003, ECLI:CZ:NS:2025:20.CDO.30.2025.1__chunk_0000, ECLI:CZ:NS:2025:20.CDO.2831.2024.1__chunk_0005, ECLI:CZ:NS:2025:20.CDO.2831.2024.1__chunk_0006 | Supported by 5 source chunks across 3 documents; evidence is concentrated in sections reasoning, header, operative_part. |

## Negative Not In Batch

| query | matching_chunk_count | matching_document_count | expected_behavior | reason_not_answerable |
| --- | ---: | ---: | --- | --- |
| mezinárodní ochrana a azyl | 0 | 0 | insufficient_support | This NS civil/criminal batch does not contain an asylum / international-protection topic slice. |
| správní vyhoštění cizince | 0 | 0 | insufficient_support | The batch is not a focused administrative-removal corpus and lacks that governing context. |
| ochrana osobních údajů podle GDPR | 0 | 0 | insufficient_support | The batch lacks a GDPR / personal-data dispute cluster with enough explicit support. |
| odpočet DPH u daně z přidané hodnoty | 0 | 0 | insufficient_support | Tax-law support is absent from this Supreme Court civil/criminal subset. |
| stavební povolení a územní rozhodnutí | 0 | 0 | insufficient_support | The batch does not contain a meaningful planning / building-permit dispute track. |

## Underspecified

| query | expected_behavior | why_underspecified | suggested_clarifying_questions |
| --- | --- | --- | --- |
| náhrada nákladů dovolacího řízení | ask_for_clarification | The phrase occurs across many civil dovolání outcomes and does not identify the legal issue or desired answer target. | Které rozhodnutí nebo právní problém v dovolacím řízení vás zajímá? / Chcete náklady po odmítnutí dovolání, po zastavení řízení, nebo po meritorním rozhodnutí? |
| zjevně neopodstatněné dovolání | ask_for_clarification | The phrase spans many criminal dovolání decisions and does not specify a statute, issue, or factual context. | Který dovolací důvod nebo trestněprávní problém máte na mysli? / Chcete rozhodnutí podle § 265i odst. 1 písm. e) tr. ř., nebo širší výklad zjevné neopodstatněnosti? |
| odmítnutí dovolání | ask_for_clarification | The query is outcome-only and can refer to many unrelated civil or criminal dovolání decisions. | Jde vám o občanskoprávní nebo trestní dovolání? / Má být dotaz navázán na konkrétní zákonné ustanovení nebo procesní důvod odmítnutí? |
| rodinný dům | ask_for_clarification | The phrase is a broad property object, not a legal issue, and can point to ownership, defects, housing, or damages. | Jde o vlastnictví, vady, bydlení, náhradu škody, nebo jiný spor o rodinný dům? / Má být dotaz zúžen na konkrétní právní otázku nebo skutkový typ? |
| dovolání | ask_for_clarification | The batch contains many dovolání contexts; the bare term is too broad for a reliable answer. | Jaký dovolací důvod nebo právní problém řešíte? / Má jít o přípustnost, odmítnutí, náklady, nebo konkrétní hmotněprávní otázku? |
| místní příslušnost | ask_for_clarification | The phrase spans multiple exekuční and other process situations without specifying the procedural context. | Jde o exekuční věc, civilní spor, nebo jiný typ řízení? / Má být dotaz zúžen na § 11 odst. 3 o. s. ř. nebo na konkrétní skutkovou situaci? |

## Current Weak Query Classification

| query | primary_classification | matching_chunk_count | matching_document_count | recommended_dataset_class | why_classified_this_way |
| --- | --- | ---: | ---: | --- | --- |
| náhrada nákladů dovolacího řízení | too_generic | 157 | 78 | underspecified | The query targets a repeated procedural outcome phrase rather than a concrete legal issue. |
| zjevně neopodstatněné dovolání | too_generic | 82 | 32 | underspecified | The query describes a broad criminal dovolání outcome without isolating the underlying issue. |
| odmítnutí dovolání | too_generic | 313 | 102 | underspecified | The query is an outcome label shared by many unrelated matters. |
| rodinný dům | too_generic | 5 | 5 | underspecified | The query names an object of dispute, not the legal question to answer. |

## Final Recommendation

- Retrieval quality testing should use: `bezdůvodné obohacení za užívání bytu`, `dovolací důvod podle § 265b odst. 1 písm. g)`, `dovolací důvod podle § 265b odst. 1 písm. h)`, `dovolací důvod podle § 265b odst. 1 písm. m)`, `místní příslušnosti chybějí nebo je nelze zjistit`, `nutná obrana vzájemné napadání`, `náhradě nákladů dovolacího řízení`, `odpovědnosti za vady jako slevy z kupní ceny`, `pověření a nařízení exekuce`, `právo bydlení`, `přípustnost dovolání podle § 237 o. s. ř.`, `trest odnětí svobody`, `určení místní příslušnosti`, `zastavení exekuce`
- Insufficient-support testing should use: `mezinárodní ochrana a azyl`, `správní vyhoštění cizince`, `ochrana osobních údajů podle GDPR`, `odpočet DPH u daně z přidané hodnoty`, `stavební povolení a územní rozhodnutí`
- Clarification behavior testing should use: `náhrada nákladů dovolacího řízení`, `zjevně neopodstatněné dovolání`, `odmítnutí dovolání`, `rodinný dům`, `dovolání`, `místní příslušnost`
- Hybrid retrieval should be added later: **no**

## Notes
- None.
