# NSoud Retrieval Decision Report

- Status: **PASS**
- Target collection: `nsoud_chunks_section_aware_test_2025_01_03`
- Dataset path: `/app/app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/relevance_eval_dataset.json`
- Categorized eval report path: `/app/app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/search_relevance_eval.json`
- Collection exists: **yes**
- Point count: **1862**
- Vector size: **768**
- Old collection unchanged: **True**
- Metadata validation: **PASS**
- JSON report path: `/app/app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/retrieval_decision_report.json`

## Decision Rules Summary

- `answerable`: direct evidence or strong source-context overlap exists in validated section-aware results.
- `insufficient_support`: the issue is legally plausible, but the retrieved results stay indirect, generic, or unsupported.
- `ask_for_clarification`: the query is too broad or ambiguous, or the dataset explicitly marks it as underspecified.

## Positive Answerable Summary

| query | expected_decision | actual_decision | validation | confidence | reason |
| --- | --- | --- | --- | ---: | --- |
| bezdůvodné obohacení za užívání bytu | answerable | answerable | PASS | 0.960 | Top results contain direct source evidence for the expected answerable query. |
| dovolací důvod podle § 265b odst. 1 písm. g) | answerable | answerable | PASS | 0.960 | Top results contain direct source evidence for the expected answerable query. |
| dovolací důvod podle § 265b odst. 1 písm. h) | answerable | answerable | PASS | 0.960 | Top results contain direct source evidence for the expected answerable query. |
| dovolací důvod podle § 265b odst. 1 písm. m) | answerable | answerable | PASS | 0.960 | Top results contain direct source evidence for the expected answerable query. |
| místní příslušnosti chybějí nebo je nelze zjistit | answerable | answerable | PASS | 0.826 | Top results contain source-term overlap in the expected section context. |
| nutná obrana vzájemné napadání | answerable | answerable | PASS | 0.840 | Top results contain source-term overlap in the expected section context. |
| náhradě nákladů dovolacího řízení | answerable | answerable | PASS | 0.960 | Top results contain direct source evidence for the expected answerable query. |
| odpovědnosti za vady jako slevy z kupní ceny | answerable | answerable | PASS | 0.838 | Top results contain source-term overlap in the expected section context. |
| pověření a nařízení exekuce | answerable | answerable | PASS | 0.960 | Top results contain direct source evidence for the expected answerable query. |
| právo bydlení | answerable | answerable | PASS | 0.960 | Top results contain direct source evidence for the expected answerable query. |
| přípustnost dovolání podle § 237 o. s. ř. | answerable | answerable | PASS | 0.960 | Top results contain direct source evidence for the expected answerable query. |
| trest odnětí svobody | answerable | answerable | PASS | 0.960 | Top results contain direct source evidence for the expected answerable query. |
| určení místní příslušnosti | answerable | answerable | PASS | 0.832 | Top results contain source-term overlap in the expected section context. |
| zastavení exekuce | answerable | answerable | PASS | 0.880 | Top results contain direct source evidence for the expected answerable query. |

## Negative Not In Batch Summary

| query | expected_decision | actual_decision | validation | confidence | reason |
| --- | --- | --- | --- | ---: | --- |
| mezinárodní ochrana a azyl | insufficient_support | insufficient_support | PASS | 0.840 | Some vocabulary overlaps appear, but the results remain too weak and indirect to support an answer. |
| správní vyhoštění cizince | insufficient_support | insufficient_support | PASS | 0.840 | Some vocabulary overlaps appear, but the results remain too weak and indirect to support an answer. |
| ochrana osobních údajů podle GDPR | insufficient_support | insufficient_support | PASS | 0.840 | Some vocabulary overlaps appear, but the results remain too weak and indirect to support an answer. |
| odpočet DPH u daně z přidané hodnoty | insufficient_support | insufficient_support | PASS | 0.840 | Some vocabulary overlaps appear, but the results remain too weak and indirect to support an answer. |
| stavební povolení a územní rozhodnutí | insufficient_support | insufficient_support | PASS | 0.840 | Some vocabulary overlaps appear, but the results remain too weak and indirect to support an answer. |

## Underspecified Summary

| query | expected_decision | actual_decision | validation | confidence | reason |
| --- | --- | --- | --- | ---: | --- |
| náhrada nákladů dovolacího řízení | ask_for_clarification | ask_for_clarification | PASS | 0.940 | The query targets a repeated procedural outcome phrase rather than a concrete legal issue. |
| zjevně neopodstatněné dovolání | ask_for_clarification | ask_for_clarification | PASS | 0.940 | The query describes a broad criminal dovolání outcome without isolating the underlying issue. |
| odmítnutí dovolání | ask_for_clarification | ask_for_clarification | PASS | 0.940 | The query is an outcome label shared by many unrelated matters. |
| rodinný dům | ask_for_clarification | ask_for_clarification | PASS | 0.940 | The query names an object of dispute, not the legal question to answer. |
| dovolání | ask_for_clarification | ask_for_clarification | PASS | 0.900 | Results span too many documents or legal contexts to justify a single direct answer. |
| místní příslušnost | ask_for_clarification | ask_for_clarification | PASS | 0.900 | Results span too many documents or legal contexts to justify a single direct answer. |

## Per-Query Decisions

### bezdůvodné obohacení za užívání bytu

- Expected decision: `answerable`
- Actual decision: `answerable`
- Validation: **PASS**
- Confidence: **0.960**
- Reason: Top results contain direct source evidence for the expected answerable query.
- Recommended user message: The available NS decisions contain relevant support for this query. The answer should be based on the retrieved chunks.
- Top result score: **0.640370**
- Top result case_number: `26 Cdo 1854/2024`
- Top result document_type: `ROZSUDEK`
- Top result legal_area: `civil`
- Top result section_type: `reasoning`
- Matched terms: bezdůvodné obohacení užívání bytu, bezdůvodné obohacení za užívání bytu
- Missing terms: -

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.640370 | 26 Cdo 1854/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2025:26.CDO.1854.2024.1__chunk_0002 | ECLI:CZ:NS:2025:26.CDO.1854.2024.1 | PASS | Odůvodnění: 1. Žalobkyně se po žalované domáhala zaplacení částky 100.000 Kč (s příslušenstvím v podobě úroku z prodlení) představující slevu z kupní ceny tam specifikovaného bytu (dále jen „předmětný byt“, resp. „byt“). Uvedla, že v bytě s... |
| 2 | 0.619628 | 26 Cdo 439/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:26.CDO.439.2024.1__chunk_0006 | ECLI:CZ:NS:2024:26.CDO.439.2024.1 | PASS | 8. Žalovaný ve vyjádření k dovolání považuje rozhodnutí odvolacího soudu za správné. Do okamžiku zajištění bytové náhrady se řídí vztah bývalého nájemce a pronajímatele ustanovením § 712a obč. zák. a teprve po zajištění bytové náhrady – jes... |
| 3 | 0.608666 | 26 Cdo 439/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:26.CDO.439.2024.1__chunk_0011 | ECLI:CZ:NS:2024:26.CDO.439.2024.1 | PASS | 20. Ze shora uvedeného se podává, že odvolací soud pochybil, jestliže žalobní nárok na zaplacení 360 000 Kč s příslušenstvím posuzoval výlučně z hlediska bezdůvodného obohacení (jako by bytová náhrada byla zajištěna), a nikoliv též (pro pří... |
| 4 | 0.596197 | 26 Cdo 2198/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:26.CDO.2198.2024.1__chunk_0002 | ECLI:CZ:NS:2024:26.CDO.2198.2024.1 | PASS | Odůvodnění: 1. Žalobce se domáhal, aby byla žalované uložena povinnost předložit mu řádná vyúčtování záloh na služby spojené s užíváním tam specifikovaného bytu (dále jen „byt“) za zúčtovací období od 1. 7. 2018 do 30. 6. 2019, od 1. 7. 201... |
| 5 | 0.593576 | 26 Cdo 439/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:26.CDO.439.2024.1__chunk_0003 | ECLI:CZ:NS:2024:26.CDO.439.2024.1 | PASS | 6. 2012, č. j. 11 Co 155/2012-78, bylo zrušeno právo společného nájmu tam specifikovaného družstevního bytu (předmětného bytu) a garáže (blíže označené) a společné členství v bytovém družstvu s tím, že výlučnou členkou družstva a výlučnou n... |

### dovolací důvod podle § 265b odst. 1 písm. g)

- Expected decision: `answerable`
- Actual decision: `answerable`
- Validation: **PASS**
- Confidence: **0.960**
- Reason: Top results contain direct source evidence for the expected answerable query.
- Recommended user message: The available NS decisions contain relevant support for this query. The answer should be based on the retrieved chunks.
- Top result score: **0.872580**
- Top result case_number: `6 Tdo 21/2025`
- Top result document_type: `USNESENÍ`
- Top result legal_area: `criminal`
- Top result section_type: `reasoning`
- Matched terms: Dovolací důvod podle § 265b odst. 1 písm. g), dovolací důvod podle § 265b odst. 1 písm. g)
- Missing terms: -

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.872580 | 6 Tdo 21/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:6.TDO.21.2025.1__chunk_0009 | ECLI:CZ:NS:2025:6.TDO.21.2025.1 | PASS | 15. Dovolací důvod podle § 265b odst. 1 písm. m) tr. ř. je dán, bylo-li rozhodnuto o zamítnutí nebo odmítnutí řádného opravného prostředku proti rozsudku nebo usnesení uvedenému v § 265a odst. 2 písm. a) až g) tr. ř., aniž byly splněny proc... |
| 2 | 0.857696 | 6 Tdo 976/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:6.TDO.976.2024.1__chunk_0005 | ECLI:CZ:NS:2024:6.TDO.976.2024.1 | PASS | 10. Důvod dovolání podle § 265b odst. 1 písm. h) tr. ř. je dán v případech, kdy rozhodnutí spočívá na nesprávném právním posouzení skutku nebo jiném nesprávném hmotněprávním posouzení. Uvedenou formulací zákon vyjadřuje, že tento dovolací d... |
| 3 | 0.854537 | 6 Tdo 936/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:6.TDO.936.2024.1__chunk_0008 | ECLI:CZ:NS:2024:6.TDO.936.2024.1 | PASS | 14. Důvod dovolání podle § 265b odst. 1 písm. h) tr. ř. je dán v případech, kdy rozhodnutí spočívá na nesprávném právním posouzení skutku nebo jiném nesprávném hmotněprávním posouzení. Uvedenou formulací zákon vyjadřuje, že tento dovolací d... |
| 4 | 0.839132 | 4 Tdo 1044/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:4.TDO.1044.2024.1__chunk_0014 | ECLI:CZ:NS:2024:4.TDO.1044.2024.1 | PASS | 33. Dovolací důvod podle § 265b odst. 1 písm. h) tr. ř. je naplněn tehdy, jestliže rozhodnutí spočívá na nesprávném právním posouzení skutku (první alternativa) nebo jiném nesprávném hmotněprávním posouzení (druhá alternativa). Uvedenou for... |
| 5 | 0.831241 | 8 Tdo 1085/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:8.TDO.1085.2024.1__chunk_0009 | ECLI:CZ:NS:2025:8.TDO.1085.2024.1 | PASS | 19. Důvod podle § 265b odst. 1 písm. m) tr. ř. spočívá v tom, že bylo rozhodnuto o zamítnutí nebo odmítnutí řádného opravného prostředku proti rozsudku nebo usnesení uvedenému v § 265a odst. 2 písm. a) až g) tr. ř., aniž byly splněny proces... |

### dovolací důvod podle § 265b odst. 1 písm. h)

- Expected decision: `answerable`
- Actual decision: `answerable`
- Validation: **PASS**
- Confidence: **0.960**
- Reason: Top results contain direct source evidence for the expected answerable query.
- Recommended user message: The available NS decisions contain relevant support for this query. The answer should be based on the retrieved chunks.
- Top result score: **0.863294**
- Top result case_number: `6 Tdo 21/2025`
- Top result document_type: `USNESENÍ`
- Top result legal_area: `criminal`
- Top result section_type: `reasoning`
- Matched terms: dovolací důvod podle § 265b odst. 1 písm. h)
- Missing terms: -

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.863294 | 6 Tdo 21/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:6.TDO.21.2025.1__chunk_0009 | ECLI:CZ:NS:2025:6.TDO.21.2025.1 | PASS | 15. Dovolací důvod podle § 265b odst. 1 písm. m) tr. ř. je dán, bylo-li rozhodnuto o zamítnutí nebo odmítnutí řádného opravného prostředku proti rozsudku nebo usnesení uvedenému v § 265a odst. 2 písm. a) až g) tr. ř., aniž byly splněny proc... |
| 2 | 0.854258 | 6 Tdo 976/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:6.TDO.976.2024.1__chunk_0005 | ECLI:CZ:NS:2024:6.TDO.976.2024.1 | PASS | 10. Důvod dovolání podle § 265b odst. 1 písm. h) tr. ř. je dán v případech, kdy rozhodnutí spočívá na nesprávném právním posouzení skutku nebo jiném nesprávném hmotněprávním posouzení. Uvedenou formulací zákon vyjadřuje, že tento dovolací d... |
| 3 | 0.854116 | 6 Tdo 936/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:6.TDO.936.2024.1__chunk_0008 | ECLI:CZ:NS:2024:6.TDO.936.2024.1 | PASS | 14. Důvod dovolání podle § 265b odst. 1 písm. h) tr. ř. je dán v případech, kdy rozhodnutí spočívá na nesprávném právním posouzení skutku nebo jiném nesprávném hmotněprávním posouzení. Uvedenou formulací zákon vyjadřuje, že tento dovolací d... |
| 4 | 0.827733 | 4 Tdo 1044/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:4.TDO.1044.2024.1__chunk_0014 | ECLI:CZ:NS:2024:4.TDO.1044.2024.1 | PASS | 33. Dovolací důvod podle § 265b odst. 1 písm. h) tr. ř. je naplněn tehdy, jestliže rozhodnutí spočívá na nesprávném právním posouzení skutku (první alternativa) nebo jiném nesprávném hmotněprávním posouzení (druhá alternativa). Uvedenou for... |
| 5 | 0.824950 | 8 Tdo 1085/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:8.TDO.1085.2024.1__chunk_0009 | ECLI:CZ:NS:2025:8.TDO.1085.2024.1 | PASS | 19. Důvod podle § 265b odst. 1 písm. m) tr. ř. spočívá v tom, že bylo rozhodnuto o zamítnutí nebo odmítnutí řádného opravného prostředku proti rozsudku nebo usnesení uvedenému v § 265a odst. 2 písm. a) až g) tr. ř., aniž byly splněny proces... |

### dovolací důvod podle § 265b odst. 1 písm. m)

- Expected decision: `answerable`
- Actual decision: `answerable`
- Validation: **PASS**
- Confidence: **0.960**
- Reason: Top results contain direct source evidence for the expected answerable query.
- Recommended user message: The available NS decisions contain relevant support for this query. The answer should be based on the retrieved chunks.
- Top result score: **0.869987**
- Top result case_number: `6 Tdo 21/2025`
- Top result document_type: `USNESENÍ`
- Top result legal_area: `criminal`
- Top result section_type: `reasoning`
- Matched terms: dovolací důvod podle § 265b odst. 1 písm. m)
- Missing terms: -

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.869987 | 6 Tdo 21/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:6.TDO.21.2025.1__chunk_0009 | ECLI:CZ:NS:2025:6.TDO.21.2025.1 | PASS | 15. Dovolací důvod podle § 265b odst. 1 písm. m) tr. ř. je dán, bylo-li rozhodnuto o zamítnutí nebo odmítnutí řádného opravného prostředku proti rozsudku nebo usnesení uvedenému v § 265a odst. 2 písm. a) až g) tr. ř., aniž byly splněny proc... |
| 2 | 0.857946 | 6 Tdo 976/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:6.TDO.976.2024.1__chunk_0005 | ECLI:CZ:NS:2024:6.TDO.976.2024.1 | PASS | 10. Důvod dovolání podle § 265b odst. 1 písm. h) tr. ř. je dán v případech, kdy rozhodnutí spočívá na nesprávném právním posouzení skutku nebo jiném nesprávném hmotněprávním posouzení. Uvedenou formulací zákon vyjadřuje, že tento dovolací d... |
| 3 | 0.852923 | 6 Tdo 936/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:6.TDO.936.2024.1__chunk_0008 | ECLI:CZ:NS:2024:6.TDO.936.2024.1 | PASS | 14. Důvod dovolání podle § 265b odst. 1 písm. h) tr. ř. je dán v případech, kdy rozhodnutí spočívá na nesprávném právním posouzení skutku nebo jiném nesprávném hmotněprávním posouzení. Uvedenou formulací zákon vyjadřuje, že tento dovolací d... |
| 4 | 0.837331 | 4 Tdo 1044/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:4.TDO.1044.2024.1__chunk_0014 | ECLI:CZ:NS:2024:4.TDO.1044.2024.1 | PASS | 33. Dovolací důvod podle § 265b odst. 1 písm. h) tr. ř. je naplněn tehdy, jestliže rozhodnutí spočívá na nesprávném právním posouzení skutku (první alternativa) nebo jiném nesprávném hmotněprávním posouzení (druhá alternativa). Uvedenou for... |
| 5 | 0.832481 | 8 Tdo 1085/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:8.TDO.1085.2024.1__chunk_0009 | ECLI:CZ:NS:2025:8.TDO.1085.2024.1 | PASS | 19. Důvod podle § 265b odst. 1 písm. m) tr. ř. spočívá v tom, že bylo rozhodnuto o zamítnutí nebo odmítnutí řádného opravného prostředku proti rozsudku nebo usnesení uvedenému v § 265a odst. 2 písm. a) až g) tr. ř., aniž byly splněny proces... |

### místní příslušnosti chybějí nebo je nelze zjistit

- Expected decision: `answerable`
- Actual decision: `answerable`
- Validation: **PASS**
- Confidence: **0.826**
- Reason: Top results contain source-term overlap in the expected section context.
- Recommended user message: The available NS decisions contain relevant support for this query. The answer should be based on the retrieved chunks.
- Top result score: **0.457301**
- Top result case_number: `6 Tdo 1057/2024`
- Top result document_type: `USNESENÍ`
- Top result legal_area: `criminal`
- Top result section_type: `reasoning`
- Matched terms: místní příslušnosti chybějí nebo je nelze zjistit
- Missing terms: -

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.457301 | 6 Tdo 1057/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:6.TDO.1057.2024.1__chunk_0010 | ECLI:CZ:NS:2025:6.TDO.1057.2024.1 | PASS | 23. Právě to však v projednávaném případě schází. Geneticky analyzované biologické stopy byly zajištěny pouze na stavebním kolečku a lahvi, nikoli na vlastní kabeláži, takže jejich propojení s krádeží není bezprostřední a onen „pozitivní dů... |
| 2 | 0.440205 | 20 Nd 18/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:20.ND.18.2025.1__chunk_0003 | ECLI:CZ:NS:2025:20.ND.18.2025.1 | PASS | 99/1963 Sb., občanský soudní řád, ve znění pozdějších předpisů – dále též jen „o. s. ř.“). V exekučním návrhu oprávněné a stejně tak v žádosti soudního exekutora je uvedena adresa povinného XY. Lustrací v informačním systému základních regi... |
| 3 | 0.438854 | 6 Tdo 827/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:6.TDO.827.2024.1__chunk_0011 | ECLI:CZ:NS:2024:6.TDO.827.2024.1 | PASS | 21. Za opodstatněnou nelze mít ani námitku dovolatelů poukazujících na to, že předmětný odpad nepřepravili přes hranice státu oni, nýbrž že uzavřeli pouze obchodní dohodu s mezinárodním prvkem. Dovolatelé zde opomíjejí, že za jednání pachat... |
| 4 | 0.405841 | 26 Nd 406/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:26.ND.406.2024.1__chunk_0004 | ECLI:CZ:NS:2025:26.ND.406.2024.1 | PASS | 12. 2009, sp. zn. 4 Nd 374/2009). Ani sama skutečnost, že v řízení může být nutné provést výslech svědků, jež bydlí či sídlí mimo obvod (místně příslušného) soudu, rovněž není důvodem pro přikázání věci soudu, v jehož obvodu svědci žijí či... |
| 5 | 0.404084 | 7 Td 6/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:7.TD.6.2025.1__chunk_0004 | ECLI:CZ:NS:2025:7.TD.6.2025.1 | PASS | 7. Pro určení místní příslušnosti soudu je podstatné místo, kde obviněná jednala, kde neznámé osobě umožnila užít k výše uvedené transakci svůj účet, tedy konkrétně především na jakém místě se nacházela, když finanční prostředky ze svého úč... |

### nutná obrana vzájemné napadání

- Expected decision: `answerable`
- Actual decision: `answerable`
- Validation: **PASS**
- Confidence: **0.840**
- Reason: Top results contain source-term overlap in the expected section context.
- Recommended user message: The available NS decisions contain relevant support for this query. The answer should be based on the retrieved chunks.
- Top result score: **0.600907**
- Top result case_number: `11 Tdo 679/2024`
- Top result document_type: `USNESENÍ`
- Top result legal_area: `criminal`
- Top result section_type: `reasoning`
- Matched terms: nutná obrana v úvahu v případech vzájemného napadání, nutná obrana vzájemné napadání
- Missing terms: -

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.600907 | 11 Tdo 679/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0011 | ECLI:CZ:NS:2024:11.TDO.679.2024.1 | PASS | 21. Existence trvajícího útoku je podle státního zástupce evidentní – poškozený tloukl do vozidla obviněného a bránil mu v odjezdu z místa. Opuštění vozidla a udeření poškozeného není v těchto souvislostech akceptací výzvy k potyčce, nýbrž... |
| 2 | 0.596968 | 11 Tdo 679/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0010 | ECLI:CZ:NS:2024:11.TDO.679.2024.1 | PASS | 19. Státní zástupce souhlasí s obviněným i v tom, že čelil útoku ze strany poškozeného. Soudy podle něj pochybily v důsledku nesprávné aplikace judikatury Nejvyššího soudu, podle které nepřichází nutná obrana v úvahu v případech vzájemného... |
| 3 | 0.573934 | 3 Tdo 19/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:3.TDO.19.2025.1__chunk_0004 | ECLI:CZ:NS:2025:3.TDO.19.2025.1 | PASS | 3. 2021, sp. zn. 6 Tdo 255/2021, podle něhož ne každé fyzické napadení druhého na veřejnosti nebo na místě veřejně přístupném musí naplňovat skutkovou podstatu přečinu výtržnictví, zvlášť jde-li o napadení, které je prostředkem, jímž pachat... |
| 4 | 0.563485 | 11 Tdo 679/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0012 | ECLI:CZ:NS:2024:11.TDO.679.2024.1 | PASS | 24. Argumenty soudů, jež vylučují nutnou obranu, tedy podle názoru státního zástupce neobstojí – subsidiarita není podmínkou nutné obrany; současně se nejedná o případ vzájemného napadání, protože obviněný reagoval na probíhající útok ze st... |
| 5 | 0.541885 | 11 Tdo 679/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0006 | ECLI:CZ:NS:2024:11.TDO.679.2024.1 | PASS | 10. Obviněný dále cituje usnesení Nejvyššího soudu ze dne 30. 10. 2019, sp. zn. 6 Tdo 1286/2019, a dodává, že v situaci, kdy proti sobě navzájem útočí dvě osoby, je rozhodující počáteční iniciativa, tedy kdo začal (případně také pohnutka či... |

### náhradě nákladů dovolacího řízení

- Expected decision: `answerable`
- Actual decision: `answerable`
- Validation: **PASS**
- Confidence: **0.960**
- Reason: Top results contain direct source evidence for the expected answerable query.
- Recommended user message: The available NS decisions contain relevant support for this query. The answer should be based on the retrieved chunks.
- Top result score: **0.779172**
- Top result case_number: `28 Cdo 3321/2024`
- Top result document_type: `USNESENÍ`
- Top result legal_area: `civil`
- Top result section_type: `reasoning`
- Matched terms: náhradě nákladů dovolacího řízení
- Missing terms: -

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.779172 | 28 Cdo 3321/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:28.CDO.3321.2024.1__chunk_0008 | ECLI:CZ:NS:2025:28.CDO.3321.2024.1 | PASS | 12. 2024 (dále „advokátní tarif“), a náhrady paušálně určených hotových výdajů za jeden úkon právní služby ve výši 300 Kč – § 11 odst. 1 písm. k) a § 13 odst. 4 advokátního tarifu. Protože zástupce dalšího účastníka řízení je plátcem daně z... |
| 2 | 0.779038 | 29 Cdo 275/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:29.CDO.275.2024.1__chunk_0004 | ECLI:CZ:NS:2025:29.CDO.275.2024.1 | PASS | 177/1996 Sb., o odměnách advokátů a náhradách advokátů za poskytování právních služeb (advokátního tarifu), ve znění účinném k datu podání vyjádření, činí (z tarifní hodnoty 421.400,- Kč) částku 10.020,- Kč, a z paušální částky náhrady hoto... |
| 3 | 0.778899 | 27 Cdo 2699/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:27.CDO.2699.2024.1__chunk_0005 | ECLI:CZ:NS:2025:27.CDO.2699.2024.1 | PASS | 13. Výrok o náhradě nákladů dovolacího řízení se opírá o § 243c odst. 3, § 224 odst. 1 a § 146 odst. 3 o. s. ř., když dovolání žalobkyně bylo odmítnuto a žalovanému vzniklo právo na náhradu účelně vynaložených nákladů dovolacího řízení. 14.... |
| 4 | 0.778514 | 28 Cdo 3513/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:28.CDO.3513.2024.1__chunk_0006 | ECLI:CZ:NS:2025:28.CDO.3513.2024.1 | PASS | 15. O nákladech dovolacího řízení bylo rozhodnuto v intencích § 243 odst. 3 věty první, § 224 odst. 1, § 151 odst. 1 části věty před středníkem a § 146 odst. 3 o. s. ř.; dovolání žalované bylo odmítnuto a na straně žalobce lze za účelně vyn... |
| 5 | 0.772786 | 25 Cdo 2348/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:25.CDO.2348.2024.1__chunk_0002 | ECLI:CZ:NS:2024:25.CDO.2348.2024.1 | PASS | Odůvodnění: 1. Žalobkyně se podanou žalobou domáhala 998.089 Kč s příslušenstvím z titulu odpovědnosti advokáta za škodu. Částka 203.186 Kč představovala marně vynaložené náklady řízení, částka 594.903 Kč kapitalizovaný úrok z prodlení z 1.... |

### odpovědnosti za vady jako slevy z kupní ceny

- Expected decision: `answerable`
- Actual decision: `answerable`
- Validation: **PASS**
- Confidence: **0.838**
- Reason: Top results contain source-term overlap in the expected section context.
- Recommended user message: The available NS decisions contain relevant support for this query. The answer should be based on the retrieved chunks.
- Top result score: **0.582669**
- Top result case_number: `33 Cdo 79/2024`
- Top result document_type: `USNESENÍ`
- Top result legal_area: `civil`
- Top result section_type: `appeal_instruction`
- Matched terms: odpovědnosti za vady jako slevy z kupní ceny
- Missing terms: -

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.582669 | 33 Cdo 79/2024 | USNESENÍ | civil | appeal_instruction | ECLI:CZ:NS:2025:33.CDO.79.2024.1__chunk_0009 | ECLI:CZ:NS:2025:33.CDO.79.2024.1 | PASS | 12. 2013, dále jen „obč. zák.“). Ve spotřebitelském právu je dodavatel ve fakticky výhodnějším postavení, neboť má odbornou převahu nad spotřebiteli, kterým své služby poskytuje. A proto kromě omezení vyplývajících z principu rovností prost... |
| 2 | 0.556213 | 33 Cdo 651/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:33.CDO.651.2024.1__chunk_0002 | ECLI:CZ:NS:2025:33.CDO.651.2024.1 | PASS | 89/2012 Sb., občanský zákoník (dále jen „o. z.“), a zák. č. 145/2010 Sb., o spotřebitelském úvěru. V důsledku nesplnění povinnosti plynoucí z § 9 zákona o spotřebitelském úvěru je nutno všechny smlouvy hodnotit jako neplatné, a žalovanému t... |
| 3 | 0.556045 | 8 Tdo 760/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:8.TDO.760.2024.1__chunk_0017 | ECLI:CZ:NS:2024:8.TDO.760.2024.1 | PASS | 8. 2004 za částku 500 Kč (srov. v podrobnostech č. l. 28947 a násl. spisu). Vzhledem k tomu, že společnost Standard nedisponovala dostatečným majetkem, se obchod realizovaný obviněnými jevil výhodným pouze zdánlivě, neboť nominální výše poh... |
| 4 | 0.540448 | 8 Tdo 760/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:8.TDO.760.2024.1__chunk_0003 | ECLI:CZ:NS:2024:8.TDO.760.2024.1 | PASS | 8. 2004 od správce konkurzní podstaty úpadce společnosti Standart s.r.o. za částku 500 Kč jako nedobytnou a prakticky bezcennou, neboť dlužník společnost Juma byl v úpadku a bylo evidentní, že konkurz bude zrušen pro nedostatek majetku, a o... |
| 5 | 0.533962 | 23 Cdo 271/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:23.CDO.271.2024.1__chunk_0011 | ECLI:CZ:NS:2024:23.CDO.271.2024.1 | PASS | 5. 2012 do 20. 5. 2014, smlouvy upravené tímto zákonem se v ostatním řídí právní úpravou závazků a úpravou smluvních typů jim nejbližších podle občanského nebo obchodního zákoníku, pokud z tohoto zákona nebo povahy věci nevyplývá něco jinéh... |

### pověření a nařízení exekuce

- Expected decision: `answerable`
- Actual decision: `answerable`
- Validation: **PASS**
- Confidence: **0.960**
- Reason: Top results contain direct source evidence for the expected answerable query.
- Recommended user message: The available NS decisions contain relevant support for this query. The answer should be based on the retrieved chunks.
- Top result score: **0.786519**
- Top result case_number: `20 Cdo 30/2025`
- Top result document_type: `USNESENÍ`
- Top result legal_area: `civil`
- Top result section_type: `reasoning`
- Matched terms: pověření a nařízení exekuce
- Missing terms: -

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.786519 | 20 Cdo 30/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:20.CDO.30.2025.1__chunk_0003 | ECLI:CZ:NS:2025:20.CDO.30.2025.1 | PASS | 120/2001 Sb., o soudních exekutorech a exekuční činnosti (exekuční řád) a o změně dalších zákonů, ve znění pozdějších předpisů (dále též jen „ex. řád“), na jeho základě není možné vést exekuci a je namístě zastavit ji podle § 268 odst. 1 pí... |
| 2 | 0.770025 | 20 Cdo 15/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:20.CDO.15.2025.1__chunk_0004 | ECLI:CZ:NS:2025:20.CDO.15.2025.1 | PASS | 120/2001 Sb., o soudních exekutorech a exekuční činnosti (exekuční řád) a o změně dalších zákonů, ve znění pozdějších předpisů]. |
| 3 | 0.770025 | 20 Cdo 13/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:20.CDO.13.2025.1__chunk_0004 | ECLI:CZ:NS:2025:20.CDO.13.2025.1 | PASS | 120/2001 Sb., o soudních exekutorech a exekuční činnosti (exekuční řád) a o změně dalších zákonů, ve znění pozdějších předpisů]. |
| 4 | 0.755956 | 20 Cdo 3450/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:20.CDO.3450.2024.1__chunk_0004 | ECLI:CZ:NS:2025:20.CDO.3450.2024.1 | PASS | 120/2001 Sb., o soudních exekutorech a exekuční činnosti (exekuční řád), ve znění pozdějších předpisů, nestanoví-li tento zákon jinak, použijí se pro exekuční řízení přiměřeně ustanovení občanského soudního řádu. Výkon rozhodnutí se provede... |
| 5 | 0.729479 | 30 Cdo 3197/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:30.CDO.3197.2024.1__chunk_0004 | ECLI:CZ:NS:2025:30.CDO.3197.2024.1 | PASS | 82/1998 Sb., o odpovědnosti za škodu způsobenou při výkonu veřejné moci rozhodnutím nebo nesprávným úředním postupem a o změně zákona České národní rady č. 358/1992 Sb., o notářích a jejich činnosti (notářský řád), ve znění pozdějších předp... |

### právo bydlení

- Expected decision: `answerable`
- Actual decision: `answerable`
- Validation: **PASS**
- Confidence: **0.960**
- Reason: Top results contain direct source evidence for the expected answerable query.
- Recommended user message: The available NS decisions contain relevant support for this query. The answer should be based on the retrieved chunks.
- Top result score: **0.728320**
- Top result case_number: `26 Cdo 439/2024`
- Top result document_type: `ROZSUDEK`
- Top result legal_area: `civil`
- Top result section_type: `reasoning`
- Matched terms: Právo bydlení, právo bydlení
- Missing terms: -

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.728320 | 26 Cdo 439/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:26.CDO.439.2024.1__chunk_0006 | ECLI:CZ:NS:2024:26.CDO.439.2024.1 | PASS | 8. Žalovaný ve vyjádření k dovolání považuje rozhodnutí odvolacího soudu za správné. Do okamžiku zajištění bytové náhrady se řídí vztah bývalého nájemce a pronajímatele ustanovením § 712a obč. zák. a teprve po zajištění bytové náhrady – jes... |
| 2 | 0.682756 | 26 Cdo 439/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:26.CDO.439.2024.1__chunk_0003 | ECLI:CZ:NS:2024:26.CDO.439.2024.1 | PASS | 6. 2012, č. j. 11 Co 155/2012-78, bylo zrušeno právo společného nájmu tam specifikovaného družstevního bytu (předmětného bytu) a garáže (blíže označené) a společné členství v bytovém družstvu s tím, že výlučnou členkou družstva a výlučnou n... |
| 3 | 0.673722 | 24 Cdo 671/2024; 24 Cdo 675/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2024:24.CDO.671.2024.1__chunk_0002 | ECLI:CZ:NS:2024:24.CDO.671.2024.1 | PASS | Odůvodnění: 1. Okresní soud v Pelhřimově rozsudkem ze dne 31. 5. 2023, č. j. 5 C 255/2015-1439, rozhodl o určení vlastnického práva k celkem devíti jednotkám v domě č. p. XY v XY. Jednalo se o byty, garáže a dílnu. Stručně řečeno, u jednote... |
| 4 | 0.664371 | 22 Cdo 1151/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2025:22.CDO.1151.2024.1__chunk_0004 | ECLI:CZ:NS:2025:22.CDO.1151.2024.1 | PASS | 7. Ve způsobu vypořádání spoluvlastnictví dospěl ke stejnému závěru jako soud prvního stupně, tedy že reálné dělení blokačních pozemků není dobře možné, svůj názor ovšem opřel o jinou skutečnost, konkrétně o veřejný zájem na vybudování ploc... |
| 5 | 0.649335 | 22 Cdo 108/2025 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2025:22.CDO.108.2025.1__chunk_0008 | ECLI:CZ:NS:2025:22.CDO.108.2025.1 | PASS | 1. 2024 na základě § 1045 odst. 2 o. z. ve spojení s § 1050 odst. 2 o. z. a § 65 zákona č. 256/2013 Sb., katastrální zákon, ve znění účinném od 1. 1. 2014, jakožto k věcem opuštěným: 25. Dovolatelka dále napadá závěry odvolacího soudu, že F... |

### přípustnost dovolání podle § 237 o. s. ř.

- Expected decision: `answerable`
- Actual decision: `answerable`
- Validation: **PASS**
- Confidence: **0.960**
- Reason: Top results contain direct source evidence for the expected answerable query.
- Recommended user message: The available NS decisions contain relevant support for this query. The answer should be based on the retrieved chunks.
- Top result score: **0.869010**
- Top result case_number: `23 Cdo 938/2024`
- Top result document_type: `USNESENÍ`
- Top result legal_area: `civil`
- Top result section_type: `reasoning`
- Matched terms: Přípustnost dovolání, Přípustnost dovolání podle § 237 o. s. ř., přípustnost dovolání, přípustnost dovolání podle § 237 o. s. ř.
- Missing terms: -

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.869010 | 23 Cdo 938/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:23.CDO.938.2024.1__chunk_0006 | ECLI:CZ:NS:2025:23.CDO.938.2024.1 | PASS | 13. Ustanovení § 241a odst. 2 o. s. ř. stanoví, že v dovolání musí být vedle obecných náležitostí (§ 42 odst. 4 o. s. ř.) uvedeno, proti kterému rozhodnutí směřuje, v jakém rozsahu se napadá, vymezení důvodu dovolání, v čem dovolatel spatřu... |
| 2 | 0.852014 | 23 Cdo 434/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2024:23.CDO.434.2024.1__chunk_0004 | ECLI:CZ:NS:2024:23.CDO.434.2024.1 | PASS | 9. Podle § 241a odst. 2 o. s. ř. v dovolání musí být vedle obecných náležitostí (§ 42 odst. 4) uvedeno, proti kterému rozhodnutí směřuje, v jakém rozsahu se rozhodnutí napadá, vymezení důvodu dovolání, v čem dovolatel spatřuje splnění předp... |
| 3 | 0.831259 | 30 Cdo 3461/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:30.CDO.3461.2024.1__chunk_0004 | ECLI:CZ:NS:2025:30.CDO.3461.2024.1 | PASS | 6. 2012, sp. zn. 30 Cdo 1486/2012). Ustanovení § 241 o. s. ř., které upravuje tzv. povinné zastoupení dovolatele při podání dovolání, představuje zvláštní podmínku dovolacího řízení, jejíž nedostatek lze odstranit, bez jejíhož splnění však... |
| 4 | 0.826377 | 30 Cdo 106/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:30.CDO.106.2025.1__chunk_0003 | ECLI:CZ:NS:2025:30.CDO.106.2025.1 | PASS | 6. Ustanovení § 241 o. s. ř., které upravuje tzv. povinné zastoupení dovolatele při podání dovolání, představuje zvláštní podmínku dovolacího řízení, jejíž nedostatek lze odstranit, bez jejíhož splnění však nelze meritorně rozhodnout o dovo... |
| 5 | 0.816218 | 21 Cdo 245/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:21.CDO.245.2024.1__chunk_0003 | ECLI:CZ:NS:2025:21.CDO.245.2024.1 | PASS | 6. 2023, č. j. 12 Co 25/2023-148, není podle § 237 o. s. ř. přípustné, neboť není splněn žádný z předpokladů přípustnosti dovolání uvedených v tomto ustanovení. 11. Dovoláním napadený rozsudek odvolacího soudu je v žalobcem předestřené otáz... |

### trest odnětí svobody

- Expected decision: `answerable`
- Actual decision: `answerable`
- Validation: **PASS**
- Confidence: **0.960**
- Reason: Top results contain direct source evidence for the expected answerable query.
- Recommended user message: The available NS decisions contain relevant support for this query. The answer should be based on the retrieved chunks.
- Top result score: **0.765762**
- Top result case_number: `3 Tdo 19/2025`
- Top result document_type: `USNESENÍ`
- Top result legal_area: `criminal`
- Top result section_type: `reasoning`
- Matched terms: Trest odnětí svobody, trest odnětí svobody
- Missing terms: -

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.765762 | 3 Tdo 19/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:3.TDO.19.2025.1__chunk_0007 | ECLI:CZ:NS:2025:3.TDO.19.2025.1 | PASS | 13. Z takto vymezených hledisek dovolatel namítl, že trest, jež mu byl uložen, neodpovídá zásadám pro ukládání trestu. S ohledem na jeho částečné doznání a odstup času od incidentu, kdy následně žil řádným životem, a s přihlédnutím k závažn... |
| 2 | 0.750247 | 3 Tdo 984/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:3.TDO.984.2024.1__chunk_0020 | ECLI:CZ:NS:2024:3.TDO.984.2024.1 | PASS | 28. Trest, který byl obviněnému uložen, ničím nevybočuje z rámce běžně se vyskytujících případů a je zcela odpovídající jak povaze a závažnosti spáchaného trestného činu, tak i poměrům pachatele. Je třeba připomenout, že trest byl obviněném... |
| 3 | 0.743396 | 8 Tdo 760/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:8.TDO.760.2024.1__chunk_0005 | ECLI:CZ:NS:2024:8.TDO.760.2024.1 | PASS | 134/2002 Sb., účinného do 30. 6. 2008), kterým byl uznán vinným napadeným rozsudkem, odsoudil podle § 252a odst. 3 tr. zák. k trestu odnětí svobody v trvání dvou let, jehož výkon mu podle § 58 odst. 1 a § 59 odst. 1 tr. zák. podmíněně odlož... |
| 4 | 0.730595 | 8 Tdo 1119/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:8.TDO.1119.2024.1__chunk_0005 | ECLI:CZ:NS:2025:8.TDO.1119.2024.1 | PASS | 10. Podle § 58 odst. 6 věty první tr. zákoníku může soud snížit trest odnětí svobody pod dolní hranici trestní sazby též tehdy, jestliže odsuzuje pachatele za přípravu k trestnému činu nebo za pokus trestného činu nebo za pomoc k trestnému... |
| 5 | 0.725357 | 3 Tdo 53/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:3.TDO.53.2025.1__chunk_0004 | ECLI:CZ:NS:2025:3.TDO.53.2025.1 | PASS | 2. Za toto jednání byl obviněný odsouzen podle § 186 odst. 5 tr. zákoníku za použití § 58 odst. 2 písm. b) tr. zákoníku a § 43 odst. 1 tr. zákoníku k úhrnnému trestu odnětí svobody v trvání 4 roků a 6 měsíců. Podle § 56 odst. 2 písm. a) tr.... |

### určení místní příslušnosti

- Expected decision: `answerable`
- Actual decision: `answerable`
- Validation: **PASS**
- Confidence: **0.832**
- Reason: Top results contain source-term overlap in the expected section context.
- Recommended user message: The available NS decisions contain relevant support for this query. The answer should be based on the retrieved chunks.
- Top result score: **0.517164**
- Top result case_number: `29 Nd 63/2025`
- Top result document_type: `USNESENÍ`
- Top result legal_area: `civil`
- Top result section_type: `reasoning`
- Matched terms: určení místní příslušnosti
- Missing terms: -

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.517164 | 29 Nd 63/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:29.ND.63.2025.1__chunk_0002 | ECLI:CZ:NS:2025:29.ND.63.2025.1 | PASS | Odůvodnění: 1. Usnesením ze dne 13. prosince 2024, č. j. 27 Nc 2451/2024-44, vyslovil Okresní soud Praha - západ svou místní nepříslušnost (bod I. výroku), rozhodl, že věc bude po právní moci usnesení předložena Nejvyššímu soudu k určení mí... |
| 2 | 0.516059 | 20 Nd 18/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:20.ND.18.2025.1__chunk_0003 | ECLI:CZ:NS:2025:20.ND.18.2025.1 | PASS | 99/1963 Sb., občanský soudní řád, ve znění pozdějších předpisů – dále též jen „o. s. ř.“). V exekučním návrhu oprávněné a stejně tak v žádosti soudního exekutora je uvedena adresa povinného XY. Lustrací v informačním systému základních regi... |
| 3 | 0.496459 | 24 Nd 34/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:24.ND.34.2025.1__chunk_0004 | ECLI:CZ:NS:2025:24.ND.34.2025.1 | PASS | 9. 2012, sp. zn. 31 Nd 200/2012, uveřejněné ve Sbírce soudních rozhodnutí a stanovisek pod č. 4, ročník 2013). 8. Nejvyšší soud v obdobných situacích vychází při určení místní příslušnosti exekučního soudu ze zásady hospodárnosti řízení zak... |
| 4 | 0.492346 | 26 Nd 573/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:26.ND.573.2024.1__chunk_0003 | ECLI:CZ:NS:2025:26.ND.573.2024.1 | PASS | 9. 2012, sp. zn. 31 Nd 200/2012, uveřejněné pod č. 4/2013 Sbírky soudních rozhodnutí a stanovisek). Nelze tak zjistit podmínky pro určení místní příslušnosti exekučního soudu (§ 45 odst. 2 exekučního řádu). 5. Nejvyšší soud proto podle § 11... |
| 5 | 0.481980 | 7 Td 6/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:7.TD.6.2025.1__chunk_0004 | ECLI:CZ:NS:2025:7.TD.6.2025.1 | PASS | 7. Pro určení místní příslušnosti soudu je podstatné místo, kde obviněná jednala, kde neznámé osobě umožnila užít k výše uvedené transakci svůj účet, tedy konkrétně především na jakém místě se nacházela, když finanční prostředky ze svého úč... |

### zastavení exekuce

- Expected decision: `answerable`
- Actual decision: `answerable`
- Validation: **PASS**
- Confidence: **0.880**
- Reason: Top results contain direct source evidence for the expected answerable query.
- Recommended user message: The available NS decisions contain relevant support for this query. The answer should be based on the retrieved chunks.
- Top result score: **0.772908**
- Top result case_number: `23 Cdo 3535/2024`
- Top result document_type: `USNESENÍ`
- Top result legal_area: `civil`
- Top result section_type: `operative_part`
- Matched terms: zastavení exekuce
- Missing terms: -

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.772908 | 23 Cdo 3535/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:23.CDO.3535.2024.1__chunk_0001 | ECLI:CZ:NS:2025:23.CDO.3535.2024.1 | PASS | takto: Dovolací řízení se zastavuje . |
| 2 | 0.772908 | 20 Cdo 3518/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:20.CDO.3518.2024.1__chunk_0001 | ECLI:CZ:NS:2025:20.CDO.3518.2024.1 | PASS | takto: Dovolací řízení se zastavuje . |
| 3 | 0.745089 | 20 Cdo 15/2025 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:20.CDO.15.2025.1__chunk_0001 | ECLI:CZ:NS:2025:20.CDO.15.2025.1 | PASS | takto: Řízení o dovolání obou povinných se zastavuje . |
| 4 | 0.745089 | 20 Cdo 13/2025 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:20.CDO.13.2025.1__chunk_0001 | ECLI:CZ:NS:2025:20.CDO.13.2025.1 | PASS | takto: Řízení o dovolání obou povinných se zastavuje . |
| 5 | 0.731205 | 20 Cdo 30/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:20.CDO.30.2025.1__chunk_0003 | ECLI:CZ:NS:2025:20.CDO.30.2025.1 | PASS | 120/2001 Sb., o soudních exekutorech a exekuční činnosti (exekuční řád) a o změně dalších zákonů, ve znění pozdějších předpisů (dále též jen „ex. řád“), na jeho základě není možné vést exekuci a je namístě zastavit ji podle § 268 odst. 1 pí... |

### mezinárodní ochrana a azyl

- Expected decision: `insufficient_support`
- Actual decision: `insufficient_support`
- Validation: **PASS**
- Confidence: **0.840**
- Reason: Some vocabulary overlaps appear, but the results remain too weak and indirect to support an answer.
- Recommended user message: The current NS collection does not contain enough direct support for this query. Add more documents or specify a different legal issue.
- Top result score: **0.597794**
- Top result case_number: `8 Tdo 1022/2024`
- Top result document_type: `USNESENÍ`
- Top result legal_area: `criminal`
- Top result section_type: `reasoning`
- Matched terms: mezinárodní ochrana a azyl
- Missing terms: azyl, cizinecké správní řízení, mezinárodní ochrana

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.597794 | 8 Tdo 1022/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:8.TDO.1022.2024.1__chunk_0004 | ECLI:CZ:NS:2024:8.TDO.1022.2024.1 | PASS | 104/2013 Sb., o mezinárodní justiční spolupráci ve věcech trestních, ve znění pozdějších předpisů (dále jen „zákon o mezinárodní justiční spolupráci“), čl. 14 sdělení č. 549/1992 Sb., federálního ministerstva zahraničních věcí o sjednání Ev... |
| 2 | 0.569177 | 30 Cdo 844/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:30.CDO.844.2024.1__chunk_0011 | ECLI:CZ:NS:2024:30.CDO.844.2024.1 | PASS | 25. Dovolání je přípustné pro řešení otázky, zda v případě zásahu do práv žalobkyně spočívajícím v tom, že byla nucena po návratu ze zahraničí strávit 5 dní doma omezena ve svobodě pohybu institutem „samoizolace“ ve smyslu ochranného opatře... |
| 3 | 0.561634 | 30 Cdo 844/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:30.CDO.844.2024.1__chunk_0002 | ECLI:CZ:NS:2024:30.CDO.844.2024.1 | PASS | Odůvodnění: I. Dosavadní průběh řízení 1. Žalobkyně se podanou žalobou domáhala zadostiučinění za nemajetkovou újmu způsobenou nezákonnými opatřeními Ministerstva zdravotnictví, v jejichž důsledku se musela po dobu 5 dnů (od 29. 8. 2021 do... |
| 4 | 0.554719 | 25 Cdo 3217/2023 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:25.CDO.3217.2023.1__chunk_0008 | ECLI:CZ:NS:2024:25.CDO.3217.2023.1 | PASS | 12. 2019, sen. zn. 29 ICdo 96/2016). V každém případě musí být obsah cizího (cizozemského) práva zjištěn tak úplně a spolehlivě, jak by tomu bylo (mělo být) v případě, kdyby spor nebo jiná právní věc byly řešeny a rozhodovány v domovské zem... |
| 5 | 0.542176 | 25 Cdo 3217/2023 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:25.CDO.3217.2023.1__chunk_0007 | ECLI:CZ:NS:2024:25.CDO.3217.2023.1 | PASS | 91/2012 Sb., o mezinárodním právu soukromém, ve znění pozdějších předpisů, pokud z jiných ustanovení tohoto zákona nevyplývá něco jiného, je třeba zahraničního práva, jehož se má použít podle ustanovení tohoto zákona, používat i bez návrhu... |

### správní vyhoštění cizince

- Expected decision: `insufficient_support`
- Actual decision: `insufficient_support`
- Validation: **PASS**
- Confidence: **0.840**
- Reason: Some vocabulary overlaps appear, but the results remain too weak and indirect to support an answer.
- Recommended user message: The current NS collection does not contain enough direct support for this query. Add more documents or specify a different legal issue.
- Top result score: **0.593625**
- Top result case_number: `6 Tdo 827/2024`
- Top result document_type: `USNESENÍ`
- Top result legal_area: `criminal`
- Top result section_type: `reasoning`
- Matched terms: správní vyhoštění cizince
- Missing terms: cizinec, pobytové správní řízení, správní vyhoštění

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.593625 | 6 Tdo 827/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:6.TDO.827.2024.1__chunk_0026 | ECLI:CZ:NS:2024:6.TDO.827.2024.1 | PASS | 51. Lichý je rovněž výklad dovolatelů dovozujících, že stíhaný přečin lze spáchat pouze, pokud dojde k porušení povinnosti spočívající v oznámení přeshraničního transportu nebo získání souhlasu s ním od příslušného orgánu. Jestliže totiž mi... |
| 2 | 0.592014 | 6 Tdo 827/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:6.TDO.827.2024.1__chunk_0011 | ECLI:CZ:NS:2024:6.TDO.827.2024.1 | PASS | 21. Za opodstatněnou nelze mít ani námitku dovolatelů poukazujících na to, že předmětný odpad nepřepravili přes hranice státu oni, nýbrž že uzavřeli pouze obchodní dohodu s mezinárodním prvkem. Dovolatelé zde opomíjejí, že za jednání pachat... |
| 3 | 0.589321 | 30 Cdo 844/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:30.CDO.844.2024.1__chunk_0013 | ECLI:CZ:NS:2024:30.CDO.844.2024.1 | PASS | 29. V daném případě je po skutkové stránce bez pochyb, že žalobkyně byla po svém návratu ze zahraničí nucena strávit pět dní v samoizolaci, a to na základě mimořádného opatření (viz výše), které bylo odvolacím soudem posouzeno jako nezákonn... |
| 4 | 0.575855 | 25 Cdo 3217/2023 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:25.CDO.3217.2023.1__chunk_0008 | ECLI:CZ:NS:2024:25.CDO.3217.2023.1 | PASS | 12. 2019, sen. zn. 29 ICdo 96/2016). V každém případě musí být obsah cizího (cizozemského) práva zjištěn tak úplně a spolehlivě, jak by tomu bylo (mělo být) v případě, kdyby spor nebo jiná právní věc byly řešeny a rozhodovány v domovské zem... |
| 5 | 0.562217 | 30 Cdo 844/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:30.CDO.844.2024.1__chunk_0011 | ECLI:CZ:NS:2024:30.CDO.844.2024.1 | PASS | 25. Dovolání je přípustné pro řešení otázky, zda v případě zásahu do práv žalobkyně spočívajícím v tom, že byla nucena po návratu ze zahraničí strávit 5 dní doma omezena ve svobodě pohybu institutem „samoizolace“ ve smyslu ochranného opatře... |

### ochrana osobních údajů podle GDPR

- Expected decision: `insufficient_support`
- Actual decision: `insufficient_support`
- Validation: **PASS**
- Confidence: **0.840**
- Reason: Some vocabulary overlaps appear, but the results remain too weak and indirect to support an answer.
- Recommended user message: The current NS collection does not contain enough direct support for this query. Add more documents or specify a different legal issue.
- Top result score: **0.599194**
- Top result case_number: `33 Cdo 79/2024`
- Top result document_type: `USNESENÍ`
- Top result legal_area: `civil`
- Top result section_type: `appeal_instruction`
- Matched terms: ochrana osobních údajů podle GDPR
- Missing terms: GDPR, osobní údaje, správce údajů

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.599194 | 33 Cdo 79/2024 | USNESENÍ | civil | appeal_instruction | ECLI:CZ:NS:2025:33.CDO.79.2024.1__chunk_0008 | ECLI:CZ:NS:2025:33.CDO.79.2024.1 | PASS | 11. 2013, sp. zn. I. ÚS 3512/11 (dostupném, stejně jako další uváděná rozhodnutí tohoto soudu na webových stránkách www.usoud.cz ), Ústavní soud (kromě jiného) uvedl, že ochrana spotřebitele spadá mezi jednu ze sdílených politik Evropské un... |
| 2 | 0.517372 | 7 Pzo 5/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:7.PZO.5.2024.1__chunk_0010 | ECLI:CZ:NS:2025:7.PZO.5.2024.1 | PASS | 15. Z tohoto důvodu je pak možné v projednávané věci dospět k závěru, že vymezení podmínek, za nichž může být vydán příkaz k odposlechu a záznamu telekomunikačního provozu, resp. příkaz ke sdělení údajů o uskutečněném telekomunikačním provo... |
| 3 | 0.511832 | 5 Tdo 1128/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:5.TDO.1128.2024.1__chunk_0016 | ECLI:CZ:NS:2025:5.TDO.1128.2024.1 | PASS | 5. 2001 o harmonizaci určitých aspektů autorského práva a práv s ním souvisejících v informační společnosti (dále jen „Směrnice“). V souladu se závěry citovaného rozhodnutí namítl, že svým jednáním vědomě do práv autorů nezasahoval, nenahrá... |
| 4 | 0.494208 | 7 Pzo 5/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:7.PZO.5.2024.1__chunk_0009 | ECLI:CZ:NS:2025:7.PZO.5.2024.1 | PASS | 7. 2024, sp. zn. 3 Pzo 1/2024). 14. Nelze přisvědčit ani námitce navrhovatele, že v příkazech absentuje výslovné zdůvodnění důvodnosti nezbytnosti zásahu do soukromí jeho pacientů. Tato námitka souvisela s postavením navrhovatele jako lékař... |
| 5 | 0.467388 | 5 Tdo 1128/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:5.TDO.1128.2024.1__chunk_0023 | ECLI:CZ:NS:2025:5.TDO.1128.2024.1 | PASS | 5. 2020, sp. zn. 5 Tdo 178/2020, ze dne 27. 2. 2013, sp. zn. 8 Tdo 137/2013, uveřejněné pod č. 7/2014 Sb. rozh. tr.), protože umožňuje, aby kdokoli mohl mít k chráněnému dílu či jeho rozmnoženině přístup na místě a v čase podle své vlastní... |

### odpočet DPH u daně z přidané hodnoty

- Expected decision: `insufficient_support`
- Actual decision: `insufficient_support`
- Validation: **PASS**
- Confidence: **0.840**
- Reason: Some vocabulary overlaps appear, but the results remain too weak and indirect to support an answer.
- Recommended user message: The current NS collection does not contain enough direct support for this query. Add more documents or specify a different legal issue.
- Top result score: **0.528100**
- Top result case_number: `28 Cdo 3321/2024`
- Top result document_type: `USNESENÍ`
- Top result legal_area: `civil`
- Top result section_type: `reasoning`
- Matched terms: -
- Missing terms: DPH, daň z přidané hodnoty, daňový odpočet

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.528100 | 28 Cdo 3321/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:28.CDO.3321.2024.1__chunk_0008 | ECLI:CZ:NS:2025:28.CDO.3321.2024.1 | PASS | 12. 2024 (dále „advokátní tarif“), a náhrady paušálně určených hotových výdajů za jeden úkon právní služby ve výši 300 Kč – § 11 odst. 1 písm. k) a § 13 odst. 4 advokátního tarifu. Protože zástupce dalšího účastníka řízení je plátcem daně z... |
| 2 | 0.453889 | 20 Cdo 3371/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:20.CDO.3371.2024.1__chunk_0004 | ECLI:CZ:NS:2025:20.CDO.3371.2024.1 | PASS | 7) Podle § 337c odst. 1 písm. a) a b) o. s. ř., ve znění zákona č. 291/2017 Sb., se z rozdělované podstaty uspokojují postupně podle těchto skupin: a) pohledávky nákladů vzniklých státu v tomto řízení, b) pohledávky související se správou d... |
| 3 | 0.448508 | 24 Cdo 3585/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:24.CDO.3585.2024.1__chunk_0006 | ECLI:CZ:NS:2025:24.CDO.3585.2024.1 | PASS | 4. 2020 ve výši 800,- Kč, za 5 úkonů vykonaných v době od 17. 4. 2020 do 31. 12. 2021 ve výši 6 000,- Kč a za 4 úkony provedené v době od 1. 1. 2022 do 3. 2. 2022 ve výši 6 750,- Kč a že jsou splněny předpoklady podle ustanovení § 12 odst.... |
| 4 | 0.439609 | 20 Cdo 2839/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2024:20.CDO.2839.2024.1__chunk_0002 | ECLI:CZ:NS:2024:20.CDO.2839.2024.1 | PASS | Odůvodnění: Oprávněná podala dne 4. června 2024 exekuční návrh na zřízení exekutorského zástavního práva na nemovitých věcech ve vlastnictví povinného (specifikované v návrhu) k zajištění pohledávky ve výši 2 124 285,77 Kč podle notářského... |
| 5 | 0.423319 | 29 Cdo 275/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:29.CDO.275.2024.1__chunk_0004 | ECLI:CZ:NS:2025:29.CDO.275.2024.1 | PASS | 177/1996 Sb., o odměnách advokátů a náhradách advokátů za poskytování právních služeb (advokátního tarifu), ve znění účinném k datu podání vyjádření, činí (z tarifní hodnoty 421.400,- Kč) částku 10.020,- Kč, a z paušální částky náhrady hoto... |

### stavební povolení a územní rozhodnutí

- Expected decision: `insufficient_support`
- Actual decision: `insufficient_support`
- Validation: **PASS**
- Confidence: **0.840**
- Reason: Some vocabulary overlaps appear, but the results remain too weak and indirect to support an answer.
- Recommended user message: The current NS collection does not contain enough direct support for this query. Add more documents or specify a different legal issue.
- Top result score: **0.739170**
- Top result case_number: `22 Cdo 1151/2024`
- Top result document_type: `ROZSUDEK`
- Top result legal_area: `civil`
- Top result section_type: `reasoning`
- Matched terms: stavební povolení a územní rozhodnutí
- Missing terms: stavební povolení, stavební úřad, územní rozhodnutí

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.739170 | 22 Cdo 1151/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2025:22.CDO.1151.2024.1__chunk_0004 | ECLI:CZ:NS:2025:22.CDO.1151.2024.1 | PASS | 7. Ve způsobu vypořádání spoluvlastnictví dospěl ke stejnému závěru jako soud prvního stupně, tedy že reálné dělení blokačních pozemků není dobře možné, svůj názor ovšem opřel o jinou skutečnost, konkrétně o veřejný zájem na vybudování ploc... |
| 2 | 0.678375 | 28 Cdo 2670/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:28.CDO.2670.2024.3__chunk_0008 | ECLI:CZ:NS:2024:28.CDO.2670.2024.3 | PASS | 10. 2021, sp. zn. 28 Cdo 2244/2021), nezabýval se odvolací soud v individuálních skutkových poměrech projednávané věci (pozemek je zatížen věcným břemenem chůze a jízdy ve prospěch sousedící stavby č. p. XY a věcným břemenem práva zřídit a... |
| 3 | 0.673138 | 22 Cdo 108/2025 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2025:22.CDO.108.2025.1__chunk_0002 | ECLI:CZ:NS:2025:22.CDO.108.2025.1 | PASS | Odůvodnění: I. Dosavadní průběh řízení 1. Okresní soud v Lounech (dále jen „soud prvního stupně“) rozsudkem ze dne 16. 5. 2024, č. j. 12 C 152/2023-52, určil, že pozemek p. č. XY a pozemek p. č. XY, jehož součástí je stavba – rodinný dům č.... |
| 4 | 0.671107 | 28 Cdo 2670/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:28.CDO.2670.2024.3__chunk_0007 | ECLI:CZ:NS:2024:28.CDO.2670.2024.3 | PASS | 12. Rozhodovací praxe dovolacího soudu i Ústavního soudu – a to již jde-li o restituci původní, nikoliv poskytnutím náhradního plnění (zde v podobě jiného pozemku) – přitom vychází z toho, že zákon o půdě, stejně tak jako jiné restituční př... |
| 5 | 0.662086 | 24 Cdo 671/2024; 24 Cdo 675/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2024:24.CDO.671.2024.1__chunk_0002 | ECLI:CZ:NS:2024:24.CDO.671.2024.1 | PASS | Odůvodnění: 1. Okresní soud v Pelhřimově rozsudkem ze dne 31. 5. 2023, č. j. 5 C 255/2015-1439, rozhodl o určení vlastnického práva k celkem devíti jednotkám v domě č. p. XY v XY. Jednalo se o byty, garáže a dílnu. Stručně řečeno, u jednote... |

### náhrada nákladů dovolacího řízení

- Expected decision: `ask_for_clarification`
- Actual decision: `ask_for_clarification`
- Validation: **PASS**
- Confidence: **0.940**
- Reason: The query targets a repeated procedural outcome phrase rather than a concrete legal issue.
- Recommended user message: The query is too broad. Please specify the legal area, factual situation, case number, or what legal question should be answered.
- Top result score: **0.770444**
- Top result case_number: `28 Cdo 3321/2024`
- Top result document_type: `USNESENÍ`
- Top result legal_area: `civil`
- Top result section_type: `reasoning`
- Matched terms: náhrada nákladů dovolacího řízení
- Missing terms: Chcete náklady po odmítnutí dovolání, po zastavení řízení, nebo po meritorním rozhodnutí?, Které rozhodnutí nebo právní problém v dovolacím řízení vás zajímá?

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.770444 | 28 Cdo 3321/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:28.CDO.3321.2024.1__chunk_0008 | ECLI:CZ:NS:2025:28.CDO.3321.2024.1 | PASS | 12. 2024 (dále „advokátní tarif“), a náhrady paušálně určených hotových výdajů za jeden úkon právní služby ve výši 300 Kč – § 11 odst. 1 písm. k) a § 13 odst. 4 advokátního tarifu. Protože zástupce dalšího účastníka řízení je plátcem daně z... |
| 2 | 0.769617 | 29 Cdo 275/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:29.CDO.275.2024.1__chunk_0004 | ECLI:CZ:NS:2025:29.CDO.275.2024.1 | PASS | 177/1996 Sb., o odměnách advokátů a náhradách advokátů za poskytování právních služeb (advokátního tarifu), ve znění účinném k datu podání vyjádření, činí (z tarifní hodnoty 421.400,- Kč) částku 10.020,- Kč, a z paušální částky náhrady hoto... |
| 3 | 0.768661 | 28 Cdo 3513/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:28.CDO.3513.2024.1__chunk_0006 | ECLI:CZ:NS:2025:28.CDO.3513.2024.1 | PASS | 15. O nákladech dovolacího řízení bylo rozhodnuto v intencích § 243 odst. 3 věty první, § 224 odst. 1, § 151 odst. 1 části věty před středníkem a § 146 odst. 3 o. s. ř.; dovolání žalované bylo odmítnuto a na straně žalobce lze za účelně vyn... |
| 4 | 0.768099 | 27 Cdo 2699/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:27.CDO.2699.2024.1__chunk_0005 | ECLI:CZ:NS:2025:27.CDO.2699.2024.1 | PASS | 13. Výrok o náhradě nákladů dovolacího řízení se opírá o § 243c odst. 3, § 224 odst. 1 a § 146 odst. 3 o. s. ř., když dovolání žalobkyně bylo odmítnuto a žalovanému vzniklo právo na náhradu účelně vynaložených nákladů dovolacího řízení. 14.... |
| 5 | 0.760259 | 25 Cdo 2348/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:25.CDO.2348.2024.1__chunk_0002 | ECLI:CZ:NS:2024:25.CDO.2348.2024.1 | PASS | Odůvodnění: 1. Žalobkyně se podanou žalobou domáhala 998.089 Kč s příslušenstvím z titulu odpovědnosti advokáta za škodu. Částka 203.186 Kč představovala marně vynaložené náklady řízení, částka 594.903 Kč kapitalizovaný úrok z prodlení z 1.... |

### zjevně neopodstatněné dovolání

- Expected decision: `ask_for_clarification`
- Actual decision: `ask_for_clarification`
- Validation: **PASS**
- Confidence: **0.940**
- Reason: The query describes a broad criminal dovolání outcome without isolating the underlying issue.
- Recommended user message: The query is too broad. Please specify the legal area, factual situation, case number, or what legal question should be answered.
- Top result score: **0.736710**
- Top result case_number: `21 Cdo 44/2025`
- Top result document_type: `USNESENÍ`
- Top result legal_area: `civil`
- Top result section_type: `operative_part`
- Matched terms: -
- Missing terms: Chcete rozhodnutí podle § 265i odst. 1 písm. e) tr. ř., nebo širší výklad zjevné neopodstatněnosti?, Který dovolací důvod nebo trestněprávní problém máte na mysli?

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.736710 | 21 Cdo 44/2025 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:21.CDO.44.2025.1__chunk_0001 | ECLI:CZ:NS:2025:21.CDO.44.2025.1 | PASS | takto: Dovolání povinného se odmítá. |
| 2 | 0.731032 | 21 Cdo 2658/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:21.CDO.2658.2024.3__chunk_0001 | ECLI:CZ:NS:2025:21.CDO.2658.2024.3 | PASS | takto: Dovolání povinného se odmítá . |
| 3 | 0.686126 | 27 Cdo 1921/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:27.CDO.1921.2024.1__chunk_0001 | ECLI:CZ:NS:2025:27.CDO.1921.2024.1 | PASS | takto: Dovolání se odmítá . |
| 4 | 0.686126 | 26 Cdo 125/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2024:26.CDO.125.2024.1__chunk_0001 | ECLI:CZ:NS:2024:26.CDO.125.2024.1 | PASS | takto: Dovolání se odmítá . |
| 5 | 0.686126 | 29 NSCR 70/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:29.NSCR.70.2024.1__chunk_0001 | ECLI:CZ:NS:2025:29.NSCR.70.2024.1 | PASS | takto: Dovolání se odmítá . |

### odmítnutí dovolání

- Expected decision: `ask_for_clarification`
- Actual decision: `ask_for_clarification`
- Validation: **PASS**
- Confidence: **0.940**
- Reason: The query is an outcome label shared by many unrelated matters.
- Recommended user message: The query is too broad. Please specify the legal area, factual situation, case number, or what legal question should be answered.
- Top result score: **0.922904**
- Top result case_number: `21 Cdo 1566/2024`
- Top result document_type: `USNESENÍ`
- Top result legal_area: `civil`
- Top result section_type: `operative_part`
- Matched terms: odmítnutí dovolání
- Missing terms: Jde vám o občanskoprávní nebo trestní dovolání?, Má být dotaz navázán na konkrétní zákonné ustanovení nebo procesní důvod odmítnutí?

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.922904 | 21 Cdo 1566/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:21.CDO.1566.2024.1__chunk_0001 | ECLI:CZ:NS:2025:21.CDO.1566.2024.1 | PASS | takto: Dovolání se odmítá . |
| 2 | 0.922904 | 27 Cdo 1921/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:27.CDO.1921.2024.1__chunk_0001 | ECLI:CZ:NS:2025:27.CDO.1921.2024.1 | PASS | takto: Dovolání se odmítá . |
| 3 | 0.922904 | 26 Cdo 125/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2024:26.CDO.125.2024.1__chunk_0001 | ECLI:CZ:NS:2024:26.CDO.125.2024.1 | PASS | takto: Dovolání se odmítá . |
| 4 | 0.922904 | 29 NSCR 1/2025 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:29.NSCR.1.2025.1__chunk_0001 | ECLI:CZ:NS:2025:29.NSCR.1.2025.1 | PASS | takto: Dovolání se odmítá . |
| 5 | 0.922904 | 20 Cdo 875/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:20.CDO.875.2024.1__chunk_0001 | ECLI:CZ:NS:2025:20.CDO.875.2024.1 | PASS | takto: Dovolání se odmítá . |

### rodinný dům

- Expected decision: `ask_for_clarification`
- Actual decision: `ask_for_clarification`
- Validation: **PASS**
- Confidence: **0.940**
- Reason: The query names an object of dispute, not the legal question to answer.
- Recommended user message: The query is too broad. Please specify the legal area, factual situation, case number, or what legal question should be answered.
- Top result score: **0.383765**
- Top result case_number: `26 Cdo 1854/2024`
- Top result document_type: `ROZSUDEK`
- Top result legal_area: `civil`
- Top result section_type: `reasoning`
- Matched terms: rodinný dům
- Missing terms: Jde o vlastnictví, vady, bydlení, náhradu škody, nebo jiný spor o rodinný dům?, Má být dotaz zúžen na konkrétní právní otázku nebo skutkový typ?

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.383765 | 26 Cdo 1854/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2025:26.CDO.1854.2024.1__chunk_0008 | ECLI:CZ:NS:2025:26.CDO.1854.2024.1 | PASS | 16. Pojem „jednotka“ použitý v § 1196 odst. 2 o. z. je však třeba vykládat ve spojení s § 1159 o. z., jenž stanoví, že jednotka zahrnuje nejen byt (jako prostorově oddělenou část domu), ale také podíl na společných částech (vzájemně spojené... |
| 2 | 0.383488 | 3 Tdo 984/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:3.TDO.984.2024.1__chunk_0006 | ECLI:CZ:NS:2024:3.TDO.984.2024.1 | PASS | 2. 2023 do 20. 2. 2023, kdy měl mít společné nezletilé děti v péči, avšak od poškozené věděl, že zdravotní stav nezletilého syna toto neumožňuje, pod záminkou být s dětmi apeloval právě opět v přítomnosti společných dětí na poškozenou, aby... |
| 3 | 0.376240 | 26 Cdo 439/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:26.CDO.439.2024.1__chunk_0003 | ECLI:CZ:NS:2024:26.CDO.439.2024.1 | PASS | 6. 2012, č. j. 11 Co 155/2012-78, bylo zrušeno právo společného nájmu tam specifikovaného družstevního bytu (předmětného bytu) a garáže (blíže označené) a společné členství v bytovém družstvu s tím, že výlučnou členkou družstva a výlučnou n... |
| 4 | 0.353452 | 7 Tdo 1096/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:7.TDO.1096.2024.1__chunk_0004 | ECLI:CZ:NS:2025:7.TDO.1096.2024.1 | PASS | 6. Pokud jde o koncept sdílené újmy, obviněný uvedl, že z rozsudku soudu prvního stupně a ze spisu vyplynulo, že celá rodina poškozeného udržuje mezi sebou velmi dobré vztahy, stojí při sobě a vzájemně se podporuje. Manželka, děti a tchán s... |
| 5 | 0.340566 | 24 Cdo 671/2024; 24 Cdo 675/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2024:24.CDO.671.2024.1__chunk_0002 | ECLI:CZ:NS:2024:24.CDO.671.2024.1 | PASS | Odůvodnění: 1. Okresní soud v Pelhřimově rozsudkem ze dne 31. 5. 2023, č. j. 5 C 255/2015-1439, rozhodl o určení vlastnického práva k celkem devíti jednotkám v domě č. p. XY v XY. Jednalo se o byty, garáže a dílnu. Stručně řečeno, u jednote... |

### dovolání

- Expected decision: `ask_for_clarification`
- Actual decision: `ask_for_clarification`
- Validation: **PASS**
- Confidence: **0.900**
- Reason: Results span too many documents or legal contexts to justify a single direct answer.
- Recommended user message: The query is too broad. Please specify the legal area, factual situation, case number, or what legal question should be answered.
- Top result score: **0.574465**
- Top result case_number: `3 Tdo 984/2024`
- Top result document_type: `USNESENÍ`
- Top result legal_area: `criminal`
- Top result section_type: `reasoning`
- Matched terms: dovolání
- Missing terms: Jaký dovolací důvod nebo právní problém řešíte?, Má jít o přípustnost, odmítnutí, náklady, nebo konkrétní hmotněprávní otázku?

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.574465 | 3 Tdo 984/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:3.TDO.984.2024.1__chunk_0022 | ECLI:CZ:NS:2024:3.TDO.984.2024.1 | PASS | 32. Jestliže obviněný ve svém dovolání požádal, aby |
| 2 | 0.567749 | 30 Cdo 308/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:30.CDO.308.2025.1__chunk_0004 | ECLI:CZ:NS:2025:30.CDO.308.2025.1 | PASS | 2. 2015, sp. zn. II. ÚS 2716/13). Ústavní soud se dále k otázce náležitostí dovolání vyjádřil v usnesení ze dne 26. 6. 2014, sp. zn. III. ÚS 1675/14, kde přiléhavě vysvětlil účel povinnosti dovolatele uvést, v čem konkrétně spatřuje splnění... |
| 3 | 0.562130 | 23 Cdo 434/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2024:23.CDO.434.2024.1__chunk_0004 | ECLI:CZ:NS:2024:23.CDO.434.2024.1 | PASS | 9. Podle § 241a odst. 2 o. s. ř. v dovolání musí být vedle obecných náležitostí (§ 42 odst. 4) uvedeno, proti kterému rozhodnutí směřuje, v jakém rozsahu se rozhodnutí napadá, vymezení důvodu dovolání, v čem dovolatel spatřuje splnění předp... |
| 4 | 0.556672 | 33 Cdo 889/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:33.CDO.889.2024.1__chunk_0005 | ECLI:CZ:NS:2025:33.CDO.889.2024.1 | PASS | 11. V rozsudku ze dne 1. |
| 5 | 0.552189 | 21 Cdo 44/2025 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:21.CDO.44.2025.1__chunk_0001 | ECLI:CZ:NS:2025:21.CDO.44.2025.1 | PASS | takto: Dovolání povinného se odmítá. |

### místní příslušnost

- Expected decision: `ask_for_clarification`
- Actual decision: `ask_for_clarification`
- Validation: **PASS**
- Confidence: **0.900**
- Reason: Results span too many documents or legal contexts to justify a single direct answer.
- Recommended user message: The query is too broad. Please specify the legal area, factual situation, case number, or what legal question should be answered.
- Top result score: **0.475141**
- Top result case_number: `29 Nd 63/2025`
- Top result document_type: `USNESENÍ`
- Top result legal_area: `civil`
- Top result section_type: `reasoning`
- Matched terms: místní příslušnost
- Missing terms: Jde o exekuční věc, civilní spor, nebo jiný typ řízení?, Má být dotaz zúžen na § 11 odst. 3 o. s. ř. nebo na konkrétní skutkovou situaci?

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.475141 | 29 Nd 63/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:29.ND.63.2025.1__chunk_0002 | ECLI:CZ:NS:2025:29.ND.63.2025.1 | PASS | Odůvodnění: 1. Usnesením ze dne 13. prosince 2024, č. j. 27 Nc 2451/2024-44, vyslovil Okresní soud Praha - západ svou místní nepříslušnost (bod I. výroku), rozhodl, že věc bude po právní moci usnesení předložena Nejvyššímu soudu k určení mí... |
| 2 | 0.458850 | 20 Nd 18/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:20.ND.18.2025.1__chunk_0003 | ECLI:CZ:NS:2025:20.ND.18.2025.1 | PASS | 99/1963 Sb., občanský soudní řád, ve znění pozdějších předpisů – dále též jen „o. s. ř.“). V exekučním návrhu oprávněné a stejně tak v žádosti soudního exekutora je uvedena adresa povinného XY. Lustrací v informačním systému základních regi... |
| 3 | 0.443376 | 24 Nd 34/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:24.ND.34.2025.1__chunk_0004 | ECLI:CZ:NS:2025:24.ND.34.2025.1 | PASS | 9. 2012, sp. zn. 31 Nd 200/2012, uveřejněné ve Sbírce soudních rozhodnutí a stanovisek pod č. 4, ročník 2013). 8. Nejvyšší soud v obdobných situacích vychází při určení místní příslušnosti exekučního soudu ze zásady hospodárnosti řízení zak... |
| 4 | 0.436403 | 25 Nd 86/2025 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:25.ND.86.2025.1__chunk_0001 | ECLI:CZ:NS:2025:25.ND.86.2025.1 | PASS | takto: Věc vedenou u Okresního soudu v Ústí nad Labem pod sp. zn. 72 EXE 3512/2024 projedná a rozhodne Okresní soud v Ústí nad Labem. |
| 5 | 0.434284 | 5 Tdo 318/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:5.TDO.318.2024.1__chunk_0024 | ECLI:CZ:NS:2024:5.TDO.318.2024.1 | PASS | 18/2006-II. Sb. rozh. tr.). Odkázat lze i na odbornou literaturu – viz např. ŠÁMAL, P., PÚRY, F., SOTOLÁŘ, A., ŠTENGLOVÁ, I. Podnikání a ekonomická kriminalita v České republice . 1. vydání. Praha: C. H. Beck, 2001, s. 266. 43. Majetek obch... |

## Final Recommendation
- PASS: the deterministic retrieval decision layer cleanly separates answerable, insufficient-support, and clarification-needed queries for the current NS collection.
