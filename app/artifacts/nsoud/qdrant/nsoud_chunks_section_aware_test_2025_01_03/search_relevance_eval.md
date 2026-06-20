# NSoud Search Relevance Evaluation

- Status: **WARN**
- Target collection: `nsoud_chunks_section_aware_test_2025_01_03`
- Dataset path: `/app/app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/relevance_eval_dataset.json`
- Collection exists: **yes**
- Collection point count: **1862**
- Collection vector size: **768**
- Old collection unchanged: **True**
- Metadata validation: **PASS**
- JSON report path: `/app/app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/search_relevance_eval.json`

## Positive Answerable Summary

| query | expected_behavior | actual_label | top_score | result_count | metadata_validation | notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| bezdůvodné obohacení za užívání bytu | retrieval_returns_relevant_chunks | PASS | 0.640370 | 10 | PASS | source case evidence found at rank 2 with source-term overlap at rank 2 |
| dovolací důvod podle § 265b odst. 1 písm. g) | retrieval_returns_relevant_chunks | PASS | 0.872580 | 10 | PASS | top result matches source terms with expected section type and score 0.873 |
| dovolací důvod podle § 265b odst. 1 písm. h) | retrieval_returns_relevant_chunks | PASS | 0.863294 | 10 | PASS | top result matches source terms with expected section type and score 0.863 |
| dovolací důvod podle § 265b odst. 1 písm. m) | retrieval_returns_relevant_chunks | PASS | 0.869987 | 10 | PASS | source case evidence found at rank 2 with source-term overlap at rank 1 |
| místní příslušnosti chybějí nebo je nelze zjistit | retrieval_returns_relevant_chunks | PASS | 0.457301 | 10 | PASS | source case evidence found at rank 2 with source-term overlap at rank 2 |
| nutná obrana vzájemné napadání | retrieval_returns_relevant_chunks | PASS | 0.600907 | 10 | PASS | source chunk evidence found at rank 6 |
| náhradě nákladů dovolacího řízení | retrieval_returns_relevant_chunks | PASS | 0.779172 | 10 | PASS | top result matches source terms with expected section type and score 0.779 |
| odpovědnosti za vady jako slevy z kupní ceny | retrieval_returns_relevant_chunks | WARN | 0.582669 | 10 | PASS | expected section context appears by rank 2 with top score 0.583, but source evidence is indirect |
| pověření a nařízení exekuce | retrieval_returns_relevant_chunks | PASS | 0.786519 | 10 | PASS | top result matches source terms with expected section type and score 0.787 |
| právo bydlení | retrieval_returns_relevant_chunks | PASS | 0.728320 | 10 | PASS | source case evidence found at rank 1 with source-term overlap at rank 1 |
| přípustnost dovolání podle § 237 o. s. ř. | retrieval_returns_relevant_chunks | PASS | 0.869010 | 10 | PASS | top result matches source terms with expected section type and score 0.869 |
| trest odnětí svobody | retrieval_returns_relevant_chunks | PASS | 0.765762 | 10 | PASS | source case evidence found at rank 2 with source-term overlap at rank 2 |
| určení místní příslušnosti | retrieval_returns_relevant_chunks | PASS | 0.517164 | 10 | PASS | source-term overlap appears within top 1 results |
| zastavení exekuce | retrieval_returns_relevant_chunks | PASS | 0.772908 | 10 | PASS | source case evidence found at rank 5 with source-term overlap at rank 5 |

## Negative Not In Batch Summary

| query | expected_behavior | actual_label | top_score | result_count | metadata_validation | notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| mezinárodní ochrana a azyl | insufficient_support | PASS | 0.597794 | 10 | PASS | results are ambiguous across legal contexts and do not show direct support |
| správní vyhoštění cizince | insufficient_support | PASS | 0.593625 | 10 | PASS | unexpected missing-context phrase overlap found at rank 6 / the phrase overlap appears incidental rather than as direct support |
| ochrana osobních údajů podle GDPR | insufficient_support | PASS | 0.599194 | 10 | PASS | results are ambiguous across legal contexts and do not show direct support |
| odpočet DPH u daně z přidané hodnoty | insufficient_support | PASS | 0.528100 | 10 | PASS | unexpected missing-context phrase overlap found at rank 3 / the phrase overlap appears incidental rather than as direct support |
| stavební povolení a územní rozhodnutí | insufficient_support | PASS | 0.739170 | 10 | PASS | results do not contain direct support and remain broad enough for insufficient-support handling |

## Underspecified Summary

| query | expected_behavior | actual_label | top_score | result_count | metadata_validation | notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| náhrada nákladů dovolacího řízení | ask_for_clarification | PASS | 0.770444 | 10 | PASS | results span 10 documents, 2 section types, and 1 legal areas |
| zjevně neopodstatněné dovolání | ask_for_clarification | PASS | 0.736710 | 10 | PASS | results span 10 documents, 1 section types, and 1 legal areas |
| odmítnutí dovolání | ask_for_clarification | PASS | 0.922904 | 10 | PASS | results span 10 documents, 1 section types, and 1 legal areas |
| rodinný dům | ask_for_clarification | PASS | 0.383765 | 10 | PASS | results span 8 documents, 2 section types, and 2 legal areas |
| dovolání | ask_for_clarification | PASS | 0.574465 | 10 | PASS | results span 10 documents, 2 section types, and 2 legal areas |
| místní příslušnost | ask_for_clarification | PASS | 0.475141 | 10 | PASS | results span 10 documents, 3 section types, and 2 legal areas |

## Per-Query Top Results

### bezdůvodné obohacení za užívání bytu

- Expected behavior: `retrieval_returns_relevant_chunks`
- Actual label: **PASS**
- Metadata validation: **PASS**
- Result count: **10**
- Top score: **0.640370**
- Notes: source case evidence found at rank 2 with source-term overlap at rank 2

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata_validation | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.640370 | 26 Cdo 1854/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2025:26.CDO.1854.2024.1__chunk_0002 | ECLI:CZ:NS:2025:26.CDO.1854.2024.1 | PASS | Odůvodnění: 1. Žalobkyně se po žalované domáhala zaplacení částky 100.000 Kč (s příslušenstvím v podobě úroku z prodlení) představující slevu z kupní ceny tam specifikovaného bytu (dále jen „předmětný byt“, resp. „byt“). Uvedla, že v bytě s... |
| 2 | 0.619628 | 26 Cdo 439/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:26.CDO.439.2024.1__chunk_0006 | ECLI:CZ:NS:2024:26.CDO.439.2024.1 | PASS | 8. Žalovaný ve vyjádření k dovolání považuje rozhodnutí odvolacího soudu za správné. Do okamžiku zajištění bytové náhrady se řídí vztah bývalého nájemce a pronajímatele ustanovením § 712a obč. zák. a teprve po zajištění bytové náhrady – jes... |
| 3 | 0.608666 | 26 Cdo 439/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:26.CDO.439.2024.1__chunk_0011 | ECLI:CZ:NS:2024:26.CDO.439.2024.1 | PASS | 20. Ze shora uvedeného se podává, že odvolací soud pochybil, jestliže žalobní nárok na zaplacení 360 000 Kč s příslušenstvím posuzoval výlučně z hlediska bezdůvodného obohacení (jako by bytová náhrada byla zajištěna), a nikoliv též (pro pří... |
| 4 | 0.596197 | 26 Cdo 2198/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:26.CDO.2198.2024.1__chunk_0002 | ECLI:CZ:NS:2024:26.CDO.2198.2024.1 | PASS | Odůvodnění: 1. Žalobce se domáhal, aby byla žalované uložena povinnost předložit mu řádná vyúčtování záloh na služby spojené s užíváním tam specifikovaného bytu (dále jen „byt“) za zúčtovací období od 1. 7. 2018 do 30. 6. 2019, od 1. 7. 201... |
| 5 | 0.593576 | 26 Cdo 439/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:26.CDO.439.2024.1__chunk_0003 | ECLI:CZ:NS:2024:26.CDO.439.2024.1 | PASS | 6. 2012, č. j. 11 Co 155/2012-78, bylo zrušeno právo společného nájmu tam specifikovaného družstevního bytu (předmětného bytu) a garáže (blíže označené) a společné členství v bytovém družstvu s tím, že výlučnou členkou družstva a výlučnou n... |
| 6 | 0.565091 | 22 Cdo 1151/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2025:22.CDO.1151.2024.1__chunk_0006 | ECLI:CZ:NS:2025:22.CDO.1151.2024.1 | PASS | 1. 2008, sp. zn. 2 Cdon 425/96 nebo usnesení Nejvyššího soudu ze dne 28. 7. 2021, sp. zn. 22 Cdo 1376/2021), odvolací soud však vycházel z obvyklé ceny jiných nemovitostí (rozvojových pozemků), kterou by hypoteticky měly, kdyby byly napojen... |
| 7 | 0.551455 | 22 Cdo 3556/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:22.CDO.3556.2024.1__chunk_0002 | ECLI:CZ:NS:2025:22.CDO.3556.2024.1 | PASS | Odůvodnění: 1. Obvodní soud pro Prahu 10 (dále jen „soud prvního stupně“) rozsudkem ze dne 19. 10. 2023, č. j. 26 C 38/2022-502, zamítl žalobu na zrušení společného jmění účastníků (výrok I). Založil žalovanému oprávnění jednat samostatně a... |
| 8 | 0.544681 | 26 Cdo 439/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:26.CDO.439.2024.1__chunk_0002 | ECLI:CZ:NS:2024:26.CDO.439.2024.1 | PASS | Odůvodnění: 1. Žalobkyně se domáhala po žalovaném zaplacení částky 428 272 Kč příslušenstvím – úrokem z prodlení. Žalobu zdůvodnila tím, že se žalovaným byli manželé a společní nájemci družstevního bytu č. 6 a garáže v domě č.p. XY v XY, ul... |
| 9 | 0.535523 | 5 Tdo 318/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:5.TDO.318.2024.1__chunk_0036 | ECLI:CZ:NS:2024:5.TDO.318.2024.1 | PASS | I. (pak by ovšem dávalo logiku, že za postoupení pohledávky reálně žádné finanční prostředky nepřeváděl a jen započetl tuto částku na dluh a k pohybu finančních prostředků by nemuselo dojít). 56. Ze skutkových zjištění soudů nižších stupňů... |
| 10 | 0.530966 | 26 Cdo 125/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2024:26.CDO.125.2024.1__chunk_0003 | ECLI:CZ:NS:2024:26.CDO.125.2024.1 | PASS | 5. V projednávané věci notář JUDr. Eduard Grygar sepsal dne 27. 1. 2021 notářský zápis sp. zn. NZ 13/2021, N 17/2021, v němž oprávněná (nájemkyně bytů) a povinná (podnájemkyně) uzavřely podle § 71b not. ř. dohodu o uznání nároků oprávněné n... |

### dovolací důvod podle § 265b odst. 1 písm. g)

- Expected behavior: `retrieval_returns_relevant_chunks`
- Actual label: **PASS**
- Metadata validation: **PASS**
- Result count: **10**
- Top score: **0.872580**
- Notes: top result matches source terms with expected section type and score 0.873

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata_validation | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.872580 | 6 Tdo 21/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:6.TDO.21.2025.1__chunk_0009 | ECLI:CZ:NS:2025:6.TDO.21.2025.1 | PASS | 15. Dovolací důvod podle § 265b odst. 1 písm. m) tr. ř. je dán, bylo-li rozhodnuto o zamítnutí nebo odmítnutí řádného opravného prostředku proti rozsudku nebo usnesení uvedenému v § 265a odst. 2 písm. a) až g) tr. ř., aniž byly splněny proc... |
| 2 | 0.857696 | 6 Tdo 976/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:6.TDO.976.2024.1__chunk_0005 | ECLI:CZ:NS:2024:6.TDO.976.2024.1 | PASS | 10. Důvod dovolání podle § 265b odst. 1 písm. h) tr. ř. je dán v případech, kdy rozhodnutí spočívá na nesprávném právním posouzení skutku nebo jiném nesprávném hmotněprávním posouzení. Uvedenou formulací zákon vyjadřuje, že tento dovolací d... |
| 3 | 0.854537 | 6 Tdo 936/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:6.TDO.936.2024.1__chunk_0008 | ECLI:CZ:NS:2024:6.TDO.936.2024.1 | PASS | 14. Důvod dovolání podle § 265b odst. 1 písm. h) tr. ř. je dán v případech, kdy rozhodnutí spočívá na nesprávném právním posouzení skutku nebo jiném nesprávném hmotněprávním posouzení. Uvedenou formulací zákon vyjadřuje, že tento dovolací d... |
| 4 | 0.839132 | 4 Tdo 1044/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:4.TDO.1044.2024.1__chunk_0014 | ECLI:CZ:NS:2024:4.TDO.1044.2024.1 | PASS | 33. Dovolací důvod podle § 265b odst. 1 písm. h) tr. ř. je naplněn tehdy, jestliže rozhodnutí spočívá na nesprávném právním posouzení skutku (první alternativa) nebo jiném nesprávném hmotněprávním posouzení (druhá alternativa). Uvedenou for... |
| 5 | 0.831241 | 8 Tdo 1085/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:8.TDO.1085.2024.1__chunk_0009 | ECLI:CZ:NS:2025:8.TDO.1085.2024.1 | PASS | 19. Důvod podle § 265b odst. 1 písm. m) tr. ř. spočívá v tom, že bylo rozhodnuto o zamítnutí nebo odmítnutí řádného opravného prostředku proti rozsudku nebo usnesení uvedenému v § 265a odst. 2 písm. a) až g) tr. ř., aniž byly splněny proces... |
| 6 | 0.819595 | 11 Tdo 679/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0014 | ECLI:CZ:NS:2024:11.TDO.679.2024.1 | PASS | 31. Dovolací důvod podle § 265b odst. 1 písm. h) tr. řádu , je naplněn tehdy, jestliže rozhodnutí spočívá na nesprávném právním posouzení skutku nebo jiném nesprávném hmotněprávním posouzení. V rámci takto vymezeného dovolacího důvodu je mo... |
| 7 | 0.817972 | 11 Tdo 875/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:11.TDO.875.2024.1__chunk_0008 | ECLI:CZ:NS:2025:11.TDO.875.2024.1 | PASS | 17. Dovolací důvod podle § 265b odst. 1 písm. h) tr. řádu , je naplněn tehdy, jestliže rozhodnutí spočívá na nesprávném právním posouzení skutku ( první alternativa ) nebo jiném nesprávném hmotněprávním posouzení ( druhá alternativa ). V rá... |
| 8 | 0.813619 | 8 Tdo 1119/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:8.TDO.1119.2024.1__chunk_0004 | ECLI:CZ:NS:2025:8.TDO.1119.2024.1 | PASS | 6. Protože dovolání lze podat jen z důvodů uvedených v § 265b tr. ř., bylo dále nutno posoudit, zda nejvyšším státním zástupcem vznesené námitky naplňují jím uplatněný zákonem stanovený dovolací důvod, jehož existence je současně nezbytnou... |
| 9 | 0.811990 | 11 Tdo 1127/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:11.TDO.1127.2024.1__chunk_0014 | ECLI:CZ:NS:2025:11.TDO.1127.2024.1 | PASS | 29. V této souvislosti je vhodné připomenout, že dovolací důvod podle § 265b odst. 1 písm. g) tr. řádu umožňuje nápravu v případech, kdy došlo k zásadním (extrémním) vadám ve skutkových zjištěních, přičemž věcně upravuje tři okruhy nejzásad... |
| 10 | 0.807609 | 11 Tdo 75/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:11.TDO.75.2025.1__chunk_0019 | ECLI:CZ:NS:2025:11.TDO.75.2025.1 | PASS | 34. Pokud jde o dovolací důvod uvedený v § 265b odst. 1 písm. g) tr. ř., pak tento uplatnil obviněný M. R. ve všech alternativách a obvinění A. v jeho první a třetí alternativě. K první z alternativ citovaného dovolacího důvodu spočívající... |

### dovolací důvod podle § 265b odst. 1 písm. h)

- Expected behavior: `retrieval_returns_relevant_chunks`
- Actual label: **PASS**
- Metadata validation: **PASS**
- Result count: **10**
- Top score: **0.863294**
- Notes: top result matches source terms with expected section type and score 0.863

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata_validation | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.863294 | 6 Tdo 21/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:6.TDO.21.2025.1__chunk_0009 | ECLI:CZ:NS:2025:6.TDO.21.2025.1 | PASS | 15. Dovolací důvod podle § 265b odst. 1 písm. m) tr. ř. je dán, bylo-li rozhodnuto o zamítnutí nebo odmítnutí řádného opravného prostředku proti rozsudku nebo usnesení uvedenému v § 265a odst. 2 písm. a) až g) tr. ř., aniž byly splněny proc... |
| 2 | 0.854258 | 6 Tdo 976/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:6.TDO.976.2024.1__chunk_0005 | ECLI:CZ:NS:2024:6.TDO.976.2024.1 | PASS | 10. Důvod dovolání podle § 265b odst. 1 písm. h) tr. ř. je dán v případech, kdy rozhodnutí spočívá na nesprávném právním posouzení skutku nebo jiném nesprávném hmotněprávním posouzení. Uvedenou formulací zákon vyjadřuje, že tento dovolací d... |
| 3 | 0.854116 | 6 Tdo 936/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:6.TDO.936.2024.1__chunk_0008 | ECLI:CZ:NS:2024:6.TDO.936.2024.1 | PASS | 14. Důvod dovolání podle § 265b odst. 1 písm. h) tr. ř. je dán v případech, kdy rozhodnutí spočívá na nesprávném právním posouzení skutku nebo jiném nesprávném hmotněprávním posouzení. Uvedenou formulací zákon vyjadřuje, že tento dovolací d... |
| 4 | 0.827733 | 4 Tdo 1044/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:4.TDO.1044.2024.1__chunk_0014 | ECLI:CZ:NS:2024:4.TDO.1044.2024.1 | PASS | 33. Dovolací důvod podle § 265b odst. 1 písm. h) tr. ř. je naplněn tehdy, jestliže rozhodnutí spočívá na nesprávném právním posouzení skutku (první alternativa) nebo jiném nesprávném hmotněprávním posouzení (druhá alternativa). Uvedenou for... |
| 5 | 0.824950 | 8 Tdo 1085/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:8.TDO.1085.2024.1__chunk_0009 | ECLI:CZ:NS:2025:8.TDO.1085.2024.1 | PASS | 19. Důvod podle § 265b odst. 1 písm. m) tr. ř. spočívá v tom, že bylo rozhodnuto o zamítnutí nebo odmítnutí řádného opravného prostředku proti rozsudku nebo usnesení uvedenému v § 265a odst. 2 písm. a) až g) tr. ř., aniž byly splněny proces... |
| 6 | 0.810542 | 11 Tdo 679/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0014 | ECLI:CZ:NS:2024:11.TDO.679.2024.1 | PASS | 31. Dovolací důvod podle § 265b odst. 1 písm. h) tr. řádu , je naplněn tehdy, jestliže rozhodnutí spočívá na nesprávném právním posouzení skutku nebo jiném nesprávném hmotněprávním posouzení. V rámci takto vymezeného dovolacího důvodu je mo... |
| 7 | 0.808662 | 11 Tdo 875/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:11.TDO.875.2024.1__chunk_0008 | ECLI:CZ:NS:2025:11.TDO.875.2024.1 | PASS | 17. Dovolací důvod podle § 265b odst. 1 písm. h) tr. řádu , je naplněn tehdy, jestliže rozhodnutí spočívá na nesprávném právním posouzení skutku ( první alternativa ) nebo jiném nesprávném hmotněprávním posouzení ( druhá alternativa ). V rá... |
| 8 | 0.801820 | 8 Tdo 1119/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:8.TDO.1119.2024.1__chunk_0004 | ECLI:CZ:NS:2025:8.TDO.1119.2024.1 | PASS | 6. Protože dovolání lze podat jen z důvodů uvedených v § 265b tr. ř., bylo dále nutno posoudit, zda nejvyšším státním zástupcem vznesené námitky naplňují jím uplatněný zákonem stanovený dovolací důvod, jehož existence je současně nezbytnou... |
| 9 | 0.799792 | 11 Tdo 1127/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:11.TDO.1127.2024.1__chunk_0014 | ECLI:CZ:NS:2025:11.TDO.1127.2024.1 | PASS | 29. V této souvislosti je vhodné připomenout, že dovolací důvod podle § 265b odst. 1 písm. g) tr. řádu umožňuje nápravu v případech, kdy došlo k zásadním (extrémním) vadám ve skutkových zjištěních, přičemž věcně upravuje tři okruhy nejzásad... |
| 10 | 0.797766 | 11 Tdo 75/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:11.TDO.75.2025.1__chunk_0019 | ECLI:CZ:NS:2025:11.TDO.75.2025.1 | PASS | 34. Pokud jde o dovolací důvod uvedený v § 265b odst. 1 písm. g) tr. ř., pak tento uplatnil obviněný M. R. ve všech alternativách a obvinění A. v jeho první a třetí alternativě. K první z alternativ citovaného dovolacího důvodu spočívající... |

### dovolací důvod podle § 265b odst. 1 písm. m)

- Expected behavior: `retrieval_returns_relevant_chunks`
- Actual label: **PASS**
- Metadata validation: **PASS**
- Result count: **10**
- Top score: **0.869987**
- Notes: source case evidence found at rank 2 with source-term overlap at rank 1

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata_validation | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.869987 | 6 Tdo 21/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:6.TDO.21.2025.1__chunk_0009 | ECLI:CZ:NS:2025:6.TDO.21.2025.1 | PASS | 15. Dovolací důvod podle § 265b odst. 1 písm. m) tr. ř. je dán, bylo-li rozhodnuto o zamítnutí nebo odmítnutí řádného opravného prostředku proti rozsudku nebo usnesení uvedenému v § 265a odst. 2 písm. a) až g) tr. ř., aniž byly splněny proc... |
| 2 | 0.857946 | 6 Tdo 976/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:6.TDO.976.2024.1__chunk_0005 | ECLI:CZ:NS:2024:6.TDO.976.2024.1 | PASS | 10. Důvod dovolání podle § 265b odst. 1 písm. h) tr. ř. je dán v případech, kdy rozhodnutí spočívá na nesprávném právním posouzení skutku nebo jiném nesprávném hmotněprávním posouzení. Uvedenou formulací zákon vyjadřuje, že tento dovolací d... |
| 3 | 0.852923 | 6 Tdo 936/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:6.TDO.936.2024.1__chunk_0008 | ECLI:CZ:NS:2024:6.TDO.936.2024.1 | PASS | 14. Důvod dovolání podle § 265b odst. 1 písm. h) tr. ř. je dán v případech, kdy rozhodnutí spočívá na nesprávném právním posouzení skutku nebo jiném nesprávném hmotněprávním posouzení. Uvedenou formulací zákon vyjadřuje, že tento dovolací d... |
| 4 | 0.837331 | 4 Tdo 1044/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:4.TDO.1044.2024.1__chunk_0014 | ECLI:CZ:NS:2024:4.TDO.1044.2024.1 | PASS | 33. Dovolací důvod podle § 265b odst. 1 písm. h) tr. ř. je naplněn tehdy, jestliže rozhodnutí spočívá na nesprávném právním posouzení skutku (první alternativa) nebo jiném nesprávném hmotněprávním posouzení (druhá alternativa). Uvedenou for... |
| 5 | 0.832481 | 8 Tdo 1085/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:8.TDO.1085.2024.1__chunk_0009 | ECLI:CZ:NS:2025:8.TDO.1085.2024.1 | PASS | 19. Důvod podle § 265b odst. 1 písm. m) tr. ř. spočívá v tom, že bylo rozhodnuto o zamítnutí nebo odmítnutí řádného opravného prostředku proti rozsudku nebo usnesení uvedenému v § 265a odst. 2 písm. a) až g) tr. ř., aniž byly splněny proces... |
| 6 | 0.818582 | 11 Tdo 679/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0014 | ECLI:CZ:NS:2024:11.TDO.679.2024.1 | PASS | 31. Dovolací důvod podle § 265b odst. 1 písm. h) tr. řádu , je naplněn tehdy, jestliže rozhodnutí spočívá na nesprávném právním posouzení skutku nebo jiném nesprávném hmotněprávním posouzení. V rámci takto vymezeného dovolacího důvodu je mo... |
| 7 | 0.816842 | 11 Tdo 875/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:11.TDO.875.2024.1__chunk_0008 | ECLI:CZ:NS:2025:11.TDO.875.2024.1 | PASS | 17. Dovolací důvod podle § 265b odst. 1 písm. h) tr. řádu , je naplněn tehdy, jestliže rozhodnutí spočívá na nesprávném právním posouzení skutku ( první alternativa ) nebo jiném nesprávném hmotněprávním posouzení ( druhá alternativa ). V rá... |
| 8 | 0.811562 | 8 Tdo 1119/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:8.TDO.1119.2024.1__chunk_0004 | ECLI:CZ:NS:2025:8.TDO.1119.2024.1 | PASS | 6. Protože dovolání lze podat jen z důvodů uvedených v § 265b tr. ř., bylo dále nutno posoudit, zda nejvyšším státním zástupcem vznesené námitky naplňují jím uplatněný zákonem stanovený dovolací důvod, jehož existence je současně nezbytnou... |
| 9 | 0.810809 | 11 Tdo 1127/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:11.TDO.1127.2024.1__chunk_0014 | ECLI:CZ:NS:2025:11.TDO.1127.2024.1 | PASS | 29. V této souvislosti je vhodné připomenout, že dovolací důvod podle § 265b odst. 1 písm. g) tr. řádu umožňuje nápravu v případech, kdy došlo k zásadním (extrémním) vadám ve skutkových zjištěních, přičemž věcně upravuje tři okruhy nejzásad... |
| 10 | 0.800744 | 11 Tdo 75/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:11.TDO.75.2025.1__chunk_0019 | ECLI:CZ:NS:2025:11.TDO.75.2025.1 | PASS | 34. Pokud jde o dovolací důvod uvedený v § 265b odst. 1 písm. g) tr. ř., pak tento uplatnil obviněný M. R. ve všech alternativách a obvinění A. v jeho první a třetí alternativě. K první z alternativ citovaného dovolacího důvodu spočívající... |

### místní příslušnosti chybějí nebo je nelze zjistit

- Expected behavior: `retrieval_returns_relevant_chunks`
- Actual label: **PASS**
- Metadata validation: **PASS**
- Result count: **10**
- Top score: **0.457301**
- Notes: source case evidence found at rank 2 with source-term overlap at rank 2

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata_validation | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.457301 | 6 Tdo 1057/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:6.TDO.1057.2024.1__chunk_0010 | ECLI:CZ:NS:2025:6.TDO.1057.2024.1 | PASS | 23. Právě to však v projednávaném případě schází. Geneticky analyzované biologické stopy byly zajištěny pouze na stavebním kolečku a lahvi, nikoli na vlastní kabeláži, takže jejich propojení s krádeží není bezprostřední a onen „pozitivní dů... |
| 2 | 0.440205 | 20 Nd 18/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:20.ND.18.2025.1__chunk_0003 | ECLI:CZ:NS:2025:20.ND.18.2025.1 | PASS | 99/1963 Sb., občanský soudní řád, ve znění pozdějších předpisů – dále též jen „o. s. ř.“). V exekučním návrhu oprávněné a stejně tak v žádosti soudního exekutora je uvedena adresa povinného XY. Lustrací v informačním systému základních regi... |
| 3 | 0.438854 | 6 Tdo 827/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:6.TDO.827.2024.1__chunk_0011 | ECLI:CZ:NS:2024:6.TDO.827.2024.1 | PASS | 21. Za opodstatněnou nelze mít ani námitku dovolatelů poukazujících na to, že předmětný odpad nepřepravili přes hranice státu oni, nýbrž že uzavřeli pouze obchodní dohodu s mezinárodním prvkem. Dovolatelé zde opomíjejí, že za jednání pachat... |
| 4 | 0.405841 | 26 Nd 406/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:26.ND.406.2024.1__chunk_0004 | ECLI:CZ:NS:2025:26.ND.406.2024.1 | PASS | 12. 2009, sp. zn. 4 Nd 374/2009). Ani sama skutečnost, že v řízení může být nutné provést výslech svědků, jež bydlí či sídlí mimo obvod (místně příslušného) soudu, rovněž není důvodem pro přikázání věci soudu, v jehož obvodu svědci žijí či... |
| 5 | 0.404084 | 7 Td 6/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:7.TD.6.2025.1__chunk_0004 | ECLI:CZ:NS:2025:7.TD.6.2025.1 | PASS | 7. Pro určení místní příslušnosti soudu je podstatné místo, kde obviněná jednala, kde neznámé osobě umožnila užít k výše uvedené transakci svůj účet, tedy konkrétně především na jakém místě se nacházela, když finanční prostředky ze svého úč... |
| 6 | 0.398429 | 6 Tdo 827/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:6.TDO.827.2024.1__chunk_0027 | ECLI:CZ:NS:2024:6.TDO.827.2024.1 | PASS | 53. Přisvědčit nelze ani námitkám dovolatelů odmítajících své zavinění. To obvinění rozporují poukazem na to, že oni nebyli těmi, kdo materiál přes hranice státu převážel, a takový status jim nelze připisovat ani úvahou odvolacího soudu o t... |
| 7 | 0.394991 | 11 Pzo 7/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:11.PZO.7.2024.1__chunk_0003 | ECLI:CZ:NS:2025:11.PZO.7.2024.1 | PASS | 10. 2024, když zcela prokazatelně nebyla s obviněným v kontaktu a od 3. 5. 2022 pracuje a žije mimo Českou republiku. Jelikož jí nebyly nikdy sděleny žádné informace, z nichž by bylo možno byť jen dovodit, že by se jakkoliv podílela na tres... |
| 8 | 0.393571 | 3 Tdo 984/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:3.TDO.984.2024.1__chunk_0005 | ECLI:CZ:NS:2024:3.TDO.984.2024.1 | PASS | 2. 2023 různými způsoby kontaktoval poškozenou, přestože od ní věděl, že si kontakt s ním vyjma řešení záležitostí ohledně společných nezletilých dětí nepřeje, proto poškozenou opakovaně vyhledával v místech, kam se poškozená přestěhovala,... |
| 9 | 0.380499 | 21 Nd 514/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2024:21.ND.514.2024.1__chunk_0003 | ECLI:CZ:NS:2024:21.ND.514.2024.1 | PASS | 5. Podle ustanovení § 11 odst. 3 o. s. ř. jde-li o věc, která patří do pravomoci soudů České republiky, ale podmínky místní příslušnosti chybějí nebo je nelze zjistit, určí Nejvyšší soud, který soud věc projedná a rozhodne. 6. V daném přípa... |
| 10 | 0.377513 | 26 Nd 573/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:26.ND.573.2024.1__chunk_0003 | ECLI:CZ:NS:2025:26.ND.573.2024.1 | PASS | 9. 2012, sp. zn. 31 Nd 200/2012, uveřejněné pod č. 4/2013 Sbírky soudních rozhodnutí a stanovisek). Nelze tak zjistit podmínky pro určení místní příslušnosti exekučního soudu (§ 45 odst. 2 exekučního řádu). 5. Nejvyšší soud proto podle § 11... |

### nutná obrana vzájemné napadání

- Expected behavior: `retrieval_returns_relevant_chunks`
- Actual label: **PASS**
- Metadata validation: **PASS**
- Result count: **10**
- Top score: **0.600907**
- Notes: source chunk evidence found at rank 6

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata_validation | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.600907 | 11 Tdo 679/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0011 | ECLI:CZ:NS:2024:11.TDO.679.2024.1 | PASS | 21. Existence trvajícího útoku je podle státního zástupce evidentní – poškozený tloukl do vozidla obviněného a bránil mu v odjezdu z místa. Opuštění vozidla a udeření poškozeného není v těchto souvislostech akceptací výzvy k potyčce, nýbrž... |
| 2 | 0.596968 | 11 Tdo 679/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0010 | ECLI:CZ:NS:2024:11.TDO.679.2024.1 | PASS | 19. Státní zástupce souhlasí s obviněným i v tom, že čelil útoku ze strany poškozeného. Soudy podle něj pochybily v důsledku nesprávné aplikace judikatury Nejvyššího soudu, podle které nepřichází nutná obrana v úvahu v případech vzájemného... |
| 3 | 0.573934 | 3 Tdo 19/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:3.TDO.19.2025.1__chunk_0004 | ECLI:CZ:NS:2025:3.TDO.19.2025.1 | PASS | 3. 2021, sp. zn. 6 Tdo 255/2021, podle něhož ne každé fyzické napadení druhého na veřejnosti nebo na místě veřejně přístupném musí naplňovat skutkovou podstatu přečinu výtržnictví, zvlášť jde-li o napadení, které je prostředkem, jímž pachat... |
| 4 | 0.563485 | 11 Tdo 679/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0012 | ECLI:CZ:NS:2024:11.TDO.679.2024.1 | PASS | 24. Argumenty soudů, jež vylučují nutnou obranu, tedy podle názoru státního zástupce neobstojí – subsidiarita není podmínkou nutné obrany; současně se nejedná o případ vzájemného napadání, protože obviněný reagoval na probíhající útok ze st... |
| 5 | 0.541885 | 11 Tdo 679/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0006 | ECLI:CZ:NS:2024:11.TDO.679.2024.1 | PASS | 10. Obviněný dále cituje usnesení Nejvyššího soudu ze dne 30. 10. 2019, sp. zn. 6 Tdo 1286/2019, a dodává, že v situaci, kdy proti sobě navzájem útočí dvě osoby, je rozhodující počáteční iniciativa, tedy kdo začal (případně také pohnutka či... |
| 6 | 0.516909 | 11 Tdo 679/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0009 | ECLI:CZ:NS:2024:11.TDO.679.2024.1 | PASS | 16. V řízení bylo prokázáno, že šlo o jeden stupňující se útok poškozeného, který započal slovní potyčkou na parkovišti, pokračoval přes přistoupení poškozeného k vozidlu obviněného, který chtěl odjet. Následné útoky pěstí do kapoty vozidla... |
| 7 | 0.496478 | 11 Tdo 679/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0017 | ECLI:CZ:NS:2024:11.TDO.679.2024.1 | PASS | 6. 2001, sp. zn. 11 Tdo 1376/2005, vyplývá, že pokud obviněný vstřícně reagoval na výzvu poškozeného („aby s ním šel ven“), která směřovala k jejich vzájemnému fyzickému konfliktu, a to tak, že obviněný odešel s poškozeným na místo, kde pot... |
| 8 | 0.454455 | 3 Tdo 650/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:3.TDO.650.2024.1__chunk_0015 | ECLI:CZ:NS:2024:3.TDO.650.2024.1 | PASS | 13. Pokud jde o námitky obviněného týkající se otázky bezprostřednosti ohrožení, státní zástupce připomněl, že obviněný v dovolání brojí proti tomu, že by svým jednáním naplnil jeden z judikaturou vyžadovaných znaků předmětného trestného či... |
| 9 | 0.448700 | 3 Tdo 1120/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:3.TDO.1120.2024.1__chunk_0012 | ECLI:CZ:NS:2025:3.TDO.1120.2024.1 | PASS | 21. Na tomto konstatování nic nemění ani patrný argumentační posun, k němuž obhajoba přistoupila při formulaci dovolání. V něm oproti řádnému opravnému prostředku nutnost opatření revizního znaleckého posudku z oboru zdravotnictví, odvětví... |
| 10 | 0.433658 | 4 Tdo 1056/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:4.TDO.1056.2024.1__chunk_0021 | ECLI:CZ:NS:2025:4.TDO.1056.2024.1 | PASS | 42. Nejvyšší soud se stručně vyjádří i k namítanému nesprávnému posouzení subjektivní stránky přisouzeného trestného činu. V obecnosti připomíná, že ustanovení § 358 tr. zákoníku chrání veřejný klid a pořádek i klidné mezilidské soužití pro... |

### náhradě nákladů dovolacího řízení

- Expected behavior: `retrieval_returns_relevant_chunks`
- Actual label: **PASS**
- Metadata validation: **PASS**
- Result count: **10**
- Top score: **0.779172**
- Notes: top result matches source terms with expected section type and score 0.779

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata_validation | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.779172 | 28 Cdo 3321/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:28.CDO.3321.2024.1__chunk_0008 | ECLI:CZ:NS:2025:28.CDO.3321.2024.1 | PASS | 12. 2024 (dále „advokátní tarif“), a náhrady paušálně určených hotových výdajů za jeden úkon právní služby ve výši 300 Kč – § 11 odst. 1 písm. k) a § 13 odst. 4 advokátního tarifu. Protože zástupce dalšího účastníka řízení je plátcem daně z... |
| 2 | 0.779038 | 29 Cdo 275/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:29.CDO.275.2024.1__chunk_0004 | ECLI:CZ:NS:2025:29.CDO.275.2024.1 | PASS | 177/1996 Sb., o odměnách advokátů a náhradách advokátů za poskytování právních služeb (advokátního tarifu), ve znění účinném k datu podání vyjádření, činí (z tarifní hodnoty 421.400,- Kč) částku 10.020,- Kč, a z paušální částky náhrady hoto... |
| 3 | 0.778899 | 27 Cdo 2699/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:27.CDO.2699.2024.1__chunk_0005 | ECLI:CZ:NS:2025:27.CDO.2699.2024.1 | PASS | 13. Výrok o náhradě nákladů dovolacího řízení se opírá o § 243c odst. 3, § 224 odst. 1 a § 146 odst. 3 o. s. ř., když dovolání žalobkyně bylo odmítnuto a žalovanému vzniklo právo na náhradu účelně vynaložených nákladů dovolacího řízení. 14.... |
| 4 | 0.778514 | 28 Cdo 3513/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:28.CDO.3513.2024.1__chunk_0006 | ECLI:CZ:NS:2025:28.CDO.3513.2024.1 | PASS | 15. O nákladech dovolacího řízení bylo rozhodnuto v intencích § 243 odst. 3 věty první, § 224 odst. 1, § 151 odst. 1 části věty před středníkem a § 146 odst. 3 o. s. ř.; dovolání žalované bylo odmítnuto a na straně žalobce lze za účelně vyn... |
| 5 | 0.772786 | 25 Cdo 2348/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:25.CDO.2348.2024.1__chunk_0002 | ECLI:CZ:NS:2024:25.CDO.2348.2024.1 | PASS | Odůvodnění: 1. Žalobkyně se podanou žalobou domáhala 998.089 Kč s příslušenstvím z titulu odpovědnosti advokáta za škodu. Částka 203.186 Kč představovala marně vynaložené náklady řízení, částka 594.903 Kč kapitalizovaný úrok z prodlení z 1.... |
| 6 | 0.766798 | 27 Cdo 3338/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:27.CDO.3338.2024.1__chunk_0001 | ECLI:CZ:NS:2025:27.CDO.3338.2024.1 | PASS | takto: I. Dovolací řízení se zastavuje . II. Žalovaná je povinna zaplatit žalobkyni na náhradě nákladů dovolacího řízení 64.323,60 Kč do tří dnů od právní moci tohoto usnesení k rukám zástupce žalobkyně. |
| 7 | 0.764918 | 20 Cdo 3371/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:20.CDO.3371.2024.1__chunk_0003 | ECLI:CZ:NS:2025:20.CDO.3371.2024.1 | PASS | 5) Podle dovolatele mu mělo být ve druhé skupině správně přiznáno 10 % z výtěžku 2 232 000Kč, tedy částka 223 200 Kč, a nikoli jen částka 187 194,30 Kč. Odvolací soud zaujal právní názor, že základ pro výpočet přihlášené pohledávky podle §... |
| 8 | 0.758485 | 23 Cdo 68/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:23.CDO.68.2025.1__chunk_0002 | ECLI:CZ:NS:2025:23.CDO.68.2025.1 | PASS | Odůvodnění: 1. Žalobkyně se podanou žalobou domáhala na žalovaném zaplacení částky 176 000 Kč s příslušenstvím jako náhrady škody. Podle tvrzení žalobkyně se uplatněný nárok měl skládat jednak z částky 140 000 Kč (představující pořizovací c... |
| 9 | 0.757433 | 23 Cdo 707/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2024:23.CDO.707.2024.1__chunk_0001 | ECLI:CZ:NS:2024:23.CDO.707.2024.1 | PASS | takto: I. Dovolání se odmítá. II. Žalována je povinna zaplatit žalobci na náhradu nákladů dovolacího řízení částku 17 714 Kč do tří dnů právní moci tohoto usnesení k rukám jeho právního zástupce. |
| 10 | 0.752414 | 23 Cdo 434/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2024:23.CDO.434.2024.1__chunk_0001 | ECLI:CZ:NS:2024:23.CDO.434.2024.1 | PASS | takto: I. Dovolání se odmítá. II. Žalobkyně je povinna zaplatit žalovanému na náhradě nákladů dovolacího řízení částku 4 114 Kč do tří dnů od právní moci tohoto usnesení k rukám právní zástupkyně žalovaného. |

### odpovědnosti za vady jako slevy z kupní ceny

- Expected behavior: `retrieval_returns_relevant_chunks`
- Actual label: **WARN**
- Metadata validation: **PASS**
- Result count: **10**
- Top score: **0.582669**
- Notes: expected section context appears by rank 2 with top score 0.583, but source evidence is indirect

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata_validation | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.582669 | 33 Cdo 79/2024 | USNESENÍ | civil | appeal_instruction | ECLI:CZ:NS:2025:33.CDO.79.2024.1__chunk_0009 | ECLI:CZ:NS:2025:33.CDO.79.2024.1 | PASS | 12. 2013, dále jen „obč. zák.“). Ve spotřebitelském právu je dodavatel ve fakticky výhodnějším postavení, neboť má odbornou převahu nad spotřebiteli, kterým své služby poskytuje. A proto kromě omezení vyplývajících z principu rovností prost... |
| 2 | 0.556213 | 33 Cdo 651/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:33.CDO.651.2024.1__chunk_0002 | ECLI:CZ:NS:2025:33.CDO.651.2024.1 | PASS | 89/2012 Sb., občanský zákoník (dále jen „o. z.“), a zák. č. 145/2010 Sb., o spotřebitelském úvěru. V důsledku nesplnění povinnosti plynoucí z § 9 zákona o spotřebitelském úvěru je nutno všechny smlouvy hodnotit jako neplatné, a žalovanému t... |
| 3 | 0.556045 | 8 Tdo 760/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:8.TDO.760.2024.1__chunk_0017 | ECLI:CZ:NS:2024:8.TDO.760.2024.1 | PASS | 8. 2004 za částku 500 Kč (srov. v podrobnostech č. l. 28947 a násl. spisu). Vzhledem k tomu, že společnost Standard nedisponovala dostatečným majetkem, se obchod realizovaný obviněnými jevil výhodným pouze zdánlivě, neboť nominální výše poh... |
| 4 | 0.540448 | 8 Tdo 760/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:8.TDO.760.2024.1__chunk_0003 | ECLI:CZ:NS:2024:8.TDO.760.2024.1 | PASS | 8. 2004 od správce konkurzní podstaty úpadce společnosti Standart s.r.o. za částku 500 Kč jako nedobytnou a prakticky bezcennou, neboť dlužník společnost Juma byl v úpadku a bylo evidentní, že konkurz bude zrušen pro nedostatek majetku, a o... |
| 5 | 0.533962 | 23 Cdo 271/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:23.CDO.271.2024.1__chunk_0011 | ECLI:CZ:NS:2024:23.CDO.271.2024.1 | PASS | 5. 2012 do 20. 5. 2014, smlouvy upravené tímto zákonem se v ostatním řídí právní úpravou závazků a úpravou smluvních typů jim nejbližších podle občanského nebo obchodního zákoníku, pokud z tohoto zákona nebo povahy věci nevyplývá něco jinéh... |
| 6 | 0.526008 | 5 Tdo 318/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:5.TDO.318.2024.1__chunk_0027 | ECLI:CZ:NS:2024:5.TDO.318.2024.1 | PASS | 2) Tato škoda (označovaná též jako tzv. reflexní škoda) je svou povahou odvozená od škody vzniklé na majetku společnosti. Její existence je závislá na existenci škody na majetku společnosti. 3) Je-li škoda vzniklá na majetku společnosti nah... |
| 7 | 0.523900 | 5 Tdo 318/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:5.TDO.318.2024.1__chunk_0030 | ECLI:CZ:NS:2024:5.TDO.318.2024.1 | PASS | 1. 2020). 50. Kromě shora naznačených sporných otázek z uvedeného ani nevyplývá vztah k nové zákonné úpravě obchodních korporací. Je tak otázkou, zda nadále mají mít před uplatňováním tzv. reflexní škody přednost tzv. derivativní žaloby, ko... |
| 8 | 0.518110 | 33 Cdo 79/2024 | USNESENÍ | civil | appeal_instruction | ECLI:CZ:NS:2025:33.CDO.79.2024.1__chunk_0006 | ECLI:CZ:NS:2025:33.CDO.79.2024.1 | PASS | 10. 2021 „dohodu o koupi nemovitosti“, v níž se prodávající a zájemce zavázali uzavřít kupní smlouvu ohledně označené nemovitosti za dohodnutou kupní cenu, včetně provize. Smluvní strany prohlásily, že uzavřením dohody jim RK obstarala příl... |
| 9 | 0.517148 | 33 Cdo 79/2024 | USNESENÍ | civil | appeal_instruction | ECLI:CZ:NS:2025:33.CDO.79.2024.1__chunk_0007 | ECLI:CZ:NS:2025:33.CDO.79.2024.1 | PASS | 11. 2021. Podle § 1753 o. z. ustanovení obchodních podmínek, které druhá strana nemohla rozumně očekávat, je neúčinné, nepřijala-li je tato strana výslovně; k opačnému ujednání se nepřihlíží. Zda se jedná o takové ustanovení, se posoudí nej... |
| 10 | 0.513648 | 29 ICdo 3/2023 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2024:29.ICDO.3.2023.1__chunk_0004 | ECLI:CZ:NS:2024:29.ICDO.3.2023.1 | PASS | 182/2006 Sb., o úpadku a způsobech jeho řešení (insolvenčního zákona). Podmínka byla formulována tak, že jde o „vznik škody z důvodu protiprávního jednání a porušení sml. povinnosti dlužníka“. [6] Pohledávka č. P968/3 byla přihlášena jako p... |

### pověření a nařízení exekuce

- Expected behavior: `retrieval_returns_relevant_chunks`
- Actual label: **PASS**
- Metadata validation: **PASS**
- Result count: **10**
- Top score: **0.786519**
- Notes: top result matches source terms with expected section type and score 0.787

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata_validation | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.786519 | 20 Cdo 30/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:20.CDO.30.2025.1__chunk_0003 | ECLI:CZ:NS:2025:20.CDO.30.2025.1 | PASS | 120/2001 Sb., o soudních exekutorech a exekuční činnosti (exekuční řád) a o změně dalších zákonů, ve znění pozdějších předpisů (dále též jen „ex. řád“), na jeho základě není možné vést exekuci a je namístě zastavit ji podle § 268 odst. 1 pí... |
| 2 | 0.770025 | 20 Cdo 15/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:20.CDO.15.2025.1__chunk_0004 | ECLI:CZ:NS:2025:20.CDO.15.2025.1 | PASS | 120/2001 Sb., o soudních exekutorech a exekuční činnosti (exekuční řád) a o změně dalších zákonů, ve znění pozdějších předpisů]. |
| 3 | 0.770025 | 20 Cdo 13/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:20.CDO.13.2025.1__chunk_0004 | ECLI:CZ:NS:2025:20.CDO.13.2025.1 | PASS | 120/2001 Sb., o soudních exekutorech a exekuční činnosti (exekuční řád) a o změně dalších zákonů, ve znění pozdějších předpisů]. |
| 4 | 0.755956 | 20 Cdo 3450/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:20.CDO.3450.2024.1__chunk_0004 | ECLI:CZ:NS:2025:20.CDO.3450.2024.1 | PASS | 120/2001 Sb., o soudních exekutorech a exekuční činnosti (exekuční řád), ve znění pozdějších předpisů, nestanoví-li tento zákon jinak, použijí se pro exekuční řízení přiměřeně ustanovení občanského soudního řádu. Výkon rozhodnutí se provede... |
| 5 | 0.729479 | 30 Cdo 3197/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:30.CDO.3197.2024.1__chunk_0004 | ECLI:CZ:NS:2025:30.CDO.3197.2024.1 | PASS | 82/1998 Sb., o odpovědnosti za škodu způsobenou při výkonu veřejné moci rozhodnutím nebo nesprávným úředním postupem a o změně zákona České národní rady č. 358/1992 Sb., o notářích a jejich činnosti (notářský řád), ve znění pozdějších předp... |
| 6 | 0.720396 | 20 Cdo 2831/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:20.CDO.2831.2024.1__chunk_0006 | ECLI:CZ:NS:2025:20.CDO.2831.2024.1 | PASS | 120/2001 Sb., o soudních exekutorech a exekuční činnosti /exekuční řád/ a o změně dalších zákonů, ve znění pozdějších předpisů - dále „ex. řád“) ve lhůtě uvedené v ustanovení § 240 odst. 1 o. s. ř., dospěl bez jednání (§ 243a odst. 1 věta p... |
| 7 | 0.715826 | 26 Cdo 125/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2024:26.CDO.125.2024.1__chunk_0005 | ECLI:CZ:NS:2024:26.CDO.125.2024.1 | PASS | 11. 2017, sp. zn. Pl. ÚS-st. 45/16, uveřejněné pod č. 460/2017 Sbírky zákonů). 8. Nejvyšší soud proto dovolání povinné podle § 243c odst. 1 věty první o. s. ř. odmítl. 9. O náhradě nákladů dovolacího řízení se rozhoduje ve zvláštním režimu... |
| 8 | 0.710013 | 28 Cdo 3321/2024 | USNESENÍ | civil | appeal_instruction | ECLI:CZ:NS:2025:28.CDO.3321.2024.1__chunk_0009 | ECLI:CZ:NS:2025:28.CDO.3321.2024.1 | PASS | Poučení: Proti tomuto rozhodnutí není opravný prostředek přípustný. Nesplní-li žalobce povinnost uloženou tímto rozhodnutím, může se další účastník řízení domáhat výkonu rozhodnutí nebo exekuce. |
| 9 | 0.706777 | 28 Nd 6/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:28.ND.6.2025.1__chunk_0003 | ECLI:CZ:NS:2025:28.ND.6.2025.1 | PASS | 6. Podle § 52 odst. 1 e. ř. nestanoví-li tento zákon jinak, použijí se pro exekuční řízení přiměřeně ustanovení o. s. ř. 7. Nejvyšší soud v usnesení ze dne 12. 9. 2012, sp. zn. 31 Nd 200/2012, uveřejněném pod číslem 4/2013 Sbírky soudních r... |
| 10 | 0.705067 | 29 Nd 59/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:29.ND.59.2025.1__chunk_0002 | ECLI:CZ:NS:2025:29.ND.59.2025.1 | PASS | Odůvodnění: 1. Exekučním návrhem ze dne 10. září 2024 podaným u soudního exekutora JUDr. Marcela Smékala, Exekutorský úřad Praha – východ, se oprávněný (Český inkasní kapitál, a. s.) domáhá provedení exekuce vůči povinnému (A. A. ) pro vymo... |

### právo bydlení

- Expected behavior: `retrieval_returns_relevant_chunks`
- Actual label: **PASS**
- Metadata validation: **PASS**
- Result count: **10**
- Top score: **0.728320**
- Notes: source case evidence found at rank 1 with source-term overlap at rank 1

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata_validation | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.728320 | 26 Cdo 439/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:26.CDO.439.2024.1__chunk_0006 | ECLI:CZ:NS:2024:26.CDO.439.2024.1 | PASS | 8. Žalovaný ve vyjádření k dovolání považuje rozhodnutí odvolacího soudu za správné. Do okamžiku zajištění bytové náhrady se řídí vztah bývalého nájemce a pronajímatele ustanovením § 712a obč. zák. a teprve po zajištění bytové náhrady – jes... |
| 2 | 0.682756 | 26 Cdo 439/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:26.CDO.439.2024.1__chunk_0003 | ECLI:CZ:NS:2024:26.CDO.439.2024.1 | PASS | 6. 2012, č. j. 11 Co 155/2012-78, bylo zrušeno právo společného nájmu tam specifikovaného družstevního bytu (předmětného bytu) a garáže (blíže označené) a společné členství v bytovém družstvu s tím, že výlučnou členkou družstva a výlučnou n... |
| 3 | 0.673722 | 24 Cdo 671/2024; 24 Cdo 675/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2024:24.CDO.671.2024.1__chunk_0002 | ECLI:CZ:NS:2024:24.CDO.671.2024.1 | PASS | Odůvodnění: 1. Okresní soud v Pelhřimově rozsudkem ze dne 31. 5. 2023, č. j. 5 C 255/2015-1439, rozhodl o určení vlastnického práva k celkem devíti jednotkám v domě č. p. XY v XY. Jednalo se o byty, garáže a dílnu. Stručně řečeno, u jednote... |
| 4 | 0.664371 | 22 Cdo 1151/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2025:22.CDO.1151.2024.1__chunk_0004 | ECLI:CZ:NS:2025:22.CDO.1151.2024.1 | PASS | 7. Ve způsobu vypořádání spoluvlastnictví dospěl ke stejnému závěru jako soud prvního stupně, tedy že reálné dělení blokačních pozemků není dobře možné, svůj názor ovšem opřel o jinou skutečnost, konkrétně o veřejný zájem na vybudování ploc... |
| 5 | 0.649335 | 22 Cdo 108/2025 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2025:22.CDO.108.2025.1__chunk_0008 | ECLI:CZ:NS:2025:22.CDO.108.2025.1 | PASS | 1. 2024 na základě § 1045 odst. 2 o. z. ve spojení s § 1050 odst. 2 o. z. a § 65 zákona č. 256/2013 Sb., katastrální zákon, ve znění účinném od 1. 1. 2014, jakožto k věcem opuštěným: 25. Dovolatelka dále napadá závěry odvolacího soudu, že F... |
| 6 | 0.645964 | 26 Cdo 1854/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2025:26.CDO.1854.2024.1__chunk_0007 | ECLI:CZ:NS:2025:26.CDO.1854.2024.1 | PASS | 1. 2024 učinila. 13. Podle § 1196 odst. 2 o. z. vzniknou-li vlastníkům jednotek práva vadou jednotky, zastupuje společenství vlastníků vlastníky jednotek při uplatňování těchto práv. 14. V usnesení ze dne 14. 9. 2022, sp. zn. 26 ICdo 28/202... |
| 7 | 0.642352 | 22 Cdo 3556/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:22.CDO.3556.2024.1__chunk_0002 | ECLI:CZ:NS:2025:22.CDO.3556.2024.1 | PASS | Odůvodnění: 1. Obvodní soud pro Prahu 10 (dále jen „soud prvního stupně“) rozsudkem ze dne 19. 10. 2023, č. j. 26 C 38/2022-502, zamítl žalobu na zrušení společného jmění účastníků (výrok I). Založil žalovanému oprávnění jednat samostatně a... |
| 8 | 0.638511 | 26 Cdo 439/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:26.CDO.439.2024.1__chunk_0008 | ECLI:CZ:NS:2024:26.CDO.439.2024.1 | PASS | 1. 2010, sp. zn. 28 Cdo 2146/2009). 14. Z § 712a obč. zák. vyplývá, že obsah vzájemných práv a povinností účastníků právního vztahu, který je uvedeným ustanovením posuzován, se řídí § 687 až § 699 obč. zák. , tedy i § 696 až § 699 obč. zák.... |
| 9 | 0.628967 | 26 Cdo 125/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2024:26.CDO.125.2024.1__chunk_0003 | ECLI:CZ:NS:2024:26.CDO.125.2024.1 | PASS | 5. V projednávané věci notář JUDr. Eduard Grygar sepsal dne 27. 1. 2021 notářský zápis sp. zn. NZ 13/2021, N 17/2021, v němž oprávněná (nájemkyně bytů) a povinná (podnájemkyně) uzavřely podle § 71b not. ř. dohodu o uznání nároků oprávněné n... |
| 10 | 0.623225 | 26 Cdo 2404/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2024:26.CDO.2404.2024.1__chunk_0003 | ECLI:CZ:NS:2024:26.CDO.2404.2024.1 | PASS | 10. 2000) a nachází se na něm (pravý) podpis jmenované svědkyně. Ztotožnil se i s názory soudu prvního stupně, že nájemní smlouva není neplatná ani z důvodu, že k ní nedala souhlas valná hromada žalované (tímto souhlasem byla podmíněna jen... |

### přípustnost dovolání podle § 237 o. s. ř.

- Expected behavior: `retrieval_returns_relevant_chunks`
- Actual label: **PASS**
- Metadata validation: **PASS**
- Result count: **10**
- Top score: **0.869010**
- Notes: top result matches source terms with expected section type and score 0.869

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata_validation | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.869010 | 23 Cdo 938/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:23.CDO.938.2024.1__chunk_0006 | ECLI:CZ:NS:2025:23.CDO.938.2024.1 | PASS | 13. Ustanovení § 241a odst. 2 o. s. ř. stanoví, že v dovolání musí být vedle obecných náležitostí (§ 42 odst. 4 o. s. ř.) uvedeno, proti kterému rozhodnutí směřuje, v jakém rozsahu se napadá, vymezení důvodu dovolání, v čem dovolatel spatřu... |
| 2 | 0.852014 | 23 Cdo 434/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2024:23.CDO.434.2024.1__chunk_0004 | ECLI:CZ:NS:2024:23.CDO.434.2024.1 | PASS | 9. Podle § 241a odst. 2 o. s. ř. v dovolání musí být vedle obecných náležitostí (§ 42 odst. 4) uvedeno, proti kterému rozhodnutí směřuje, v jakém rozsahu se rozhodnutí napadá, vymezení důvodu dovolání, v čem dovolatel spatřuje splnění předp... |
| 3 | 0.831259 | 30 Cdo 3461/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:30.CDO.3461.2024.1__chunk_0004 | ECLI:CZ:NS:2025:30.CDO.3461.2024.1 | PASS | 6. 2012, sp. zn. 30 Cdo 1486/2012). Ustanovení § 241 o. s. ř., které upravuje tzv. povinné zastoupení dovolatele při podání dovolání, představuje zvláštní podmínku dovolacího řízení, jejíž nedostatek lze odstranit, bez jejíhož splnění však... |
| 4 | 0.826377 | 30 Cdo 106/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:30.CDO.106.2025.1__chunk_0003 | ECLI:CZ:NS:2025:30.CDO.106.2025.1 | PASS | 6. Ustanovení § 241 o. s. ř., které upravuje tzv. povinné zastoupení dovolatele při podání dovolání, představuje zvláštní podmínku dovolacího řízení, jejíž nedostatek lze odstranit, bez jejíhož splnění však nelze meritorně rozhodnout o dovo... |
| 5 | 0.816218 | 21 Cdo 245/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:21.CDO.245.2024.1__chunk_0003 | ECLI:CZ:NS:2025:21.CDO.245.2024.1 | PASS | 6. 2023, č. j. 12 Co 25/2023-148, není podle § 237 o. s. ř. přípustné, neboť není splněn žádný z předpokladů přípustnosti dovolání uvedených v tomto ustanovení. 11. Dovoláním napadený rozsudek odvolacího soudu je v žalobcem předestřené otáz... |
| 6 | 0.813386 | 23 Cdo 271/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:23.CDO.271.2024.1__chunk_0008 | ECLI:CZ:NS:2024:23.CDO.271.2024.1 | PASS | 18. Dovolání bylo podáno včas, osobou k tomu oprávněnou, za splnění podmínky § 241 odst. 1 o. s. ř. Dovolací soud rovněž shledal, že dovolání obsahuje náležitosti vyžadované ustanovením § 241a odst. 2 o. s. ř. a dále se proto zabýval jeho p... |
| 7 | 0.807662 | 27 Cdo 395/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:27.CDO.395.2024.1__chunk_0004 | ECLI:CZ:NS:2025:27.CDO.395.2024.1 | PASS | [15] Podle § 241b odst. 3 o. s. ř. dovolání, které neobsahuje údaje o tom, v jakém rozsahu se rozhodnutí odvolacího soudu napadá, v čem dovolatel spatřuje splnění předpokladů přípustnosti dovolání (§ 237 až 238a) nebo které neobsahuje vymez... |
| 8 | 0.806783 | 28 Cdo 1880/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:28.CDO.1880.2024.1__chunk_0006 | ECLI:CZ:NS:2025:28.CDO.1880.2024.1 | PASS | 2. 2024, sp. zn. 28 Cdo 2593/2023). 15. Z vylíčeného je zjevné, že na předmětné dovolání nelze pohlížet jako na přípustné, pročež je Nejvyšší soud podle § 243c odst. 1 o. s. ř. odmítl. 16. O náhradě nákladů dovolacího řízení bylo rozhodnuto... |
| 9 | 0.793445 | 8 Tdo 1119/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:8.TDO.1119.2024.1__chunk_0004 | ECLI:CZ:NS:2025:8.TDO.1119.2024.1 | PASS | 6. Protože dovolání lze podat jen z důvodů uvedených v § 265b tr. ř., bylo dále nutno posoudit, zda nejvyšším státním zástupcem vznesené námitky naplňují jím uplatněný zákonem stanovený dovolací důvod, jehož existence je současně nezbytnou... |
| 10 | 0.792795 | 28 Cdo 3321/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:28.CDO.3321.2024.1__chunk_0005 | ECLI:CZ:NS:2025:28.CDO.3321.2024.1 | PASS | 7. Dovolání není přípustné. 8. Dovolatel zakládá přípustnost dovolání ve smyslu ustanovení § 237 o. s. ř. na odklonu odvolacího soudu od judikatury Nejvyššího soudu představované - ve věci týchž účastníků, jež je vymezena zásadně shodnými s... |

### trest odnětí svobody

- Expected behavior: `retrieval_returns_relevant_chunks`
- Actual label: **PASS**
- Metadata validation: **PASS**
- Result count: **10**
- Top score: **0.765762**
- Notes: source case evidence found at rank 2 with source-term overlap at rank 2

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata_validation | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.765762 | 3 Tdo 19/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:3.TDO.19.2025.1__chunk_0007 | ECLI:CZ:NS:2025:3.TDO.19.2025.1 | PASS | 13. Z takto vymezených hledisek dovolatel namítl, že trest, jež mu byl uložen, neodpovídá zásadám pro ukládání trestu. S ohledem na jeho částečné doznání a odstup času od incidentu, kdy následně žil řádným životem, a s přihlédnutím k závažn... |
| 2 | 0.750247 | 3 Tdo 984/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:3.TDO.984.2024.1__chunk_0020 | ECLI:CZ:NS:2024:3.TDO.984.2024.1 | PASS | 28. Trest, který byl obviněnému uložen, ničím nevybočuje z rámce běžně se vyskytujících případů a je zcela odpovídající jak povaze a závažnosti spáchaného trestného činu, tak i poměrům pachatele. Je třeba připomenout, že trest byl obviněném... |
| 3 | 0.743396 | 8 Tdo 760/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:8.TDO.760.2024.1__chunk_0005 | ECLI:CZ:NS:2024:8.TDO.760.2024.1 | PASS | 134/2002 Sb., účinného do 30. 6. 2008), kterým byl uznán vinným napadeným rozsudkem, odsoudil podle § 252a odst. 3 tr. zák. k trestu odnětí svobody v trvání dvou let, jehož výkon mu podle § 58 odst. 1 a § 59 odst. 1 tr. zák. podmíněně odlož... |
| 4 | 0.730595 | 8 Tdo 1119/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:8.TDO.1119.2024.1__chunk_0005 | ECLI:CZ:NS:2025:8.TDO.1119.2024.1 | PASS | 10. Podle § 58 odst. 6 věty první tr. zákoníku může soud snížit trest odnětí svobody pod dolní hranici trestní sazby též tehdy, jestliže odsuzuje pachatele za přípravu k trestnému činu nebo za pokus trestného činu nebo za pomoc k trestnému... |
| 5 | 0.725357 | 3 Tdo 53/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:3.TDO.53.2025.1__chunk_0004 | ECLI:CZ:NS:2025:3.TDO.53.2025.1 | PASS | 2. Za toto jednání byl obviněný odsouzen podle § 186 odst. 5 tr. zákoníku za použití § 58 odst. 2 písm. b) tr. zákoníku a § 43 odst. 1 tr. zákoníku k úhrnnému trestu odnětí svobody v trvání 4 roků a 6 měsíců. Podle § 56 odst. 2 písm. a) tr.... |
| 6 | 0.722160 | 3 Tdo 1120/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:3.TDO.1120.2024.1__chunk_0018 | ECLI:CZ:NS:2025:3.TDO.1120.2024.1 | PASS | 12. 2018, sp. zn. IV. ÚS 3227/18, nebo ze dne 27. 4. 2021, sp. zn. III. ÚS 817/21). Ve shodě s Nejvyšším soudem konstantně vycházel z toho, že ustanovení trestního zákoníku umožňující snížení trestu odnětí svobody pod dolní hranici trestní... |
| 7 | 0.721585 | 3 Tdo 650/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:3.TDO.650.2024.1__chunk_0005 | ECLI:CZ:NS:2024:3.TDO.650.2024.1 | PASS | 3. Za uvedený přečin byl obviněný M. H. soudem druhého stupně odsouzen podle § 273 odst. 2 tr. zákoníku k trestu odnětí svobody v trvání jednoho roku a šesti měsíců, jehož výkon byl podle § 81 odst. 1 a § 82 odst. 1 tr. zákoníku podmíněně o... |
| 8 | 0.719716 | 8 Tdo 1022/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:8.TDO.1022.2024.1__chunk_0019 | ECLI:CZ:NS:2024:8.TDO.1022.2024.1 | PASS | 19. Obviněný taktéž uplatnil námitky stran přiměřenosti trestu odnětí svobody. Nejvyšší soud v této souvislosti připomíná, že námitky vůči druhu a výměře uloženého trestu s výjimkou trestu odnětí svobody na doživotí lze v dovolání úspěšně u... |
| 9 | 0.702818 | 11 Tdo 1114/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:11.TDO.1114.2024.1__chunk_0015 | ECLI:CZ:NS:2025:11.TDO.1114.2024.1 | PASS | 5. 2014 sp. zn. 3 Tdo 448/2014 . 42. Je zřejmé, že za první téměř roční skutek v době od druhé poloviny roku 2020 do 7. 6. 2021 byl dovolatel již odsouzen shora uvedeným rozsudkem ze dne 30. 3. 2022 ve své předchozí trestní věci sp. zn. 3 T... |
| 10 | 0.694623 | 4 Tdo 1018/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:4.TDO.1018.2024.1__chunk_0003 | ECLI:CZ:NS:2024:4.TDO.1018.2024.1 | PASS | 2. Za uvedené jednání byla obviněná odsouzena dle § 147 odst. 2 tr. zákoníku k trestu odnětí svobody v trvání 20 (dvaceti) měsíců. Dle § 81 odst. 1 a § 82 odst. 1 tr. zákoníku byl výkon tohoto trestu podmíněně odložen na zkušební dobu v trv... |

### určení místní příslušnosti

- Expected behavior: `retrieval_returns_relevant_chunks`
- Actual label: **PASS**
- Metadata validation: **PASS**
- Result count: **10**
- Top score: **0.517164**
- Notes: source-term overlap appears within top 1 results

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata_validation | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.517164 | 29 Nd 63/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:29.ND.63.2025.1__chunk_0002 | ECLI:CZ:NS:2025:29.ND.63.2025.1 | PASS | Odůvodnění: 1. Usnesením ze dne 13. prosince 2024, č. j. 27 Nc 2451/2024-44, vyslovil Okresní soud Praha - západ svou místní nepříslušnost (bod I. výroku), rozhodl, že věc bude po právní moci usnesení předložena Nejvyššímu soudu k určení mí... |
| 2 | 0.516059 | 20 Nd 18/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:20.ND.18.2025.1__chunk_0003 | ECLI:CZ:NS:2025:20.ND.18.2025.1 | PASS | 99/1963 Sb., občanský soudní řád, ve znění pozdějších předpisů – dále též jen „o. s. ř.“). V exekučním návrhu oprávněné a stejně tak v žádosti soudního exekutora je uvedena adresa povinného XY. Lustrací v informačním systému základních regi... |
| 3 | 0.496459 | 24 Nd 34/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:24.ND.34.2025.1__chunk_0004 | ECLI:CZ:NS:2025:24.ND.34.2025.1 | PASS | 9. 2012, sp. zn. 31 Nd 200/2012, uveřejněné ve Sbírce soudních rozhodnutí a stanovisek pod č. 4, ročník 2013). 8. Nejvyšší soud v obdobných situacích vychází při určení místní příslušnosti exekučního soudu ze zásady hospodárnosti řízení zak... |
| 4 | 0.492346 | 26 Nd 573/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:26.ND.573.2024.1__chunk_0003 | ECLI:CZ:NS:2025:26.ND.573.2024.1 | PASS | 9. 2012, sp. zn. 31 Nd 200/2012, uveřejněné pod č. 4/2013 Sbírky soudních rozhodnutí a stanovisek). Nelze tak zjistit podmínky pro určení místní příslušnosti exekučního soudu (§ 45 odst. 2 exekučního řádu). 5. Nejvyšší soud proto podle § 11... |
| 5 | 0.481980 | 7 Td 6/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:7.TD.6.2025.1__chunk_0004 | ECLI:CZ:NS:2025:7.TD.6.2025.1 | PASS | 7. Pro určení místní příslušnosti soudu je podstatné místo, kde obviněná jednala, kde neznámé osobě umožnila užít k výše uvedené transakci svůj účet, tedy konkrétně především na jakém místě se nacházela, když finanční prostředky ze svého úč... |
| 6 | 0.477732 | 26 Cdo 1854/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2025:26.CDO.1854.2024.1__chunk_0007 | ECLI:CZ:NS:2025:26.CDO.1854.2024.1 | PASS | 1. 2024 učinila. 13. Podle § 1196 odst. 2 o. z. vzniknou-li vlastníkům jednotek práva vadou jednotky, zastupuje společenství vlastníků vlastníky jednotek při uplatňování těchto práv. 14. V usnesení ze dne 14. 9. 2022, sp. zn. 26 ICdo 28/202... |
| 7 | 0.460264 | 6 Tdo 827/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:6.TDO.827.2024.1__chunk_0011 | ECLI:CZ:NS:2024:6.TDO.827.2024.1 | PASS | 21. Za opodstatněnou nelze mít ani námitku dovolatelů poukazujících na to, že předmětný odpad nepřepravili přes hranice státu oni, nýbrž že uzavřeli pouze obchodní dohodu s mezinárodním prvkem. Dovolatelé zde opomíjejí, že za jednání pachat... |
| 8 | 0.449468 | 22 Cdo 108/2025 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2025:22.CDO.108.2025.1__chunk_0002 | ECLI:CZ:NS:2025:22.CDO.108.2025.1 | PASS | Odůvodnění: I. Dosavadní průběh řízení 1. Okresní soud v Lounech (dále jen „soud prvního stupně“) rozsudkem ze dne 16. 5. 2024, č. j. 12 C 152/2023-52, určil, že pozemek p. č. XY a pozemek p. č. XY, jehož součástí je stavba – rodinný dům č.... |
| 9 | 0.448212 | 7 Td 6/2025 | USNESENÍ | criminal | header | ECLI:CZ:NS:2025:7.TD.6.2025.1__chunk_0000 | ECLI:CZ:NS:2025:7.TD.6.2025.1 | PASS | 7 Td 6/2025-109 USNESENÍ Nejvyšší soud rozhodl dne 29. 1. 2025 v neveřejném zasedání v trestní věci obviněné M. P. vedené u Okresního soudu v Náchodě pod sp. zn. 1 T 242/2024, o návrhu soudu na určení místní příslušnosti |
| 10 | 0.445020 | 5 Tdo 318/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:5.TDO.318.2024.1__chunk_0043 | ECLI:CZ:NS:2024:5.TDO.318.2024.1 | PASS | 12. 2021. Pro úplnost lze jen dodat, že lze v tomto ohledu souhlasit se státním zástupcem, podle nějž soud prvního stupně mohl (a též musel) ve věci rozhodnout, a i kdyby nebyl místně příslušný, pokud to obviněný v hlavním líčení nenamítl (... |

### zastavení exekuce

- Expected behavior: `retrieval_returns_relevant_chunks`
- Actual label: **PASS**
- Metadata validation: **PASS**
- Result count: **10**
- Top score: **0.772908**
- Notes: source case evidence found at rank 5 with source-term overlap at rank 5

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata_validation | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.772908 | 23 Cdo 3535/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:23.CDO.3535.2024.1__chunk_0001 | ECLI:CZ:NS:2025:23.CDO.3535.2024.1 | PASS | takto: Dovolací řízení se zastavuje . |
| 2 | 0.772908 | 20 Cdo 3518/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:20.CDO.3518.2024.1__chunk_0001 | ECLI:CZ:NS:2025:20.CDO.3518.2024.1 | PASS | takto: Dovolací řízení se zastavuje . |
| 3 | 0.745089 | 20 Cdo 15/2025 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:20.CDO.15.2025.1__chunk_0001 | ECLI:CZ:NS:2025:20.CDO.15.2025.1 | PASS | takto: Řízení o dovolání obou povinných se zastavuje . |
| 4 | 0.745089 | 20 Cdo 13/2025 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:20.CDO.13.2025.1__chunk_0001 | ECLI:CZ:NS:2025:20.CDO.13.2025.1 | PASS | takto: Řízení o dovolání obou povinných se zastavuje . |
| 5 | 0.731205 | 20 Cdo 30/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:20.CDO.30.2025.1__chunk_0003 | ECLI:CZ:NS:2025:20.CDO.30.2025.1 | PASS | 120/2001 Sb., o soudních exekutorech a exekuční činnosti (exekuční řád) a o změně dalších zákonů, ve znění pozdějších předpisů (dále též jen „ex. řád“), na jeho základě není možné vést exekuci a je namístě zastavit ji podle § 268 odst. 1 pí... |
| 6 | 0.622976 | 29 Cdo 275/2024 | USNESENÍ | civil | appeal_instruction | ECLI:CZ:NS:2025:29.CDO.275.2024.1__chunk_0005 | ECLI:CZ:NS:2025:29.CDO.275.2024.1 | PASS | Poučení: Proti tomuto usnesení není přípustný opravný prostředek. Nesplní-li povinný dobrovolně, co mu ukládá vykonatelné rozhodnutí, může se oprávněný domáhat exekuce (výkonu rozhodnutí). |
| 7 | 0.616189 | 20 Ncu 177/2024 | ROZSUDEK | - | appeal_instruction | ECLI:CZ:NS:2024:20.NCU.177.2024.1__chunk_0004 | ECLI:CZ:NS:2024:20.NCU.177.2024.1 | PASS | Poučení: Proti tomuto rozsudku není opravný prostředek přípustný. Tento rozsudek nabývá právní moci doručením. |
| 8 | 0.616189 | 20 Ncu 185/2024 | ROZSUDEK | - | appeal_instruction | ECLI:CZ:NS:2025:20.NCU.185.2024.1__chunk_0003 | ECLI:CZ:NS:2025:20.NCU.185.2024.1 | PASS | Poučení: Proti tomuto rozsudku není opravný prostředek přípustný. Tento rozsudek nabývá právní moci doručením. |
| 9 | 0.616189 | 20 Ncu 165/2024 | ROZSUDEK | - | appeal_instruction | ECLI:CZ:NS:2025:20.NCU.165.2024.1__chunk_0002 | ECLI:CZ:NS:2025:20.NCU.165.2024.1 | PASS | Poučení: Proti tomuto rozsudku není opravný prostředek přípustný. Tento rozsudek nabývá právní moci doručením. |
| 10 | 0.614507 | 21 Cdo 2658/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:21.CDO.2658.2024.3__chunk_0002 | ECLI:CZ:NS:2025:21.CDO.2658.2024.3 | PASS | Odůvodnění: 1. Okresní soud ve Vsetíně usnesením ze dne 7. 7. 2023, č. j. 10 EXE 557/2021-151, zamítl návrh povinného na zastavení exekuce vedené JUDr. Lukášem Jíchou, soudním exekutorem Exekutorského úřadu v Přerově, na základě pověření vy... |

### mezinárodní ochrana a azyl

- Expected behavior: `insufficient_support`
- Actual label: **PASS**
- Metadata validation: **PASS**
- Result count: **10**
- Top score: **0.597794**
- Notes: results are ambiguous across legal contexts and do not show direct support

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata_validation | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.597794 | 8 Tdo 1022/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:8.TDO.1022.2024.1__chunk_0004 | ECLI:CZ:NS:2024:8.TDO.1022.2024.1 | PASS | 104/2013 Sb., o mezinárodní justiční spolupráci ve věcech trestních, ve znění pozdějších předpisů (dále jen „zákon o mezinárodní justiční spolupráci“), čl. 14 sdělení č. 549/1992 Sb., federálního ministerstva zahraničních věcí o sjednání Ev... |
| 2 | 0.569177 | 30 Cdo 844/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:30.CDO.844.2024.1__chunk_0011 | ECLI:CZ:NS:2024:30.CDO.844.2024.1 | PASS | 25. Dovolání je přípustné pro řešení otázky, zda v případě zásahu do práv žalobkyně spočívajícím v tom, že byla nucena po návratu ze zahraničí strávit 5 dní doma omezena ve svobodě pohybu institutem „samoizolace“ ve smyslu ochranného opatře... |
| 3 | 0.561634 | 30 Cdo 844/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:30.CDO.844.2024.1__chunk_0002 | ECLI:CZ:NS:2024:30.CDO.844.2024.1 | PASS | Odůvodnění: I. Dosavadní průběh řízení 1. Žalobkyně se podanou žalobou domáhala zadostiučinění za nemajetkovou újmu způsobenou nezákonnými opatřeními Ministerstva zdravotnictví, v jejichž důsledku se musela po dobu 5 dnů (od 29. 8. 2021 do... |
| 4 | 0.554719 | 25 Cdo 3217/2023 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:25.CDO.3217.2023.1__chunk_0008 | ECLI:CZ:NS:2024:25.CDO.3217.2023.1 | PASS | 12. 2019, sen. zn. 29 ICdo 96/2016). V každém případě musí být obsah cizího (cizozemského) práva zjištěn tak úplně a spolehlivě, jak by tomu bylo (mělo být) v případě, kdyby spor nebo jiná právní věc byly řešeny a rozhodovány v domovské zem... |
| 5 | 0.542176 | 25 Cdo 3217/2023 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:25.CDO.3217.2023.1__chunk_0007 | ECLI:CZ:NS:2024:25.CDO.3217.2023.1 | PASS | 91/2012 Sb., o mezinárodním právu soukromém, ve znění pozdějších předpisů, pokud z jiných ustanovení tohoto zákona nevyplývá něco jiného, je třeba zahraničního práva, jehož se má použít podle ustanovení tohoto zákona, používat i bez návrhu... |
| 6 | 0.523139 | 30 Cdo 844/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:30.CDO.844.2024.1__chunk_0013 | ECLI:CZ:NS:2024:30.CDO.844.2024.1 | PASS | 29. V daném případě je po skutkové stránce bez pochyb, že žalobkyně byla po svém návratu ze zahraničí nucena strávit pět dní v samoizolaci, a to na základě mimořádného opatření (viz výše), které bylo odvolacím soudem posouzeno jako nezákonn... |
| 7 | 0.504580 | 21 Cdo 2841/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:21.CDO.2841.2024.1__chunk_0010 | ECLI:CZ:NS:2025:21.CDO.2841.2024.1 | PASS | 11. 2017, sp. zn. Pl. ÚS-st. 45/16). O výjimečný případ, kdy skutková otázka s ohledem na její průmět do základních lidských práv a svobod je způsobilá založit přípustnost dovolání podle § 237 o. s. ř. (srov. například nález Ústavního soudu... |
| 8 | 0.499144 | 5 Tdo 1128/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:5.TDO.1128.2024.1__chunk_0033 | ECLI:CZ:NS:2025:5.TDO.1128.2024.1 | PASS | 26/2013 Sb. rozh. tr., neboť s vědomím všech okolností spáchání trestného činu nemohl stát rezignovat na svou roli při ochraně oprávněných zájmů fyzických a právnických osob s odkazem na primární existenci institutů občanského práva či jiný... |
| 9 | 0.492092 | 33 Cdo 79/2024 | USNESENÍ | civil | appeal_instruction | ECLI:CZ:NS:2025:33.CDO.79.2024.1__chunk_0014 | ECLI:CZ:NS:2025:33.CDO.79.2024.1 | PASS | 6. 2003, sp. zn. 21 Cdo 121/2003, rozsudek Nejvyššího soudu ze dne 31. 3. 2010, sp. zn. 33 Cdo 689/2008). V dané věci přitom závěr o začlenění ujednání dle § 1837 písm. a) o. z. do článku 14.2 smlouvy obsahujícím naprosto nesouvisející proh... |
| 10 | 0.489238 | 11 Tcu 96/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:11.TCU.96.2024.1__chunk_0005 | ECLI:CZ:NS:2025:11.TCU.96.2024.1 | PASS | 10. Podmínka oboustranné trestnosti je též splněna ohledně odsouzení z rozsudku druhého cizozemského soudu. V tomto případě jednání odsouzeného vykazuje nejméně znaky přečinu krádeže podle § 205 odst. 1 písm. b) tr. zákoníku a přečinu poruš... |

### správní vyhoštění cizince

- Expected behavior: `insufficient_support`
- Actual label: **PASS**
- Metadata validation: **PASS**
- Result count: **10**
- Top score: **0.593625**
- Notes: unexpected missing-context phrase overlap found at rank 6 | the phrase overlap appears incidental rather than as direct support

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata_validation | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.593625 | 6 Tdo 827/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:6.TDO.827.2024.1__chunk_0026 | ECLI:CZ:NS:2024:6.TDO.827.2024.1 | PASS | 51. Lichý je rovněž výklad dovolatelů dovozujících, že stíhaný přečin lze spáchat pouze, pokud dojde k porušení povinnosti spočívající v oznámení přeshraničního transportu nebo získání souhlasu s ním od příslušného orgánu. Jestliže totiž mi... |
| 2 | 0.592014 | 6 Tdo 827/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:6.TDO.827.2024.1__chunk_0011 | ECLI:CZ:NS:2024:6.TDO.827.2024.1 | PASS | 21. Za opodstatněnou nelze mít ani námitku dovolatelů poukazujících na to, že předmětný odpad nepřepravili přes hranice státu oni, nýbrž že uzavřeli pouze obchodní dohodu s mezinárodním prvkem. Dovolatelé zde opomíjejí, že za jednání pachat... |
| 3 | 0.589321 | 30 Cdo 844/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:30.CDO.844.2024.1__chunk_0013 | ECLI:CZ:NS:2024:30.CDO.844.2024.1 | PASS | 29. V daném případě je po skutkové stránce bez pochyb, že žalobkyně byla po svém návratu ze zahraničí nucena strávit pět dní v samoizolaci, a to na základě mimořádného opatření (viz výše), které bylo odvolacím soudem posouzeno jako nezákonn... |
| 4 | 0.575855 | 25 Cdo 3217/2023 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:25.CDO.3217.2023.1__chunk_0008 | ECLI:CZ:NS:2024:25.CDO.3217.2023.1 | PASS | 12. 2019, sen. zn. 29 ICdo 96/2016). V každém případě musí být obsah cizího (cizozemského) práva zjištěn tak úplně a spolehlivě, jak by tomu bylo (mělo být) v případě, kdyby spor nebo jiná právní věc byly řešeny a rozhodovány v domovské zem... |
| 5 | 0.562217 | 30 Cdo 844/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:30.CDO.844.2024.1__chunk_0011 | ECLI:CZ:NS:2024:30.CDO.844.2024.1 | PASS | 25. Dovolání je přípustné pro řešení otázky, zda v případě zásahu do práv žalobkyně spočívajícím v tom, že byla nucena po návratu ze zahraničí strávit 5 dní doma omezena ve svobodě pohybu institutem „samoizolace“ ve smyslu ochranného opatře... |
| 6 | 0.552608 | 20 Nd 18/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:20.ND.18.2025.1__chunk_0003 | ECLI:CZ:NS:2025:20.ND.18.2025.1 | PASS | 99/1963 Sb., občanský soudní řád, ve znění pozdějších předpisů – dále též jen „o. s. ř.“). V exekučním návrhu oprávněné a stejně tak v žádosti soudního exekutora je uvedena adresa povinného XY. Lustrací v informačním systému základních regi... |
| 7 | 0.544320 | 30 Cdo 844/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:30.CDO.844.2024.1__chunk_0002 | ECLI:CZ:NS:2024:30.CDO.844.2024.1 | PASS | Odůvodnění: I. Dosavadní průběh řízení 1. Žalobkyně se podanou žalobou domáhala zadostiučinění za nemajetkovou újmu způsobenou nezákonnými opatřeními Ministerstva zdravotnictví, v jejichž důsledku se musela po dobu 5 dnů (od 29. 8. 2021 do... |
| 8 | 0.527260 | 11 Tcu 96/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:11.TCU.96.2024.1__chunk_0005 | ECLI:CZ:NS:2025:11.TCU.96.2024.1 | PASS | 10. Podmínka oboustranné trestnosti je též splněna ohledně odsouzení z rozsudku druhého cizozemského soudu. V tomto případě jednání odsouzeného vykazuje nejméně znaky přečinu krádeže podle § 205 odst. 1 písm. b) tr. zákoníku a přečinu poruš... |
| 9 | 0.523261 | 6 Tdo 21/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:6.TDO.21.2025.1__chunk_0004 | ECLI:CZ:NS:2025:6.TDO.21.2025.1 | PASS | 12. 2022 a o „ blíže nezjištěném způsobu “ vniknutí do parku zámeckého areálu. Následkem toho byl obviněný zkrácen na právu na obhajobě, neboť se nemohl vyjádřit k dostatečně určitému popisu skutku, který současně pro svou neurčitost nebyl... |
| 10 | 0.522260 | 25 Cdo 3217/2023 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:25.CDO.3217.2023.1__chunk_0007 | ECLI:CZ:NS:2024:25.CDO.3217.2023.1 | PASS | 91/2012 Sb., o mezinárodním právu soukromém, ve znění pozdějších předpisů, pokud z jiných ustanovení tohoto zákona nevyplývá něco jiného, je třeba zahraničního práva, jehož se má použít podle ustanovení tohoto zákona, používat i bez návrhu... |

### ochrana osobních údajů podle GDPR

- Expected behavior: `insufficient_support`
- Actual label: **PASS**
- Metadata validation: **PASS**
- Result count: **10**
- Top score: **0.599194**
- Notes: results are ambiguous across legal contexts and do not show direct support

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata_validation | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.599194 | 33 Cdo 79/2024 | USNESENÍ | civil | appeal_instruction | ECLI:CZ:NS:2025:33.CDO.79.2024.1__chunk_0008 | ECLI:CZ:NS:2025:33.CDO.79.2024.1 | PASS | 11. 2013, sp. zn. I. ÚS 3512/11 (dostupném, stejně jako další uváděná rozhodnutí tohoto soudu na webových stránkách www.usoud.cz ), Ústavní soud (kromě jiného) uvedl, že ochrana spotřebitele spadá mezi jednu ze sdílených politik Evropské un... |
| 2 | 0.517372 | 7 Pzo 5/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:7.PZO.5.2024.1__chunk_0010 | ECLI:CZ:NS:2025:7.PZO.5.2024.1 | PASS | 15. Z tohoto důvodu je pak možné v projednávané věci dospět k závěru, že vymezení podmínek, za nichž může být vydán příkaz k odposlechu a záznamu telekomunikačního provozu, resp. příkaz ke sdělení údajů o uskutečněném telekomunikačním provo... |
| 3 | 0.511832 | 5 Tdo 1128/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:5.TDO.1128.2024.1__chunk_0016 | ECLI:CZ:NS:2025:5.TDO.1128.2024.1 | PASS | 5. 2001 o harmonizaci určitých aspektů autorského práva a práv s ním souvisejících v informační společnosti (dále jen „Směrnice“). V souladu se závěry citovaného rozhodnutí namítl, že svým jednáním vědomě do práv autorů nezasahoval, nenahrá... |
| 4 | 0.494208 | 7 Pzo 5/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:7.PZO.5.2024.1__chunk_0009 | ECLI:CZ:NS:2025:7.PZO.5.2024.1 | PASS | 7. 2024, sp. zn. 3 Pzo 1/2024). 14. Nelze přisvědčit ani námitce navrhovatele, že v příkazech absentuje výslovné zdůvodnění důvodnosti nezbytnosti zásahu do soukromí jeho pacientů. Tato námitka souvisela s postavením navrhovatele jako lékař... |
| 5 | 0.467388 | 5 Tdo 1128/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:5.TDO.1128.2024.1__chunk_0023 | ECLI:CZ:NS:2025:5.TDO.1128.2024.1 | PASS | 5. 2020, sp. zn. 5 Tdo 178/2020, ze dne 27. 2. 2013, sp. zn. 8 Tdo 137/2013, uveřejněné pod č. 7/2014 Sb. rozh. tr.), protože umožňuje, aby kdokoli mohl mít k chráněnému dílu či jeho rozmnoženině přístup na místě a v čase podle své vlastní... |
| 6 | 0.461959 | 11 Pzo 7/2024 | USNESENÍ | criminal | operative_part | ECLI:CZ:NS:2025:11.PZO.7.2024.1__chunk_0001 | ECLI:CZ:NS:2025:11.PZO.7.2024.1 | PASS | takto: Podle § 314n odst. 1 tr. řádu příkazem k zjištění údajů o telekomunikačním provozu vydaným soudkyní Okresního soudu ve Znojmě dne 24. 2. 2022 pod sp. zn. 3 Nt 20003/2022, nebyl porušen zákon . |
| 7 | 0.447030 | 5 Tdo 1128/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:5.TDO.1128.2024.1__chunk_0012 | ECLI:CZ:NS:2025:5.TDO.1128.2024.1 | PASS | 10. 2017, sp. zn. 5 Tdo 1167/2017, a ze dne 8. 10. 2014, sp. zn. 5 Tdo 171/2014. Zdůraznil, že obviněný spáchal uvedený trestný čin prostřednictvím veřejně přístupné počítačové sítě, a to umístěním obsahu chráněných autorských děl do veřejn... |
| 8 | 0.441087 | 11 Pzo 7/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:11.PZO.7.2024.1__chunk_0003 | ECLI:CZ:NS:2025:11.PZO.7.2024.1 | PASS | 10. 2024, když zcela prokazatelně nebyla s obviněným v kontaktu a od 3. 5. 2022 pracuje a žije mimo Českou republiku. Jelikož jí nebyly nikdy sděleny žádné informace, z nichž by bylo možno byť jen dovodit, že by se jakkoliv podílela na tres... |
| 9 | 0.435796 | 23 Cdo 938/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:23.CDO.938.2024.1__chunk_0010 | ECLI:CZ:NS:2025:23.CDO.938.2024.1 | PASS | 7. 2020, sp. zn. 23 Cdo 3944/2019, nelze omezení účinků ochranné známky podle § 10 odst. 2 ZOZ považovat za subjektivní právo, s nímž by bylo možno disponovat. Z citovaného judikátu tak vyplývá, že na otázku přechodu omezení účinků ochranné... |
| 10 | 0.434586 | 25 Cdo 3217/2023 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:25.CDO.3217.2023.1__chunk_0006 | ECLI:CZ:NS:2024:25.CDO.3217.2023.1 | PASS | 8. Řešením otázky výkladu a uplatnění článku 85 nařízení č. 883/2004, a toho v jakém rozsahu vstupuje instituce poskytující dávky sociálního zabezpečení (podle předpisů jednoho členského státu Evropské unie), na které vznikl osobě nárok v d... |

### odpočet DPH u daně z přidané hodnoty

- Expected behavior: `insufficient_support`
- Actual label: **PASS**
- Metadata validation: **PASS**
- Result count: **10**
- Top score: **0.528100**
- Notes: unexpected missing-context phrase overlap found at rank 3 | the phrase overlap appears incidental rather than as direct support

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata_validation | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.528100 | 28 Cdo 3321/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:28.CDO.3321.2024.1__chunk_0008 | ECLI:CZ:NS:2025:28.CDO.3321.2024.1 | PASS | 12. 2024 (dále „advokátní tarif“), a náhrady paušálně určených hotových výdajů za jeden úkon právní služby ve výši 300 Kč – § 11 odst. 1 písm. k) a § 13 odst. 4 advokátního tarifu. Protože zástupce dalšího účastníka řízení je plátcem daně z... |
| 2 | 0.453889 | 20 Cdo 3371/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:20.CDO.3371.2024.1__chunk_0004 | ECLI:CZ:NS:2025:20.CDO.3371.2024.1 | PASS | 7) Podle § 337c odst. 1 písm. a) a b) o. s. ř., ve znění zákona č. 291/2017 Sb., se z rozdělované podstaty uspokojují postupně podle těchto skupin: a) pohledávky nákladů vzniklých státu v tomto řízení, b) pohledávky související se správou d... |
| 3 | 0.448508 | 24 Cdo 3585/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:24.CDO.3585.2024.1__chunk_0006 | ECLI:CZ:NS:2025:24.CDO.3585.2024.1 | PASS | 4. 2020 ve výši 800,- Kč, za 5 úkonů vykonaných v době od 17. 4. 2020 do 31. 12. 2021 ve výši 6 000,- Kč a za 4 úkony provedené v době od 1. 1. 2022 do 3. 2. 2022 ve výši 6 750,- Kč a že jsou splněny předpoklady podle ustanovení § 12 odst.... |
| 4 | 0.439609 | 20 Cdo 2839/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2024:20.CDO.2839.2024.1__chunk_0002 | ECLI:CZ:NS:2024:20.CDO.2839.2024.1 | PASS | Odůvodnění: Oprávněná podala dne 4. června 2024 exekuční návrh na zřízení exekutorského zástavního práva na nemovitých věcech ve vlastnictví povinného (specifikované v návrhu) k zajištění pohledávky ve výši 2 124 285,77 Kč podle notářského... |
| 5 | 0.423319 | 29 Cdo 275/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:29.CDO.275.2024.1__chunk_0004 | ECLI:CZ:NS:2025:29.CDO.275.2024.1 | PASS | 177/1996 Sb., o odměnách advokátů a náhradách advokátů za poskytování právních služeb (advokátního tarifu), ve znění účinném k datu podání vyjádření, činí (z tarifní hodnoty 421.400,- Kč) částku 10.020,- Kč, a z paušální částky náhrady hoto... |
| 6 | 0.416858 | 24 Cdo 3585/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:24.CDO.3585.2024.1__chunk_0005 | ECLI:CZ:NS:2025:24.CDO.3585.2024.1 | PASS | 177/1996 Sb., o odměnách advokátů a náhradách advokátů za poskytování právních služeb (advokátní tarif) ve znění pozdějších předpisů (dále též jen „advokátní tarif“) a dovodil, že činí 1 200 Kč za jeden úkon právní služby a že jsou dány dův... |
| 7 | 0.415820 | 20 Cdo 3371/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:20.CDO.3371.2024.1__chunk_0003 | ECLI:CZ:NS:2025:20.CDO.3371.2024.1 | PASS | 5) Podle dovolatele mu mělo být ve druhé skupině správně přiznáno 10 % z výtěžku 2 232 000Kč, tedy částka 223 200 Kč, a nikoli jen částka 187 194,30 Kč. Odvolací soud zaujal právní názor, že základ pro výpočet přihlášené pohledávky podle §... |
| 8 | 0.411192 | 29 Nd 541/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:29.ND.541.2024.1__chunk_0002 | ECLI:CZ:NS:2025:29.ND.541.2024.1 | PASS | Odůvodnění: Návrhem ze dne 18. září 2024 se oprávněný domáhá nařízení a provedení exekuce vůči povinnému pro pohledávku ve výši 2.123,32 Kč s příslušenstvím. Dne 25. září 2024 požádal soudní exekutor JUDr. Lukáš Jícha, Exekutorský úřad Přer... |
| 9 | 0.410050 | 24 Cdo 3585/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:24.CDO.3585.2024.1__chunk_0009 | ECLI:CZ:NS:2025:24.CDO.3585.2024.1 | PASS | 2. 2020 do 21. 10. 2020 v rozsahu 4 úkonů podle ustanovení § 7, § 9 odst. 1 a § 12a advokátního tarifu ve znění účinném do 31. 12. 2021činí 4 800 Kč, že odměna za úkony provedené ve dnech 4. 2. 2021, 23. 11. 2021 a 16. 11. 2021 podle ustano... |
| 10 | 0.404894 | 24 Cdo 3585/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:24.CDO.3585.2024.1__chunk_0008 | ECLI:CZ:NS:2025:24.CDO.3585.2024.1 | PASS | 4. 2020, že ustanovení § 12a advokátního tarifu platilo v řízení o úschovách i v době po 1. 1. 2014 až do svého zrušení dnem 31. 12. 2021 a že při rozhodování o výši odměny procesního opatrovníka ustanoveného soudem se neuplatňuje zásada zá... |

### stavební povolení a územní rozhodnutí

- Expected behavior: `insufficient_support`
- Actual label: **PASS**
- Metadata validation: **PASS**
- Result count: **10**
- Top score: **0.739170**
- Notes: results do not contain direct support and remain broad enough for insufficient-support handling

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata_validation | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.739170 | 22 Cdo 1151/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2025:22.CDO.1151.2024.1__chunk_0004 | ECLI:CZ:NS:2025:22.CDO.1151.2024.1 | PASS | 7. Ve způsobu vypořádání spoluvlastnictví dospěl ke stejnému závěru jako soud prvního stupně, tedy že reálné dělení blokačních pozemků není dobře možné, svůj názor ovšem opřel o jinou skutečnost, konkrétně o veřejný zájem na vybudování ploc... |
| 2 | 0.678375 | 28 Cdo 2670/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:28.CDO.2670.2024.3__chunk_0008 | ECLI:CZ:NS:2024:28.CDO.2670.2024.3 | PASS | 10. 2021, sp. zn. 28 Cdo 2244/2021), nezabýval se odvolací soud v individuálních skutkových poměrech projednávané věci (pozemek je zatížen věcným břemenem chůze a jízdy ve prospěch sousedící stavby č. p. XY a věcným břemenem práva zřídit a... |
| 3 | 0.673138 | 22 Cdo 108/2025 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2025:22.CDO.108.2025.1__chunk_0002 | ECLI:CZ:NS:2025:22.CDO.108.2025.1 | PASS | Odůvodnění: I. Dosavadní průběh řízení 1. Okresní soud v Lounech (dále jen „soud prvního stupně“) rozsudkem ze dne 16. 5. 2024, č. j. 12 C 152/2023-52, určil, že pozemek p. č. XY a pozemek p. č. XY, jehož součástí je stavba – rodinný dům č.... |
| 4 | 0.671107 | 28 Cdo 2670/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:28.CDO.2670.2024.3__chunk_0007 | ECLI:CZ:NS:2024:28.CDO.2670.2024.3 | PASS | 12. Rozhodovací praxe dovolacího soudu i Ústavního soudu – a to již jde-li o restituci původní, nikoliv poskytnutím náhradního plnění (zde v podobě jiného pozemku) – přitom vychází z toho, že zákon o půdě, stejně tak jako jiné restituční př... |
| 5 | 0.662086 | 24 Cdo 671/2024; 24 Cdo 675/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2024:24.CDO.671.2024.1__chunk_0002 | ECLI:CZ:NS:2024:24.CDO.671.2024.1 | PASS | Odůvodnění: 1. Okresní soud v Pelhřimově rozsudkem ze dne 31. 5. 2023, č. j. 5 C 255/2015-1439, rozhodl o určení vlastnického práva k celkem devíti jednotkám v domě č. p. XY v XY. Jednalo se o byty, garáže a dílnu. Stručně řečeno, u jednote... |
| 6 | 0.656905 | 28 Cdo 2670/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:28.CDO.2670.2024.3__chunk_0006 | ECLI:CZ:NS:2024:28.CDO.2670.2024.3 | PASS | 5. 2017, sp. zn. 28 Cdo 5045/2015, či usnesení Nejvyššího soudu ze dne 26. 4. 2007, sp. zn. 28 Cdo 220/2005). 11. Judikatura dovolacího soudu dále dovodila, že překážkou vydání náhradního zemědělského pozemku ve smyslu zákona o půdě může bý... |
| 7 | 0.650130 | 28 Cdo 2670/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:28.CDO.2670.2024.3__chunk_0003 | ECLI:CZ:NS:2024:28.CDO.2670.2024.3 | PASS | 3. Proti výroku III rozsudku odvolacího soudu podala dovolání žalovaná. Předestřela otázku, zda pozemek parc. č. XY v k. ú. XY je vhodným náhradním pozemkem ve smyslu § 11a zákona o půdě. Namítala, že jeho vydání brání překážka spočívající... |
| 8 | 0.648119 | 3 Tdo 650/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:3.TDO.650.2024.1__chunk_0011 | ECLI:CZ:NS:2024:3.TDO.650.2024.1 | PASS | 9. Poslední okruh dovolacích námitek se týká postavení obviněného jako stavbyvedoucího. Obviněný zdůrazňuje, že pro oblast XY–XY nebyl stavbyvedoucím, který by zodpovídal za bezpečnost při provádění stavby. Jako stavbyvedoucí byl zodpovědný... |
| 9 | 0.642305 | 33 Cdo 1129/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:33.CDO.1129.2024.1__chunk_0005 | ECLI:CZ:NS:2025:33.CDO.1129.2024.1 | PASS | 40/1964 Sb., občanského zákoníku, ve znění pozdějších předpisů (dále jen „obč. zák.“), je třeba na něj pohlížet jako na součást pozemku - parcely č. 3/6 (§ 505, § 506 odst. 1 o. z.). Z toho vyplývá závěr, že nebylo možné na základě smlouvy... |
| 10 | 0.641288 | 28 Cdo 2670/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:28.CDO.2670.2024.3__chunk_0005 | ECLI:CZ:NS:2024:28.CDO.2670.2024.3 | PASS | 9. 2009, sp. zn. 28 Cdo 4876/2008, či usnesení Nejvyššího soudu ze dne 3. 1. 2011, sp. zn. 28 Cdo 99/2010, ze dne 24. 9. 2014, sp. zn. 28 Cdo 3304/2014, a ze dne 2. 5. 2016, sp. zn. 28 Cdo 4400/2015). Ani oprávněná osoba se tudíž nemůže neo... |

### náhrada nákladů dovolacího řízení

- Expected behavior: `ask_for_clarification`
- Actual label: **PASS**
- Metadata validation: **PASS**
- Result count: **10**
- Top score: **0.770444**
- Notes: results span 10 documents, 2 section types, and 1 legal areas

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata_validation | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.770444 | 28 Cdo 3321/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:28.CDO.3321.2024.1__chunk_0008 | ECLI:CZ:NS:2025:28.CDO.3321.2024.1 | PASS | 12. 2024 (dále „advokátní tarif“), a náhrady paušálně určených hotových výdajů za jeden úkon právní služby ve výši 300 Kč – § 11 odst. 1 písm. k) a § 13 odst. 4 advokátního tarifu. Protože zástupce dalšího účastníka řízení je plátcem daně z... |
| 2 | 0.769617 | 29 Cdo 275/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:29.CDO.275.2024.1__chunk_0004 | ECLI:CZ:NS:2025:29.CDO.275.2024.1 | PASS | 177/1996 Sb., o odměnách advokátů a náhradách advokátů za poskytování právních služeb (advokátního tarifu), ve znění účinném k datu podání vyjádření, činí (z tarifní hodnoty 421.400,- Kč) částku 10.020,- Kč, a z paušální částky náhrady hoto... |
| 3 | 0.768661 | 28 Cdo 3513/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:28.CDO.3513.2024.1__chunk_0006 | ECLI:CZ:NS:2025:28.CDO.3513.2024.1 | PASS | 15. O nákladech dovolacího řízení bylo rozhodnuto v intencích § 243 odst. 3 věty první, § 224 odst. 1, § 151 odst. 1 části věty před středníkem a § 146 odst. 3 o. s. ř.; dovolání žalované bylo odmítnuto a na straně žalobce lze za účelně vyn... |
| 4 | 0.768099 | 27 Cdo 2699/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:27.CDO.2699.2024.1__chunk_0005 | ECLI:CZ:NS:2025:27.CDO.2699.2024.1 | PASS | 13. Výrok o náhradě nákladů dovolacího řízení se opírá o § 243c odst. 3, § 224 odst. 1 a § 146 odst. 3 o. s. ř., když dovolání žalobkyně bylo odmítnuto a žalovanému vzniklo právo na náhradu účelně vynaložených nákladů dovolacího řízení. 14.... |
| 5 | 0.760259 | 25 Cdo 2348/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:25.CDO.2348.2024.1__chunk_0002 | ECLI:CZ:NS:2024:25.CDO.2348.2024.1 | PASS | Odůvodnění: 1. Žalobkyně se podanou žalobou domáhala 998.089 Kč s příslušenstvím z titulu odpovědnosti advokáta za škodu. Částka 203.186 Kč představovala marně vynaložené náklady řízení, částka 594.903 Kč kapitalizovaný úrok z prodlení z 1.... |
| 6 | 0.752328 | 27 Cdo 3338/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:27.CDO.3338.2024.1__chunk_0001 | ECLI:CZ:NS:2025:27.CDO.3338.2024.1 | PASS | takto: I. Dovolací řízení se zastavuje . II. Žalovaná je povinna zaplatit žalobkyni na náhradě nákladů dovolacího řízení 64.323,60 Kč do tří dnů od právní moci tohoto usnesení k rukám zástupce žalobkyně. |
| 7 | 0.750306 | 20 Cdo 3371/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:20.CDO.3371.2024.1__chunk_0003 | ECLI:CZ:NS:2025:20.CDO.3371.2024.1 | PASS | 5) Podle dovolatele mu mělo být ve druhé skupině správně přiznáno 10 % z výtěžku 2 232 000Kč, tedy částka 223 200 Kč, a nikoli jen částka 187 194,30 Kč. Odvolací soud zaujal právní názor, že základ pro výpočet přihlášené pohledávky podle §... |
| 8 | 0.749727 | 23 Cdo 68/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:23.CDO.68.2025.1__chunk_0002 | ECLI:CZ:NS:2025:23.CDO.68.2025.1 | PASS | Odůvodnění: 1. Žalobkyně se podanou žalobou domáhala na žalovaném zaplacení částky 176 000 Kč s příslušenstvím jako náhrady škody. Podle tvrzení žalobkyně se uplatněný nárok měl skládat jednak z částky 140 000 Kč (představující pořizovací c... |
| 9 | 0.742144 | 23 Cdo 707/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2024:23.CDO.707.2024.1__chunk_0001 | ECLI:CZ:NS:2024:23.CDO.707.2024.1 | PASS | takto: I. Dovolání se odmítá. II. Žalována je povinna zaplatit žalobci na náhradu nákladů dovolacího řízení částku 17 714 Kč do tří dnů právní moci tohoto usnesení k rukám jeho právního zástupce. |
| 10 | 0.737311 | 23 Cdo 434/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2024:23.CDO.434.2024.1__chunk_0001 | ECLI:CZ:NS:2024:23.CDO.434.2024.1 | PASS | takto: I. Dovolání se odmítá. II. Žalobkyně je povinna zaplatit žalovanému na náhradě nákladů dovolacího řízení částku 4 114 Kč do tří dnů od právní moci tohoto usnesení k rukám právní zástupkyně žalovaného. |

### zjevně neopodstatněné dovolání

- Expected behavior: `ask_for_clarification`
- Actual label: **PASS**
- Metadata validation: **PASS**
- Result count: **10**
- Top score: **0.736710**
- Notes: results span 10 documents, 1 section types, and 1 legal areas

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata_validation | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.736710 | 21 Cdo 44/2025 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:21.CDO.44.2025.1__chunk_0001 | ECLI:CZ:NS:2025:21.CDO.44.2025.1 | PASS | takto: Dovolání povinného se odmítá. |
| 2 | 0.731032 | 21 Cdo 2658/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:21.CDO.2658.2024.3__chunk_0001 | ECLI:CZ:NS:2025:21.CDO.2658.2024.3 | PASS | takto: Dovolání povinného se odmítá . |
| 3 | 0.686126 | 27 Cdo 1921/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:27.CDO.1921.2024.1__chunk_0001 | ECLI:CZ:NS:2025:27.CDO.1921.2024.1 | PASS | takto: Dovolání se odmítá . |
| 4 | 0.686126 | 26 Cdo 125/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2024:26.CDO.125.2024.1__chunk_0001 | ECLI:CZ:NS:2024:26.CDO.125.2024.1 | PASS | takto: Dovolání se odmítá . |
| 5 | 0.686126 | 29 NSCR 70/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:29.NSCR.70.2024.1__chunk_0001 | ECLI:CZ:NS:2025:29.NSCR.70.2024.1 | PASS | takto: Dovolání se odmítá . |
| 6 | 0.686126 | 22 Cdo 3556/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:22.CDO.3556.2024.1__chunk_0001 | ECLI:CZ:NS:2025:22.CDO.3556.2024.1 | PASS | takto: Dovolání se odmítá . |
| 7 | 0.686126 | 29 NSCR 1/2025 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:29.NSCR.1.2025.1__chunk_0001 | ECLI:CZ:NS:2025:29.NSCR.1.2025.1 | PASS | takto: Dovolání se odmítá . |
| 8 | 0.686126 | 26 Cdo 84/2025 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:26.CDO.84.2025.1__chunk_0001 | ECLI:CZ:NS:2025:26.CDO.84.2025.1 | PASS | takto: Dovolání se odmítá . |
| 9 | 0.686126 | 21 Cdo 1566/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:21.CDO.1566.2024.1__chunk_0001 | ECLI:CZ:NS:2025:21.CDO.1566.2024.1 | PASS | takto: Dovolání se odmítá . |
| 10 | 0.686126 | 20 Cdo 875/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:20.CDO.875.2024.1__chunk_0001 | ECLI:CZ:NS:2025:20.CDO.875.2024.1 | PASS | takto: Dovolání se odmítá . |

### odmítnutí dovolání

- Expected behavior: `ask_for_clarification`
- Actual label: **PASS**
- Metadata validation: **PASS**
- Result count: **10**
- Top score: **0.922904**
- Notes: results span 10 documents, 1 section types, and 1 legal areas

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata_validation | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.922904 | 21 Cdo 1566/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:21.CDO.1566.2024.1__chunk_0001 | ECLI:CZ:NS:2025:21.CDO.1566.2024.1 | PASS | takto: Dovolání se odmítá . |
| 2 | 0.922904 | 27 Cdo 1921/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:27.CDO.1921.2024.1__chunk_0001 | ECLI:CZ:NS:2025:27.CDO.1921.2024.1 | PASS | takto: Dovolání se odmítá . |
| 3 | 0.922904 | 26 Cdo 125/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2024:26.CDO.125.2024.1__chunk_0001 | ECLI:CZ:NS:2024:26.CDO.125.2024.1 | PASS | takto: Dovolání se odmítá . |
| 4 | 0.922904 | 29 NSCR 1/2025 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:29.NSCR.1.2025.1__chunk_0001 | ECLI:CZ:NS:2025:29.NSCR.1.2025.1 | PASS | takto: Dovolání se odmítá . |
| 5 | 0.922904 | 20 Cdo 875/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:20.CDO.875.2024.1__chunk_0001 | ECLI:CZ:NS:2025:20.CDO.875.2024.1 | PASS | takto: Dovolání se odmítá . |
| 6 | 0.922904 | 29 NSCR 70/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:29.NSCR.70.2024.1__chunk_0001 | ECLI:CZ:NS:2025:29.NSCR.70.2024.1 | PASS | takto: Dovolání se odmítá . |
| 7 | 0.922904 | 22 Cdo 3556/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:22.CDO.3556.2024.1__chunk_0001 | ECLI:CZ:NS:2025:22.CDO.3556.2024.1 | PASS | takto: Dovolání se odmítá . |
| 8 | 0.922904 | 26 Cdo 84/2025 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:26.CDO.84.2025.1__chunk_0001 | ECLI:CZ:NS:2025:26.CDO.84.2025.1 | PASS | takto: Dovolání se odmítá . |
| 9 | 0.893802 | 21 Cdo 2658/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:21.CDO.2658.2024.3__chunk_0001 | ECLI:CZ:NS:2025:21.CDO.2658.2024.3 | PASS | takto: Dovolání povinného se odmítá . |
| 10 | 0.887667 | 21 Cdo 44/2025 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:21.CDO.44.2025.1__chunk_0001 | ECLI:CZ:NS:2025:21.CDO.44.2025.1 | PASS | takto: Dovolání povinného se odmítá. |

### rodinný dům

- Expected behavior: `ask_for_clarification`
- Actual label: **PASS**
- Metadata validation: **PASS**
- Result count: **10**
- Top score: **0.383765**
- Notes: results span 8 documents, 2 section types, and 2 legal areas

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata_validation | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.383765 | 26 Cdo 1854/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2025:26.CDO.1854.2024.1__chunk_0008 | ECLI:CZ:NS:2025:26.CDO.1854.2024.1 | PASS | 16. Pojem „jednotka“ použitý v § 1196 odst. 2 o. z. je však třeba vykládat ve spojení s § 1159 o. z., jenž stanoví, že jednotka zahrnuje nejen byt (jako prostorově oddělenou část domu), ale také podíl na společných částech (vzájemně spojené... |
| 2 | 0.383488 | 3 Tdo 984/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:3.TDO.984.2024.1__chunk_0006 | ECLI:CZ:NS:2024:3.TDO.984.2024.1 | PASS | 2. 2023 do 20. 2. 2023, kdy měl mít společné nezletilé děti v péči, avšak od poškozené věděl, že zdravotní stav nezletilého syna toto neumožňuje, pod záminkou být s dětmi apeloval právě opět v přítomnosti společných dětí na poškozenou, aby... |
| 3 | 0.376240 | 26 Cdo 439/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:26.CDO.439.2024.1__chunk_0003 | ECLI:CZ:NS:2024:26.CDO.439.2024.1 | PASS | 6. 2012, č. j. 11 Co 155/2012-78, bylo zrušeno právo společného nájmu tam specifikovaného družstevního bytu (předmětného bytu) a garáže (blíže označené) a společné členství v bytovém družstvu s tím, že výlučnou členkou družstva a výlučnou n... |
| 4 | 0.353452 | 7 Tdo 1096/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:7.TDO.1096.2024.1__chunk_0004 | ECLI:CZ:NS:2025:7.TDO.1096.2024.1 | PASS | 6. Pokud jde o koncept sdílené újmy, obviněný uvedl, že z rozsudku soudu prvního stupně a ze spisu vyplynulo, že celá rodina poškozeného udržuje mezi sebou velmi dobré vztahy, stojí při sobě a vzájemně se podporuje. Manželka, děti a tchán s... |
| 5 | 0.340566 | 24 Cdo 671/2024; 24 Cdo 675/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2024:24.CDO.671.2024.1__chunk_0002 | ECLI:CZ:NS:2024:24.CDO.671.2024.1 | PASS | Odůvodnění: 1. Okresní soud v Pelhřimově rozsudkem ze dne 31. 5. 2023, č. j. 5 C 255/2015-1439, rozhodl o určení vlastnického práva k celkem devíti jednotkám v domě č. p. XY v XY. Jednalo se o byty, garáže a dílnu. Stručně řečeno, u jednote... |
| 6 | 0.328666 | 26 Cdo 439/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:26.CDO.439.2024.1__chunk_0006 | ECLI:CZ:NS:2024:26.CDO.439.2024.1 | PASS | 8. Žalovaný ve vyjádření k dovolání považuje rozhodnutí odvolacího soudu za správné. Do okamžiku zajištění bytové náhrady se řídí vztah bývalého nájemce a pronajímatele ustanovením § 712a obč. zák. a teprve po zajištění bytové náhrady – jes... |
| 7 | 0.325900 | 4 Tdo 1044/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:4.TDO.1044.2024.1__chunk_0005 | ECLI:CZ:NS:2024:4.TDO.1044.2024.1 | PASS | 14. Podle dovolatele je celé řízení vedeno tendenčně a je poplatné době, ve které žijeme, reagující na společenskou objednávku pod tíhou různých hnutí za hranicí demagogie. Příkladmo akcentuje, že soudy nijak nehodnotily skutečnost, že pošk... |
| 8 | 0.312357 | 22 Cdo 108/2025 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2025:22.CDO.108.2025.1__chunk_0002 | ECLI:CZ:NS:2025:22.CDO.108.2025.1 | PASS | Odůvodnění: I. Dosavadní průběh řízení 1. Okresní soud v Lounech (dále jen „soud prvního stupně“) rozsudkem ze dne 16. 5. 2024, č. j. 12 C 152/2023-52, určil, že pozemek p. č. XY a pozemek p. č. XY, jehož součástí je stavba – rodinný dům č.... |
| 9 | 0.310003 | 3 Tdo 980/2024 | USNESENÍ | criminal | signature | ECLI:CZ:NS:2024:3.TDO.980.2024.1__chunk_0028 | ECLI:CZ:NS:2024:3.TDO.980.2024.1 | PASS | 51. Obdobná situace pak nastává u vniknutí do garáže v XY ulici, kterou měl pronajatou svědek O. H. od svědka J. H., a ve které se nacházely věci obou těchto osob [skutek pod bodem 2) výroku o vině]. Předně je potřebné uvést, že věci tam od... |
| 10 | 0.299560 | 26 Cdo 439/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:26.CDO.439.2024.1__chunk_0008 | ECLI:CZ:NS:2024:26.CDO.439.2024.1 | PASS | 1. 2010, sp. zn. 28 Cdo 2146/2009). 14. Z § 712a obč. zák. vyplývá, že obsah vzájemných práv a povinností účastníků právního vztahu, který je uvedeným ustanovením posuzován, se řídí § 687 až § 699 obč. zák. , tedy i § 696 až § 699 obč. zák.... |

### dovolání

- Expected behavior: `ask_for_clarification`
- Actual label: **PASS**
- Metadata validation: **PASS**
- Result count: **10**
- Top score: **0.574465**
- Notes: results span 10 documents, 2 section types, and 2 legal areas

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata_validation | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.574465 | 3 Tdo 984/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:3.TDO.984.2024.1__chunk_0022 | ECLI:CZ:NS:2024:3.TDO.984.2024.1 | PASS | 32. Jestliže obviněný ve svém dovolání požádal, aby |
| 2 | 0.567749 | 30 Cdo 308/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:30.CDO.308.2025.1__chunk_0004 | ECLI:CZ:NS:2025:30.CDO.308.2025.1 | PASS | 2. 2015, sp. zn. II. ÚS 2716/13). Ústavní soud se dále k otázce náležitostí dovolání vyjádřil v usnesení ze dne 26. 6. 2014, sp. zn. III. ÚS 1675/14, kde přiléhavě vysvětlil účel povinnosti dovolatele uvést, v čem konkrétně spatřuje splnění... |
| 3 | 0.562130 | 23 Cdo 434/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2024:23.CDO.434.2024.1__chunk_0004 | ECLI:CZ:NS:2024:23.CDO.434.2024.1 | PASS | 9. Podle § 241a odst. 2 o. s. ř. v dovolání musí být vedle obecných náležitostí (§ 42 odst. 4) uvedeno, proti kterému rozhodnutí směřuje, v jakém rozsahu se rozhodnutí napadá, vymezení důvodu dovolání, v čem dovolatel spatřuje splnění předp... |
| 4 | 0.556672 | 33 Cdo 889/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:33.CDO.889.2024.1__chunk_0005 | ECLI:CZ:NS:2025:33.CDO.889.2024.1 | PASS | 11. V rozsudku ze dne 1. |
| 5 | 0.552189 | 21 Cdo 44/2025 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:21.CDO.44.2025.1__chunk_0001 | ECLI:CZ:NS:2025:21.CDO.44.2025.1 | PASS | takto: Dovolání povinného se odmítá. |
| 6 | 0.550429 | 21 Cdo 2658/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:21.CDO.2658.2024.3__chunk_0001 | ECLI:CZ:NS:2025:21.CDO.2658.2024.3 | PASS | takto: Dovolání povinného se odmítá . |
| 7 | 0.537154 | 11 Tdo 875/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:11.TDO.875.2024.1__chunk_0003 | ECLI:CZ:NS:2025:11.TDO.875.2024.1 | PASS | 12. 2023, sp. zn. 1 ZT 64/2023, pro skutek uvedený pod bodem II. obžaloby. II. Dovolání a vyjádření k němu 4. Proti rozsudku odvolacího soudu podává nyní nejvyšší státní zástupce dovolání, a to z důvodů podle § 265b odst. 1 písm. g) a h) tr... |
| 8 | 0.536745 | 20 Cdo 15/2025 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:20.CDO.15.2025.1__chunk_0001 | ECLI:CZ:NS:2025:20.CDO.15.2025.1 | PASS | takto: Řízení o dovolání obou povinných se zastavuje . |
| 9 | 0.536745 | 20 Cdo 13/2025 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:20.CDO.13.2025.1__chunk_0001 | ECLI:CZ:NS:2025:20.CDO.13.2025.1 | PASS | takto: Řízení o dovolání obou povinných se zastavuje . |
| 10 | 0.534765 | 23 Cdo 3535/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:23.CDO.3535.2024.1__chunk_0001 | ECLI:CZ:NS:2025:23.CDO.3535.2024.1 | PASS | takto: Dovolací řízení se zastavuje . |

### místní příslušnost

- Expected behavior: `ask_for_clarification`
- Actual label: **PASS**
- Metadata validation: **PASS**
- Result count: **10**
- Top score: **0.475141**
- Notes: results span 10 documents, 3 section types, and 2 legal areas

| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata_validation | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.475141 | 29 Nd 63/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:29.ND.63.2025.1__chunk_0002 | ECLI:CZ:NS:2025:29.ND.63.2025.1 | PASS | Odůvodnění: 1. Usnesením ze dne 13. prosince 2024, č. j. 27 Nc 2451/2024-44, vyslovil Okresní soud Praha - západ svou místní nepříslušnost (bod I. výroku), rozhodl, že věc bude po právní moci usnesení předložena Nejvyššímu soudu k určení mí... |
| 2 | 0.458850 | 20 Nd 18/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:20.ND.18.2025.1__chunk_0003 | ECLI:CZ:NS:2025:20.ND.18.2025.1 | PASS | 99/1963 Sb., občanský soudní řád, ve znění pozdějších předpisů – dále též jen „o. s. ř.“). V exekučním návrhu oprávněné a stejně tak v žádosti soudního exekutora je uvedena adresa povinného XY. Lustrací v informačním systému základních regi... |
| 3 | 0.443376 | 24 Nd 34/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:24.ND.34.2025.1__chunk_0004 | ECLI:CZ:NS:2025:24.ND.34.2025.1 | PASS | 9. 2012, sp. zn. 31 Nd 200/2012, uveřejněné ve Sbírce soudních rozhodnutí a stanovisek pod č. 4, ročník 2013). 8. Nejvyšší soud v obdobných situacích vychází při určení místní příslušnosti exekučního soudu ze zásady hospodárnosti řízení zak... |
| 4 | 0.436403 | 25 Nd 86/2025 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:25.ND.86.2025.1__chunk_0001 | ECLI:CZ:NS:2025:25.ND.86.2025.1 | PASS | takto: Věc vedenou u Okresního soudu v Ústí nad Labem pod sp. zn. 72 EXE 3512/2024 projedná a rozhodne Okresní soud v Ústí nad Labem. |
| 5 | 0.434284 | 5 Tdo 318/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:5.TDO.318.2024.1__chunk_0024 | ECLI:CZ:NS:2024:5.TDO.318.2024.1 | PASS | 18/2006-II. Sb. rozh. tr.). Odkázat lze i na odbornou literaturu – viz např. ŠÁMAL, P., PÚRY, F., SOTOLÁŘ, A., ŠTENGLOVÁ, I. Podnikání a ekonomická kriminalita v České republice . 1. vydání. Praha: C. H. Beck, 2001, s. 266. 43. Majetek obch... |
| 6 | 0.423219 | 26 Nd 573/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:26.ND.573.2024.1__chunk_0003 | ECLI:CZ:NS:2025:26.ND.573.2024.1 | PASS | 9. 2012, sp. zn. 31 Nd 200/2012, uveřejněné pod č. 4/2013 Sbírky soudních rozhodnutí a stanovisek). Nelze tak zjistit podmínky pro určení místní příslušnosti exekučního soudu (§ 45 odst. 2 exekučního řádu). 5. Nejvyšší soud proto podle § 11... |
| 7 | 0.422494 | 7 Td 6/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:7.TD.6.2025.1__chunk_0004 | ECLI:CZ:NS:2025:7.TD.6.2025.1 | PASS | 7. Pro určení místní příslušnosti soudu je podstatné místo, kde obviněná jednala, kde neznámé osobě umožnila užít k výše uvedené transakci svůj účet, tedy konkrétně především na jakém místě se nacházela, když finanční prostředky ze svého úč... |
| 8 | 0.421469 | 4 Tdo 1018/2024 | USNESENÍ | criminal | signature | ECLI:CZ:NS:2024:4.TDO.1018.2024.1__chunk_0016 | ECLI:CZ:NS:2024:4.TDO.1018.2024.1 | PASS | V Brně dne 17. 12. 2024 JUDr. Jiří Pácal předseda senátu |
| 9 | 0.420128 | 29 Nd 461/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2024:29.ND.461.2024.1__chunk_0001 | ECLI:CZ:NS:2024:29.ND.461.2024.1 | PASS | takto: Věc vedenou u Okresního soudu v Ústí nad Labem pod sp. zn. 73 EXE 2135/2024 projedná a rozhodne Okresní soud v Ústí nad Labem. |
| 10 | 0.419401 | 22 Nd 435/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2024:22.ND.435.2024.1__chunk_0001 | ECLI:CZ:NS:2024:22.ND.435.2024.1 | PASS | takto: Věc, vedenou u Okresního soudu v Ústí nad Labem pod sp. zn. 72 EXE 1636/2024, projedná a rozhodne Městský soud v Brně. |

## Final Recommendation
- WARN: collection integrity and metadata are intact, but some queries remain indirect or need dataset-review follow-up before this evaluation can be treated as a stronger regression gate.
