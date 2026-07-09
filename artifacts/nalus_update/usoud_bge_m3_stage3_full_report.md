# Ustavni soud / NALUS - BGE-M3 Stage 3 Full Report

Generated: 2026-07-09T08:35:42Z

## Goal

- Run a guarded full-corpus candidate collection without touching production.

## Run Summary

- Script path: `scripts/build_usoud_bge_m3_candidate.py`
- Builder version: `usoud-bge-m3-guarded-v5`
- Stage 1 commit reference: `4290559 Add guarded ÚS BGE-M3 smoke builder`
- Mode: `full`
- Action: `execute`
- Dry-run command: `python scripts/build_usoud_bge_m3_candidate.py --mode full --limit 600 --collection-name nalus_us_bge_m3_mvp_recent_3h_20260709 --source-manifest batches/manifest.json --output-dir artifacts/nalus_update/usoud_bge_m3_mvp_recent_3h_20260709 --no-alias-update --ingest-slice mvp_recent_3h --decision-date-to 2026-07-09 --newest-first --embedding-batch-size 16 --full-record-batch-size 50 --dry-run`
- Execute command: `python scripts/build_usoud_bge_m3_candidate.py --mode full --limit 600 --collection-name nalus_us_bge_m3_mvp_recent_3h_20260709 --source-manifest batches/manifest.json --output-dir artifacts/nalus_update/usoud_bge_m3_mvp_recent_3h_20260709 --no-alias-update --ingest-slice mvp_recent_3h --decision-date-to 2026-07-09 --newest-first --embedding-batch-size 16 --full-record-batch-size 50 --execute --recreate-full-collection`
- Input: `batches/manifest.json`
- Limit: `600`
- Selected records: `600`
- Source files: `44` listed in JSON summary
- Generated chunks: `4980`
- Estimated Qdrant points: `4980`
- Estimated embedding texts: `4980`
- Embedding model: `BAAI/bge-m3`
- Vector dimension validation: `PASS (1024)`
- Qdrant collection: `nalus_us_bge_m3_mvp_recent_3h_20260709`
- Collection point count before: `None`
- Collection point count after: `4980`
- Inserted point count: `4980`
- Qdrant write occurred: `True`
- `nalus_live` before/after: `784812` / `784812`
- `nalus_stable_20260326` before/after: `784812` / `784812`
- `nalus_live` target before/after: `nalus_stable_20260326` / `nalus_stable_20260326`
- BM25 status: `available`
- Hybrid/RRF status: `available_rrf`
- Payload metadata validation: `PASS`
- Production API touched: `False`
- Aliases touched: `False`
- Aliases changed by verification: `False`
- Retrieval logic changed: `False`
- Clarification gate changed: `False`
- Production safety touched: `False`
- Final status: `PASS`
- Stage recommendation: `full_candidate_complete_review_required`

## Qdrant Aliases

- `nalus_live` -> `nalus_stable_20260326`

## Sample Payloads

- doc=`ECLI:CZ:US:2026:4.US.922.26.1` date=`6. 5. 2026` chunk=`2784` snippet="rozhodnutí je zřejmé, že okresní soud rodinné poměry stěžovatele zjišťoval (zejména výpověďmi stěžovatele a jeho družky), z jeho zjištění však nevyplývá nic, co by značilo, že b..."
- doc=`ECLI:CZ:US:2026:2.US.129.26.1` date=`15. 4. 2026` chunk=`4485` snippet="NALUS - databáze rozhodnutí Ústavního soudu II.ÚS 129/26 ze dne 15. 4. 2026 Česká republika USNESENÍ Ústavního soudu Ústavní soud rozhodl v senátu složeném z předsedy Jiřího Při..."
- doc=`ECLI:CZ:US:2026:2.US.730.26.1` date=`22. 4. 2026` chunk=`4151` snippet="potřeby nijak nezohlednily, nebo kdyby bez výpovědi třetího ze sourozenců považovaly bez dalšího jeho názor za nutně totožný. Nad rámec právě uvedeného je přitom nutné uvést, že..."

## Full Retrieval Validation Queries

### `právo na spravedlivý proces`

- Dense results from scoped collection: `True`
- Dense results from candidate collection: `True`
- All reported hits from candidate collection: `True`
- BM25 results: `5`
- Hybrid results: `5`
- Qualitative relevance: `plausible_top_hits_keyword_overlap`

- dense score=`0.684431` doc=`ECLI:CZ:US:2026:1.US.927.25.1` date=`13. 5. 2026` snippet="spravedlivý proces. Právo na soudní ochranu (čl. 36 odst. 1 Listiny), potažmo právo na spravedlivý proces (čl. 6 Úmluvy o ochraně lidských práv a základních svobod), představuje právo procesního charakteru. Právo na s..."
- dense score=`0.626534` doc=`ECLI:CZ:US:2026:3.US.3031.24.1` date=`21. 4. 2026` snippet="může každý domáhat stanoveným postupem svého práva u nezávislého a nestranného soudu. Naplnění záruk plynoucích z tohoto práva Ústavní soud posuzuje s ohledem na průběh celého řízení. Z hlediska zachování základního p..."
- dense score=`0.612255` doc=`ECLI:CZ:US:2026:3.US.3031.24.1` date=`21. 4. 2026` snippet="právního předchůdce nedošlo k popření práva na obhajobu či dalších aspektů vyplývajících z práva na spravedlivý proces. Konkrétně poskytnou nástupnické právnické osobě veškerá příslušná práva i dostatečná poučení s oh..."
- dense score=`0.610062` doc=`ECLI:CZ:US:2026:2.US.3174.25.1` date=`6. 5. 2026` snippet="Stěžovatel nesouhlasí se závěry správních soudů a namítá, že napadenými rozsudky bylo porušeno jeho právo na spravedlivý proces dle čl. 36 odst. 1 Listiny, čl. 6 odst. 1 Úmluvy a čl. 47 Listiny základních práv EU, prá..."
- dense score=`0.589194` doc=`ECLI:CZ:US:2026:3.US.952.26.1` date=`22. 5. 2026` snippet="221 odst. 2 o. s. ř. nařídil, aby v dalším řízení věc projednal a rozhodl jiný samosoudce. II. Argumentace stěžovatelky 8. Stěžovatelka spatřuje porušení svého práva na spravedlivý proces podle čl. 36 odst. 1 Listiny ..."

- hybrid score=`0.032787` doc=`ECLI:CZ:US:2026:1.US.927.25.1` date=`13. 5. 2026` snippet="spravedlivý proces. Právo na soudní ochranu (čl. 36 odst. 1 Listiny), potažmo právo na spravedlivý proces (čl. 6 Úmluvy o ochraně lidských práv a základních svobod), představuje právo procesního charakteru. Právo na s..."
- hybrid score=`0.031498` doc=`ECLI:CZ:US:2026:2.US.3174.25.1` date=`6. 5. 2026` snippet="Stěžovatel nesouhlasí se závěry správních soudů a namítá, že napadenými rozsudky bylo porušeno jeho právo na spravedlivý proces dle čl. 36 odst. 1 Listiny, čl. 6 odst. 1 Úmluvy a čl. 47 Listiny základních práv EU, prá..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:3.US.3031.24.1` date=`21. 4. 2026` snippet="může každý domáhat stanoveným postupem svého práva u nezávislého a nestranného soudu. Naplnění záruk plynoucích z tohoto práva Ústavní soud posuzuje s ohledem na průběh celého řízení. Z hlediska zachování základního p..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:3.US.952.26.1` date=`22. 5. 2026` snippet="jen tehdy, bude-li to pro účastníky řízení z hlediska naplnění jejich práva na spravedlivý proces příznivější. [srov. např. nález ze dne 19. 4. 2021 sp. zn. II. ÚS 52/21 (N 79/105 SbNU 322); usnesení ze dne 21. 10. 20..."
- hybrid score=`0.015873` doc=`ECLI:CZ:US:2026:3.US.3031.24.1` date=`21. 4. 2026` snippet="právního předchůdce nedošlo k popření práva na obhajobu či dalších aspektů vyplývajících z práva na spravedlivý proces. Konkrétně poskytnou nástupnické právnické osobě veškerá příslušná práva i dostatečná poučení s oh..."

### `opomenuté důkazy`

- Dense results from scoped collection: `True`
- Dense results from candidate collection: `True`
- All reported hits from candidate collection: `True`
- BM25 results: `5`
- Hybrid results: `5`
- Qualitative relevance: `plausible_top_hits_keyword_overlap`

- dense score=`0.545561` doc=`ECLI:CZ:US:2026:1.US.3310.25.1` date=`28. 4. 2026` snippet="Obecné soudy naopak skutkový stav ustálený na základě řádně provedeného dokazování subsumovaly správně pod zákonné znaky skutkové podstaty loupeže podle § 173 odst. 1 trestního zákoníku (b. 28 a b. 32 až 33 odůvodnění..."
- dense score=`0.511988` doc=`ECLI:CZ:US:2026:3.US.3649.25.1` date=`15. 4. 2026` snippet="věnovat (srov. rozsudek velkého senátu Evropského soudu pro lidská práva ze dne 18. 12. 2018 ve věci Murtazaliyeva proti Rusku, stížnost č. 36658/05, body 143, 162 a 166, podobně též rozsudek Evropského soudu pro lids..."
- dense score=`0.504426` doc=`ECLI:CZ:US:2026:2.US.719.26.2` date=`22. 4. 2026` snippet="výhrůžkami buď nereagovali vůbec a dívali se jinam, nebo přikyvovali a maximálně ke konci pobytu v nemocnici stěžovatele napomenuli, ať nedělá ostudu a zanechá svého dosavadního jednání. Z provedeného dokazování tedy ..."
- dense score=`0.504379` doc=`ECLI:CZ:US:2026:2.US.248.26.2` date=`15. 4. 2026` snippet="stěžovatel poškozené opakovaně vyhrožoval a dožadoval se její pozornosti (bod 17 napadeného rozsudku odvolacího soudu), poškozená proto měla ze stěžovatele oprávněný strach a o partnerský vztah tak podle obecných soud..."
- dense score=`0.503727` doc=`ECLI:CZ:US:2026:3.US.418.26.1` date=`29. 4. 2026` snippet="pochybení nezjistil. 13. Na základě uvedeného je zřejmé, že stěžovatel v ústavní stížnosti opakovaně předkládá shodnou argumentaci, jako v obhajobě a v opravných prostředcích, jíž se obecné soudy dostatečně zabývaly a..."

- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:1.US.3310.25.1` date=`28. 4. 2026` snippet="Obecné soudy naopak skutkový stav ustálený na základě řádně provedeného dokazování subsumovaly správně pod zákonné znaky skutkové podstaty loupeže podle § 173 odst. 1 trestního zákoníku (b. 28 a b. 32 až 33 odůvodnění..."
- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:3.US.454.26.3` date=`15. 4. 2026` snippet="vrchního soudu napadli dovoláním. Nejvyšší soud je napadeným usnesením podle § 265i odst. 1 písm. e) trestního řádu odmítl jako zjevně neopodstatněná. II. Argumentace stěžovatelů 8. Argumentace obou stěžovatelů se z v..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:3.US.3649.25.1` date=`15. 4. 2026` snippet="věnovat (srov. rozsudek velkého senátu Evropského soudu pro lidská práva ze dne 18. 12. 2018 ve věci Murtazaliyeva proti Rusku, stížnost č. 36658/05, body 143, 162 a 166, podobně též rozsudek Evropského soudu pro lids..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:2.US.730.26.1` date=`22. 4. 2026` snippet="zn. IV. ÚS 3749/17 ze dne 9. 1. 2018). Případné porušení práv dětí by se ve sféře základních práv otce mohlo projevit, nikoli však přímo, nýbrž pouze zprostředkovaně (viz např. nález sp. zn. II. ÚS 1931/17 ze dne 19. ..."
- hybrid score=`0.015873` doc=`ECLI:CZ:US:2026:2.US.719.26.2` date=`22. 4. 2026` snippet="výhrůžkami buď nereagovali vůbec a dívali se jinam, nebo přikyvovali a maximálně ke konci pobytu v nemocnici stěžovatele napomenuli, ať nedělá ostudu a zanechá svého dosavadního jednání. Z provedeného dokazování tedy ..."

### `odůvodnění rozhodnutí`

- Dense results from scoped collection: `True`
- Dense results from candidate collection: `True`
- All reported hits from candidate collection: `True`
- BM25 results: `5`
- Hybrid results: `5`
- Qualitative relevance: `plausible_top_hits_keyword_overlap`

- dense score=`0.623365` doc=`ECLI:CZ:US:2026:4.US.1779.25.2` date=`3. 6. 2026` snippet="nález ze dne 20. 6. 1995 sp. zn. III. ÚS 84/94 ). Prostřednictvím odůvodnění totiž soud seznamuje účastníky řízení s úvahami, které jej vedly k vydání rozhodnutí. Soud se musí vypořádat se všemi relevantními okolnostm..."
- dense score=`0.621716` doc=`ECLI:CZ:US:2026:3.US.3515.24.1` date=`22. 4. 2026` snippet="soudního exekutora, shledají-li existenci mimořádných okolností zvláštního zřetele hodných, pro které by povinnému náhrada nákladů exekučního řízení (vzniklých nejčastěji v souvislosti s podáním návrhu na zastavení ex..."
- dense score=`0.598229` doc=`ECLI:CZ:US:2026:4.US.422.26.1` date=`15. 4. 2026` snippet="neodůvodňovat rozhodnutí o návrhu na upuštění od výkonu trestu odnětí svobody jen na případy, ve kterých je buď žadateli vyhověno, nebo vznáší-li irelevantní argumenty nebo je-li jeho podání neodůvodněné, popřípadě od..."
- dense score=`0.595898` doc=`ECLI:CZ:US:2026:3.US.995.26.1` date=`3. 6. 2026` snippet="nelze pojímat jako rozhodování o vině obviněného a jemu uloženém trestu. 13. Z čl. 36 Listiny a z čl. 6 Úmluvy o ochraně lidských práv a základních svobod vyplývá, že rozhodnutí soudu, na jehož základě došlo k zbavení..."
- dense score=`0.589874` doc=`ECLI:CZ:US:2026:1.US.3093.25.1` date=`19. 5. 2026` snippet="rozhodnutí řádně odůvodnit a adekvátně se vypořádat s argumenty uplatněnými účastníky řízení (srov. např. nález ze dne 23. 6. 2006 sp. zn. III. ÚS 521/05 ). Závazek odůvodnit rozhodnutí však nemůže být chápán tak, že ..."

- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:4.US.1779.25.2` date=`3. 6. 2026` snippet="nález ze dne 20. 6. 1995 sp. zn. III. ÚS 84/94 ). Prostřednictvím odůvodnění totiž soud seznamuje účastníky řízení s úvahami, které jej vedly k vydání rozhodnutí. Soud se musí vypořádat se všemi relevantními okolnostm..."
- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:4.US.3060.25.2` date=`15. 4. 2026` snippet="oslabení rodičovských kompetencí otce jako nepřiměřený a nedostatečně odůvodněný. 30. Ze zvukového záznamu z jednání před krajským soudem plyne, že soud výše popsané jednání otce v průběhu řízení také poměrně emotivně..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:3.US.3515.24.1` date=`22. 4. 2026` snippet="soudního exekutora, shledají-li existenci mimořádných okolností zvláštního zřetele hodných, pro které by povinnému náhrada nákladů exekučního řízení (vzniklých nejčastěji v souvislosti s podáním návrhu na zastavení ex..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:4.US.1193.26.1` date=`18. 5. 2026` snippet="(SU) s tím, že je takové využití charakteristické pro konkrétní území města. Dále jen odkázal na budoucí soutěžní dialog, v němž bude řešena konkrétní podoba využití území. Nejvyšší správní soud zmínil vlastní judikat..."
- hybrid score=`0.015873` doc=`ECLI:CZ:US:2026:4.US.422.26.1` date=`15. 4. 2026` snippet="neodůvodňovat rozhodnutí o návrhu na upuštění od výkonu trestu odnětí svobody jen na případy, ve kterých je buď žadateli vyhověno, nebo vznáší-li irelevantní argumenty nebo je-li jeho podání neodůvodněné, popřípadě od..."

### `porušení základních práv`

- Dense results from scoped collection: `True`
- Dense results from candidate collection: `True`
- All reported hits from candidate collection: `True`
- BM25 results: `5`
- Hybrid results: `5`
- Qualitative relevance: `plausible_top_hits_keyword_overlap`

- dense score=`0.677265` doc=`ECLI:CZ:US:2026:2.US.1295.26.1` date=`18. 5. 2026` snippet="ředitelství policie Středočeského kraje, Služby kriminální policie a vyšetřování, Odboru hospodářské kriminality (policejní orgán), Krajského státního zastupitelství v Praze a Krajského soudu v Praze v rámci trestního..."
- dense score=`0.645268` doc=`ECLI:CZ:US:2026:1.US.2196.25.1` date=`19. 6. 2026` snippet="Domažlicích ze dne 23. května 2024 č. j. 6 C 59/2018-685 a rozsudkem Nejvyššího soudu ze dne 13. března 2024 č. j. 28 Cdo 3880/2023-656, bylo porušeno základní právo stěžovatelky na soudní ochranu podle čl. 36 odst. 1..."
- dense score=`0.634675` doc=`ECLI:CZ:US:2026:1.US.30.26.1` date=`27. 5. 2026` snippet="záhlaví uvedeným usnesením odmítl stěžovatelovo dovolání. III. Argumentace stěžovatele 6. Stěžovatel v ústavní stížnosti navrhl zrušení napadených rozhodnutí s tvrzením, že jimi byla porušena jeho ústavně zaručená prá..."
- dense score=`0.629094` doc=`ECLI:CZ:US:2026:3.US.1024.26.1` date=`3. 6. 2026` snippet="zákona o Ústavním soudu se stěžovatelka domáhá zrušení v záhlaví označených rozhodnutí s tvrzením, že jimi došlo k porušení jejích ústavně zaručených základních práv zakotvených v čl. 1, čl. 3 odst. 1, čl. 7 odst. 1 a..."
- dense score=`0.623336` doc=`ECLI:CZ:US:2026:3.US.2824.25.1` date=`29. 4. 2026` snippet="II. Argumentace stěžovatele 6. Řádně zastoupený stěžovatel ve své včas podané ústavní stížnosti splňující požadavky zákona č. 182/1993 Sb., o Ústavním soudu, ve znění pozdějších předpisů (dále jen "zákon o Ústavním so..."

- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:2.US.1295.26.1` date=`18. 5. 2026` snippet="ředitelství policie Středočeského kraje, Služby kriminální policie a vyšetřování, Odboru hospodářské kriminality (policejní orgán), Krajského státního zastupitelství v Praze a Krajského soudu v Praze v rámci trestního..."
- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:3.US.439.26.1` date=`22. 5. 2026` snippet="1470/24 ; nález ze dne 12. 1. 2005 sp. zn. III. ÚS 441/04 (N 6/36 SbNU 53)]. Z toho důvodu Ústavní soud typicky odmítá ústavní stížnosti podané proti dílčím (procesním) rozhodnutím pro nepřípustnost. 12. Princip subsi..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:1.US.2196.25.1` date=`19. 6. 2026` snippet="Domažlicích ze dne 23. května 2024 č. j. 6 C 59/2018-685 a rozsudkem Nejvyššího soudu ze dne 13. března 2024 č. j. 28 Cdo 3880/2023-656, bylo porušeno základní právo stěžovatelky na soudní ochranu podle čl. 36 odst. 1..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:1.US.729.26.1` date=`29. 4. 2026` snippet="svobod není garance úspěchu v řízení. Na tom bez dalšího nemohou nic změnit ani tvrzená dotčení základních práv stěžovatelky a základních práv jejího syna. Stěžovatelka ostatně tyto své námitky v ústavní stížnosti ani..."
- hybrid score=`0.015873` doc=`ECLI:CZ:US:2026:1.US.30.26.1` date=`27. 5. 2026` snippet="záhlaví uvedeným usnesením odmítl stěžovatelovo dovolání. III. Argumentace stěžovatele 6. Stěžovatel v ústavní stížnosti navrhl zrušení napadených rozhodnutí s tvrzením, že jimi byla porušena jeho ústavně zaručená prá..."

### `extrémní nesoulad`

- Dense results from scoped collection: `True`
- Dense results from candidate collection: `True`
- All reported hits from candidate collection: `True`
- BM25 results: `5`
- Hybrid results: `5`
- Qualitative relevance: `plausible_top_hits_keyword_overlap`

- dense score=`0.507896` doc=`ECLI:CZ:US:2026:2.US.3645.25.1` date=`3. 6. 2026` snippet="se řídí neúčinnou právní úpravou. 8. Stěžovatelka namítá, že vedlejší účastnice v době jednání se stěžovatelkou o uzavření dodatku k pojistné smlouvě věděla, že pojistná smlouva má vady, které způsobují její absolutní..."
- dense score=`0.498484` doc=`ECLI:CZ:US:2026:3.US.3515.24.1` date=`22. 4. 2026` snippet="smlouva o úvěru shledána nemravná a tudíž neplatná, a to i když bylo stěžovatelkou zaplaceno několikanásobně více, než kolik činila původně půjčená částka. Oprávněná se proti rozhodnutí okresního soudu o zastavení exe..."
- dense score=`0.496027` doc=`ECLI:CZ:US:2026:3.US.2824.25.1` date=`29. 4. 2026` snippet="možný (viz usnesení Nejvyššího soudu sp. zn. 8 Tdo 1409/2016)." (bod 313, str. 52). Krajský soud rovněž odkázal na usnesení Nejvyššího soudu sp. zn. 3 Tdo 859/2018, podle kterého: "Při posuzování rozhraničení vědomé n..."
- dense score=`0.489178` doc=`ECLI:CZ:US:2026:3.US.2144.25.1` date=`28. 5. 2026` snippet="souladu se zákonem. 34. V řízení nevyvstal stěžovateli tvrzený extrémní nesoulad mezi provedenými důkazy a skutkovými zjištěními. Ten by spočíval v racionálně neobhajitelném úsudku soudů o jejich vztahu. Extrémní rozp..."
- dense score=`0.486669` doc=`ECLI:CZ:US:2026:4.US.897.26.1` date=`29. 4. 2026` snippet="dohody mezi společnostmi a informací od britského finančního úřadu, což krajský soud nevypořádal a neústavně opomněl tyto důkazy. 13. Řady neústavních pochybení se dopustil i Nejvyšší soud a jeho pochybení se z velké ..."

- hybrid score=`0.031498` doc=`ECLI:CZ:US:2026:3.US.2144.25.1` date=`28. 5. 2026` snippet="souladu se zákonem. 34. V řízení nevyvstal stěžovateli tvrzený extrémní nesoulad mezi provedenými důkazy a skutkovými zjištěními. Ten by spočíval v racionálně neobhajitelném úsudku soudů o jejich vztahu. Extrémní rozp..."
- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:2.US.3645.25.1` date=`3. 6. 2026` snippet="se řídí neúčinnou právní úpravou. 8. Stěžovatelka namítá, že vedlejší účastnice v době jednání se stěžovatelkou o uzavření dodatku k pojistné smlouvě věděla, že pojistná smlouva má vady, které způsobují její absolutní..."
- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:2.US.945.26.1` date=`28. 4. 2026` snippet="se odvolací soud z hlediska dokazování fakticky nezabýval. Po Ústavním soudu tak stěžovatel požaduje, aby jako další přezkumná instance zhodnotil, zda shromáždění konkrétního důkazu je účelné či nikoliv. Taková role Ú..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:3.US.3515.24.1` date=`22. 4. 2026` snippet="smlouva o úvěru shledána nemravná a tudíž neplatná, a to i když bylo stěžovatelkou zaplaceno několikanásobně více, než kolik činila původně půjčená částka. Oprávněná se proti rozhodnutí okresního soudu o zastavení exe..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:1.US.862.26.1` date=`23. 4. 2026` snippet="není další instancí v systému obecného soudnictví (srov. např. nález sp. zn. III. ÚS 23/93 ). K zásahu do hodnocení důkazů přistupuje pouze tehdy, je-li dán extrémní nesoulad skutkových zjištění s právními závěry nebo..."

### `rovnost účastníků řízení`

- Dense results from scoped collection: `True`
- Dense results from candidate collection: `True`
- All reported hits from candidate collection: `True`
- BM25 results: `5`
- Hybrid results: `5`
- Qualitative relevance: `plausible_top_hits_keyword_overlap`

- dense score=`0.582796` doc=`ECLI:CZ:US:2026:2.US.818.26.1` date=`15. 4. 2026` snippet="rovnosti účastníků řízení, když stěžovatelka by byla povinna vrátit nemovitosti insolvenčnímu správci a ten by však již neměl prostředky pro vrácení celé kupní ceny. Takový stav je v rozporu se zásadou rovnosti účastn..."
- dense score=`0.567984` doc=`ECLI:CZ:US:2026:Pl.US.46.25.1` date=`29. 4. 2026` snippet="srovnatelné skupiny členy zastupitelstev vykonávající souběžně funkce na různých úrovních územní samosprávy na straně jedné a členy zastupitelstev, kteří současně vykonávají odlišné veřejné funkce, jsou zaměstnanci ne..."
- dense score=`0.556444` doc=`ECLI:CZ:US:2026:4.US.3787.25.1` date=`20. 5. 2026` snippet="strukturální součást právní úpravy civilního dovolání, která významně napomáhá naplňovat zákonné a ústavní účely tohoto institutu. Omezení práva na přístup k soudu, jež z něj nutně plyne, je však vyváženo závažností v..."
- dense score=`0.555948` doc=`ECLI:CZ:US:2026:Pl.US.46.25.1` date=`29. 4. 2026` snippet="již dříve přezkoumával ústavnost souběhu funkcí v samosprávě s členstvím v zákonodárném či vládním sboru. Podle navrhovatelky Ústavní soud také výslovně zdůraznil, že právní úpravu shledává za ústavně přijatelnou s oh..."
- dense score=`0.555808` doc=`ECLI:CZ:US:2026:Pl.US.39.25.1` date=`24. 6. 2026` snippet="vymezený v čl. 3 odst. 1 Listiny. Jedná se o kritérium neutrální, nikoliv podezřelé, a tedy nevytvářející prostor pro posouzení diskriminace podle čl. 3 odst. 1 Listiny. Nelze než uzavřít, že toto ustanovení není apli..."

- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:2.US.818.26.1` date=`15. 4. 2026` snippet="rovnosti účastníků řízení, když stěžovatelka by byla povinna vrátit nemovitosti insolvenčnímu správci a ten by však již neměl prostředky pro vrácení celé kupní ceny. Takový stav je v rozporu se zásadou rovnosti účastn..."
- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:2.US.403.26.1` date=`3. 6. 2026` snippet="NALUS - databáze rozhodnutí Ústavního soudu II.ÚS 403/26 ze dne 3. 6. 2026 Česká republika USNESENÍ Ústavního soudu Ústavní soud rozhodl v senátu složeném z předsedy Jiřího Přibáně, soudce zpravodaje Martina Smolka a ..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:Pl.US.46.25.1` date=`29. 4. 2026` snippet="srovnatelné skupiny členy zastupitelstev vykonávající souběžně funkce na různých úrovních územní samosprávy na straně jedné a členy zastupitelstev, kteří současně vykonávají odlišné veřejné funkce, jsou zaměstnanci ne..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:Pl.US.46.25.1` date=`29. 4. 2026` snippet="99-102), podrobně zabýval vymezením toho, jakým způsobem je třeba přezkoumávat namítané zásahy do práva na rovné zacházení, s cílem sjednotit dosavadní judikaturní přístupy. 84. Jak plyne z citovaného nálezu Pl. ÚS 18..."
- hybrid score=`0.015873` doc=`ECLI:CZ:US:2026:4.US.3787.25.1` date=`20. 5. 2026` snippet="strukturální součást právní úpravy civilního dovolání, která významně napomáhá naplňovat zákonné a ústavní účely tohoto institutu. Omezení práva na přístup k soudu, jež z něj nutně plyne, je však vyváženo závažností v..."

### `právo na zákonného soudce`

- Dense results from scoped collection: `True`
- Dense results from candidate collection: `True`
- All reported hits from candidate collection: `True`
- BM25 results: `5`
- Hybrid results: `5`
- Qualitative relevance: `plausible_top_hits_keyword_overlap`

- dense score=`0.667826` doc=`ECLI:CZ:US:2026:1.US.1901.25.1` date=`20. 5. 2026` snippet="variant, kterou stěžovatelé zmiňují (ať již vzájemná koordinace či přijetí "pilotního" rozhodnutí, které potom do jisté míry převezme těleso druhé), jejich námitka se s obsahem práva na zákonného soudce zcela míjí. Ob..."
- dense score=`0.643879` doc=`ECLI:CZ:US:2026:4.US.2718.25.1` date=`13. 5. 2026` snippet="Listiny. Příslušnost soudu i soudce stanoví zákon. Podstatou této záruky je, že podávání návrhů soudům a přidělování případů soudcům se odehrává podle předem stanovených pravidel, čímž má být minimalizována možnost je..."
- dense score=`0.618192` doc=`ECLI:CZ:US:2026:1.US.1901.25.1` date=`20. 5. 2026` snippet="tomto rozsahu téměř zcela překrývá s námitkami uplatněnými ostatními stěžovateli, musí Ústavní soud s ohledem na některé odlišnosti obou typů ústavní stížnosti věnovat zvláštní pozornost i komunálnímu rozměru věci. IV..."
- dense score=`0.614922` doc=`ECLI:CZ:US:2026:3.US.2429.24.1` date=`15. 4. 2026` snippet="pravidla jen tam, kde to zákon výslovně připouští [srov. nález Ústavního soudu ze dne 11. 10. 2016 sp. zn. II. ÚS 849/16 (N 188/83 SbNU 81), body 32 a 36]. K přijetí usnesení o odmítnutí dovolání z důvodu, že dovolání..."
- dense score=`0.591069` doc=`ECLI:CZ:US:2026:Pl.US.18.25.1` date=`24. 6. 2026` snippet="nespočívá pouze v právu jednotlivce zahájit řízení před soudem. Jde především o právo na to, aby soud o věci meritorně rozhodl, pokud jednotlivec dodržel zákonem stanovený (a ústavně souladný) postup a podmínky pro po..."

- hybrid score=`0.032787` doc=`ECLI:CZ:US:2026:1.US.1901.25.1` date=`20. 5. 2026` snippet="variant, kterou stěžovatelé zmiňují (ať již vzájemná koordinace či přijetí "pilotního" rozhodnutí, které potom do jisté míry převezme těleso druhé), jejich námitka se s obsahem práva na zákonného soudce zcela míjí. Ob..."
- hybrid score=`0.031754` doc=`ECLI:CZ:US:2026:4.US.2718.25.1` date=`13. 5. 2026` snippet="Listiny. Příslušnost soudu i soudce stanoví zákon. Podstatou této záruky je, že podávání návrhů soudům a přidělování případů soudcům se odehrává podle předem stanovených pravidel, čímž má být minimalizována možnost je..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:4.US.2338.25.1` date=`23. 4. 2026` snippet="základních práv a svobod. Usnesení Nejvyššího soudu porušilo též stěžovatelovo základní právo na zákonného soudce podle čl. 38 odst. 1 Listiny. II. Usnesení Nejvyššího soudu ze dne 30. dubna 2025 č. j. 3 Tdo 242/2025-..."
- hybrid score=`0.015873` doc=`ECLI:CZ:US:2026:1.US.1901.25.1` date=`20. 5. 2026` snippet="tomto rozsahu téměř zcela překrývá s námitkami uplatněnými ostatními stěžovateli, musí Ústavní soud s ohledem na některé odlišnosti obou typů ústavní stížnosti věnovat zvláštní pozornost i komunálnímu rozměru věci. IV..."
- hybrid score=`0.015873` doc=`ECLI:CZ:US:2026:4.US.2338.25.1` date=`23. 4. 2026` snippet="takových případů je, pokud trestní soudy hodnotí důkazy svévolně, poruší zásadu bezprostřednosti (přímosti) dokazování či pokud hodnocení (alespoň potenciálně významného) důkazu v napadeném rozhodnutí zcela chybí. 31...."

### `odmítnutí dovolání`

- Dense results from scoped collection: `True`
- Dense results from candidate collection: `True`
- All reported hits from candidate collection: `True`
- BM25 results: `5`
- Hybrid results: `5`
- Qualitative relevance: `partial_keyword_overlap`

- dense score=`0.694767` doc=`ECLI:CZ:US:2026:2.US.1074.26.1` date=`6. 5. 2026` snippet="dovolání stěžovatele jako nepřípustné. V odůvodnění usnesení přiléhavě poukázal na to, že podle § 238 odst. 1 písm. e) o. s. ř. není podle § 237 o. s. ř. dovolání přípustné proti usnesením, proti nimž je přípustná žal..."
- dense score=`0.657885` doc=`ECLI:CZ:US:2026:3.US.268.26.1` date=`15. 4. 2026` snippet="legitimaci k podání odvolání nepřiznává. Naopak usnesení, jímž městský soud podle § 29 odst. 2 insolvenčního zákona potvrdil usnesení schůze věřitelů o ustanovení nového insolvenčního správce, má povahu rozhodnutí vyd..."
- dense score=`0.6495` doc=`ECLI:CZ:US:2026:4.US.1179.26.1` date=`27. 5. 2026` snippet="stížností téhož stěžovatele, která byla založena na totožné argumentaci, se zabýval v usnesení ze dne 6. 5. 2026 sp. zn. II. ÚS 1074/26 , kterým byla ústavní stížnost odmítnuta. Ústavní soud nemá důvod se od uvedených..."
- dense score=`0.644119` doc=`ECLI:CZ:US:2026:4.US.851.26.1` date=`29. 4. 2026` snippet="jde o zjevně neopodstatněný návrh ve smyslu § 43 odst. 2 písm. a) zákona o Ústavním soudu. Uvedené ustanovení v zájmu racionality a efektivity řízení před Ústavním soudem dává tomuto soudu pravomoc posoudit opodstatně..."
- dense score=`0.641358` doc=`ECLI:CZ:US:2026:1.US.1086.26.1` date=`28. 4. 2026` snippet="ÚS 1160/24 , kde jsou důvody odmítnutí obdobných ústavních stížností téže stěžovatelky podrobně vysvětleny. 3. Ústavní soud proto návrh odmítl pro vady za přiměřeného užití § 43 odst. 1 písm. a) zákona o Ústavním soud..."

- hybrid score=`0.032266` doc=`ECLI:CZ:US:2026:2.US.1074.26.1` date=`6. 5. 2026` snippet="dovolání stěžovatele jako nepřípustné. V odůvodnění usnesení přiléhavě poukázal na to, že podle § 238 odst. 1 písm. e) o. s. ř. není podle § 237 o. s. ř. dovolání přípustné proti usnesením, proti nimž je přípustná žal..."
- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:3.US.2429.24.1` date=`15. 4. 2026` snippet="soudu [srov. nález ze dne 31. 10. 2023 sp. zn. III. ÚS 647/23 (N 153/120 SbNU 215) bod 27], kam spadá i posuzování subjektivní přípustnosti dovolání. Zda tedy bude dovolání posouzeno jako subjektivně přípustné či nepř..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:3.US.268.26.1` date=`15. 4. 2026` snippet="legitimaci k podání odvolání nepřiznává. Naopak usnesení, jímž městský soud podle § 29 odst. 2 insolvenčního zákona potvrdil usnesení schůze věřitelů o ustanovení nového insolvenčního správce, má povahu rozhodnutí vyd..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:3.US.2429.24.1` date=`15. 4. 2026` snippet="o. s. ř. došlo se souhlasem všech členů senátu. 23. Nejvyšší soud však v nyní posuzované věci tuto fázi řízení obešel, když otázku aktivní legitimace k podání žaloby pro zmatečnost, která měla být předmětem právního p..."
- hybrid score=`0.015873` doc=`ECLI:CZ:US:2026:4.US.1179.26.1` date=`27. 5. 2026` snippet="stížností téhož stěžovatele, která byla založena na totožné argumentaci, se zabýval v usnesení ze dne 6. 5. 2026 sp. zn. II. ÚS 1074/26 , kterým byla ústavní stížnost odmítnuta. Ústavní soud nemá důvod se od uvedených..."

### `náklady řízení`

- Dense results from scoped collection: `True`
- Dense results from candidate collection: `True`
- All reported hits from candidate collection: `True`
- BM25 results: `5`
- Hybrid results: `5`
- Qualitative relevance: `plausible_top_hits_keyword_overlap`

- dense score=`0.61779` doc=`ECLI:CZ:US:2026:2.US.614.26.1` date=`20. 5. 2026` snippet="částku 8 606 621,69 Kč (výrok I.) a na náhradu nákladů řízení částku 893 701,50 Kč (výrok II.) a dále na náhradu nákladů řízení vzniklých státu částku 6 231,50 Kč (výrok III.). 3. Vedlejší účastnice napadla rozsudek o..."
- dense score=`0.607618` doc=`ECLI:CZ:US:2026:3.US.307.26.1` date=`29. 4. 2026` snippet="sama o sobě nezpůsobuje protiústavnost napadených rozhodnutí. Soudy stanovily výši nákladů řízení na základě zákonných pravidel, která v rozhodnutích výslovně označily. Stěžovatel v ústavní stížnosti nezpochybnil ani ..."
- dense score=`0.607322` doc=`ECLI:CZ:US:2026:2.US.263.26.3` date=`13. 5. 2026` snippet="zaplatit žalobci na ztížení společenského uplatnění částku 2 900 000 Kč s příslušenstvím (výrok I.) a na bolestném částku 600 840 Kč s příslušenstvím (výrok II.). V částkách 3 790 000 Kč a 400 560 Kč bylo řízení zasta..."
- dense score=`0.588893` doc=`ECLI:CZ:US:2026:4.US.864.26.1` date=`6. 5. 2026` snippet="vyhlášen při druhém jednání, na jehož základě stěžovatelka zaslala soudu vyčíslení nákladů právního zastoupení v celkové výši 43 088,10 Kč. Písemné vyhotovení napadeného rozsudku bylo stěžovatelce doručeno dne 18. 2. ..."
- dense score=`0.587775` doc=`ECLI:CZ:US:2026:4.US.722.26.1` date=`29. 4. 2026` snippet="citelný dopad do majetkové sféry účastníků řízení, samotný spor o nákladech řízení zpravidla nedosahuje intenzity způsobilé porušit jejich základní práva a svobody (srov. např. nález ze dne 28. 8. 2024 sp. zn. I. ÚS 2..."

- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:2.US.614.26.1` date=`20. 5. 2026` snippet="částku 8 606 621,69 Kč (výrok I.) a na náhradu nákladů řízení částku 893 701,50 Kč (výrok II.) a dále na náhradu nákladů řízení vzniklých státu částku 6 231,50 Kč (výrok III.). 3. Vedlejší účastnice napadla rozsudek o..."
- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:1.US.1570.25.1` date=`29. 4. 2026` snippet="požadoval pověřit exekutora pro náklady exekuce a náklady oprávněné v exekučním řízení, aniž určoval způsob vymožení těchto nákladů. 7. Okresní soud v Příbrami považoval stížnostní námitky za nedůvodné. Exekutor mohl ..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:3.US.307.26.1` date=`29. 4. 2026` snippet="sama o sobě nezpůsobuje protiústavnost napadených rozhodnutí. Soudy stanovily výši nákladů řízení na základě zákonných pravidel, která v rozhodnutích výslovně označily. Stěžovatel v ústavní stížnosti nezpochybnil ani ..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:Pl.US.18.25.1` date=`24. 6. 2026` snippet="prostředky a zajišťovat efektivitu rozhodovacího procesu. Nezajištění tlumočníka prodlužuje řízení, zvyšuje náklady a blokuje rozhodování. V řízení o žádosti je naprosto adekvátní, aby náklady na tlumočníka nesl žadat..."
- hybrid score=`0.015873` doc=`ECLI:CZ:US:2026:2.US.263.26.3` date=`13. 5. 2026` snippet="zaplatit žalobci na ztížení společenského uplatnění částku 2 900 000 Kč s příslušenstvím (výrok I.) a na bolestném částku 600 840 Kč s příslušenstvím (výrok II.). V částkách 3 790 000 Kč a 400 560 Kč bylo řízení zasta..."

### `vlastnické právo`

- Dense results from scoped collection: `True`
- Dense results from candidate collection: `True`
- All reported hits from candidate collection: `True`
- BM25 results: `5`
- Hybrid results: `5`
- Qualitative relevance: `plausible_top_hits_keyword_overlap`

- dense score=`0.531692` doc=`ECLI:CZ:US:2026:2.US.3174.25.1` date=`6. 5. 2026` snippet="zdrojovky. V důsledku toho mělo dojít k újmě na majetkových hodnotách stěžovatele, konkrétně ke snížení hodnoty jeho majetku. Stěžovatel odkazuje na nález Ústavního soudu sp. zn. II. ÚS 2144/14 ze dne 4. 2. 2016 a uvá..."
- dense score=`0.528708` doc=`ECLI:CZ:US:2026:2.US.3198.25.1` date=`22. 4. 2026` snippet="vlastnické právo k zajištěným věcem, nýbrž z jakých finančních prostředků byly věci pořízeny. Proto je rovněž irelevantní, že manželé mají vypořádané společné jmění manželů. Vrchní soud označil rozsah zkoumání vlastni..."
- dense score=`0.518693` doc=`ECLI:CZ:US:2026:2.US.947.26.1` date=`6. 5. 2026` snippet="vlastnického práva vedlejších účastnic není v konkrétních okolnostech dané věci zjevně excesivní, svévolný nebo zneužívající. Podle ustálené judikatury Ústavního soudu jsou obecné soudy povinny při střetu základních p..."
- dense score=`0.518654` doc=`ECLI:CZ:US:2026:4.US.384.26.1` date=`13. 5. 2026` snippet="výrok byl podle § 159a odst. 1 občanského soudního řádu závazný jen pro účastníky daného řízení, tedy pro tehdejší povinný subjekt (Zemědělské družstvo Kolovraty), nikoli pro současné vlastníky zapsané v katastru nemo..."
- dense score=`0.510867` doc=`ECLI:CZ:US:2026:1.US.2302.25.1` date=`22. 4. 2026` snippet="Pohledávku vůči stěžovatelce měla vydražit společnost AGHAMA. Rozhodným právem je právo slovenské, a byť jde o originární způsob nabytí, šlo o dobrovolnou dražbu se soukromoprávním charakterem. Nejvyšší soud Slovenské..."

- hybrid score=`0.031498` doc=`ECLI:CZ:US:2026:4.US.384.26.1` date=`13. 5. 2026` snippet="výrok byl podle § 159a odst. 1 občanského soudního řádu závazný jen pro účastníky daného řízení, tedy pro tehdejší povinný subjekt (Zemědělské družstvo Kolovraty), nikoli pro současné vlastníky zapsané v katastru nemo..."
- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:2.US.3174.25.1` date=`6. 5. 2026` snippet="zdrojovky. V důsledku toho mělo dojít k újmě na majetkových hodnotách stěžovatele, konkrétně ke snížení hodnoty jeho majetku. Stěžovatel odkazuje na nález Ústavního soudu sp. zn. II. ÚS 2144/14 ze dne 4. 2. 2016 a uvá..."
- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:1.US.1115.26.1` date=`27. 5. 2026` snippet="podmínky řádné držby, a nedošlo tak k vydržení. I kdyby právní předchůdci stěžovatelky předmětnou část pozemku vydrželi, stěžovatelka by si za této situace nemohla započítat dobu, po kterou měli předmětnou část pozemk..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:2.US.3198.25.1` date=`22. 4. 2026` snippet="vlastnické právo k zajištěným věcem, nýbrž z jakých finančních prostředků byly věci pořízeny. Proto je rovněž irelevantní, že manželé mají vypořádané společné jmění manželů. Vrchní soud označil rozsah zkoumání vlastni..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:1.US.1115.26.1` date=`27. 5. 2026` snippet="zda je takový výklad ústavně souladný, opakovaně zabýval. Dospěl přitom k závěru, že takový výklad není svévolný a neopodstatňuje jeho kasační zásah (srov. v podrobnostech usnesení ze dne 24. 11. 2023 sp. zn. IV. ÚS 1..."

### `svoboda projevu`

- Dense results from scoped collection: `True`
- Dense results from candidate collection: `True`
- All reported hits from candidate collection: `True`
- BM25 results: `5`
- Hybrid results: `5`
- Qualitative relevance: `partial_keyword_overlap`

- dense score=`0.570068` doc=`ECLI:CZ:US:2026:1.US.2699.25.1` date=`15. 4. 2026` snippet="Vedle toho stěžovatelka odkazovala i na garance těchto práv obsažené v čl. 6 Úmluvy o ochraně lidských práv a základních svobod (dále i jen "Úmluva"). 6. Stěžovatelka sice přímo netvrdí, že by napadenými rozhodnutími ..."
- dense score=`0.548823` doc=`ECLI:CZ:US:2026:1.US.2699.25.1` date=`15. 4. 2026` snippet="stěžovatele za spáchání trestného činu šíření poplašné zprávy. Stěžovatelka měla za to, že podobný náhled na proporcionalitu trestněprávní sankce za politický projev měly obecné soudy přijmout i v její věci. 21. Tento..."
- dense score=`0.545919` doc=`ECLI:CZ:US:2026:1.US.2699.25.1` date=`15. 4. 2026` snippet="kontextu občanskoprávní žaloby, jsou ovšem aplikovatelné i v situaci, kdy k zásahu do svobody projevu došlo prostředky trestního práva. Na druhé straně však Ústavní soud musí při vyvažování v kolizi stojících práv vzí..."
- dense score=`0.538798` doc=`ECLI:CZ:US:2026:1.US.2699.25.1` date=`15. 4. 2026` snippet="pod čl. 8 Úmluvy, resp. čl. 10 Listiny. Otázkou proto zůstává, zda trestní soudy v rámci svého rozhodování dostatečně zohlednily i proporcionalitu zásahu do stěžovatelčiny svobody projevu. 17. Ústavní soud v nálezu sp..."
- dense score=`0.538707` doc=`ECLI:CZ:US:2026:1.US.2699.25.1` date=`15. 4. 2026` snippet="straně druhé stojí lidská důstojnost osob - cílů jejích verbálních útoků. Poskytnutí trestněprávní ochrany těmto právům lze vnímat a má být vnímáno jako realizace pozitivní povinnosti státu chránit základní práva jedn..."

- hybrid score=`0.032787` doc=`ECLI:CZ:US:2026:1.US.2699.25.1` date=`15. 4. 2026` snippet="Vedle toho stěžovatelka odkazovala i na garance těchto práv obsažené v čl. 6 Úmluvy o ochraně lidských práv a základních svobod (dále i jen "Úmluva"). 6. Stěžovatelka sice přímo netvrdí, že by napadenými rozhodnutími ..."
- hybrid score=`0.032258` doc=`ECLI:CZ:US:2026:1.US.2699.25.1` date=`15. 4. 2026` snippet="stěžovatele za spáchání trestného činu šíření poplašné zprávy. Stěžovatelka měla za to, že podobný náhled na proporcionalitu trestněprávní sankce za politický projev měly obecné soudy přijmout i v její věci. 21. Tento..."
- hybrid score=`0.031746` doc=`ECLI:CZ:US:2026:1.US.2699.25.1` date=`15. 4. 2026` snippet="kontextu občanskoprávní žaloby, jsou ovšem aplikovatelné i v situaci, kdy k zásahu do svobody projevu došlo prostředky trestního práva. Na druhé straně však Ústavní soud musí při vyvažování v kolizi stojících práv vzí..."
- hybrid score=`0.015625` doc=`ECLI:CZ:US:2026:1.US.2699.25.1` date=`15. 4. 2026` snippet="pod čl. 8 Úmluvy, resp. čl. 10 Listiny. Otázkou proto zůstává, zda trestní soudy v rámci svého rozhodování dostatečně zohlednily i proporcionalitu zásahu do stěžovatelčiny svobody projevu. 17. Ústavní soud v nálezu sp..."
- hybrid score=`0.015625` doc=`ECLI:CZ:US:2026:1.US.2699.25.1` date=`15. 4. 2026` snippet="konstatuje, že si soud prvního stupně při zjišťování skutkového stavu počínal ústavně konformně, důkazními návrhy stěžovatelky se řádně zabýval a jejich zamítnutí korektně a ústavně konformně odůvodnil (srov. bod 17. ..."

### `ochrana soukromí`

- Dense results from scoped collection: `True`
- Dense results from candidate collection: `True`
- All reported hits from candidate collection: `True`
- BM25 results: `5`
- Hybrid results: `5`
- Qualitative relevance: `partial_keyword_overlap`

- dense score=`0.526922` doc=`ECLI:CZ:US:2026:4.US.837.26.1` date=`29. 4. 2026` snippet="stručně řečeno - upravil styk nezletilého se stěžovatelkou (výrok I). Okresní soud naopak nařídil předběžné opatření, jímž zakázal jakkoli zveřejňovat a přikázal odstranit již zveřejněné zvukové nebo obrazové záznamy,..."
- dense score=`0.51` doc=`ECLI:CZ:US:2026:3.US.3424.25.1` date=`23. 4. 2026` snippet="nepřísluší (...)" [nález ze dne 8. 8. 2017 sp. zn. Pl. ÚS 32/16 (N 139/86 SbNU 369; 345/2018 Sb.), bod 67]. 23. Ochrana soukromé sféry jednotlivce je v Listině roztříštěna mezi několik vzájemně se doplňujících článků ..."
- dense score=`0.505442` doc=`ECLI:CZ:US:2026:1.US.1055.26.1` date=`15. 5. 2026` snippet="státní zastupitelství stěžovateli sdělilo, že k výkonu vnitřního dohledu důvod neshledává. 3. V ústavní stížnosti stěžovatel brojí proti postupu policejního orgánu i státních zastupitelství. Namítá, že komunikace s ad..."
- dense score=`0.503074` doc=`ECLI:CZ:US:2026:1.US.1055.26.1` date=`15. 5. 2026` snippet="zastupitelství v Praze s žádostí o přezkum postupu policejního orgánu, v níž mimo jiné namítal, že komunikace s advokátem je chráněná, a požádal, "aby plná audionahrávka této komunikace a přepis uvedené komunikace byl..."
- dense score=`0.502362` doc=`ECLI:CZ:US:2026:2.US.1113.26.1` date=`22. 5. 2026` snippet="NALUS - databáze rozhodnutí Ústavního soudu II.ÚS 1113/26 ze dne 22. 5. 2026 Česká republika USNESENÍ Ústavního soudu Ústavní soud rozhodl soudcem zpravodajem Jiřím Přibáněm o ústavní stížnosti stěžovatele JUDr. Pavla..."

- hybrid score=`0.032522` doc=`ECLI:CZ:US:2026:3.US.3424.25.1` date=`23. 4. 2026` snippet="nepřísluší (...)" [nález ze dne 8. 8. 2017 sp. zn. Pl. ÚS 32/16 (N 139/86 SbNU 369; 345/2018 Sb.), bod 67]. 23. Ochrana soukromé sféry jednotlivce je v Listině roztříštěna mezi několik vzájemně se doplňujících článků ..."
- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:4.US.837.26.1` date=`29. 4. 2026` snippet="stručně řečeno - upravil styk nezletilého se stěžovatelkou (výrok I). Okresní soud naopak nařídil předběžné opatření, jímž zakázal jakkoli zveřejňovat a přikázal odstranit již zveřejněné zvukové nebo obrazové záznamy,..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:3.US.739.26.1` date=`3. 6. 2026` snippet="odepřít výpověď a procesní rovnost]. In: HUSSEINI, Faisal, BARTOŇ, Michal, KOKEŠ, Marian, KOPA, Martin a kol. Listina základních práv a svobod. 1. vyd. Praha: C. H. Beck, 2021, marg. č. 29.) Skutečnost, že v souběžně ..."
- hybrid score=`0.015873` doc=`ECLI:CZ:US:2026:1.US.1055.26.1` date=`15. 5. 2026` snippet="státní zastupitelství stěžovateli sdělilo, že k výkonu vnitřního dohledu důvod neshledává. 3. V ústavní stížnosti stěžovatel brojí proti postupu policejního orgánu i státních zastupitelství. Namítá, že komunikace s ad..."
- hybrid score=`0.015873` doc=`ECLI:CZ:US:2026:3.US.1205.26.1` date=`22. 5. 2026` snippet="Stěžovatel podal ústavní stížnost a následná doplnění, ve kterých (1) navrhl zrušit všech osm popsaných rozhodnutí, (2) navrhl odklad vykonatelnosti výroků o povinnosti zaplatit soudní poplatek a (3) požádal o přiznán..."

### `vazba`

- Dense results from scoped collection: `True`
- Dense results from candidate collection: `True`
- All reported hits from candidate collection: `True`
- BM25 results: `5`
- Hybrid results: `5`
- Qualitative relevance: `plausible_top_hits_keyword_overlap`

- dense score=`0.475454` doc=`ECLI:CZ:US:2026:4.US.883.26.1` date=`29. 4. 2026` snippet="uvádí též nepřesné a zavádějící informace, ze spisu neplyne, že byl stěžovatel odsouzen v Nizozemí či v Německu. Začtvrté, soudy ignorovaly též to, že si stěžovatel zajistil v Česku nájemní bydlení, nezvažovaly dostat..."
- dense score=`0.463701` doc=`ECLI:CZ:US:2026:2.US.645.26.1` date=`15. 4. 2026` snippet="identické argumentaci jako na počátku vazby. Takový postup je v rozporu s požadavkem proporcionality zásahu do osobní svobody. Stěžovatel dodává, že vrchní soud ve svém vyjádření pouze odkázal na napadené usnesení, kt..."
- dense score=`0.451015` doc=`ECLI:CZ:US:2026:2.US.645.26.1` date=`15. 4. 2026` snippet="170/25 ). 20. Ústavní soud judikuje, že opakují-li vazební soudy jen počáteční důvody vazby a nevysvětlí-li v pozdějších vazebních rozhodnutích, proč je vazba nadále nutná, nedostojí formálním požadavkům, které na ně ..."
- dense score=`0.44865` doc=`ECLI:CZ:US:2026:3.US.995.26.1` date=`3. 6. 2026` snippet="sp. zn. IV. ÚS 161/04 ). 11. Vazba představuje významný zásah do života obviněného, neboť jej izoluje od rodinného a sociálního prostředí a nezřídka jej stigmatizuje, což má pro něj závažné sociální, psychologické a e..."
- dense score=`0.447617` doc=`ECLI:CZ:US:2026:4.US.612.26.1` date=`15. 4. 2026` snippet="se vedlo trestní stíhání jen proti třem dílčím útokům (resp. skutkům ve smyslu trestního řádu). U nich však celkově způsobená škoda nepřesáhla zákonnou hranici škody velkého rozsahu. Zatřetí, pokud původní usnesení o ..."

- hybrid score=`0.031514` doc=`ECLI:CZ:US:2026:4.US.612.26.1` date=`15. 4. 2026` snippet="se vedlo trestní stíhání jen proti třem dílčím útokům (resp. skutkům ve smyslu trestního řádu). U nich však celkově způsobená škoda nepřesáhla zákonnou hranici škody velkého rozsahu. Zatřetí, pokud původní usnesení o ..."
- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:4.US.883.26.1` date=`29. 4. 2026` snippet="uvádí též nepřesné a zavádějící informace, ze spisu neplyne, že byl stěžovatel odsouzen v Nizozemí či v Německu. Začtvrté, soudy ignorovaly též to, že si stěžovatel zajistil v Česku nájemní bydlení, nezvažovaly dostat..."
- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:2.US.645.26.1` date=`15. 4. 2026` snippet="trestný čin, o který se pokusil, nebo vykoná trestný čin, který připravoval nebo kterým hrozil. Tento vazební důvod je založen na předpokladu, že účelem trestního řízení je i předcházení trestné činnosti. Mezi konkrét..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:2.US.645.26.1` date=`15. 4. 2026` snippet="identické argumentaci jako na počátku vazby. Takový postup je v rozporu s požadavkem proporcionality zásahu do osobní svobody. Stěžovatel dodává, že vrchní soud ve svém vyjádření pouze odkázal na napadené usnesení, kt..."
- hybrid score=`0.015873` doc=`ECLI:CZ:US:2026:2.US.645.26.1` date=`15. 4. 2026` snippet="170/25 ). 20. Ústavní soud judikuje, že opakují-li vazební soudy jen počáteční důvody vazby a nevysvětlí-li v pozdějších vazebních rozhodnutích, proč je vazba nadále nutná, nedostojí formálním požadavkům, které na ně ..."

### `trestní řízení`

- Dense results from scoped collection: `True`
- Dense results from candidate collection: `True`
- All reported hits from candidate collection: `True`
- BM25 results: `5`
- Hybrid results: `5`
- Qualitative relevance: `plausible_top_hits_keyword_overlap`

- dense score=`0.61916` doc=`ECLI:CZ:US:2026:4.US.3452.25.1` date=`29. 4. 2026` snippet="trestního řízení volně intervenuje a supluje úvahy příslušných orgánů ve všech věcech, které spadají do působnosti práva trestního (nález ze dne 3. 9. 2025 sp. zn. IV. ÚS 2582/24 , bod 42). Tato teze platí o to silněj..."
- dense score=`0.609219` doc=`ECLI:CZ:US:2026:4.US.566.26.1` date=`10. 6. 2026` snippet="trestním řízení je zajistit, aby při praktické aplikaci ustanovení trestního řádu, jimiž je upraveno sjednávání a schvalování dohody o vině a trestu, byly zájmy poškozených (v míře zákonem vyžadované) skutečně chráněn..."
- dense score=`0.606736` doc=`ECLI:CZ:US:2026:3.US.3031.24.1` date=`21. 4. 2026` snippet="trestním řízení v nynější věci sledovaly důležitý cíl, tedy zabránit trestně stíhaným právnickým osobám vyhýbat se následkům trestního řízení účelovými přesuny majetku na další právnické osoby. Sledování tohoto cíle v..."
- dense score=`0.605844` doc=`ECLI:CZ:US:2026:1.US.3579.25.1` date=`29. 4. 2026` snippet="řízení, a zda tohoto účelu nelze dosáhnout jinak. Ústavnímu soudu nepřísluší zasahovat do takto vymezené pravomoci orgánů činných v trestním řízení, pokud jejich postupem nedošlo k porušení základních práv a svobod (s..."
- dense score=`0.593407` doc=`ECLI:CZ:US:2026:4.US.1196.26.1` date=`13. 5. 2026` snippet="řízení, ve kterém se rozhoduje o uložení penále, je řízením s typově zvýšeným významem předmětu řízení, neboť tato otázka nebyla dosud v judikatuře dovolacího soudu vyřešena. Dospěl přitom k závěru, že takové řízení, ..."

- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:4.US.3452.25.1` date=`29. 4. 2026` snippet="trestního řízení volně intervenuje a supluje úvahy příslušných orgánů ve všech věcech, které spadají do působnosti práva trestního (nález ze dne 3. 9. 2025 sp. zn. IV. ÚS 2582/24 , bod 42). Tato teze platí o to silněj..."
- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:3.US.3031.24.1` date=`21. 4. 2026` snippet="projednávané věci tedy platí závěr obecných soudů, že převod části závodu (Divize 4) z původně stíhané právnické osoby na stěžovatelku založil právní nástupnictví stěžovatelky v trestním řízení, a došlo k přechodu tre..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:4.US.566.26.1` date=`10. 6. 2026` snippet="trestním řízení je zajistit, aby při praktické aplikaci ustanovení trestního řádu, jimiž je upraveno sjednávání a schvalování dohody o vině a trestu, byly zájmy poškozených (v míře zákonem vyžadované) skutečně chráněn..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:3.US.3031.24.1` date=`21. 4. 2026` snippet="mezeru, v jejímž důsledku by orgány činné v trestním řízení vůbec nemohly proti nástupnické právnické osobě procesně postupovat. Jde-li o procesní ustanovení v řízení proti právnické osobě, navazuje zákon o trestní od..."
- hybrid score=`0.015873` doc=`ECLI:CZ:US:2026:3.US.3031.24.1` date=`21. 4. 2026` snippet="trestním řízení v nynější věci sledovaly důležitý cíl, tedy zabránit trestně stíhaným právnickým osobám vyhýbat se následkům trestního řízení účelovými přesuny majetku na další právnické osoby. Sledování tohoto cíle v..."

### `civilní řízení`

- Dense results from scoped collection: `True`
- Dense results from candidate collection: `True`
- All reported hits from candidate collection: `True`
- BM25 results: `5`
- Hybrid results: `5`
- Qualitative relevance: `plausible_top_hits_keyword_overlap`

- dense score=`0.563638` doc=`ECLI:CZ:US:2026:2.US.899.26.1` date=`29. 4. 2026` snippet="kterých by v řízení, z něhož příslušné rozhodnutí cizozemského soudu vzešlo, byla porušena základní práva účastníka řízení. Rozpor s veřejným pořádkem totiž musí být takového stupně, že porušuje základní principy práv..."
- dense score=`0.551072` doc=`ECLI:CZ:US:2026:4.US.897.26.1` date=`29. 4. 2026` snippet="nedošlo. 17. V prvé řadě třeba uvést, že není porušením práva na řádný proces, jestliže obecné soudy nebudují vlastní závěry na podrobné oponentuře (a vyvracení) jednotlivě vznesených námitek, pakliže proti nim staví ..."
- dense score=`0.548506` doc=`ECLI:CZ:US:2026:4.US.2338.25.1` date=`23. 4. 2026` snippet="18. Nález IV. ÚS 1247/20 není přenositelný na trestní věci. Nález řešil civilní spor. Trestní řízení a civilní řízení se řídí odlišnými procesními principy i pravidly dokazování. Přistoupit na stěžovatelův požadavek p..."
- dense score=`0.543137` doc=`ECLI:CZ:US:2026:3.US.2752.24.1` date=`13. 5. 2026` snippet="nepřípustně některé z ústavně zaručených základních práv či svobod nebo jsou v rozporu s požadavky řádného procesu či s obecně sdílenými zásadami spravedlnosti. Postup v soudním řízení, zjišťování a hodnocení skutkové..."
- dense score=`0.541617` doc=`ECLI:CZ:US:2026:2.US.899.26.1` date=`29. 4. 2026` snippet="řízení podle kritérií pro neuznání rozhodnutí stanovených nařízením Brusel I bis, se tak míjí s průběhem řízení. 23. Civilní soudy po vymezení hypotézy právní normy následně v souladu s ústavními požadavky zkoumaly, z..."

- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:2.US.899.26.1` date=`29. 4. 2026` snippet="kterých by v řízení, z něhož příslušné rozhodnutí cizozemského soudu vzešlo, byla porušena základní práva účastníka řízení. Rozpor s veřejným pořádkem totiž musí být takového stupně, že porušuje základní principy práv..."
- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:4.US.897.26.1` date=`29. 4. 2026` snippet="ústavními požadavky vysvětlily, že vadou není ani neprojednání stěžovatelčina protinávrhu, neboť britský soud vyšel z toho, že za ukončení smlouvy je odpovědná stěžovatelka. Tento nárok je nadto možné případně projedn..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:4.US.897.26.1` date=`29. 4. 2026` snippet="nedošlo. 17. V prvé řadě třeba uvést, že není porušením práva na řádný proces, jestliže obecné soudy nebudují vlastní závěry na podrobné oponentuře (a vyvracení) jednotlivě vznesených námitek, pakliže proti nim staví ..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:2.US.899.26.1` date=`29. 4. 2026` snippet="sankční prvek, neboť výsledek soudního řízení závisel na vyřešení odborných otázek soudními znalci. Nelze proto ani tvrdit, že by tento výsledek měl být v extrémním rozporu s obsahem důkazů (srov. bod 55 usnesení Nejv..."
- hybrid score=`0.015873` doc=`ECLI:CZ:US:2026:4.US.2338.25.1` date=`23. 4. 2026` snippet="18. Nález IV. ÚS 1247/20 není přenositelný na trestní věci. Nález řešil civilní spor. Trestní řízení a civilní řízení se řídí odlišnými procesními principy i pravidly dokazování. Přistoupit na stěžovatelův požadavek p..."

### `ústavní stížnost`

- Dense results from scoped collection: `True`
- Dense results from candidate collection: `True`
- All reported hits from candidate collection: `True`
- BM25 results: `5`
- Hybrid results: `5`
- Qualitative relevance: `plausible_top_hits_keyword_overlap`

- dense score=`0.660455` doc=`ECLI:CZ:US:2026:2.US.657.26.2` date=`22. 4. 2026` snippet="věci a předchozí průběh řízení 1. Stěžovatelé v ústavní stížnosti navrhují, aby Ústavní soud zrušil v záhlaví označené usnesení Vrchního soudu v Praze ("vrchní soud"), nebo jeho výrok nařizující projednání v jiném slo..."
- dense score=`0.651104` doc=`ECLI:CZ:US:2026:4.US.651.26.1` date=`27. 5. 2026` snippet="na náhradě nákladů řízení před soudy obou stupňů částku 341 304 Kč (výrok III.). Výrokem IV. a V. rozhodl o nákladech řízení státu. 3. Stěžovatel ústavní stížností napadá výrok III. rozsudku městského soudu o náhradě ..."
- dense score=`0.650649` doc=`ECLI:CZ:US:2026:3.US.1145.26.1` date=`7. 5. 2026` snippet="zprostit ustanoveného advokáta povinnosti obhajovat navrhovatele a novému obhájci ustanovil přiměřenou lhůtu k nastudování věci; v souvislosti s tím požaduje předběžným opatřením uložit soudu, aby odročil hlavní líčen..."
- dense score=`0.647053` doc=`ECLI:CZ:US:2026:1.US.306.26.1` date=`20. 4. 2026` snippet="ústavní stížnosti začíná běžet dnem doručení rozhodnutí o posledním procesním prostředku, který zákon stěžovateli k ochraně jeho práva poskytuje (§ 72 odst. 3 zákona o Ústavním soudu). Tímto posledním prostředkem ochr..."
- dense score=`0.642765` doc=`ECLI:CZ:US:2026:2.US.614.26.1` date=`20. 5. 2026` snippet="NALUS - databáze rozhodnutí Ústavního soudu II.ÚS 614/26 ze dne 20. 5. 2026 Česká republika USNESENÍ Ústavního soudu Ústavní soud rozhodl v senátu složeném z předsedy Jiřího Přibáně (soudce zpravodaje) a soudců Martin..."

- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:2.US.657.26.2` date=`22. 4. 2026` snippet="věci a předchozí průběh řízení 1. Stěžovatelé v ústavní stížnosti navrhují, aby Ústavní soud zrušil v záhlaví označené usnesení Vrchního soudu v Praze ("vrchní soud"), nebo jeho výrok nařizující projednání v jiném slo..."
- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:1.US.1218.26.1` date=`14. 5. 2026` snippet="ÚS 2900/25 , bod 3). 8. Soudce zpravodaj shrnuje, že odmítl stěžovatelovu ústavní stížnost podle § 43 odst. 1 písm. e) ve spojení s § 75 odst. 1 zákona o Ústavním soudu. Neshledal přitom důvod pro aplikaci § 75 odst. ..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:4.US.651.26.1` date=`27. 5. 2026` snippet="na náhradě nákladů řízení před soudy obou stupňů částku 341 304 Kč (výrok III.). Výrokem IV. a V. rozhodl o nákladech řízení státu. 3. Stěžovatel ústavní stížností napadá výrok III. rozsudku městského soudu o náhradě ..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:2.US.1074.26.1` date=`6. 5. 2026` snippet="dvouměsíční lhůta k podání ústavní stížnosti již uplynula a ústavní stížnost proti usnesení krajského soudu je proto opožděná. 17. Ústavní soud uzavírá, že přezkoumal ústavní stížnost z hlediska kompetencí daných mu Ú..."
- hybrid score=`0.015873` doc=`ECLI:CZ:US:2026:3.US.1145.26.1` date=`7. 5. 2026` snippet="zprostit ustanoveného advokáta povinnosti obhajovat navrhovatele a novému obhájci ustanovil přiměřenou lhůtu k nastudování věci; v souvislosti s tím požaduje předběžným opatřením uložit soudu, aby odročil hlavní líčen..."

### `proporcionalita`

- Dense results from scoped collection: `True`
- Dense results from candidate collection: `True`
- All reported hits from candidate collection: `True`
- BM25 results: `5`
- Hybrid results: `5`
- Qualitative relevance: `plausible_top_hits_keyword_overlap`

- dense score=`0.492816` doc=`ECLI:CZ:US:2026:2.US.1002.26.2` date=`20. 5. 2026` snippet="výnosu z trestné činnosti ve výši 38 milionů Kč po provedených korekcích. Stížnostní soud tak zjevně provedl přezkum proporcionality zajištění a jeho rozsah přizpůsobil dosavadním skutkovým zjištěním. Nelze proto uzav..."
- dense score=`0.469471` doc=`ECLI:CZ:US:2026:1.US.2699.25.1` date=`15. 4. 2026` snippet="stěžovatele za spáchání trestného činu šíření poplašné zprávy. Stěžovatelka měla za to, že podobný náhled na proporcionalitu trestněprávní sankce za politický projev měly obecné soudy přijmout i v její věci. 21. Tento..."
- dense score=`0.468248` doc=`ECLI:CZ:US:2026:4.US.1949.25.1` date=`19. 5. 2026` snippet="pozemků v RÚIAN a v katastru nemovitostí. Změna územního plánu toliko odrážela skutečnost, že pozemky stěžovatelky již nebyly zastavěnými stavebními pozemky a nesplňovaly ani žádné jiné znaky pro zařazení do zastavěné..."
- dense score=`0.465097` doc=`ECLI:CZ:US:2026:1.US.3707.25.1` date=`28. 4. 2026` snippet="ze dne 25. 9. 2007 sp. zn. Pl. ÚS 85/06 nebo nález ze dne 14. 5. 2018 sp. zn. I. ÚS 2502/17 ). Z napadených rozhodnutí přítomnost kvalifikované vady nevyplývá. Obecné soudy srozumitelně vysvětlily, z jakých důvodů roz..."
- dense score=`0.464061` doc=`ECLI:CZ:US:2026:3.US.895.26.1` date=`22. 5. 2026` snippet="sankci, prostředek psychického zlomení procesního odporu stěžovatele a jeho exemplární potrestání. Stěžovatel přitom měl k odmítnutí předání nezletilých řadu důvodů, které průběžně dokládal a obracel se na veřejné org..."

- hybrid score=`0.032018` doc=`ECLI:CZ:US:2026:2.US.1002.26.2` date=`20. 5. 2026` snippet="výnosu z trestné činnosti ve výši 38 milionů Kč po provedených korekcích. Stížnostní soud tak zjevně provedl přezkum proporcionality zajištění a jeho rozsah přizpůsobil dosavadním skutkovým zjištěním. Nelze proto uzav..."
- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:2.US.1002.26.2` date=`20. 5. 2026` snippet="rozhodnutí jako celek obstojí z hlediska ústavních požadavků kladených na použití majetkových zajišťovacích institutů. Důvody zajištění byly konkretizovány, jejich proporcionalita byla stížnostním soudem přezkoumána a..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:1.US.2699.25.1` date=`15. 4. 2026` snippet="stěžovatele za spáchání trestného činu šíření poplašné zprávy. Stěžovatelka měla za to, že podobný náhled na proporcionalitu trestněprávní sankce za politický projev měly obecné soudy přijmout i v její věci. 21. Tento..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:2.US.3190.25.1` date=`10. 6. 2026` snippet="se opírají i soudy obou stupňů, je právě existence duální diagnózy (drogová závislost a schizofrenie) klíčovým problémem, který odůvodňuje uložení zabezpečovací detence. Okresní soud s odvoláním na znalce tak na str. ..."
- hybrid score=`0.015873` doc=`ECLI:CZ:US:2026:4.US.1949.25.1` date=`19. 5. 2026` snippet="pozemků v RÚIAN a v katastru nemovitostí. Změna územního plánu toliko odrážela skutečnost, že pozemky stěžovatelky již nebyly zastavěnými stavebními pozemky a nesplňovaly ani žádné jiné znaky pro zařazení do zastavěné..."

### `retroaktivita`

- Dense results from scoped collection: `True`
- Dense results from candidate collection: `True`
- All reported hits from candidate collection: `True`
- BM25 results: `5`
- Hybrid results: `5`
- Qualitative relevance: `semantic_or_weak_keyword_overlap_review_needed`

- dense score=`0.55632` doc=`ECLI:CZ:US:2026:2.US.1089.26.1` date=`20. 5. 2026` snippet="kvalifikovaných vad. 8. V nynější věci Ústavní soud žádné kvalifikované vady neshledal. Výklad nyní vyslovený správními soudy, tedy že se i ve věci stěžovatele vychází z právního stavu v době vydání konstitutivního ro..."
- dense score=`0.552255` doc=`ECLI:CZ:US:2026:Pl.US.46.25.1` date=`29. 4. 2026` snippet="nedochází k zásahu do práva na přístup k voleným a jiným veřejným funkcím za rovných podmínek. VII. c) Právní jistota a princip zákazu zpětné účinnosti právních norem 75. Podle navrhovatelky bylo napadenými ustanovení..."
- dense score=`0.551084` doc=`ECLI:CZ:US:2026:2.US.1089.26.1` date=`20. 5. 2026` snippet="nepřípustně odebráno nabyté právo, protože to dosud před povolením záměru nevzniklo. Nadto již existovala uvedená ustálená judikatura, že po změně územního plánu je třeba vycházet z této nové, změněné podoby. Mohlo jí..."
- dense score=`0.540997` doc=`ECLI:CZ:US:2026:Pl.US.46.25.1` date=`29. 4. 2026` snippet="právní následky pro minulost, v minulosti nastalé skutečnosti však právně kvalifikuje jako podmínku budoucího právního následku nebo pro budoucnost modifikuje právní následky založené podle dřívějších předpisů (srov. ..."
- dense score=`0.534489` doc=`ECLI:CZ:US:2026:Pl.US.46.25.1` date=`29. 4. 2026` snippet="změna pravidel pro odměňování tak má ve vztahu k uvedeným členům zastupitelstev charakter nepravé retroaktivity. 79. Nepravá retroaktivita je obecně přípustná a výjimečné okolnosti, proč by tomu tak nemělo v posuzovan..."

- hybrid score=`0.032787` doc=`ECLI:CZ:US:2026:2.US.1089.26.1` date=`20. 5. 2026` snippet="kvalifikovaných vad. 8. V nynější věci Ústavní soud žádné kvalifikované vady neshledal. Výklad nyní vyslovený správními soudy, tedy že se i ve věci stěžovatele vychází z právního stavu v době vydání konstitutivního ro..."
- hybrid score=`0.031754` doc=`ECLI:CZ:US:2026:Pl.US.46.25.1` date=`29. 4. 2026` snippet="právní následky pro minulost, v minulosti nastalé skutečnosti však právně kvalifikuje jako podmínku budoucího právního následku nebo pro budoucnost modifikuje právní následky založené podle dřívějších předpisů (srov. ..."
- hybrid score=`0.031746` doc=`ECLI:CZ:US:2026:2.US.1089.26.1` date=`20. 5. 2026` snippet="nepřípustně odebráno nabyté právo, protože to dosud před povolením záměru nevzniklo. Nadto již existovala uvedená ustálená judikatura, že po změně územního plánu je třeba vycházet z této nové, změněné podoby. Mohlo jí..."
- hybrid score=`0.03101` doc=`ECLI:CZ:US:2026:Pl.US.46.25.1` date=`29. 4. 2026` snippet="změna pravidel pro odměňování tak má ve vztahu k uvedeným členům zastupitelstev charakter nepravé retroaktivity. 79. Nepravá retroaktivita je obecně přípustná a výjimečné okolnosti, proč by tomu tak nemělo v posuzovan..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:Pl.US.46.25.1` date=`29. 4. 2026` snippet="nedochází k zásahu do práva na přístup k voleným a jiným veřejným funkcím za rovných podmínek. VII. c) Právní jistota a princip zákazu zpětné účinnosti právních norem 75. Podle navrhovatelky bylo napadenými ustanovení..."

### `legitimní očekávání`

- Dense results from scoped collection: `True`
- Dense results from candidate collection: `True`
- All reported hits from candidate collection: `True`
- BM25 results: `5`
- Hybrid results: `5`
- Qualitative relevance: `plausible_top_hits_keyword_overlap`

- dense score=`0.638147` doc=`ECLI:CZ:US:2026:Pl.US.39.25.1` date=`24. 6. 2026` snippet="očekávání je legitimní pouze tehdy, bylo-li založeno zákonným ustanovením nebo právním aktem vztahujícím se k dotčenému majetkovému zájmu (nález sp. zn. Pl. ÚS 21/21 , bod 96). 77. K těmto závěrům se Ústavní soud přih..."
- dense score=`0.590004` doc=`ECLI:CZ:US:2026:2.US.263.26.3` date=`13. 5. 2026` snippet="legitimního očekávání podle čl. 11 odst. 1 Listiny [srov. např. nálezy ze dne 8. 3. 2023 sp. zn. I. ÚS 3281/22 (N 37/117 SbNU 45), ze dne 13. 12. 2023 sp. zn. III. ÚS 2040/22 (N 185/121 SbNU 290), ze dne 10. 4. 2024 s..."
- dense score=`0.586143` doc=`ECLI:CZ:US:2026:1.US.1100.26.1` date=`21. 5. 2026` snippet="legitimního očekávání vzniklého jednáním vedlejší účastnice, toto je vyvráceno jednoznačným a podloženým závěrem obecných soudů o tom, že si stěžovatel musel být od počátku vědom, že není vlastníkem žalovaných pozemků..."
- dense score=`0.551959` doc=`ECLI:CZ:US:2026:4.US.1106.26.1` date=`27. 5. 2026` snippet="pochybení. 12. Ústavní soud nemůže přisvědčit námitce stěžovatele, že soudy nejednaly v souladu s principem legitimního očekávání a "tuzemským právním řádem", neboť nesprávně aplikovaly vybraná ustanovení zákonů v jeh..."
- dense score=`0.550806` doc=`ECLI:CZ:US:2026:1.US.2813.25.1` date=`16. 4. 2026` snippet="takto neurčité formulace nemohl být stěžovatelům zřejmý dopad nového územního rozhodnutí (taktéž neurčitého) - tedy jaké konkrétní podmínky byly konzumovány. 7. Nejvyšší správní soud podle stěžovatelů nesprávně interp..."

- hybrid score=`0.031514` doc=`ECLI:CZ:US:2026:1.US.2813.25.1` date=`16. 4. 2026` snippet="takto neurčité formulace nemohl být stěžovatelům zřejmý dopad nového územního rozhodnutí (taktéž neurčitého) - tedy jaké konkrétní podmínky byly konzumovány. 7. Nejvyšší správní soud podle stěžovatelů nesprávně interp..."
- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:Pl.US.39.25.1` date=`24. 6. 2026` snippet="očekávání je legitimní pouze tehdy, bylo-li založeno zákonným ustanovením nebo právním aktem vztahujícím se k dotčenému majetkovému zájmu (nález sp. zn. Pl. ÚS 21/21 , bod 96). 77. K těmto závěrům se Ústavní soud přih..."
- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:Pl.US.46.25.1` date=`29. 4. 2026` snippet="právní následky pro minulost, v minulosti nastalé skutečnosti však právně kvalifikuje jako podmínku budoucího právního následku nebo pro budoucnost modifikuje právní následky založené podle dřívějších předpisů (srov. ..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:2.US.263.26.3` date=`13. 5. 2026` snippet="legitimního očekávání podle čl. 11 odst. 1 Listiny [srov. např. nálezy ze dne 8. 3. 2023 sp. zn. I. ÚS 3281/22 (N 37/117 SbNU 45), ze dne 13. 12. 2023 sp. zn. III. ÚS 2040/22 (N 185/121 SbNU 290), ze dne 10. 4. 2024 s..."
- hybrid score=`0.015873` doc=`ECLI:CZ:US:2026:1.US.1100.26.1` date=`21. 5. 2026` snippet="legitimního očekávání vzniklého jednáním vedlejší účastnice, toto je vyvráceno jednoznačným a podloženým závěrem obecných soudů o tom, že si stěžovatel musel být od počátku vědom, že není vlastníkem žalovaných pozemků..."

### `právo na účinný prostředek nápravy`

- Dense results from scoped collection: `True`
- Dense results from candidate collection: `True`
- All reported hits from candidate collection: `True`
- BM25 results: `5`
- Hybrid results: `5`
- Qualitative relevance: `plausible_top_hits_keyword_overlap`

- dense score=`0.59516` doc=`ECLI:CZ:US:2026:1.US.1055.26.1` date=`15. 5. 2026` snippet="efektivní prostředek ochrany, od jehož vyřízení by bylo možné ve smyslu § 72 odst. 3 zákona o Ústavním soudu odvíjet lhůtu k podání ústavní stížnosti. Zejména tak tomu nebude u tzv. vnitřního dohledu podle § 12e zákon..."
- dense score=`0.588468` doc=`ECLI:CZ:US:2026:2.US.1038.26.1` date=`17. 6. 2026` snippet="hodné, pro které by stěžovatelka nemusela všechny dostupné procesní prostředky vyčerpat, tyto však neshledal. 11. K odvolání stěžovatelky Městský soud v Praze napadeným rozsudkem rozsudek obvodního soudu potvrdil. Sou..."
- dense score=`0.584726` doc=`ECLI:CZ:US:2026:2.US.3134.25.1` date=`15. 4. 2026` snippet="smyslu, nýbrž obsahuje i požadavek předestření relevantních námitek obecným soudům, jimž ochrana všech základních práv a svobod přísluší (čl. 4 Ústavy). 27. Ústavní soud se konečně nemůže ztotožnit ani s námitkou stěž..."
- dense score=`0.579533` doc=`ECLI:CZ:US:2026:2.US.1038.26.1` date=`17. 6. 2026` snippet="čl. 36 odst. 3 Listiny může vzniknout odpovědnost státu za škodu způsobenou nezákonným opatřením obecné povahy, přestože toto opatření nebylo formálně zrušeno nebo změněno, a to za předpokladu, že stěžovatelé nezákonn..."
- dense score=`0.573343` doc=`ECLI:CZ:US:2026:3.US.825.26.1` date=`15. 4. 2026` snippet="opravný prostředek. Český právní řád dokonce konstruuje opravný prostředek pro případ, kdy bylo pravomocně rozhodnuto v neprospěch účastníka v důsledku trestného činu soudce [§ 229 odst. 1 písm. g) občanského soudního..."

- hybrid score=`0.032266` doc=`ECLI:CZ:US:2026:1.US.1055.26.1` date=`15. 5. 2026` snippet="efektivní prostředek ochrany, od jehož vyřízení by bylo možné ve smyslu § 72 odst. 3 zákona o Ústavním soudu odvíjet lhůtu k podání ústavní stížnosti. Zejména tak tomu nebude u tzv. vnitřního dohledu podle § 12e zákon..."
- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:3.US.307.26.1` date=`29. 4. 2026` snippet="posouzení věci bezpředmětné. II. Argumentace stěžovatele 7. Stěžovatel podal ústavní stížnost proti rozhodnutí městského a Nejvyššího soudu. Tato rozhodnutí podle něj porušují jeho právo na vlastnictví majetku, na spr..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:2.US.1038.26.1` date=`17. 6. 2026` snippet="hodné, pro které by stěžovatelka nemusela všechny dostupné procesní prostředky vyčerpat, tyto však neshledal. 11. K odvolání stěžovatelky Městský soud v Praze napadeným rozsudkem rozsudek obvodního soudu potvrdil. Sou..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:2.US.129.26.1` date=`15. 4. 2026` snippet="že trestní stíhání bylo vedeno svévolně, nelze se domáhat náhrady škody. Stěžovatel je však toho názoru, že trestní stíhání v jeho věci svévolně vedeno bylo, a to z důvodu, že k promlčení trestní odpovědnosti stěžovat..."
- hybrid score=`0.015873` doc=`ECLI:CZ:US:2026:2.US.3134.25.1` date=`15. 4. 2026` snippet="smyslu, nýbrž obsahuje i požadavek předestření relevantních námitek obecným soudům, jimž ochrana všech základních práv a svobod přísluší (čl. 4 Ústavy). 27. Ústavní soud se konečně nemůže ztotožnit ani s námitkou stěž..."

## Failures / Warnings

- None.
