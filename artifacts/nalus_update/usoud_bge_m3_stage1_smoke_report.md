# Ustavni soud / NALUS - BGE-M3 Stage 1 Smoke Report

Generated: 2026-07-08T14:09:10Z

- Script path: `scripts/build_usoud_bge_m3_candidate.py`
- Builder version: `usoud-bge-m3-smoke-v1`
- Action: `execute`
- Dry-run command: `python scripts/build_usoud_bge_m3_candidate.py --mode smoke --limit 20 --collection-name nalus_us_bge_m3_smoke_20260708 --source-batch batches/year_2026_20260708_124949.json --output-dir artifacts/nalus_update/usoud_bge_m3_smoke_20260708 --dry-run`
- Execute command: `python scripts/build_usoud_bge_m3_candidate.py --mode smoke --limit 20 --collection-name nalus_us_bge_m3_smoke_20260708 --source-batch batches/year_2026_20260708_124949.json --output-dir artifacts/nalus_update/usoud_bge_m3_smoke_20260708 --execute --recreate-smoke-collection --no-alias-update`
- Input: `batches/year_2026_20260708_124949.json`
- Selected records: `20`
- Generated chunks: `445`
- Embedding model: `BAAI/bge-m3`
- Vector dimension validation: `PASS (1024)`
- Qdrant collection: `nalus_us_bge_m3_smoke_20260708`
- Collection point count before: `None`
- Collection point count after: `445`
- `nalus_live` before/after: `784812` / `784812`
- `nalus_stable_20260326` before/after: `784812` / `784812`
- BM25 status: `available`
- Hybrid/RRF status: `available_rrf`
- Production API touched: `False`
- Aliases touched: `False`
- Aliases changed by verification: `False`
- Retrieval logic changed: `False`
- Clarification gate changed: `False`
- Stage 2 recommendation: `safe_after_review`

## Smoke Query Results

### `právo na spravedlivý proces`

- Dense results from smoke collection: `True`
- BM25 results: `5`
- Hybrid results: `5`

- dense score=`0.555627` doc=`ECLI:CZ:US:2026:3.US.1988.25.1` date=`11. 6. 2026` snippet="nebo veřejného zájmu. Ostatně toto právo nachází ve všech základních procesních kodexech své zákonné provedení (kromě zmíněného § 65 trestního řádu srov. např. § 44 občanského soudního řádu, § 95 notářského řádu, § 94..."
- dense score=`0.540252` doc=`ECLI:CZ:US:2026:3.US.1537.26.1` date=`17. 6. 2026` snippet="trestní řízení, jichž se týkaly informace poskytované obžalovaným K. Domnívá se, že se obecné soudy nedostatečně vypořádaly s otázkou, zda jednání, které mu bylo přičítáno, svou intenzitou a společenskou škodlivostí s..."
- dense score=`0.539761` doc=`ECLI:CZ:US:2026:4.US.566.26.1` date=`10. 6. 2026` snippet="soustavy obecných soudů a není ani povolán k instančnímu přezkumu jejich rozhodnutí. Pravomoc Ústavního soudu je založena výlučně k přezkumu rozhodnutí z hlediska dodržení ústavněprávních principů, tj. zda v řízení (r..."
- dense score=`0.53963` doc=`ECLI:CZ:US:2026:1.US.2196.25.1` date=`19. 6. 2026` snippet="Domažlicích ze dne 23. května 2024 č. j. 6 C 59/2018-685 a rozsudkem Nejvyššího soudu ze dne 13. března 2024 č. j. 28 Cdo 3880/2023-656, bylo porušeno základní právo stěžovatelky na soudní ochranu podle čl. 36 odst. 1..."
- dense score=`0.539316` doc=`ECLI:CZ:US:2026:Pl.US.18.25.1` date=`24. 6. 2026` snippet="nespočívá pouze v právu jednotlivce zahájit řízení před soudem. Jde především o právo na to, aby soud o věci meritorně rozhodl, pokud jednotlivec dodržel zákonem stanovený (a ústavně souladný) postup a podmínky pro po..."

- hybrid score=`0.032522` doc=`ECLI:CZ:US:2026:3.US.1988.25.1` date=`11. 6. 2026` snippet="nebo veřejného zájmu. Ostatně toto právo nachází ve všech základních procesních kodexech své zákonné provedení (kromě zmíněného § 65 trestního řádu srov. např. § 44 občanského soudního řádu, § 95 notářského řádu, § 94..."
- hybrid score=`0.03125` doc=`ECLI:CZ:US:2026:1.US.2196.25.1` date=`19. 6. 2026` snippet="Domažlicích ze dne 23. května 2024 č. j. 6 C 59/2018-685 a rozsudkem Nejvyššího soudu ze dne 13. března 2024 č. j. 28 Cdo 3880/2023-656, bylo porušeno základní právo stěžovatelky na soudní ochranu podle čl. 36 odst. 1..."
- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:3.US.1320.25.1` date=`11. 6. 2026` snippet="usnesení ze dne 30. 3. 2021 sp. zn. IV. ÚS 1142/20 , odkazující na závěry stanoviska pléna ze dne 28. 11. 2017 sp. zn. Pl. ÚS-st. 45/16 (ST 45/87 SbNU 905; 460/2017 Sb.). Z něj mj. plyne, že vadu řízení lze v dovolání..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:3.US.1537.26.1` date=`17. 6. 2026` snippet="trestní řízení, jichž se týkaly informace poskytované obžalovaným K. Domnívá se, že se obecné soudy nedostatečně vypořádaly s otázkou, zda jednání, které mu bylo přičítáno, svou intenzitou a společenskou škodlivostí s..."
- hybrid score=`0.015873` doc=`ECLI:CZ:US:2026:4.US.566.26.1` date=`10. 6. 2026` snippet="soustavy obecných soudů a není ani povolán k instančnímu přezkumu jejich rozhodnutí. Pravomoc Ústavního soudu je založena výlučně k přezkumu rozhodnutí z hlediska dodržení ústavněprávních principů, tj. zda v řízení (r..."

### `opomenuté důkazy`

- Dense results from smoke collection: `True`
- BM25 results: `5`
- Hybrid results: `5`

- dense score=`0.498451` doc=`ECLI:CZ:US:2026:3.US.1007.25.1` date=`11. 6. 2026` snippet="předložené stěžovatelkou zobrazují předmětný pozemek jako - zjednodušeně řečeno - osázenou součást sousedního pole. Prima facie tak šlo o důkazy, které byly s to prokázat skutečnosti významné pro rozhodnutí ve věci, n..."
- dense score=`0.488453` doc=`ECLI:CZ:US:2026:3.US.1007.25.1` date=`11. 6. 2026` snippet="absence nepoctivého úmyslu, nýbrž je především nutno, aby vydržitel věc fakticky ovládal (corpus possessionis) s vůlí mít jej pro sebe (animus possidendi). 11. Tvrzení, že hl. m. Praha pozemek nikdy nedrželo a že tent..."
- dense score=`0.486019` doc=`ECLI:CZ:US:2026:3.US.1007.25.1` date=`11. 6. 2026` snippet="zákonem (Petrov, J., Výtisk, M., Beran, V. a kol. Občanský zákoník, s. 1156, marg. 7), zatímco protistrana - tady stěžovatelka - musí tvrdit a prokazovat buď zánik držby, nebo nepoctivý úmysl domnělého vydržitele (roz..."
- dense score=`0.484998` doc=`ECLI:CZ:US:2026:3.US.1007.25.1` date=`11. 6. 2026` snippet="o privatizaci majetku státu ze dne 7. 11. 1997, úplné znění smlouvy o bezúplatném převodu ze dne 31. 12. 1997 a listinu nazvanou ohlášení nabytí vlastnického práva pro záznam do katastru nemovitostí ze dne 20. 7. 2004..."
- dense score=`0.482164` doc=`ECLI:CZ:US:2026:3.US.3503.25.1` date=`18. 6. 2026` snippet="situace byl výslech syna stěžovatele způsobilý objasnit skutkové okolnosti významné pro posouzení dobré víry podle § 984 odst. 1 občanského zákoníku, zejména jakými konkrétními informacemi syn stěžovatele disponoval, ..."

- hybrid score=`0.032002` doc=`ECLI:CZ:US:2026:3.US.1007.25.1` date=`11. 6. 2026` snippet="absence nepoctivého úmyslu, nýbrž je především nutno, aby vydržitel věc fakticky ovládal (corpus possessionis) s vůlí mít jej pro sebe (animus possidendi). 11. Tvrzení, že hl. m. Praha pozemek nikdy nedrželo a že tent..."
- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:3.US.1007.25.1` date=`11. 6. 2026` snippet="předložené stěžovatelkou zobrazují předmětný pozemek jako - zjednodušeně řečeno - osázenou součást sousedního pole. Prima facie tak šlo o důkazy, které byly s to prokázat skutečnosti významné pro rozhodnutí ve věci, n..."
- hybrid score=`0.016393` doc=`ECLI:CZ:US:2026:3.US.1007.25.1` date=`11. 6. 2026` snippet="NALUS - databáze rozhodnutí Ústavního soudu III.ÚS 1007/25 ze dne 11. 6. 2026 Opomenuté důkazy ve sporu o (mimořádném) vydržení vlastnického práva Česká republika NÁLEZ Ústavního soudu Jménem republiky Ústavní soud ro..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:3.US.1007.25.1` date=`11. 6. 2026` snippet="SbNU 51); nález sp. zn. III. ÚS 95/97 ze dne 12. 6. 1997 (N 76/8 SbNU 231)]. Požadavek na relevanci, způsobilost prokázat tvrzenou skutečnost a nezbytnost provedení navrženého důkazu současně brání tomu, aby byla posk..."
- hybrid score=`0.015873` doc=`ECLI:CZ:US:2026:3.US.1007.25.1` date=`11. 6. 2026` snippet="zákonem (Petrov, J., Výtisk, M., Beran, V. a kol. Občanský zákoník, s. 1156, marg. 7), zatímco protistrana - tady stěžovatelka - musí tvrdit a prokazovat buď zánik držby, nebo nepoctivý úmysl domnělého vydržitele (roz..."

### `odůvodnění rozhodnutí`

- Dense results from smoke collection: `True`
- BM25 results: `5`
- Hybrid results: `5`

- dense score=`0.58155` doc=`ECLI:CZ:US:2026:4.US.566.26.1` date=`10. 6. 2026` snippet="7. 2025 sp. zn. IV.ÚS 1323/25 , bod 16, sp. zn. I. ÚS 1456/25 , cit. výše, bod 20 a násl.). 23. I při rozhodování o adhezním nároku je součástí práva na soudní ochranu právo na odůvodnění soudního rozhodnutí, neboť je..."
- dense score=`0.568842` doc=`ECLI:CZ:US:2026:1.US.2196.25.1` date=`19. 6. 2026` snippet="Tato podmínka byla vícekrát připomenuta i v souvislosti s dovoláním. Pokud Nejvyšší soud své rozhodnutí odůvodní nedostatečným způsobem, je takové rozhodnutí ve své podstatě nepřezkoumatelné a zasahuje do základního p..."
- dense score=`0.561509` doc=`ECLI:CZ:US:2026:3.US.1187.26.1` date=`11. 6. 2026` snippet="oprávněna doplnit své blanketní odvolání. Na tuto otázku ovšem již detailně odpověděl Nejvyšší správní soud v napadeném rozsudku. S odkazem na svou judikaturu konstatoval, že časový prostor pro doplnění odvolání je od..."
- dense score=`0.560819` doc=`ECLI:CZ:US:2026:3.US.1320.25.1` date=`11. 6. 2026` snippet="1472/23 ]. 20. Podle judikatury Ústavního soudu se o překvapivé rozhodnutí jedná mimo jiné tehdy, nedostanou-li účastníci řízení příležitost vyjádřit se k odlišnému hodnocení důkazů (nález ze dne 9. 1. 2014 sp. zn. II..."
- dense score=`0.559089` doc=`ECLI:CZ:US:2026:3.US.1320.25.1` date=`11. 6. 2026` snippet="rozhodování vedlejší účastnice uvádí, že překvapivost rozhodnutí podle rozhodovací praxe Ústavního soudu nastane, pokud odvolací soud dospěje k závěru o věcné správnosti rozsudku na základě zcela odlišného právního po..."

- hybrid score=`0.032787` doc=`ECLI:CZ:US:2026:4.US.566.26.1` date=`10. 6. 2026` snippet="7. 2025 sp. zn. IV.ÚS 1323/25 , bod 16, sp. zn. I. ÚS 1456/25 , cit. výše, bod 20 a násl.). 23. I při rozhodování o adhezním nároku je součástí práva na soudní ochranu právo na odůvodnění soudního rozhodnutí, neboť je..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:1.US.2196.25.1` date=`19. 6. 2026` snippet="Tato podmínka byla vícekrát připomenuta i v souvislosti s dovoláním. Pokud Nejvyšší soud své rozhodnutí odůvodní nedostatečným způsobem, je takové rozhodnutí ve své podstatě nepřezkoumatelné a zasahuje do základního p..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:4.US.566.26.1` date=`10. 6. 2026` snippet="způsobem náhrady škody nebo nemajetkové újmy nebo vydáním bezdůvodného obohacení a tato dohoda byla soudem schválena v podobě, s níž souhlasil (§ 245 odst. 2 trestního řádu). Odvolací soud sice při zrušení výroku o ná..."
- hybrid score=`0.015873` doc=`ECLI:CZ:US:2026:3.US.1187.26.1` date=`11. 6. 2026` snippet="oprávněna doplnit své blanketní odvolání. Na tuto otázku ovšem již detailně odpověděl Nejvyšší správní soud v napadeném rozsudku. S odkazem na svou judikaturu konstatoval, že časový prostor pro doplnění odvolání je od..."
- hybrid score=`0.015873` doc=`ECLI:CZ:US:2026:3.US.1007.25.1` date=`11. 6. 2026` snippet="SbNU 51); nález sp. zn. III. ÚS 95/97 ze dne 12. 6. 1997 (N 76/8 SbNU 231)]. Požadavek na relevanci, způsobilost prokázat tvrzenou skutečnost a nezbytnost provedení navrženého důkazu současně brání tomu, aby byla posk..."

### `porušení základních práv`

- Dense results from smoke collection: `True`
- BM25 results: `5`
- Hybrid results: `5`

- dense score=`0.645268` doc=`ECLI:CZ:US:2026:1.US.2196.25.1` date=`19. 6. 2026` snippet="Domažlicích ze dne 23. května 2024 č. j. 6 C 59/2018-685 a rozsudkem Nejvyššího soudu ze dne 13. března 2024 č. j. 28 Cdo 3880/2023-656, bylo porušeno základní právo stěžovatelky na soudní ochranu podle čl. 36 odst. 1..."
- dense score=`0.610715` doc=`ECLI:CZ:US:2026:3.US.3315.25.2` date=`25. 6. 2026` snippet="zaručená práva dotčené osoby. II. Skutkové okolnosti posuzované věci 3. Ústavní stížností podle čl. 87 odst. 1 písm. d) Ústavy České republiky (dále jen "Ústava") a § 72 a násl. zákona č. 182/1993 Sb., o Ústavním soud..."
- dense score=`0.6018` doc=`ECLI:CZ:US:2026:3.US.1007.25.1` date=`11. 6. 2026` snippet="215/2022-83 byla porušena ústavně zaručená práva stěžovatelky podle čl. 36 odst. 1 Listiny základních práv a svobod. II. Tato rozhodnutí se ruší. Odůvodnění I. Skutkové okolnosti věci a obsah napadených rozhodnutí 1. ..."
- dense score=`0.599971` doc=`ECLI:CZ:US:2026:2.US.1038.26.1` date=`17. 6. 2026` snippet="porušeno právo stěžovatelky na soudní ochranu a na náhradu škody způsobené nezákonným rozhodnutím podle čl. 36 odst. 1 a 3 Listiny základních práv a svobod. II. Usnesení Nejvyššího soudu č. j. 30 Cdo 2498/2025-274 ze ..."
- dense score=`0.595656` doc=`ECLI:CZ:US:2026:3.US.1988.25.1` date=`11. 6. 2026` snippet="zastupitelství dále dodalo, že státní zástupkyně okresního státního zastupitelství měla odkázat stěžovatelku na Policii ČR, kterou může stěžovatelka požádat o informace podle zákona o svobodném přístupu k informacím. ..."

- hybrid score=`0.032787` doc=`ECLI:CZ:US:2026:1.US.2196.25.1` date=`19. 6. 2026` snippet="Domažlicích ze dne 23. května 2024 č. j. 6 C 59/2018-685 a rozsudkem Nejvyššího soudu ze dne 13. března 2024 č. j. 28 Cdo 3880/2023-656, bylo porušeno základní právo stěžovatelky na soudní ochranu podle čl. 36 odst. 1..."
- hybrid score=`0.031746` doc=`ECLI:CZ:US:2026:3.US.1007.25.1` date=`11. 6. 2026` snippet="215/2022-83 byla porušena ústavně zaručená práva stěžovatelky podle čl. 36 odst. 1 Listiny základních práv a svobod. II. Tato rozhodnutí se ruší. Odůvodnění I. Skutkové okolnosti věci a obsah napadených rozhodnutí 1. ..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:3.US.3315.25.2` date=`25. 6. 2026` snippet="zaručená práva dotčené osoby. II. Skutkové okolnosti posuzované věci 3. Ústavní stížností podle čl. 87 odst. 1 písm. d) Ústavy České republiky (dále jen "Ústava") a § 72 a násl. zákona č. 182/1993 Sb., o Ústavním soud..."
- hybrid score=`0.016129` doc=`ECLI:CZ:US:2026:2.US.1027.26.1` date=`24. 6. 2026` snippet="(srov. čl. 81 a 90 Ústavy České republiky). Pokud soudy postupují v souladu s obsahem hlavy páté Listiny základních práv a svobod, nemůže na sebe atrahovat právo přezkumného dohledu nad jejich činností (čl. 83 Ústavy ..."
- hybrid score=`0.015625` doc=`ECLI:CZ:US:2026:2.US.1038.26.1` date=`17. 6. 2026` snippet="porušeno právo stěžovatelky na soudní ochranu a na náhradu škody způsobené nezákonným rozhodnutím podle čl. 36 odst. 1 a 3 Listiny základních práv a svobod. II. Usnesení Nejvyššího soudu č. j. 30 Cdo 2498/2025-274 ze ..."

## Failures / Warnings

- None.
