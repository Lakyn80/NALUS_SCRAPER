# NSoud Weak Relevance Diagnostics

- Status: **PASS**
- Target collection: `nsoud_chunks_section_aware_test_2025_01_03`
- Point count: **1862**
- Vector size: **768**
- Weak query count: **4**
- Queries with exact matches: **4**
- Queries where semantic missed exact matches: **2**
- Queries likely needing hybrid retrieval: **0**
- Old collection count before/after: **1785 -> 1785**
- Markdown path: `/app/app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/weak_relevance_diagnostics.md`
- JSON path: `/app/app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/weak_relevance_diagnostics.json`

## náhrada nákladů dovolacího řízení

- Likely failure reason: `query_too_generic`
- Recommendation: `rewrite query`
- Exact match exists: **yes**
- Exact match candidate count: **1228**
- Exact phrase-match count: **0**
- Semantic top 10 exact overlap count: **1**
- Semantic missed exact matches: **no**
- Notes: The query terms are broad and match many chunks exactly, which weakens diagnostic precision.

### Semantic Top 10

| rank | score | case_number | document_type | legal_area | section_type | structure_status | chunk_id | document_id | metadata_present | matched_important_terms | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.770444 | 28 Cdo 3321/2024 | USNESENÍ | civil | reasoning | strong | ECLI:CZ:NS:2025:28.CDO.3321.2024.1__chunk_0008 | ECLI:CZ:NS:2025:28.CDO.3321.2024.1 | yes | - | 12. 2024 (dále „advokátní tarif“), a náhrady paušálně určených hotových výdajů za jeden úkon právní služby ve výši 300 Kč – § 11 odst. 1 písm. k) a § 13 odst. 4 advokátního tarifu. Protože zástupce dalšího účastníka říze... |
| 2 | 0.769617 | 29 Cdo 275/2024 | USNESENÍ | civil | reasoning | medium | ECLI:CZ:NS:2025:29.CDO.275.2024.1__chunk_0004 | ECLI:CZ:NS:2025:29.CDO.275.2024.1 | yes | - | 177/1996 Sb., o odměnách advokátů a náhradách advokátů za poskytování právních služeb (advokátního tarifu), ve znění účinném k datu podání vyjádření, činí (z tarifní hodnoty 421.400,- Kč) částku 10.020,- Kč, a z paušální... |
| 3 | 0.768661 | 28 Cdo 3513/2024 | USNESENÍ | civil | reasoning | strong | ECLI:CZ:NS:2025:28.CDO.3513.2024.1__chunk_0006 | ECLI:CZ:NS:2025:28.CDO.3513.2024.1 | yes | dovolacího řízení | 15. O nákladech dovolacího řízení bylo rozhodnuto v intencích § 243 odst. 3 věty první, § 224 odst. 1, § 151 odst. 1 části věty před středníkem a § 146 odst. 3 o. s. ř.; dovolání žalované bylo odmítnuto a na straně žalob... |
| 4 | 0.768099 | 27 Cdo 2699/2024 | USNESENÍ | civil | reasoning | strong | ECLI:CZ:NS:2025:27.CDO.2699.2024.1__chunk_0005 | ECLI:CZ:NS:2025:27.CDO.2699.2024.1 | yes | dovolacího řízení | 13. Výrok o náhradě nákladů dovolacího řízení se opírá o § 243c odst. 3, § 224 odst. 1 a § 146 odst. 3 o. s. ř., když dovolání žalobkyně bylo odmítnuto a žalovanému vzniklo právo na náhradu účelně vynaložených nákladů do... |
| 5 | 0.760259 | 25 Cdo 2348/2024 | ROZSUDEK | civil | reasoning | strong | ECLI:CZ:NS:2024:25.CDO.2348.2024.1__chunk_0002 | ECLI:CZ:NS:2024:25.CDO.2348.2024.1 | yes | - | Odůvodnění: 1. Žalobkyně se podanou žalobou domáhala 998.089 Kč s příslušenstvím z titulu odpovědnosti advokáta za škodu. Částka 203.186 Kč představovala marně vynaložené náklady řízení, částka 594.903 Kč kapitalizovaný... |
| 6 | 0.752328 | 27 Cdo 3338/2024 | USNESENÍ | civil | operative_part | strong | ECLI:CZ:NS:2025:27.CDO.3338.2024.1__chunk_0001 | ECLI:CZ:NS:2025:27.CDO.3338.2024.1 | yes | dovolacího řízení | takto: I. Dovolací řízení se zastavuje . II. Žalovaná je povinna zaplatit žalobkyni na náhradě nákladů dovolacího řízení 64.323,60 Kč do tří dnů od právní moci tohoto usnesení k rukám zástupce žalobkyně. |
| 7 | 0.750306 | 20 Cdo 3371/2024 | USNESENÍ | civil | reasoning | medium | ECLI:CZ:NS:2025:20.CDO.3371.2024.1__chunk_0003 | ECLI:CZ:NS:2025:20.CDO.3371.2024.1 | yes | - | 5) Podle dovolatele mu mělo být ve druhé skupině správně přiznáno 10 % z výtěžku 2 232 000Kč, tedy částka 223 200 Kč, a nikoli jen částka 187 194,30 Kč. Odvolací soud zaujal právní názor, že základ pro výpočet přihlášené... |
| 8 | 0.749727 | 23 Cdo 68/2025 | USNESENÍ | civil | reasoning | strong | ECLI:CZ:NS:2025:23.CDO.68.2025.1__chunk_0002 | ECLI:CZ:NS:2025:23.CDO.68.2025.1 | yes | - | Odůvodnění: 1. Žalobkyně se podanou žalobou domáhala na žalovaném zaplacení částky 176 000 Kč s příslušenstvím jako náhrady škody. Podle tvrzení žalobkyně se uplatněný nárok měl skládat jednak z částky 140 000 Kč (předst... |
| 9 | 0.742144 | 23 Cdo 707/2024 | USNESENÍ | civil | operative_part | strong | ECLI:CZ:NS:2024:23.CDO.707.2024.1__chunk_0001 | ECLI:CZ:NS:2024:23.CDO.707.2024.1 | yes | dovolacího řízení | takto: I. Dovolání se odmítá. II. Žalována je povinna zaplatit žalobci na náhradu nákladů dovolacího řízení částku 17 714 Kč do tří dnů právní moci tohoto usnesení k rukám jeho právního zástupce. |
| 10 | 0.737311 | 23 Cdo 434/2024 | USNESENÍ | civil | operative_part | strong | ECLI:CZ:NS:2024:23.CDO.434.2024.1__chunk_0001 | ECLI:CZ:NS:2024:23.CDO.434.2024.1 | yes | dovolacího řízení | takto: I. Dovolání se odmítá. II. Žalobkyně je povinna zaplatit žalovanému na náhradě nákladů dovolacího řízení částku 4 114 Kč do tří dnů od právní moci tohoto usnesení k rukám právní zástupkyně žalovaného. |

### Exact-match Diagnostics

| rank | exact_score | phrase_match | in_semantic_top_10 | semantic_rank | case_number | document_type | legal_area | section_type | chunk_id | matched_important_terms | matched_generic_terms | preview |
| --- | ---: | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 35 | no | no | 0 | 20 Cdo 3450/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:20.CDO.3450.2024.1__chunk_0004 | dovolacího řízení | náklady, dovolání, řízení | 120/2001 Sb., o soudních exekutorech a exekuční činnosti (exekuční řád), ve znění pozdějších předpisů, nestanoví-li tento zákon jinak, použijí se pro exekuční řízení přiměřeně ustanovení občanského soudního řádu. Výkon r... |
| 2 | 35 | no | no | 0 | 23 Cdo 68/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:23.CDO.68.2025.1__chunk_0003 | dovolacího řízení | náklady, dovolání, řízení | 6. Žalovaný v podaném vyjádření k dovolání označil dovolání žalobkyně za nepřípustné a navrhl Nejvyššímu soudu, aby je odmítl, případně aby je zamítl a uložil žalobkyni povinnost nahradit žalovanému náklady dovolacího ří... |
| 3 | 35 | no | no | 0 | 24 Cdo 2633/2024 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2024:24.CDO.2633.2024.1__chunk_0011 | dovolacího řízení | náklady, dovolání, řízení | 7. 2015). Soudy v závěru, který nebyl dovolatelkou zpochybněn, dovodily, že žalobce nebyl platně vyděděn, a to nejen důvodem uplatněným v prohlášení o vydědění /z důvodu uvedeného v ustanovení § 1646 odst. 1 písm. b) o.... |
| 4 | 35 | no | yes | 4 | 27 Cdo 2699/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:27.CDO.2699.2024.1__chunk_0005 | dovolacího řízení | náklady, dovolání, řízení | 13. Výrok o náhradě nákladů dovolacího řízení se opírá o § 243c odst. 3, § 224 odst. 1 a § 146 odst. 3 o. s. ř., když dovolání žalobkyně bylo odmítnuto a žalovanému vzniklo právo na náhradu účelně vynaložených nákladů do... |
| 5 | 35 | no | no | 0 | 27 Cdo 3338/2024 | USNESENÍ | civil | signature | ECLI:CZ:NS:2025:27.CDO.3338.2024.1__chunk_0003 | dovolacího řízení | náklady, dovolání, řízení | Předseda senátu Nejvyššího soudu proto dovolací řízení v souladu s § 243f odst. 2 zákona č. 99/1963 Sb., občanského soudního řádu (dále jen „o. s. ř.“), zastavil podle § 243c odst. 3 věty druhé o. s. ř. 5. Výrok o náhrad... |
| 6 | 35 | no | no | 0 | 27 Cdo 395/2024 | USNESENÍ | civil | signature | ECLI:CZ:NS:2025:27.CDO.395.2024.1__chunk_0007 | dovolacího řízení | náklady, dovolání, řízení | 11. 2022 a ze dne 14. 12. 2022 tudíž Nejvyšší soud nemohl přihlížet. [21] O náhradě nákladů dovolacího řízení ve vztahu mezi žalobcem a žalovanými 1), 2), 4) 6) až 9) bylo rozhodnuto podle § 243c odst. 3 věty první, § 22... |
| 7 | 35 | no | no | 0 | 28 Cdo 1880/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:28.CDO.1880.2024.1__chunk_0006 | dovolacího řízení | náklady, dovolání, řízení | 2. 2024, sp. zn. 28 Cdo 2593/2023). 15. Z vylíčeného je zjevné, že na předmětné dovolání nelze pohlížet jako na přípustné, pročež je Nejvyšší soud podle § 243c odst. 1 o. s. ř. odmítl. 16. O náhradě nákladů dovolacího ří... |
| 8 | 35 | no | no | 0 | 28 Cdo 2866/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2024:28.CDO.2866.2024.1__chunk_0007 | dovolacího řízení | náklady, dovolání, řízení | 16. Napadá-li snad dovolatel rozsudek odvolacího soudu i v části jeho výroku o nákladech řízení (uvádí-li výslovně, že rozhodnutí odvolacího soudu je napadeno v jeho celém rozsahu, přestože v této části žádnou dovolací a... |
| 9 | 35 | no | no | 0 | 28 Cdo 3321/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:28.CDO.3321.2024.1__chunk_0001 | dovolacího řízení | náklady, dovolání, řízení | takto: I. Dovolání se odmítá . II. Žalobce je povinen nahradit Římskokatolické farnosti Rožmberk nad Vltavou náklady dovolacího řízení ve výši 4.114,- Kč k rukám jejího zástupce, JUDr. Jakuba Kříže, Ph.D., advokáta se sí... |
| 10 | 35 | no | no | 0 | 28 Cdo 3321/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:28.CDO.3321.2024.1__chunk_0007 | dovolacího řízení | náklady, dovolání, řízení | 12/1945 Sb., nikterak neodchýlil. Jeho postup přitom plně koresponduje výše citované judikatuře Ústavního soudu o kasační závaznosti jeho nálezů. K ní se ve své recentní judikatuře hlásí i dovolací soud (srovnej např. us... |

## zjevně neopodstatněné dovolání

- Likely failure reason: `query_too_generic`
- Recommendation: `rewrite query`
- Exact match exists: **yes**
- Exact match candidate count: **867**
- Exact phrase-match count: **0**
- Semantic top 10 exact overlap count: **0**
- Semantic missed exact matches: **yes**
- Notes: The query terms are broad and match many chunks exactly, which weakens diagnostic precision.

### Semantic Top 10

| rank | score | case_number | document_type | legal_area | section_type | structure_status | chunk_id | document_id | metadata_present | matched_important_terms | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.736710 | 21 Cdo 44/2025 | USNESENÍ | civil | operative_part | strong | ECLI:CZ:NS:2025:21.CDO.44.2025.1__chunk_0001 | ECLI:CZ:NS:2025:21.CDO.44.2025.1 | yes | dovolání | takto: Dovolání povinného se odmítá. |
| 2 | 0.731032 | 21 Cdo 2658/2024 | USNESENÍ | civil | operative_part | strong | ECLI:CZ:NS:2025:21.CDO.2658.2024.3__chunk_0001 | ECLI:CZ:NS:2025:21.CDO.2658.2024.3 | yes | dovolání | takto: Dovolání povinného se odmítá . |
| 3 | 0.686126 | 27 Cdo 1921/2024 | USNESENÍ | civil | operative_part | strong | ECLI:CZ:NS:2025:27.CDO.1921.2024.1__chunk_0001 | ECLI:CZ:NS:2025:27.CDO.1921.2024.1 | yes | dovolání | takto: Dovolání se odmítá . |
| 4 | 0.686126 | 26 Cdo 125/2024 | USNESENÍ | civil | operative_part | strong | ECLI:CZ:NS:2024:26.CDO.125.2024.1__chunk_0001 | ECLI:CZ:NS:2024:26.CDO.125.2024.1 | yes | dovolání | takto: Dovolání se odmítá . |
| 5 | 0.686126 | 29 NSCR 70/2024 | USNESENÍ | civil | operative_part | medium | ECLI:CZ:NS:2025:29.NSCR.70.2024.1__chunk_0001 | ECLI:CZ:NS:2025:29.NSCR.70.2024.1 | yes | dovolání | takto: Dovolání se odmítá . |
| 6 | 0.686126 | 22 Cdo 3556/2024 | USNESENÍ | civil | operative_part | strong | ECLI:CZ:NS:2025:22.CDO.3556.2024.1__chunk_0001 | ECLI:CZ:NS:2025:22.CDO.3556.2024.1 | yes | dovolání | takto: Dovolání se odmítá . |
| 7 | 0.686126 | 29 NSCR 1/2025 | USNESENÍ | civil | operative_part | strong | ECLI:CZ:NS:2025:29.NSCR.1.2025.1__chunk_0001 | ECLI:CZ:NS:2025:29.NSCR.1.2025.1 | yes | dovolání | takto: Dovolání se odmítá . |
| 8 | 0.686126 | 26 Cdo 84/2025 | USNESENÍ | civil | operative_part | strong | ECLI:CZ:NS:2025:26.CDO.84.2025.1__chunk_0001 | ECLI:CZ:NS:2025:26.CDO.84.2025.1 | yes | dovolání | takto: Dovolání se odmítá . |
| 9 | 0.686126 | 21 Cdo 1566/2024 | USNESENÍ | civil | operative_part | strong | ECLI:CZ:NS:2025:21.CDO.1566.2024.1__chunk_0001 | ECLI:CZ:NS:2025:21.CDO.1566.2024.1 | yes | dovolání | takto: Dovolání se odmítá . |
| 10 | 0.686126 | 20 Cdo 875/2024 | USNESENÍ | civil | operative_part | strong | ECLI:CZ:NS:2025:20.CDO.875.2024.1__chunk_0001 | ECLI:CZ:NS:2025:20.CDO.875.2024.1 | yes | dovolání | takto: Dovolání se odmítá . |

### Exact-match Diagnostics

| rank | exact_score | phrase_match | in_semantic_top_10 | semantic_rank | case_number | document_type | legal_area | section_type | chunk_id | matched_important_terms | matched_generic_terms | preview |
| --- | ---: | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 45 | no | no | 0 | 11 Tdo 1114/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:11.TDO.1114.2024.1__chunk_0013 | zjevně neopodstatněné, dovolání | dovolání | 34. Označil však za zjevně nesprávné pokoušet se o nápravu vytvořením nové nezákonnosti – posunutím hranice mezi skutky vůči svědkyni J. ze 7. 6. 2021 na 30. 3. 2022 v nyní posuzované věci. Když v souladu s § 12 odst. 11... |
| 2 | 45 | no | no | 0 | 11 Tdo 1127/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:11.TDO.1127.2024.1__chunk_0013 | zjevně neopodstatněné, dovolání | dovolání | 24. Z těchto důvodů státní zástupkyně navrhuje, aby Nejvyšší soud podané dovolání podle § 265i odst. 1 písm. e) tr. řádu odmítl, protože je zjevně neopodstatněné, a aby tak rozhodl v souladu s § 265r odst. 1 písm. a) tr.... |
| 3 | 45 | no | no | 0 | 11 Tdo 679/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:11.TDO.679.2024.1__chunk_0014 | zjevně neopodstatněné, dovolání | dovolání | 31. Dovolací důvod podle § 265b odst. 1 písm. h) tr. řádu , je naplněn tehdy, jestliže rozhodnutí spočívá na nesprávném právním posouzení skutku nebo jiném nesprávném hmotněprávním posouzení. V rámci takto vymezeného dov... |
| 4 | 45 | no | no | 0 | 11 Tdo 75/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:11.TDO.75.2025.1__chunk_0012 | zjevně neopodstatněné, dovolání | dovolání | 17. Závěrem svého dovolání proto obvinění A. navrhli, aby Nejvyšší soud zrušil napadené rozhodnutí Vrchního soudu v Praze a věc přikázal příslušnému soudu k novému projednání a rozhodnutí. Zároveň oba jmenovaní obvinění... |
| 5 | 45 | no | no | 0 | 11 Tdo 75/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:11.TDO.75.2025.1__chunk_0018 | zjevně neopodstatněné, dovolání | dovolání | 9. 2004, sp. zn. II. ÚS 279/03). 32. Nadto Nejvyšší soud i při respektování shora uvedeného interpretuje a aplikuje podmínky připuštění dovolání tak, aby dodržel maximy práva na spravedlivý proces vymezené Úmluvou o ochr... |
| 6 | 45 | no | no | 0 | 11 Tdo 75/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:11.TDO.75.2025.1__chunk_0039 | zjevně neopodstatněné, dovolání | dovolání | 5. 2005, sp. zn. II. ÚS 681/04, podle kterého právo na spravedlivý proces není možné vykládat tak, že garantuje úspěch v řízení či zaručuje právo na rozhodnutí, jež odpovídá představám stěžovatele (tj. obviněných). Uvede... |
| 7 | 45 | no | no | 0 | 11 Tdo 765/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:11.TDO.765.2024.1__chunk_0008 | zjevně neopodstatněné, dovolání | dovolání | 10. Konečně se státní zástupce neztotožňuje ani s výtkou dovolatele, že soudy vzaly v potaz jeho trestní minulost. Poukazuje na to, že soud prvního stupně tuto skutečnost reflektoval jakožto přitěžující okolnost podle §... |
| 8 | 45 | no | no | 0 | 3 Tdo 1120/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:3.TDO.1120.2024.1__chunk_0021 | zjevně neopodstatněné, dovolání | dovolání | V. Způsob rozhodnutí 32. Nejvyšší soud proto předložené dovolání odmítl podle § 265i odst. 1 písm. e) tr. ř. jako zjevně neopodstatněné. Toto rozhodnutí učinil v souladu s ustanovením § 265r odst. 1 písm. a) tr. ř. v nev... |
| 9 | 45 | no | no | 0 | 3 Tdo 19/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:3.TDO.19.2025.1__chunk_0009 | zjevně neopodstatněné, dovolání | dovolání | 20. Státní zástupce proto navrhl, aby Nejvyšší soud dovolání obviněného odmítl podle § 265e tr. ř. jako zjevně neopodstatněné a aby tak učinil v neveřejném zasedání za podmínek § 265r odst. 1, písm. a) tr. ř. S rozhodová... |
| 10 | 45 | no | no | 0 | 3 Tdo 19/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:3.TDO.19.2025.1__chunk_0023 | zjevně neopodstatněné, dovolání | dovolání | 47. Dovolací důvod podle tohoto ustanovení obviněný neuplatnil v prvé alternativě – tedy že mu bylo odepřeno právo na přístup k druhé instanci. Uplatnil jej v alternativně druhé – totiž že odvolací soud v rámci svého pře... |

## odmítnutí dovolání

- Likely failure reason: `query_too_generic`
- Recommendation: `rewrite query`
- Exact match exists: **yes**
- Exact match candidate count: **868**
- Exact phrase-match count: **27**
- Semantic top 10 exact overlap count: **0**
- Semantic missed exact matches: **yes**
- Notes: The query terms are broad and match many chunks exactly, which weakens diagnostic precision.

### Semantic Top 10

| rank | score | case_number | document_type | legal_area | section_type | structure_status | chunk_id | document_id | metadata_present | matched_important_terms | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.922904 | 21 Cdo 1566/2024 | USNESENÍ | civil | operative_part | strong | ECLI:CZ:NS:2025:21.CDO.1566.2024.1__chunk_0001 | ECLI:CZ:NS:2025:21.CDO.1566.2024.1 | yes | - | takto: Dovolání se odmítá . |
| 2 | 0.922904 | 27 Cdo 1921/2024 | USNESENÍ | civil | operative_part | strong | ECLI:CZ:NS:2025:27.CDO.1921.2024.1__chunk_0001 | ECLI:CZ:NS:2025:27.CDO.1921.2024.1 | yes | - | takto: Dovolání se odmítá . |
| 3 | 0.922904 | 26 Cdo 125/2024 | USNESENÍ | civil | operative_part | strong | ECLI:CZ:NS:2024:26.CDO.125.2024.1__chunk_0001 | ECLI:CZ:NS:2024:26.CDO.125.2024.1 | yes | - | takto: Dovolání se odmítá . |
| 4 | 0.922904 | 29 NSCR 1/2025 | USNESENÍ | civil | operative_part | strong | ECLI:CZ:NS:2025:29.NSCR.1.2025.1__chunk_0001 | ECLI:CZ:NS:2025:29.NSCR.1.2025.1 | yes | - | takto: Dovolání se odmítá . |
| 5 | 0.922904 | 20 Cdo 875/2024 | USNESENÍ | civil | operative_part | strong | ECLI:CZ:NS:2025:20.CDO.875.2024.1__chunk_0001 | ECLI:CZ:NS:2025:20.CDO.875.2024.1 | yes | - | takto: Dovolání se odmítá . |
| 6 | 0.922904 | 29 NSCR 70/2024 | USNESENÍ | civil | operative_part | medium | ECLI:CZ:NS:2025:29.NSCR.70.2024.1__chunk_0001 | ECLI:CZ:NS:2025:29.NSCR.70.2024.1 | yes | - | takto: Dovolání se odmítá . |
| 7 | 0.922904 | 22 Cdo 3556/2024 | USNESENÍ | civil | operative_part | strong | ECLI:CZ:NS:2025:22.CDO.3556.2024.1__chunk_0001 | ECLI:CZ:NS:2025:22.CDO.3556.2024.1 | yes | - | takto: Dovolání se odmítá . |
| 8 | 0.922904 | 26 Cdo 84/2025 | USNESENÍ | civil | operative_part | strong | ECLI:CZ:NS:2025:26.CDO.84.2025.1__chunk_0001 | ECLI:CZ:NS:2025:26.CDO.84.2025.1 | yes | - | takto: Dovolání se odmítá . |
| 9 | 0.893802 | 21 Cdo 2658/2024 | USNESENÍ | civil | operative_part | strong | ECLI:CZ:NS:2025:21.CDO.2658.2024.3__chunk_0001 | ECLI:CZ:NS:2025:21.CDO.2658.2024.3 | yes | - | takto: Dovolání povinného se odmítá . |
| 10 | 0.887667 | 21 Cdo 44/2025 | USNESENÍ | civil | operative_part | strong | ECLI:CZ:NS:2025:21.CDO.44.2025.1__chunk_0001 | ECLI:CZ:NS:2025:21.CDO.44.2025.1 | yes | - | takto: Dovolání povinného se odmítá. |

### Exact-match Diagnostics

| rank | exact_score | phrase_match | in_semantic_top_10 | semantic_rank | case_number | document_type | legal_area | section_type | chunk_id | matched_important_terms | matched_generic_terms | preview |
| --- | ---: | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 130 | yes | no | 0 | 20 Cdo 13/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:20.CDO.13.2025.1__chunk_0003 | odmítnutí dovolání | odmítnutí, dovolání | 99/1963 Sb., občanský soudní řád, ve znění účinném od 30. září 2017 (srov. část první čl. II bod 2 zákona č. 296/2017 Sb.), dále jen „o. s. ř.“. Dovolacímu soudu je z jeho činnosti známo a plyne to i z obsahu spisu, že p... |
| 2 | 130 | yes | no | 0 | 20 Cdo 15/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:20.CDO.15.2025.1__chunk_0003 | odmítnutí dovolání | odmítnutí, dovolání | 99/1963 Sb., občanský soudní řád, ve znění účinném od 30. září 2017 (srov. část první čl. II bod 2 zákona č. 296/2017 Sb.), dále jen „o. s. ř.“. Dovolacímu soudu je z jeho činnosti známo a plyne to i z obsahu spisu, že p... |
| 3 | 130 | yes | no | 0 | 20 Cdo 3061/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2024:20.CDO.3061.2024.1__chunk_0006 | odmítnutí dovolání | odmítnutí, dovolání | 8/ Rozhodnutí odvolacího soudu (jeho závěr o tom, že žalobou pro zmatečnost napadené rozhodnutí není postiženo zmatečností podle ustanovení § 229 odst. 4 o. s. ř.) je v souladu s ustálenou rozhodovací praxí dovolacího so... |
| 4 | 130 | yes | no | 0 | 21 Cdo 245/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:21.CDO.245.2024.1__chunk_0002 | odmítnutí dovolání | odmítnutí, dovolání | 3. Proti rozsudku odvolacího soudu podali dovolání oba účastníci. 4. Žalobce přípustnost dovolání spatřuje k vyřešení otázky, zda § 650 občanského zákoníku „se vztahuje také na případy, kdy dlužník věřiteli hrozbou brání... |
| 5 | 130 | yes | no | 0 | 21 Cdo 245/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:21.CDO.245.2024.1__chunk_0005 | odmítnutí dovolání | odmítnutí, dovolání | 6. 2023, č. j. 12 Co 25/2023-148, se podává, že směřuje toliko proti části výroku I, v níž bylo rozhodnuto o změně rozsudku Okresního soudu v Chomutově ze dne 16. 9. 2022, č. j. 34 C 7/2021-112, ve výroku III tak, že žád... |
| 6 | 130 | yes | no | 0 | 21 Cdo 2841/2024 | USNESENÍ | civil | operative_part | ECLI:CZ:NS:2025:21.CDO.2841.2024.1__chunk_0010 | odmítnutí dovolání | odmítnutí, dovolání | 11. 2017, sp. zn. Pl. ÚS-st. 45/16). O výjimečný případ, kdy skutková otázka s ohledem na její průmět do základních lidských práv a svobod je způsobilá založit přípustnost dovolání podle § 237 o. s. ř. (srov. například n... |
| 7 | 130 | yes | no | 0 | 21 Cdo 44/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:21.CDO.44.2025.1__chunk_0006 | odmítnutí dovolání | odmítnutí, dovolání | 10. 2014, sp. zn. 29 Cdo 4097/2014). 18. Nejvyšší soud České republiky proto dovolání povinného podle ustanovení § 243c odst. 1 o. s. ř. odmítl. 19. Dovolatel navrhl odklad právní moci usnesení odvolacího soudu; Ústavní... |
| 8 | 130 | yes | no | 0 | 3 Tdo 19/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:3.TDO.19.2025.1__chunk_0023 | odmítnutí dovolání | odmítnutí, dovolání | 47. Dovolací důvod podle tohoto ustanovení obviněný neuplatnil v prvé alternativě – tedy že mu bylo odepřeno právo na přístup k druhé instanci. Uplatnil jej v alternativně druhé – totiž že odvolací soud v rámci svého pře... |
| 9 | 130 | yes | no | 0 | 3 Tdo 53/2025 | USNESENÍ | criminal | signature | ECLI:CZ:NS:2025:3.TDO.53.2025.1__chunk_0012 | odmítnutí dovolání | odmítnutí, dovolání | předseda senátu Nejvyššího soudu podle § 265o odst. 1 tr. ř. odložil výkon napadeného rozhodnutí, pak je třeba uvést, že se z jeho strany jednalo o podnět, nikoli o návrh, o němž by bylo nutné učinit formální rozhodnutí.... |
| 10 | 130 | yes | no | 0 | 3 Tdo 650/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2024:3.TDO.650.2024.1__chunk_0031 | odmítnutí dovolání | odmítnutí, dovolání | 47. Je tak možno uzavřít, že vzhledem k tomu, že dovolací námitky obviněného M. H., ať už je bylo možno podřadit pod uplatněné důvody dovolání či nikoliv, vyhodnotil Nejvyšší soud jako neopodstatněné, nemohlo dojít k nap... |

## rodinný dům

- Likely failure reason: `query_too_generic`
- Recommendation: `rewrite query`
- Exact match exists: **yes**
- Exact match candidate count: **543**
- Exact phrase-match count: **5**
- Semantic top 10 exact overlap count: **1**
- Semantic missed exact matches: **no**
- Notes: The query terms are broad and match many chunks exactly, which weakens diagnostic precision.

### Semantic Top 10

| rank | score | case_number | document_type | legal_area | section_type | structure_status | chunk_id | document_id | metadata_present | matched_important_terms | preview |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.383765 | 26 Cdo 1854/2024 | ROZSUDEK | civil | reasoning | strong | ECLI:CZ:NS:2025:26.CDO.1854.2024.1__chunk_0008 | ECLI:CZ:NS:2025:26.CDO.1854.2024.1 | yes | - | 16. Pojem „jednotka“ použitý v § 1196 odst. 2 o. z. je však třeba vykládat ve spojení s § 1159 o. z., jenž stanoví, že jednotka zahrnuje nejen byt (jako prostorově oddělenou část domu), ale také podíl na společných částe... |
| 2 | 0.383488 | 3 Tdo 984/2024 | USNESENÍ | criminal | reasoning | strong | ECLI:CZ:NS:2024:3.TDO.984.2024.1__chunk_0006 | ECLI:CZ:NS:2024:3.TDO.984.2024.1 | yes | - | 2. 2023 do 20. 2. 2023, kdy měl mít společné nezletilé děti v péči, avšak od poškozené věděl, že zdravotní stav nezletilého syna toto neumožňuje, pod záminkou být s dětmi apeloval právě opět v přítomnosti společných dětí... |
| 3 | 0.376240 | 26 Cdo 439/2024 | ROZSUDEK | civil | reasoning | strong | ECLI:CZ:NS:2024:26.CDO.439.2024.1__chunk_0003 | ECLI:CZ:NS:2024:26.CDO.439.2024.1 | yes | - | 6. 2012, č. j. 11 Co 155/2012-78, bylo zrušeno právo společného nájmu tam specifikovaného družstevního bytu (předmětného bytu) a garáže (blíže označené) a společné členství v bytovém družstvu s tím, že výlučnou členkou d... |
| 4 | 0.353452 | 7 Tdo 1096/2024 | USNESENÍ | criminal | reasoning | strong | ECLI:CZ:NS:2025:7.TDO.1096.2024.1__chunk_0004 | ECLI:CZ:NS:2025:7.TDO.1096.2024.1 | yes | - | 6. Pokud jde o koncept sdílené újmy, obviněný uvedl, že z rozsudku soudu prvního stupně a ze spisu vyplynulo, že celá rodina poškozeného udržuje mezi sebou velmi dobré vztahy, stojí při sobě a vzájemně se podporuje. Manž... |
| 5 | 0.340566 | 24 Cdo 671/2024; 24 Cdo 675/2024 | USNESENÍ | civil | reasoning | strong | ECLI:CZ:NS:2024:24.CDO.671.2024.1__chunk_0002 | ECLI:CZ:NS:2024:24.CDO.671.2024.1 | yes | - | Odůvodnění: 1. Okresní soud v Pelhřimově rozsudkem ze dne 31. 5. 2023, č. j. 5 C 255/2015-1439, rozhodl o určení vlastnického práva k celkem devíti jednotkám v domě č. p. XY v XY. Jednalo se o byty, garáže a dílnu. Struč... |
| 6 | 0.328666 | 26 Cdo 439/2024 | ROZSUDEK | civil | reasoning | strong | ECLI:CZ:NS:2024:26.CDO.439.2024.1__chunk_0006 | ECLI:CZ:NS:2024:26.CDO.439.2024.1 | yes | - | 8. Žalovaný ve vyjádření k dovolání považuje rozhodnutí odvolacího soudu za správné. Do okamžiku zajištění bytové náhrady se řídí vztah bývalého nájemce a pronajímatele ustanovením § 712a obč. zák. a teprve po zajištění... |
| 7 | 0.325900 | 4 Tdo 1044/2024 | USNESENÍ | criminal | reasoning | strong | ECLI:CZ:NS:2024:4.TDO.1044.2024.1__chunk_0005 | ECLI:CZ:NS:2024:4.TDO.1044.2024.1 | yes | - | 14. Podle dovolatele je celé řízení vedeno tendenčně a je poplatné době, ve které žijeme, reagující na společenskou objednávku pod tíhou různých hnutí za hranicí demagogie. Příkladmo akcentuje, že soudy nijak nehodnotily... |
| 8 | 0.312357 | 22 Cdo 108/2025 | ROZSUDEK | civil | reasoning | strong | ECLI:CZ:NS:2025:22.CDO.108.2025.1__chunk_0002 | ECLI:CZ:NS:2025:22.CDO.108.2025.1 | yes | rodinný dům | Odůvodnění: I. Dosavadní průběh řízení 1. Okresní soud v Lounech (dále jen „soud prvního stupně“) rozsudkem ze dne 16. 5. 2024, č. j. 12 C 152/2023-52, určil, že pozemek p. č. XY a pozemek p. č. XY, jehož součástí je sta... |
| 9 | 0.310003 | 3 Tdo 980/2024 | USNESENÍ | criminal | signature | strong | ECLI:CZ:NS:2024:3.TDO.980.2024.1__chunk_0028 | ECLI:CZ:NS:2024:3.TDO.980.2024.1 | yes | - | 51. Obdobná situace pak nastává u vniknutí do garáže v XY ulici, kterou měl pronajatou svědek O. H. od svědka J. H., a ve které se nacházely věci obou těchto osob [skutek pod bodem 2) výroku o vině]. Předně je potřebné u... |
| 10 | 0.299560 | 26 Cdo 439/2024 | ROZSUDEK | civil | reasoning | strong | ECLI:CZ:NS:2024:26.CDO.439.2024.1__chunk_0008 | ECLI:CZ:NS:2024:26.CDO.439.2024.1 | yes | - | 1. 2010, sp. zn. 28 Cdo 2146/2009). 14. Z § 712a obč. zák. vyplývá, že obsah vzájemných práv a povinností účastníků právního vztahu, který je uvedeným ustanovením posuzován, se řídí § 687 až § 699 obč. zák. , tedy i § 69... |

### Exact-match Diagnostics

| rank | exact_score | phrase_match | in_semantic_top_10 | semantic_rank | case_number | document_type | legal_area | section_type | chunk_id | matched_important_terms | matched_generic_terms | preview |
| --- | ---: | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 135 | yes | no | 0 | 30 Cdo 308/2025 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2025:30.CDO.308.2025.1__chunk_0002 | rodinný dům | dům, nemovitost, byt | Odůvodnění: Žalobkyně se žalobou domáhala po žalované zaplacení částky 6 130 000 Kč s příslušenstvím jako náhrady škody, která jí měla být způsobena v souvislosti s nesprávným úředním postupem soudního exekutora v exekuč... |
| 2 | 130 | yes | yes | 8 | 22 Cdo 108/2025 | ROZSUDEK | civil | reasoning | ECLI:CZ:NS:2025:22.CDO.108.2025.1__chunk_0002 | rodinný dům | dům, byt | Odůvodnění: I. Dosavadní průběh řízení 1. Okresní soud v Lounech (dále jen „soud prvního stupně“) rozsudkem ze dne 16. 5. 2024, č. j. 12 C 152/2023-52, určil, že pozemek p. č. XY a pozemek p. č. XY, jehož součástí je sta... |
| 3 | 130 | yes | no | 0 | 23 Cdo 3170/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2024:23.CDO.3170.2024.1__chunk_0002 | rodinný dům | dům, byt | Odůvodnění: 1. Žalobce se podanou žalobou na žalované domáhal zaplacení částky 336 744 Kč z titulu odpovědnosti za vady jako slevy z kupní ceny, kterou zaplatil na základě kupní smlouvy uzavřené mezi žalobcem a žalovanou... |
| 4 | 130 | yes | no | 0 | 26 Cdo 2404/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2024:26.CDO.2404.2024.1__chunk_0003 | rodinný dům | dům, byt | 10. 2000) a nachází se na něm (pravý) podpis jmenované svědkyně. Ztotožnil se i s názory soudu prvního stupně, že nájemní smlouva není neplatná ani z důvodu, že k ní nedala souhlas valná hromada žalované (tímto souhlasem... |
| 5 | 125 | yes | no | 0 | 23 Cdo 434/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2024:23.CDO.434.2024.1__chunk_0002 | rodinný dům | dům | Odůvodnění: 1. Žalobkyně se podanou žalobou domáhala určení vlastnického práva ke stavbě č. p. XY (rodinný dům) na pozemcích parc. č. st. XY, parc. č. st. XY a parc. č. st. XY, vše v k. ú. XY. Tvrdila, že dne 8. 6. 1992... |
| 6 | 10 | no | no | 0 | 11 Tdo 75/2025 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:11.TDO.75.2025.1__chunk_0024 | - | nemovitost, byt | 39. S ohledem na výše uvedené důkazy objektivní povahy, jimiž byl zdokumentovaný průběh návštěv obviněného M. R. v pěstírně v obci XY, odvoz sušiny konopí jeho osobou, jeho podíl na dodání sazenic konopí nezbytných pro j... |
| 7 | 10 | no | no | 0 | 24 Cdo 671/2024; 24 Cdo 675/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2024:24.CDO.671.2024.1__chunk_0004 | - | nemovitost, byt | 5. Žalovaná 1) ve svém dovolání předložila Nejvyššímu soud dvě otázky. První z nich dosud neměla být v rozhodovací praxi dovolacího soudu vyřešena. Žalovaná 1) nesouhlasila se závěrem odvolacího soudu, že při stanovení o... |
| 8 | 10 | no | no | 0 | 26 Cdo 125/2024 | USNESENÍ | civil | reasoning | ECLI:CZ:NS:2024:26.CDO.125.2024.1__chunk_0002 | - | nemovitost, byt | Odůvodnění: 1. Krajský soud v Praze (odvolací soud) usnesením ze dne 30. 5. 2023, č. j. 17 Co 96/2023-316, potvrdil usnesení Okresního soudu Praha-západ (soud prvního stupně) ze dne 8. 2. 2023, č. j. 206 EXE 6445/2021-27... |
| 9 | 5 | no | no | 0 | 11 Pzo 7/2024 | USNESENÍ | criminal | reasoning | ECLI:CZ:NS:2025:11.PZO.7.2024.1__chunk_0003 | - | byt | 10. 2024, když zcela prokazatelně nebyla s obviněným v kontaktu a od 3. 5. 2022 pracuje a žije mimo Českou republiku. Jelikož jí nebyly nikdy sděleny žádné informace, z nichž by bylo možno byť jen dovodit, že by se jakko... |
| 10 | 5 | no | no | 0 | 11 Pzo 7/2024 | USNESENÍ | criminal | signature | ECLI:CZ:NS:2025:11.PZO.7.2024.1__chunk_0005 | - | byt | předseda senátu a v přípravném řízení nařídí jejich vydání státnímu zástupci nebo policejnímu orgánu soudce na návrh státního zástupce. 1. 2. Příkaz k zjištění údajů o telekomunikačním provozu přitom musí být vydán písem... |

## Final Summary

- Real retrieval problems: none
- Generic or sparse-topic weak queries: `náhrada nákladů dovolacího řízení`, `zjevně neopodstatněné dovolání`, `odmítnutí dovolání`, `rodinný dům`
- Hybrid retrieval recommended before production-scale scrape: **no**
