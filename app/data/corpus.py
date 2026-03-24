"""
Seed corpus — rozhodnutí Ústavního soudu ČR.

Formát: list[(str_id, text)]
  str_id — používá KeywordRetriever i jako výsledný chunk.id
  text   — plný text úryvku rozhodnutí

Tato data jsou ingestována do Qdrant při startu aplikace (app/api/startup.py).
Přidej další rozhodnutí sem — startup je automaticky načte.
"""

CORPUS: list[tuple[str, str]] = [
    (
        "1",
        "III.ÚS 255/22 — Ústavní soud se zabýval případem mezinárodního únosu dítěte matkou "
        "do Ruska. Matka jako cizinka neoprávněně přemístila nezletilé dítě na území Ruské "
        "federace bez souhlasu otce. Soud aplikoval Haagskou úmluvu o mezinárodním únosu "
        "dítěte a nařídil neprodlený návrat dítěte do země jeho obvyklého bydliště. "
        "Rodičovská odpovědnost náleží oběma rodičům společně.",
    ),
    (
        "2",
        "II.ÚS 100/21 — Případ řeší rodičovskou odpovědnost a střídavou péči o nezletilé "
        "dítě. Ústavní soud zdůraznil, že zájem dítěte je prvořadý při rozhodování o péči. "
        "Rodičovská odpovědnost obou rodičů musí být zachována i po rozvodu manželství. "
        "Střídavá péče je preferována, pokud to nejlepší zájem dítěte nevylučuje.",
    ),
    (
        "3",
        "I.ÚS 88/23 — Stěžovatel se domáhá náhrady škody způsobené státem v důsledku "
        "nezákonného rozhodnutí orgánu veřejné moci. Odpovědnost státu za škodu je "
        "zakotvena v zákoně č. 82/1998 Sb. Náhrada škody zahrnuje skutečnou škodu i ušlý "
        "zisk. Stát odpovídá za škodu způsobenou nezákonnými rozhodnutími.",
    ),
    (
        "4",
        "IV.ÚS 312/20 — Ústavní soud shledal porušení práva na spravedlivý proces z důvodu "
        "nepřiměřené délky řízení. Délka soudního řízení přesáhla rozumnou dobu a porušila "
        "základní právo stěžovatele. Průtahy v řízení před obecnými soudy jsou nepřípustné. "
        "Stát je povinen zajistit projednání věci v přiměřené lhůtě.",
    ),
    (
        "5",
        "III.ÚS 401/22 — Haagská úmluva o občanskoprávních aspektech mezinárodních únosů "
        "dítěte zavazuje smluvní státy k rychlému návratu neoprávněně přemístěných dětí. "
        "Únos dítěte jedním z rodičů porušuje právo druhého rodiče na styk s dítětem. "
        "Opatrovnický soud musí rozhodnout o návratu dítěte bez zbytečných průtahů.",
    ),
    (
        "6",
        "II.ÚS 77/19 — Soud posuzoval přiměřenost délky trestního řízení a právo na "
        "projednání věci bez průtahů. Nepřiměřená délka řízení představuje porušení "
        "článku 38 odst. 2 Listiny základních práv a svobod. Obviněnému přísluší náhrada "
        "nemajetkové újmy způsobené nepřiměřeně dlouhým řízením.",
    ),
    (
        "7",
        "I.ÚS 502/21 — Stát nese odpovědnost za škodu způsobenou nesprávným úředním "
        "postupem státních orgánů. Náhrada škody od státu se přiznává i za nemajetkovou "
        "újmu v případech zvláštního zřetele hodných. Odpovědnost státu za škodu je "
        "objektivní a nevyžaduje zavinění konkrétního úředníka.",
    ),
    (
        "8",
        "IV.ÚS 190/23 — Ústavní soud se zabýval otázkou rodičovské odpovědnosti "
        "v přeshraničním kontextu. Mezinárodní únos dítěte cizinkou porušuje práva otce "
        "jako druhého z rodičů. Soud aplikoval nařízení Brusel IIa při určování příslušnosti "
        "k rozhodnutí o návratu dítěte. Rodičovská odpovědnost nesmí být vykonávána "
        "v rozporu se základními právy druhého rodiče.",
    ),
]
