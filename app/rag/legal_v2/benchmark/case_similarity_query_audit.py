"""Query audit and natural-language rewrite helpers for golden v3."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.rag.legal_v2.benchmark.case_similarity_golden_v2 import (
    MAX_QUERY_WORDS,
    MIN_QUERY_WORDS,
    audit_query_leakage,
    count_words,
)

BOILERPLATE_CLOSING = "Chci podobnou judikaturu bez uvádění konkrétní spisové značky."
AUTO_PREFIXES = (
    "Zajímá mě, jak Ústavní soud posoudil tuto situaci: ",
    "Hledám ústavní usnesení k následující procesní nebo věcné otázce: ",
    "Potřebuji najít ústavní rozhodnutí k této právní otázce: ",
)
REASONING_HEADERS = (
    "Odůvodnění:",
    "Odůvodnění ",
    "Ústavní soud dospěl k závěru",
    "Ústavní soud je přesvědčen",
    "Stěžovatel namítá",
    "Maje na zřeteli",
)
_CORRUPT_MARKERS = ("dané věci , dané věci", "dané věci dané věci")
_DANE_VECI_RE = re.compile(r"\bdané věci\b", re.IGNORECASE)


@dataclass(frozen=True)
class QueryAuditResult:
    query_id: str
    status: str  # approved | needs_edit | rejected
    flags: list[str]
    rewritten_text: str
    rewrite_applied: bool


# Curated rewrites keyed by query_id — natural Czech legal-search wording.
# Derived from v2 legal_area + relevance_notes; not from retrieval ranks.
CURATED_QUERY_REWRITES: dict[str, str] = {
    "nalus-cs-v2-001": (
        "Podal jsem k Ústavnímu soudu podání, které nesplňuje náležitosti řádného návrhu "
        "a současně se domáhám vyloučení soudců. Může ÚS takové podání odmítnout bez dalšího "
        "poučování, pokud jsem už dříve opakovaně dostal informace o vadách podání?"
    ),
    "nalus-cs-v2-002": (
        "Od kdy přestává platit vyživovací povinnost rodiče vůči dceři, která po ukončení "
        "studia a dovršení zletilosti sama obstarává obživu? Hledám judikaturu Ústavního soudu "
        "k posouzení okamžiku zániku vyživovací povinnosti."
    ),
    "nalus-cs-v2-003": (
        "Nejvyšší soud odmítl dovolání pro nedostatečné vymezení přípustnosti. "
        "Může Ústavní soud přezkoumat, zda odmítnutí dovolání nebylo v rozporu s judikaturou "
        "k přípustnosti dovolání ve věcech občanskoprávních?"
    ),
    "nalus-cs-v2-004": (
        "Sociálka a soudy řeší odebrání dětí a já jsem proti tomu hned podal stížnost, "
        "ale bez advokáta, spíš naléhavě a chaoticky. V podání mi chybí pořádné odůvodnění "
        "a zastoupení. Potřebuju najít podobné věci, kde rodič napadá zásah do péče o děti, "
        "ale návrh padne už na formálních vadách — ne meritorní spor o péči."
    ),
    "nalus-cs-v2-005": (
        "Jak Ústavní soud posuzuje přípustnost ústavní stížnosti proti rozhodnutí o nepřípustnosti "
        "dovolání Nejvyššího soudu, pokud stěžovatel brojí proti formalistickému postupu?"
    ),
    "nalus-cs-v2-006": (
        "Za jakých podmínek Ústavní soud nenařizuje ústní jednání podle § 44 zákona o Ústavním "
        "soudu, pokud od něj nelze očekávat další objasnění věci?"
    ),
    "nalus-cs-v2-007": (
        "Jak Ústavní soud vysvětluje rozdíl mezi užíváním společné věci v rámci spoluvlastnického "
        "podílu a užíváním nad rámec podílu bez právního důvodu podle občanského zákoníku?"
    ),
    "nalus-cs-v2-008": (
        "V restitučním řízení stěžovatelé nedoložili původní vlastnické právo k nemovitostem. "
        "Jak Ústavní soud posuzuje ústavní stížnost, když orgány veřejné moci tvrdí, "
        "že chybí knihovní vložka a kupní smlouva z roku 1948?"
    ),
    "nalus-cs-v2-009": (
        "Může stěžovatel v ústavní stížnosti úspěšně brojit proti rozhodnutí obecných soudů "
        "tím, že nezletilému dítěti nebyl ustanoven kolizní opatrovník, když dítě samo "
        "ústavní stížnost nepodalo?"
    ),
    "nalus-cs-v2-010": (
        "Jak Ústavní soud vysvětluje institut nuceného výkupu účastnických cenných papírů "
        "(squeeze-out) v kontextu novely obchodního zákoníku z roku 2005?"
    ),
    "nalus-cs-v2-011": (
        "Jaké jsou meze přezkumu dokazování trestními soudy z hlediska ústavní stížnosti, "
        "zejména ve vztahu k zásadám bezprostřednosti a ústnosti trestního řízení?"
    ),
    "nalus-cs-v2-012": (
        "Může daňový orgán vymáhat daňovou povinnost plátce daně i tehdy, kdy část daně "
        "zaplatili jednotliví poplatníci, aniž by šlo o nepřípustné dvojí zdanění?"
    ),
    "nalus-cs-v2-013": (
        "Jak Ústavní soud aplikuje princip nejlepšího zájmu dítěte podle čl. 3 odst. 1 "
        "Úmluvy o právech dítěte v soudním rozhodování?"
    ),
    "nalus-cs-v2-014": (
        "Musí Ústavní soud znovu poučit stěžovatele o povinném zastoupení advokátem a náležitostech "
        "ústavní stížnosti, pokud tentýž člověk opakovaně podává ústavní stížnosti se stejnými "
        "formálními vadami?"
    ),
    "nalus-cs-v2-015": (
        "Jak Ústavní soud posuzuje námitku podjatosti, když zástupkyně účastníka jedná "
        "současně jako přísedící u téhož okresního soudu?"
    ),
    "nalus-cs-v2-016": (
        "Jak Ústavní soud přistupuje k ústavním stížnostem proti rozhodnutím obecných soudů "
        "o nákladech řízení a k otázce zákazu reformace in peius u nákladů?"
    ),
    "nalus-cs-v2-017": (
        "Může Ústavní soud projednat ústavní stížnost proti rozsudku, proti němuž stěžovatel "
        "ještě podal odvolání a nevyčerpal všechny procesní prostředky?"
    ),
    "nalus-cs-v2-018": (
        "Jak Ústavní soud posuzuje rozhodování soudu o správní žalobě bez ústního jednání "
        "podle § 250f občanského soudního řádu a povinnost řádného odůvodnění?"
    ),
    "nalus-cs-v2-019": (
        "Je v rozporu se závaznou judikaturou odmítnout část ústavní stížnosti směřující "
        "proti usnesení obsahujícímu výzvu k odstranění vad, aniž by se věc posoudila "
        "v souladu s dřívějšími nálezy?"
    ),
    "nalus-cs-v2-020": (
        "Jak Ústavní soud posuzuje registraci kandidáta do Senátu, když v podání chybí "
        "úplné rodné číslo a registrace byla zamítnuta?"
    ),
    "nalus-cs-v2-021": (
        "Jak Ústavní soud posuzuje dopad nepřesného označení právního subjektu ve správním "
        "rozhodnutí na vykonatelnost rozhodnutí a ochranu vlastnického práva?"
    ),
    "nalus-cs-v2-022": (
        "Od kdy běží lhůta pro vrácení zákona prezidentem republiky Poslanecké sněmovně "
        "a jak Ústavní soud posuzuje následné nehlasování o znovu projednaném zákonu?"
    ),
    "nalus-cs-v2-023": (
        "Může Ústavní soud projednat stížnost proti rozsudku, který dosud nenabyl právní moci "
        "proto, že stěžovatel podal odvolání a odvolací soud nařídil hlavní líčení?"
    ),
    "nalus-cs-v2-024": (
        "Za jakých podmínek Ústavní soud neshledá extrémní nesoulad nebo zjevnou arbitrárnost "
        "při hodnocení důkazů trestními soudy?"
    ),
    "nalus-cs-v2-025": (
        "Jak Ústavní soud posuzuje postup Nejvyššího správního soudu při přezkumu voleb "
        "a algoritmus pro posouzení vlivu příspěvků na sociálních sítích na volbu kandidáta?"
    ),
    "nalus-cs-v2-026": (
        "Jak Ústavní soud aplikuje subjektivní a objektivní test nestrannosti soudce "
        "v řízení o vyloučení soudce?"
    ),
    "nalus-cs-v2-027": (
        "Co když odvolací soud po vytknutí nesprávného vymezení faktického pracovního poměru "
        "setrval na původním rozhodnutí a závazný právní názor dovolacího soudu použil "
        "v rozporu s jeho smyslem?"
    ),
    "nalus-cs-v2-028": (
        "Jak Ústavní soud postupuje při formálně vadném podání označeném jako stížnost "
        "proti postupu soudu a zadržování ve vazbě, které nesplňuje podmínky ústavní stížnosti?"
    ),
    "nalus-cs-v2-029": (
        "Může soud rozhodovat o okamžitém umístění dítěte do péče nahrazující výchovu rodičů "
        "až dodatečně poté, co správní orgán vydal předběžné opatření o umístění?"
    ),
    "nalus-cs-v2-030": (
        "Může Ústavní soud projednat ústavní stížnost proti platebnímu výměru, pokud stěžovatel "
        "nevyčerpal řádné ani mimořádné opravné prostředky a lhůty již uplynuly?"
    ),
    "nalus-cs-v2-031": (
        "Jak Ústavní soud posuzuje, zda odvolací soud měl uvažovat o kvalifikaci skutku "
        "jako zpronevěry, když pochyboval o naplnění znaků podvodu?"
    ),
    "nalus-cs-v2-032": (
        "Jak Ústavní soud posuzuje vymáhání celního dluhu po nesplnění povinnosti celním "
        "dlžníkem a přechod z nalézacího do vymáhacího řízení?"
    ),
    "nalus-cs-v2-033": (
        "Stačí k prokázání včasnosti elektronického podání kopie hlavičky e-mailu, "
        "nebo Ústavní soud vyžaduje spolehlivější důkaz o doručení?"
    ),
    "nalus-cs-v2-034": (
        "Jaké požadavky Ústavní soud klade na výklad právních jednání a na rozhodnutí "
        "Nejvyššího soudu o nepřípustnosti dovolání?"
    ),
    "nalus-cs-v2-035": (
        "Jak Ústavní soud posuzuje běh lhůty k podání ústavní stížnosti po podání dovolání "
        "a souběh s mimořádnými opravnými prostředky?"
    ),
    "nalus-cs-v2-036": (
        "Jak Ústavní soud posuzuje ústavní stížnost brojící proti rozhodnutí s tvrzením "
        "porušení práva podnikat a ochrany majetku podle Listiny?"
    ),
    "nalus-cs-v2-037": (
        "Může Ústavní soud přezkoumat rozhodnutí obecných soudů o návrhu na zrušení "
        "věcného břemene podle § 151p občanského zákoníku?"
    ),
    "nalus-cs-v2-038": (
        "Jak Ústavní soud posuzuje ústavní stížnost směřující proti rozhodnutí obecných soudů "
        "s námitkou porušení práva na ochranu vlastnictví a spravedlivého procesu?"
    ),
    "nalus-cs-v2-039": (
        "Jak Ústavní soud vnímá doplňující odlišné stanovisko soudce k stanovisku pléna "
        "a jeho vztah k řešení ústavněprávní otázky?"
    ),
    "nalus-cs-v2-040": (
        "Jak Ústavní soud posuzuje právo dítěte vyjádřit názor ve věcech, které se ho týkají, "
        "a vyrovnání nerovného postavení dítěte vůči rodičům?"
    ),
    "nalus-cs-v2-041": (
        "Musí soudy odůvodnit, proč nevyhověly důkazním návrhům účastníků, a jak Ústavní soud "
        "posuzuje zásadu volného hodnocení důkazů?"
    ),
    "nalus-cs-v2-042": (
        "Může volič podat návrh na neplatnost volby kandidáta ve volebním obvodu, "
        "do něhož nespadá volební okrsek, v němž je zapsán ve voličském seznamu?"
    ),
    "nalus-cs-v2-043": (
        "Má Ústavní soud sjednotit postup tak, aby při vyhovění ústavní stížnosti zrušil "
        "vazební rozhodnutí i tehdy, kdy stěžovatel již ve vazbě není?"
    ),
    "nalus-cs-v2-044": (
        "Jak obecné soudy a Ústavní soud posuzují, zda stavba zasahuje výlučně do pozemku "
        "parcellního vlastníka podle geometrických plánů?"
    ),
    "nalus-cs-v2-045": (
        "Jak Ústavní soud posuzuje právo na soudní ochranu ve vztahu k judikatuře "
        "Evropského soudu pro lidská práva o přístupu k soudu?"
    ),
    "nalus-cs-v2-046": (
        "Jak Ústavní soud chrání právo územních samosprávných celků na samosprávu "
        "podle čl. 101 Ústavy?"
    ),
    "nalus-cs-v2-047": (
        "Jak Ústavní soud posuzuje podání navrhovatelů domáhajících se zrušení rozsudku "
        "a přezkumu postupu Nejvyššího soudu, který neprojednal dovolání?"
    ),
    "nalus-cs-v2-048": (
        "Jak Ústavní soud posuzuje stížnost proti usnesení vrchního soudu o zamítnutí "
        "žádosti o propuštění z vazby?"
    ),
    "nalus-cs-v2-049": (
        "Jaká je povinnost obecného soudu zkoumat nejen přezkoumatelnost správního rozhodnutí, "
        "ale i to, kdo je účastníkem správního soudnictví?"
    ),
    "nalus-cs-v2-050": (
        "Jak Ústavní soud posuzuje restituční spor, kdy vedlejší účastnice tvrdí, "
        "že o vydání převáděných pozemků neusilovala?"
    ),
    "nalus-cs-v2-051": (
        "Jak Ústavní soud posuzuje námitku podjatosti soudce krajského soudu v civilní věci "
        "a rozhodnutí vrchního soudu o vyloučení?"
    ),
    "nalus-cs-v2-052": (
        "Jak Nejvyšší soud a Ústavní soud posuzují přípustnost dovolání podle § 238 odst. 1 "
        "o. s. ř. u peněžitého nároku sestávajícího z více dílčích nároků?"
    ),
    "nalus-cs-v2-053": (
        "Co Ústavní soud považuje za ústavně nekonformní výklad práva a kdy výklad "
        "nedůvodně vybočuje z respektované soudní praxe?"
    ),
    "nalus-cs-v2-054": (
        "Může schvalování rozpočtu centrální banky jiným orgánem než bankou samotnou "
        "nepřímo ovlivňovat její nezávislost?"
    ),
    "nalus-cs-v2-055": (
        "Jak Ústavní soud posuzuje procesní předpoklady řízení o ústavní stížnosti — "
        "včasnost, aktivní legitimaci a zastoupení advokátem?"
    ),
    "nalus-cs-v2-056": (
        "Může Ústavní soud přezkoumat postup správních orgánů, které nevydaly rozhodnutí "
        "o vyloučení z řízení o odstranění stavby?"
    ),
    "nalus-cs-v2-057": (
        "Jak Ústavní soud v plenárním stanovisku sjednocuje postup senátů při rozhodování "
        "o ústavní stížnosti v obdobných procesních otázkách?"
    ),
    "nalus-cs-v2-058": (
        "Musí odvolací soud zjišťovat, proč rodič využívá prostředky k vynucení styku s nezletilou, "
        "a posoudit vliv komunikace rodičů na nejlepší zájem dítěte?"
    ),
    "nalus-cs-v2-059": (
        "Jak Ústavní soud posuzuje kázeňské trestání ve výkonu trestu odnětí svobody "
        "a hranice omezení osobní svobody nad rámec trvání trestu?"
    ),
    "nalus-cs-v2-060": (
        "Jak Ústavní soud posuzuje ústavní stížnost brojící proti rozhodnutí obecných soudů "
        "s tvrzením porušení práva na soudní ochranu?"
    ),
}


def _strip_boilerplate(text: str) -> str:
    cleaned = text.strip()
    for prefix in AUTO_PREFIXES:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    cleaned = cleaned.replace(BOILERPLATE_CLOSING, "").strip()
    for header in REASONING_HEADERS:
        if cleaned.lower().startswith(header.lower()):
            cleaned = cleaned[len(header) :].strip()
    cleaned = _DANE_VECI_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;")
    return cleaned


def classify_query(
    query_id: str,
    query_text: str,
    *,
    target_ecli: str | None = None,
    case_reference: str | None = None,
    target_reasoning: str = "",
) -> tuple[str, list[str]]:
    flags: list[str] = []
    flags.extend(audit_query_leakage(query_text, target_ecli=target_ecli, case_reference=case_reference))
    if BOILERPLATE_CLOSING in query_text:
        flags.append("boilerplate_closing_phrase")
    if any(query_text.startswith(prefix) for prefix in AUTO_PREFIXES):
        flags.append("auto_generated_template_prefix")
    if any(query_text.strip().lower().startswith(h.lower()) for h in REASONING_HEADERS):
        flags.append("reasoning_header_opening")
    if any(marker in query_text for marker in _CORRUPT_MARKERS):
        flags.append("corrupted_placeholder_text")
    if _DANE_VECI_RE.search(query_text):
        flags.append("placeholder_dane_veci")
    if count_words(query_text) > 120 and flags:
        flags.append("possibly_over_close_auto_paraphrase")

    if query_id == "nalus-cs-v2-004" and not flags:
        return "approved", flags
    if "corrupted_placeholder_text" in flags and query_id not in CURATED_QUERY_REWRITES:
        return "rejected", flags
    if flags:
        return "needs_edit", flags
    if count_words(query_text) < MIN_QUERY_WORDS:
        return "needs_edit", flags + ["query_too_short"]
    return "approved", flags


def rewrite_query(
    query_id: str,
    query_text: str,
    *,
    legal_area: str = "",
    relevance_notes: str = "",
) -> tuple[str, bool]:
    if query_id in CURATED_QUERY_REWRITES:
        return CURATED_QUERY_REWRITES[query_id], True
    status, _ = classify_query(query_id, query_text)
    if status == "approved":
        return query_text, False
    core = _strip_boilerplate(query_text)
    if len(core) < 40 and relevance_notes:
        core = relevance_notes.strip()
    question = (
        f"Hledám judikaturu Ústavního soudu k otázce: {core[:320]}"
        if core
        else f"Hledám judikaturu Ústavního soudu v oblasti {legal_area or 'práva'}."
    )
    if count_words(question) > MAX_QUERY_WORDS:
        words = question.split()
        question = " ".join(words[:MAX_QUERY_WORDS])
    return question, True


def audit_and_rewrite_v2_query(item: Any, *, target_reasoning: str = "") -> QueryAuditResult:
    status, flags = classify_query(
        item.query_id,
        item.query_text,
        target_ecli=item.expected_primary_ecli,
        case_reference=item.case_reference,
        target_reasoning=target_reasoning,
    )
    rewritten, applied = rewrite_query(
        item.query_id,
        item.query_text,
        legal_area=item.legal_area,
        relevance_notes=item.relevance_notes,
    )
    if status == "approved":
        rewritten = item.query_text
        applied = False
    elif status == "needs_edit":
        status = "edited" if applied else "needs_edit"
    return QueryAuditResult(
        query_id=item.query_id,
        status="approved" if status == "approved" else ("edited" if applied else status),
        flags=flags,
        rewritten_text=rewritten,
        rewrite_applied=applied and status != "approved",
    )
