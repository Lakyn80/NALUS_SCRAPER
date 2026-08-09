#!/usr/bin/env python3
"""Build a full pilot_600 judgment inventory with descriptions and search queries.

Offline only — no paid LLM. Descriptions and queries are derived from indexed
metadata + non-boilerplate judgment text heuristics.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BM25 = (
    PROJECT_ROOT.parent
    / "nalus-scraper"
    / "storage"
    / "rag"
    / "bm25"
    / "nalus_legal_paragraph_bm25_v2_pilot_600.sqlite"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "artifacts" / "legal_v2" / "pilot_600_judgment_inventory"

WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+")

BOILERPLATE_SENTENCE_RE = re.compile(
    r"(?i)"
    r"^(NALUS\b|Česká republika\b|"
    r"Ústavní(?:ho)?\s+soud(?:u)?\s+(?:rozhodl|Jménem)|"
    r"Nález\s+Ústavní\s+soud|"
    r"Usnesení\s+Ústavní\s+soud|"
    r"USNESENÍ\s+Nejvyšší\s+soud|"
    r"ROZSUDEK\s+Nejvyšší\s+soud|"
    r"Jménem\s+republiky|"
    r"\d+\s+(?:Tdo|Cdo|Nd|Ncu|ICdo|Tcu|Tvo|Td|Pzo)\s+\d+/\d+|"
    r"takto\s*:)"
)
SENATE_JUNK_RE = re.compile(
    r"(?i)"
    r"rozhodl\s+(?:v\s+senát|v\s+plénu)|"
    r"soudce\s+zpravodaj|"
    r"složeném\s+z\s+předsedy|"
    r"zákon(?:ě)?\s+č\.\s*182/1993|"
    r"formální\s+náležitosti|"
    r"Včas\s+podanou|"
    r"splňuje\s+formální|"
    r"NALUS\s*-\s*databáze|"
    r"žádný\s+z\s+účastníků\s+nemá\s+právo\s+na\s+náhradu\s+nákladů"
)
PROCEDURAL_OPENER_RE = re.compile(
    r"(?i)^(?:\d+\.\s*)?(?:Usnesením|Rozsudkem|Usnesení|Rozsudek|Nálezem)\b"
)

HEADER_CASE_RE = re.compile(
    r"(?i)\b((?:[IVX]+\.|Pl\.)\s*ÚS\s+\d+/\d+(?:\s*#\d+)?)|"
    r"(\d+\s+(?:Tdo|Cdo|Nd|Ncu|ICdo|Tcu|Tvo|Td|Pzo|NScr)\s+\d+/\d+)"
)
ECLI_US_RE = re.compile(
    r"^ECLI:CZ:US:(?P<year>\d{4}):(?P<senate>\d+|Pl)\.US\.(?P<num>\d+)\.(?P<year2>\d{2})\.(?P<seq>\d+)$",
    re.IGNORECASE,
)
ECLI_NS_RE = re.compile(
    r"^ECLI:CZ:NS:(?P<year>\d{4}):(?P<senate>\d+)\.(?P<kind>[A-Z]+)\.(?P<num>\d+)\.(?P<year2>\d{4})\.(?P<seq>\d+)$",
    re.IGNORECASE,
)

ABOUT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)((?:se\s+)?domáhá(?:\s+se)?\s+zrušení[^\.]{15,240})"),
    re.compile(r"(?i)(navrhovatel[^\.]{0,40}domáhá[^\.]{15,220})"),
    re.compile(r"(?i)((?:ve věci|v řízení o|řízení o)[^\.]{15,220})"),
    re.compile(r"(?i)(uznán(?:a)?\s+vinným[^\.]{10,200})"),
    re.compile(r"(?i)(porušen[íi][^\.]{8,200}(?:Listiny|Úmluvy|čl\.|základních\s+práv))"),
    re.compile(r"(?i)(náhrad[ay]\s+škody[^\.]{10,180})"),
    re.compile(r"(?i)(exekučn[^\.]{10,180})"),
    re.compile(r"(?i)(vazební[^\.]{8,160}|vazb[ayě][^\.]{8,160})"),
    re.compile(r"(?i)(nezletil[^\.]{10,200})"),
    re.compile(r"(?i)(mezinárodní\s+ochran[^\.]{8,160}|azyl[^\.]{8,160})"),
    re.compile(r"(?i)(uznání\s+(?:cizího|zahraničního)\s+(?:rozsudku|rozhodnutí)[^\.]{8,160})"),
]

TOPIC_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("mezinarodni_unos_ditete", re.compile(r"(?i)únos(?:u|em)?|Haagsk\w+\s+úmluv|návrat\s+dítěte\s+do")),
    ("vycestovani_zahranici", re.compile(r"(?i)zákaz[^\.]{0,40}vycestov|vycestován[^\.]{0,50}(bez souhlasu|zahrani)")),
    ("pece_o_nezletile", re.compile(r"(?i)svěřil\w*\s+nezletil|svěření\s+do\s+péče|střídavé?\s+péč|péči\s+matky|péči\s+otce")),
    ("uprava_styku", re.compile(r"(?i)úprav[ay]\s+styk|styk\s+(?:otce|matky)\s+s\s+nezletil")),
    ("vyzivne", re.compile(r"(?i)výživn\w+|aliment")),
    ("opatrovnictvi", re.compile(r"(?i)opatrovnick\w+\s+řízen|kolizní\s+opatrovník")),
    ("rodinne_pravo", re.compile(r"(?i)rodinnoprávn|rozvod\s+manželství|společné\s+jmění\s+manželů")),
    ("vazba", re.compile(r"(?i)\bvazební\b|\bvazbě\b|\bvazba\b|\bvazby\b")),
    ("trestni_rizeni", re.compile(r"(?i)trestní\s+stíhání|trest\s+odnětí\s+svobody|obviněn\w+|uznán\w*\s+vinným|dovolání\s+v\s+trestní")),
    ("bezplatna_obhajoba", re.compile(r"(?i)bezplatn\w+\s+obhajob|právo\s+na\s+obhajobu")),
    ("azyl_mezinarodni_ochrana", re.compile(r"(?i)mezinárodní\s+ochran|azylov\w+\s+řízení|udělení\s+azylu")),
    ("vydani_do_ciziny", re.compile(r"(?i)vydání\s+do\s+(?:ciziny|cizího)|řízení\s+o\s+vydání")),
    ("uznani_ciziho_rozhodnuti", re.compile(r"(?i)uznání\s+(?:cizího|zahraničního)\s+(?:rozsudku|rozhodnutí)|prohlášení\s+vykonatelnosti")),
    ("obcanstvi", re.compile(r"(?i)státního?\s+občanství|udělení\s+občanství")),
    ("rusko_rf", re.compile(r"(?i)Ruské\s+federace|Ruské\s+federaci|občan\w*\s+Rus|do\s+Ruska|v\s+Rusku")),
    ("exekuce", re.compile(r"(?i)exekučn\w+\s+řízen|exekutor|nařízení\s+exekuce")),
    ("nahrada_skody", re.compile(r"(?i)náhrad[ay]\s+škody|ušlý\s+nájem|nemajetková\s+újma")),
    ("najemn_byt", re.compile(r"(?i)nájem\s+(?:bytu|bytu)|výpověď\s+z\s+nájmu|bytov\w+\s+náhrad")),
    ("pracovni_pravo", re.compile(r"(?i)pracovní\s+poměr|výpověď\s+z\s+pracovního|odstupn[ée]|neplatnost\s+výpovědi")),
    ("danove_rizeni", re.compile(r"(?i)daňov\w+\s+řízen|správce\s+daně|daň\s+z\s+příjm|DPH\b")),
    ("spravni_soudnictvi", re.compile(r"(?i)Nejvyšší(?:ho)?\s+správní(?:ho)?\s+soud|správní\s+žalob")),
    ("insolvence", re.compile(r"(?i)insolvenční\s+řízen|oddlužení|úpad(?:ek|ku)")),
    ("pravomoc_prislusnost", re.compile(r"(?i)místní\s+příslušnost|věcná\s+příslušnost|určení\s+příslušnosti|mezinárodní\s+příslušnost")),
    ("predbezne_opatreni", re.compile(r"(?i)předběžn\w+\s+opatřen")),
    ("nejlepsi_zajemy_ditete", re.compile(r"(?i)nejlepš\w+\s+zájm\w+\s+dítěte")),
    ("svoboda_projevu", re.compile(r"(?i)svoboda\s+projevu|ochrana\s+osobnosti|zásah\s+do\s+soukromí")),
    ("diskriminace", re.compile(r"(?i)diskriminac|nerovné\s+zacházení")),
    ("bezplatna_obhajoba", re.compile(r"(?i)bezplatn\w+\s+obhajob|právo\s+na\s+bezplatnou\s+obhajobu")),
    ("ustavni_stiznost_formalni_vady", re.compile(r"(?i)odmítá\s+se\s+pro\s+neodstranění\s+vad|odmítnut[^\.]{0,60}formáln")),
    ("pravni_zastoupeni_advokat", re.compile(r"(?i)není\s+zastoupen\s+advokátem|bez\s+právního\s+zastoupení|výzva[^\.]{0,40}advokát")),
    ("naklady_rizeni", re.compile(r"(?i)náhrad[ay]\s+nákladů\s+řízení")),
]

PROCEDURAL_TOPICS = {
    "ustavni_stiznost_formalni_vady",
    "pravni_zastoupeni_advokat",
    "naklady_rizeni",
}

TOPIC_LABELS_CS: dict[str, str] = {
    "mezinarodni_unos_ditete": "mezinárodní únos dítěte",
    "vycestovani_zahranici": "zákaz vycestování s dítětem",
    "pece_o_nezletile": "péče o nezletilé dítě",
    "uprava_styku": "úprava styku s dítětem",
    "vyzivne": "výživné",
    "opatrovnictvi": "opatrovnické řízení",
    "rodinne_pravo": "rodinné právo",
    "vazba": "vazba",
    "trestni_rizeni": "trestní řízení",
    "azyl_mezinarodni_ochrana": "azyl / mezinárodní ochrana",
    "vydani_do_ciziny": "vydání do ciziny",
    "uznani_ciziho_rozhodnuti": "uznání cizího rozhodnutí",
    "obcanstvi": "státní občanství",
    "rusko_rf": "vazba na Rusko",
    "exekuce": "exekuce",
    "nahrada_skody": "náhrada škody",
    "najemn_byt": "nájem bytu",
    "pracovni_pravo": "pracovní právo",
    "danove_rizeni": "daňové řízení",
    "spravni_soudnictvi": "správní soudnictví",
    "insolvence": "insolvence",
    "pravomoc_prislusnost": "příslušnost soudu",
    "predbezne_opatreni": "předběžné opatření",
    "nejlepsi_zajemy_ditete": "nejlepší zájem dítěte",
    "svoboda_projevu": "svoboda projevu / ochrana osobnosti",
    "diskriminace": "diskriminace",
    "bezplatna_obhajoba": "bezplatná obhajoba",
    "ustavni_stiznost_formalni_vady": "formální vady ústavní stížnosti",
    "pravni_zastoupeni_advokat": "povinné zastoupení advokátem",
    "naklady_rizeni": "náklady řízení",
}

TOPIC_QUERY_HINTS: dict[str, tuple[str, str]] = {
    "mezinarodni_unos_ditete": (
        "matka unesla dítě do zahraničí",
        "Hledám rozhodnutí, kde jeden rodič odvezl dítě do ciziny bez souhlasu druhého a řeší se návrat dítěte.",
    ),
    "vycestovani_zahranici": (
        "zákaz vycestování s dítětem",
        "Potřebuji případy, kde soud řešil zákaz vycestování nezletilého dítěte do zahraničí bez souhlasu rodiče.",
    ),
    "pece_o_nezletile": (
        "svěření dítěte do péče",
        "Hledám rozhodnutí o tom, komu svěřit nezletilé dítě do péče po rozchodu rodičů.",
    ),
    "uprava_styku": (
        "úprava styku s dítětem",
        "Chci podobná rozhodnutí o úpravě styku rodiče s nezletilým dítětem.",
    ),
    "vyzivne": (
        "výživné na dítě",
        "Hledám spory o výživné na nezletilé dítě a jak soudy určují výši alimentů.",
    ),
    "opatrovnictvi": (
        "opatrovnické řízení",
        "Potřebuji rozhodnutí z opatrovnického řízení o poměrech k nezletilému dítěti.",
    ),
    "rodinne_pravo": (
        "rodinné právo rozvod majetek",
        "Hledám podobné rodinněprávní spory – rozvod, majetek manželů nebo poměry k dětem.",
    ),
    "vazba": (
        "vazba ústavní stížnost",
        "Hledám rozhodnutí proti vazbě nebo prodloužení vazby.",
    ),
    "trestni_rizeni": (
        "trestní stíhání podobný případ",
        "Hledám podobná trestní rozhodnutí – stíhání, vina nebo trest.",
    ),
    "azyl_mezinarodni_ochrana": (
        "azyl a mezinárodní ochrana",
        "Hledám rozhodnutí o žádosti o azyl nebo mezinárodní ochranu v Česku.",
    ),
    "vydani_do_ciziny": (
        "vydání osoby do ciziny",
        "Potřebuji případy o vydání osoby k trestnímu stíhání do cizího státu.",
    ),
    "uznani_ciziho_rozhodnuti": (
        "uznání cizího rozsudku v Česku",
        "Hledám, jak české soudy uznávají zahraniční rozsudek nebo rozhodnutí.",
    ),
    "obcanstvi": (
        "žádost o české občanství",
        "Hledám spory o udělení nebo odepření státního občanství.",
    ),
    "rusko_rf": (
        "věc s vazbou na Rusko",
        "Hledám rozhodnutí, kde hraje roli Rusko nebo ruská státní příslušnost.",
    ),
    "exekuce": (
        "exekuce a exekuční řízení",
        "Potřebuji podobné případy o exekuci, exekutorovi nebo zastavení exekuce.",
    ),
    "nahrada_skody": (
        "náhrada škody",
        "Hledám spory o náhradu škody nebo nemajetkové újmy.",
    ),
    "najemn_byt": (
        "nájem bytu výpověď",
        "Hledám spory o nájem bytu, výpověď z nájmu nebo bytovou náhradu.",
    ),
    "pracovni_pravo": (
        "výpověď z práce",
        "Hledám pracovní spory o výpověď, odstupné nebo neplatnost skončení pracovního poměru.",
    ),
    "danove_rizeni": (
        "daňové řízení spor se správcem daně",
        "Hledám rozhodnutí o daňovém řízení nebo sporu se správcem daně.",
    ),
    "spravni_soudnictvi": (
        "správní žaloba Nejvyšší správní soud",
        "Hledám podobné správní spory řešené správními soudy.",
    ),
    "insolvence": (
        "insolvence oddlužení",
        "Hledám rozhodnutí o insolvenčním řízení nebo oddlužení.",
    ),
    "pravomoc_prislusnost": (
        "který soud je příslušný",
        "Potřebuji případy o místní nebo věcné příslušnosti soudu.",
    ),
    "predbezne_opatreni": (
        "předběžné opatření",
        "Hledám rozhodnutí o předběžném opatření, ideálně v rodinné věci.",
    ),
    "nejlepsi_zajemy_ditete": (
        "nejlepší zájem dítěte",
        "Hledám, jak soudy posuzují nejlepší zájem dítěte při rozhodování o péči.",
    ),
    "svoboda_projevu": (
        "ochrana osobnosti a svoboda projevu",
        "Hledám spory o ochranu osobnosti, zásah do soukromí nebo svobodu projevu.",
    ),
    "diskriminace": (
        "diskriminace nerovné zacházení",
        "Hledám rozhodnutí o diskriminaci nebo nerovném zacházení.",
    ),
    "bezplatna_obhajoba": (
        "právo na bezplatnou obhajobu",
        "Hledám rozhodnutí o právu na bezplatnou obhajobu v trestní věci.",
    ),
    "ustavni_stiznost_formalni_vady": (
        "ústavní stížnost odmítnutá pro vady",
        "Podal jsem ústavní stížnost a odmítli ji pro formální vady. Hledám podobné případy.",
    ),
    "pravni_zastoupeni_advokat": (
        "povinné zastoupení advokátem",
        "Ústavní soud vyžaduje advokáta. Hledám, kdy stížnost padne kvůli chybějícímu zastoupení.",
    ),
    "naklady_rizeni": (
        "náhrada nákladů řízení",
        "Nejde mi o meritorní spor, ale o náhradu nákladů řízení před soudem.",
    ),
}

NS_KIND_FALLBACK: dict[str, tuple[str, str, str]] = {
    "TDO": ("trestni_rizeni", "trestní dovolání Nejvyšší soud", "Hledám podobná trestní dovolání u Nejvyššího soudu."),
    "TD": ("trestni_rizeni", "trestní věc Nejvyšší soud", "Hledám podobná trestní rozhodnutí Nejvyššího soudu."),
    "TVO": ("vazba", "vazba Nejvyšší soud", "Hledám rozhodnutí Nejvyššího soudu o vazbě."),
    "TCU": ("trestni_rizeni", "trestní věc Nejvyšší soud", "Hledám podobná trestní rozhodnutí Nejvyššího soudu."),
    "CDO": ("nahrada_skody", "civilní dovolání Nejvyšší soud", "Hledám podobná civilní dovolání u Nejvyššího soudu."),
    "ICDO": ("insolvence", "insolvenční dovolání", "Hledám insolvenční dovolání u Nejvyššího soudu."),
    "ND": ("pravomoc_prislusnost", "určení příslušnosti soudu", "Hledám, jak Nejvyšší soud určuje příslušnost soudu."),
    "NCU": ("uznani_ciziho_rozhodnuti", "uznání cizího rozsudku", "Hledám uznání zahraničního rozsudku českým soudem."),
    "PZO": ("predbezne_opatreni", "předběžné opatření Nejvyšší soud", "Hledám předběžná opatření řešená Nejvyšším soudem."),
    "NSCR": ("trestni_rizeni", "trestní věc Nejvyšší soud", "Hledám podobná trestní rozhodnutí Nejvyššího soudu."),
}

SENATE_MAP = {
    "1": "I.",
    "2": "II.",
    "3": "III.",
    "4": "IV.",
}

NS_KIND_CASE_LABEL = {
    "TDO": "Tdo",
    "CDO": "Cdo",
    "ND": "Nd",
    "NCU": "Ncu",
    "ICDO": "ICdo",
    "TCU": "Tcu",
    "TVO": "Tvo",
    "TD": "Td",
    "PZO": "Pzo",
    "NSCR": "NSCR",
}


@dataclass
class JudgmentInventoryItem:
    ecli: str
    canonical_document_id: str
    case_number: str | None
    court: str | None
    decision_date: str | None
    document_type: str | None
    chunk_count: int
    topic_tags: list[str]
    description: str
    search_queries: list[str]
    evidence_excerpts: list[str] = field(default_factory=list)
    source: str = "bm25_pilot_600"


def _clean(text: str) -> str:
    return WHITESPACE_RE.sub(" ", (text or "").strip())


def _optional(value: Any) -> str | None:
    text = _clean(str(value or ""))
    return text or None


def _is_junk_sentence(text: str) -> bool:
    cleaned = _clean(text)
    if len(cleaned) < 40:
        return True
    if BOILERPLATE_SENTENCE_RE.search(cleaned):
        return True
    if SENATE_JUNK_RE.search(cleaned):
        return True
    if PROCEDURAL_OPENER_RE.search(cleaned) and len(cleaned) < 120:
        return True
    return False


def _load_rows(bm25_path: Path) -> dict[str, dict[str, Any]]:
    con = sqlite3.connect(f"file:{bm25_path}?mode=ro", uri=True)
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT document_id, section_type, text, metadata
        FROM bm25_chunks
        ORDER BY document_id, chunk_id
        """
    ).fetchall()
    con.close()

    by_doc: dict[str, dict[str, Any]] = {}
    for document_id, section_type, text, metadata_raw in rows:
        ecli = str(document_id).strip()
        entry = by_doc.get(ecli)
        if entry is None:
            meta: dict[str, Any] = {}
            if metadata_raw:
                try:
                    loaded = json.loads(metadata_raw)
                    if isinstance(loaded, dict):
                        meta = loaded
                except json.JSONDecodeError:
                    meta = {}
            entry = {"ecli": ecli, "meta": meta, "chunks": []}
            by_doc[ecli] = entry
        entry["chunks"].append(
            {
                "section_type": str(section_type or ""),
                "text": str(text or ""),
            }
        )
    return by_doc


def _iter_sentences(chunks: list[dict[str, str]]) -> list[str]:
    preferred_sections = {
        "participants",
        "facts",
        "procedural_history",
        "party_arguments",
        "legal_reasoning",
        "decision",
        "introduction",
    }
    ordered = sorted(
        chunks,
        key=lambda c: (0 if c["section_type"] in preferred_sections else 1),
    )
    sentences: list[str] = []
    seen: set[str] = set()
    for chunk in ordered:
        for part in SENTENCE_SPLIT_RE.split(_clean(chunk["text"])):
            cleaned = _clean(part)
            if _is_junk_sentence(cleaned):
                continue
            key = cleaned[:120].lower()
            if key in seen:
                continue
            seen.add(key)
            sentences.append(cleaned)
            if len(sentences) >= 40:
                return sentences
    return sentences


def _extract_about(sentences: list[str], full_blob: str) -> str:
    candidates: list[tuple[int, str]] = []
    for sentence in sentences:
        for rank, pattern in enumerate(ABOUT_PATTERNS):
            match = pattern.search(sentence)
            if not match:
                continue
            about = _clean(match.group(1))
            if len(about) < 40 or _is_junk_sentence(about):
                continue
            # Prefer spans that start near a legal cue, not mid-parenthesis fragments.
            if about.startswith(")") or about[:1].islower() and "domáhá" not in about.lower():
                # Allow lowercase starts only for known cue openers.
                if not re.match(r"(?i)^(ve věci|v řízení|řízení|uznán|porušen|náhrad|exekuč|vazeb|vazb|nezletil|mezinárodní|azyl|uznání|domáhá|navrhovatel|se domáhá)", about):
                    continue
            if len(about) > 260:
                about = about[:259].rsplit(" ", 1)[0] + "…"
            candidates.append((rank, about))
            break
    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]
    for rank, pattern in enumerate(ABOUT_PATTERNS):
        match = pattern.search(full_blob)
        if not match:
            continue
        about = _clean(match.group(1))
        if len(about) >= 40 and not SENATE_JUNK_RE.search(about):
            if about.startswith(")"):
                continue
            if len(about) > 260:
                about = about[:259].rsplit(" ", 1)[0] + "…"
            return about
    for sentence in sentences:
        if len(sentence) >= 60 and "záhlaví" not in sentence.lower():
            return sentence[:259].rsplit(" ", 1)[0] + ("…" if len(sentence) > 260 else "")
    return ""


def _detect_topics(blob: str) -> list[str]:
    tags: list[str] = []
    for name, pattern in TOPIC_RULES:
        if pattern.search(blob):
            tags.append(name)
    substantive = [tag for tag in tags if tag not in PROCEDURAL_TOPICS]
    procedural = [tag for tag in tags if tag in PROCEDURAL_TOPICS]
    return (substantive + procedural)[:8]


def _ns_kind(ecli: str) -> str | None:
    match = ECLI_NS_RE.match(ecli)
    if not match:
        return None
    return match.group("kind").upper()


def _ensure_topics(topics: list[str], ecli: str) -> list[str]:
    if topics:
        return topics
    kind = _ns_kind(ecli)
    if kind and kind in NS_KIND_FALLBACK:
        return [NS_KIND_FALLBACK[kind][0]]
    return topics


def _topic_labels(topics: list[str]) -> str:
    labels = [TOPIC_LABELS_CS.get(tag, tag.replace("_", " ")) for tag in topics[:3]]
    return ", ".join(labels) if labels else "obecné soudní řízení"


def _topics_for_queries(topics: list[str]) -> list[str]:
    substantive = [tag for tag in topics if tag not in PROCEDURAL_TOPICS]
    return substantive or topics


def _build_description(
    *,
    case_number: str | None,
    court: str | None,
    decision_date: str | None,
    document_type: str | None,
    topics: list[str],
    about: str,
) -> str:
    head_parts = [court or "Ústavní soud"]
    if case_number:
        head_parts.append(case_number)
    if decision_date:
        head_parts.append(f"ze dne {decision_date}")
    if document_type:
        head_parts.append(f"({document_type})")
    head = " ".join(head_parts)
    topic_label = _topic_labels(topics)
    if about:
        return f"{head}. Témata: {topic_label}. {about}"
    return f"{head}. Témata: {topic_label}."


def _natural_short_from_about(about: str) -> str | None:
    if not about or len(about) < 50:
        return None
    text = about
    # Weak procedural-only cues are useless as search queries.
    if re.search(r"(?i)záhlaví\s+(?:označen|uveden|citovan)", text) and not re.search(
        r"(?i)nezletil|výživn|vazb|exekuc|azyl|nájem|pracovn|daň|únos|občanst",
        text,
    ):
        return None
    text = re.sub(r"(?i)^(?:se\s+)?domáhá(?:\s+se)?\s+zrušení\s+", "spor o zrušení ", text)
    text = re.sub(r"(?i)^ve věci\s+", "", text)
    text = re.sub(r"(?i)^v řízení o\s+", "řízení o ", text)
    text = re.sub(r"ECLI:[^\s]+", "", text)
    text = re.sub(r"\b\d+\.\s*ÚS\s+\d+/\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[IVX]+\.?\s*ÚS\s+\d+/\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"č\.\s*j\.[^,]{0,50}", "", text, flags=re.IGNORECASE)
    text = _clean(text)
    if len(text) < 35 or SENATE_JUNK_RE.search(text):
        return None
    if len(text) > 110:
        text = text[:109].rsplit(" ", 1)[0]
    return text


def _build_queries(
    *,
    court: str | None,
    topics: list[str],
    about: str,
    ecli: str,
) -> list[str]:
    """Build 2 natural queries.

    q1 = how a layperson typically searches (topic-oriented)
    q2 = longer narrative with case-specific cues (better for self-retrieval smoke)
    """
    query_topics = _topics_for_queries(topics)
    topic_short = ""
    topic_long = ""
    if query_topics:
        primary = query_topics[0]
        topic_short, topic_long = TOPIC_QUERY_HINTS.get(
            primary,
            (
                "podobné soudní rozhodnutí",
                "Hledám podobné soudní rozhodnutí k mé právní situaci.",
            ),
        )
        for extra in query_topics[1:4]:
            if extra == "rusko_rf" and "Rusko" not in topic_short:
                topic_short = f"{topic_short} Rusko"
            elif extra in {"vycestovani_zahranici", "pece_o_nezletile"} and extra != primary:
                extra_short, _ = TOPIC_QUERY_HINTS[extra]
                if extra_short.lower() not in topic_short.lower() and len(topic_short) < 55:
                    topic_short = f"{topic_short} {extra_short}"
                    break

    about_short = _natural_short_from_about(about)
    distinctive = _distinctive_query_cues(about, topics=query_topics)
    cue = distinctive or about_short

    kind = _ns_kind(ecli)
    if topic_short:
        short = topic_short
    elif cue:
        short = cue
    elif kind and kind in NS_KIND_FALLBACK:
        short = NS_KIND_FALLBACK[kind][1]
    elif court and "Nejvyšší" in court:
        short = "dovolání Nejvyšší soud podobný případ"
    else:
        short = "ústavní stížnost podobný případ"

    if topic_long and cue:
        long = f"{topic_long} Konkrétně situace jako: {cue}."
    elif cue:
        long = f"Hledám podobná soudní rozhodnutí k této situaci: {cue}."
    elif topic_long:
        long = topic_long
    elif kind and kind in NS_KIND_FALLBACK:
        long = NS_KIND_FALLBACK[kind][2]
    elif court and "Nejvyšší" in court:
        long = "Hledám podobná rozhodnutí Nejvyššího soudu k mé situaci."
    else:
        long = "Mám spor před Ústavním soudem a hledám podobná rozhodnutí ke své situaci."

    return [_clean(short)[:180], _clean(long)[:420]]


def _distinctive_query_cues(about: str, *, topics: list[str]) -> str | None:
    """Extract a short distinctive cue suitable for self-retrieval smoke tests."""
    if not about:
        return None
    text = about
    text = re.sub(r"ECLI:[^\s]+", "", text)
    text = re.sub(r"č\.\s*j\.[^,]{0,50}", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[IVX]+\.?\s*ÚS\s+\d+/\d+(?:\s*#\d+)?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?i)\b(?:záhlaví\s+(?:označen|uveden|citovan)\w*)\b", "", text)
    text = _clean(text)
    if len(text) < 45:
        return None
    if re.search(r"(?i)záhlaví|formální náležitosti|soudce zpravodaj", text):
        return None

    # Keep noun-heavy factual fragments.
    lowered = text.lower()
    keep_markers = (
        "nezletil",
        "dítě",
        "matk",
        "otec",
        "výživn",
        "vazb",
        "exekuc",
        "azyl",
        "občanst",
        "rus",
        "nájem",
        "byt",
        "pracov",
        "daň",
        "únos",
        "vycestov",
        "péč",
        "styk",
        "škod",
        "obhajob",
        "insolven",
        "vydání",
        "uznání",
    )
    if topics and not any(marker in lowered for marker in keep_markers):
        # Still allow if about mentions a concrete court/outcome beyond boilerplate.
        if not re.search(r"(?i)(okresní|krajský|vrchní|správní|náhrad|zrušení rozhodnutí)", text):
            return None

    # Soft rewrite into a search-like phrase.
    text = re.sub(r"(?i)^(?:se\s+)?domáhá(?:\s+se)?\s+zrušení\s+", "", text)
    text = re.sub(r"(?i)^ve věci\s+", "", text)
    text = _clean(text)
    if len(text) < 40:
        return None
    if len(text) > 120:
        text = text[:119].rsplit(" ", 1)[0]
    return text


def _case_number_from_ecli(ecli: str) -> str | None:
    match = ECLI_US_RE.match(ecli)
    if match:
        senate = match.group("senate")
        num = match.group("num")
        year2 = match.group("year2")
        if senate.lower() == "pl":
            return f"Pl.ÚS {num}/{year2}"
        prefix = SENATE_MAP.get(senate, f"{senate}.")
        return f"{prefix}ÚS {num}/{year2}"
    match = ECLI_NS_RE.match(ecli)
    if match:
        kind = match.group("kind").upper()
        label = NS_KIND_CASE_LABEL.get(kind, kind.capitalize())
        return f"{match.group('senate')} {label} {match.group('num')}/{match.group('year2')}"
    return None


def _infer_case_number(meta: dict[str, Any], chunks: list[dict[str, str]], ecli: str) -> str | None:
    direct = _optional(meta.get("case_reference") or meta.get("case_number"))
    if direct:
        return direct
    for chunk in chunks[:8]:
        match = HEADER_CASE_RE.search(chunk.get("text") or "")
        if match:
            return _clean(next(g for g in match.groups() if g))
    return _case_number_from_ecli(ecli)


def _infer_court(meta: dict[str, Any], ecli: str) -> str:
    court = _optional(meta.get("court") or meta.get("court_name"))
    if court:
        return court
    if ":NS:" in ecli.upper():
        return "Nejvyšší soud"
    return "Ústavní soud"


def build_inventory(bm25_path: Path) -> list[JudgmentInventoryItem]:
    by_doc = _load_rows(bm25_path)
    items: list[JudgmentInventoryItem] = []
    for ecli in sorted(by_doc.keys()):
        entry = by_doc[ecli]
        meta = entry["meta"]
        chunks = entry["chunks"]
        sentences = _iter_sentences(chunks)
        full_blob = "\n".join(chunk["text"] for chunk in chunks)
        topics = _detect_topics(full_blob)
        topics = _ensure_topics(topics, ecli)
        about = _extract_about(sentences, full_blob)
        case_number = _infer_case_number(meta, chunks, ecli)
        court = _infer_court(meta, ecli)
        decision_date = _optional(meta.get("decision_date") or meta.get("date"))
        document_type = _optional(meta.get("document_type") or meta.get("decision_type"))
        description = _build_description(
            case_number=case_number,
            court=court,
            decision_date=decision_date,
            document_type=document_type,
            topics=topics,
            about=about,
        )
        queries = _build_queries(
            court=court,
            topics=topics,
            about=about,
            ecli=ecli,
        )
        excerpts = [s[:280] for s in sentences[:2]]
        items.append(
            JudgmentInventoryItem(
                ecli=ecli,
                canonical_document_id=ecli,
                case_number=case_number,
                court=court,
                decision_date=decision_date,
                document_type=document_type,
                chunk_count=len(chunks),
                topic_tags=topics,
                description=description,
                search_queries=queries,
                evidence_excerpts=excerpts,
            )
        )
    return items


def write_json(path: Path, items: list[JudgmentInventoryItem], *, bm25_path: Path) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "scripts/legal_v2/build_pilot_600_judgment_inventory.py",
        "bm25_path": str(bm25_path),
        "bm25_index_id": "nalus_legal_paragraph_bm25_v2_pilot_600",
        "qdrant_collection": "nalus_legal_paragraph_chunks_v2_pilot_600",
        "document_count": len(items),
        "notes": [
            "Descriptions and search queries are deterministic extractions/heuristics, not LLM-authored legal advice.",
            "search_queries intentionally omit ECLI and case numbers to mimic real user search.",
            "Use this file as an input seed for Stage 1 retrieval smoke / future golden expansion.",
        ],
        "documents": [asdict(item) for item in items],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, items: list[JudgmentInventoryItem]) -> None:
    lines: list[str] = [
        "# Pilot 600 — inventář rozsudků",
        "",
        f"Počet dokumentů: **{len(items)}**",
        "",
        "Zdroj: BM25 sidecar `nalus_legal_paragraph_bm25_v2_pilot_600`.",
        "",
        "Každý záznam obsahuje krátký popis (z metadat + úvodních pasáží) a **2 přirozené vyhledávací otázky**.",
        "",
        "Poznámka: popisy a otázky jsou offline heuristiky (bez placeného LLM), vhodné jako seed pro testování Stage 1.",
        "",
        "---",
        "",
    ]
    for index, item in enumerate(items, start=1):
        topic_label = _topic_labels(item.topic_tags)
        lines.extend(
            [
                f"## {index}. {item.case_number or item.ecli}",
                "",
                f"- **ECLI:** `{item.ecli}`",
                f"- **Soud:** {item.court or '—'}",
                f"- **Datum:** {item.decision_date or '—'}",
                f"- **Typ:** {item.document_type or '—'}",
                f"- **Chunky:** {item.chunk_count}",
                f"- **Témata:** {topic_label}",
                "",
                "### Popis",
                "",
                item.description,
                "",
                "### Jak by to člověk hledal",
                "",
                f"1. `{item.search_queries[0]}`",
                f"2. {item.search_queries[1]}",
                "",
                "---",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_queries_jsonl(path: Path, items: list[JudgmentInventoryItem]) -> None:
    """Flat eval seed: one row per search query."""
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            for query_index, query in enumerate(item.search_queries, start=1):
                row = {
                    "query_id": f"{item.ecli}::q{query_index}",
                    "ecli": item.ecli,
                    "case_number": item.case_number,
                    "query_index": query_index,
                    "query": query,
                    "topic_tags": item.topic_tags,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bm25-path", type=Path, default=DEFAULT_BM25)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    bm25_path = args.bm25_path
    if not bm25_path.exists():
        raise SystemExit(f"BM25 sidecar not found: {bm25_path}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    items = build_inventory(bm25_path)
    if not items:
        raise SystemExit("No documents found in BM25 sidecar")

    json_path = output_dir / "pilot_600_judgment_inventory.json"
    md_path = output_dir / "pilot_600_judgment_inventory.md"
    queries_path = output_dir / "pilot_600_search_queries.jsonl"
    write_json(json_path, items, bm25_path=bm25_path)
    write_markdown(md_path, items)
    write_queries_jsonl(queries_path, items)

    topic_counts: dict[str, int] = defaultdict(int)
    missing_case = 0
    for item in items:
        if not item.case_number:
            missing_case += 1
        if not item.topic_tags:
            topic_counts["(untagged)"] += 1
        for tag in item.topic_tags:
            topic_counts[tag] += 1

    print(f"documents={len(items)}")
    print(f"missing_case_number={missing_case}")
    print(f"json={json_path}")
    print(f"md={md_path}")
    print(f"queries_jsonl={queries_path}")
    print("top_topics=")
    for tag, count in sorted(topic_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]:
        print(f"  {tag}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
