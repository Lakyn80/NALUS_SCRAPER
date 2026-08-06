from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any

from app.rag.legal_v2.models import normalize_legal_text


class QueryIntent(str, Enum):
    LEGAL_DOCUMENT_SEARCH = "legal_document_search"
    FACT_PATTERN_MATCH = "fact_pattern_match"
    LEGAL_PROVISION_SEARCH = "legal_provision_search"
    CASE_CITATION_SEARCH = "case_citation_search"
    UNKNOWN = "unknown"


class ConstraintPolarity(str, Enum):
    HARD = "hard"
    SOFT = "soft"
    NEGATIVE = "negative"


class ConstraintCategory(str, Enum):
    ENTITY = "entity"
    EVENT = "event"
    RELATION = "relation"
    LOCATION = "location"
    DATE_RANGE = "date_range"
    DURATION = "duration"
    LEGAL_PROVISION = "legal_provision"
    COURT = "court"
    DOCUMENT_TYPE = "document_type"
    PROCEDURAL_POSTURE = "procedural_posture"
    DECISION_OUTCOME = "decision_outcome"
    NEGATION = "negation"
    MODALITY = "modality"
    SOURCE_OF_CLAIM = "source_of_claim"
    CITED_CASE = "cited_case"
    CURRENT_CASE = "current_case"


@dataclass(frozen=True)
class QueryEntity:
    entity_id: str
    text: str
    normalized_text: str
    entity_type: str = "generic"
    role: str | None = None
    source_start: int | None = None
    source_end: int | None = None


@dataclass(frozen=True)
class QueryEvent:
    event_id: str
    event_type: str
    trigger_text: str
    normalized_trigger: str
    actor_entity_id: str | None = None
    object_entity_id: str | None = None
    location_entity_ids: list[str] = field(default_factory=list)
    modality: str | None = None
    negated: bool = False


@dataclass(frozen=True)
class QueryRelation:
    relation_id: str
    subject_entity_id: str
    action: str
    object_entity_id: str | None = None
    qualifiers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryDateRange:
    date_range_id: str
    start: str | None = None
    end: str | None = None
    original_text: str | None = None


@dataclass(frozen=True)
class QueryConstraint:
    constraint_id: str
    category: ConstraintCategory
    value: str
    normalized_value: str
    polarity: ConstraintPolarity
    attribute: str | None = None
    source_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class QuerySpecV2:
    original_query: str
    normalized_query: str
    structured_query: dict[str, Any]
    retrieval_queries: list[str]
    intent: QueryIntent
    entities: list[QueryEntity] = field(default_factory=list)
    events: list[QueryEvent] = field(default_factory=list)
    relations: list[QueryRelation] = field(default_factory=list)
    locations: list[QueryEntity] = field(default_factory=list)
    origin: QueryEntity | None = None
    destination: QueryEntity | None = None
    movement_direction: str | None = None
    date_ranges: list[QueryDateRange] = field(default_factory=list)
    durations: list[str] = field(default_factory=list)
    legal_provisions: list[str] = field(default_factory=list)
    courts: list[str] = field(default_factory=list)
    document_types: list[str] = field(default_factory=list)
    procedural_posture: list[str] = field(default_factory=list)
    decision_outcome: list[str] = field(default_factory=list)
    negations: list[str] = field(default_factory=list)
    modalities: list[str] = field(default_factory=list)
    source_of_claims: list[str] = field(default_factory=list)
    cited_cases: list[str] = field(default_factory=list)
    current_case_identifiers: list[str] = field(default_factory=list)
    hard_constraints: list[QueryConstraint] = field(default_factory=list)
    soft_constraints: list[QueryConstraint] = field(default_factory=list)
    negative_constraints: list[QueryConstraint] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    requires_verification: bool = True

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QuerySpecV2":
        return cls(
            original_query=str(payload.get("original_query") or ""),
            normalized_query=str(payload.get("normalized_query") or ""),
            structured_query=dict(payload.get("structured_query") or {}),
            retrieval_queries=[str(item) for item in payload.get("retrieval_queries") or []],
            intent=_intent_from_value(payload.get("intent")),
            entities=[_entity_from_dict(item) for item in payload.get("entities") or []],
            events=[_event_from_dict(item) for item in payload.get("events") or []],
            relations=[
                _relation_from_dict(item) for item in payload.get("relations") or []
            ],
            locations=[
                _entity_from_dict(item) for item in payload.get("locations") or []
            ],
            origin=_optional_entity(payload.get("origin")),
            destination=_optional_entity(payload.get("destination")),
            movement_direction=_optional_str(payload.get("movement_direction")),
            date_ranges=[
                _date_range_from_dict(item)
                for item in payload.get("date_ranges") or []
            ],
            durations=[str(item) for item in payload.get("durations") or []],
            legal_provisions=[
                str(item) for item in payload.get("legal_provisions") or []
            ],
            courts=[str(item) for item in payload.get("courts") or []],
            document_types=[
                str(item) for item in payload.get("document_types") or []
            ],
            procedural_posture=[
                str(item) for item in payload.get("procedural_posture") or []
            ],
            decision_outcome=[
                str(item) for item in payload.get("decision_outcome") or []
            ],
            negations=[str(item) for item in payload.get("negations") or []],
            modalities=[str(item) for item in payload.get("modalities") or []],
            source_of_claims=[
                str(item) for item in payload.get("source_of_claims") or []
            ],
            cited_cases=[str(item) for item in payload.get("cited_cases") or []],
            current_case_identifiers=[
                str(item) for item in payload.get("current_case_identifiers") or []
            ],
            hard_constraints=[
                _constraint_from_dict(item)
                for item in payload.get("hard_constraints") or []
            ],
            soft_constraints=[
                _constraint_from_dict(item)
                for item in payload.get("soft_constraints") or []
            ],
            negative_constraints=[
                _constraint_from_dict(item)
                for item in payload.get("negative_constraints") or []
            ],
            ambiguities=[str(item) for item in payload.get("ambiguities") or []],
            requires_verification=bool(payload.get("requires_verification", True)),
        )

    def all_constraints(self) -> list[QueryConstraint]:
        return [
            *self.hard_constraints,
            *self.soft_constraints,
            *self.negative_constraints,
        ]


_CASE_ID_RE = re.compile(r"\b(?:[IVXLCDM]+\.\s*)?ÚS\s*\d+/\d+\b", re.IGNORECASE)
_LEGAL_PROVISION_RE = re.compile(
    r"(?:§\s*\d+[a-zA-Z]*(?:\s*odst\.\s*\d+)?|čl\.\s*\d+[a-zA-Z]*)",
    re.IGNORECASE,
)
_LOCATION_WORD = r"[A-Za-zÁ-ž][A-Za-zÁ-ž.-]*"
_ORIGIN_DESTINATION_RE = re.compile(
    rf"\b(?:z|ze)\s+(?P<origin>{_LOCATION_WORD}(?:\s+{_LOCATION_WORD}){{0,2}}?)"
    rf"\s+do\s+(?P<destination>{_LOCATION_WORD}(?:\s+{_LOCATION_WORD}){{0,2}}?)"
    r"(?=\s|$|[,.])",
    re.IGNORECASE,
)
_DATE_RE = re.compile(r"\b\d{1,2}\.\s*\d{1,2}\.\s*\d{4}\b")
_LEGAL_CONCEPT_RULES: tuple[dict[str, Any], ...] = (
    {
        "name": "international_child_removal",
        "label": "mezinárodní únos dítěte",
        "patterns": (
            "unos ditete",
            "unesla dite",
            "unesl",
            "unos",
            "neopravnene premisteni ditete",
            "premistila dite",
            "odvezla dite",
            "haagsk",
            "obvykle bydliste",
        ),
        "expansions": ("mezinárodní únos dítěte", "neoprávněné přemístění dítěte", "obvyklé bydliště dítěte", "Haagská úmluva"),
    },
    {
        "name": "domestic_custody",
        "label": "péče o nezletilé dítě",
        "patterns": (
            "pece o",
            "peci o",
            "o peci",
            "spor o peci",
            "zasah do pece",
            "styk",
            "opatrovnick",
            "svereni nezletileho",
            "kontakt s ditetem",
        ),
        "expansions": ("péče o nezletilé dítě", "úprava styku rodiče s dítětem", "opatrovnické řízení"),
        "merits_expansions": ("úprava styku rodiče s dítětem", "opatrovnické řízení", "meritorní spor o péči"),
    },
    {
        "name": "mandatory_lawyer_representation",
        "label": "povinné zastoupení advokátem",
        "patterns": (
            "bez advokata",
            "nemel jsem advokata",
            "nemela jsem advokata",
            "chybelo zastoupeni",
            "chybelo zastoupeni advokatem",
            "nebyl zastoupen advokatem",
            "nebyla zastoupena advokatem",
            "povinne pravni zastoupeni",
            "povinne zastoupeni advokatem",
            "zastoupeni, takze",
        ),
        "expansions": (
            "povinné zastoupení advokátem",
            "chybějící advokát u ústavní stížnosti",
            "nezastoupený stěžovatel",
        ),
    },
    {
        "name": "defective_filing",
        "label": "vadné podání",
        "patterns": (
            "vadne podani",
            "vadnych stiznosti",
            "vadne stiznosti",
            "formalni vady",
            "formalnich vadach",
            "nesplnovalo nalezitosti",
            "nalezitosti podani",
            "chybi oduvodneni",
            "podani mi chybi",
            "neprojednateln",
            "padne uz na formalnich",
            "padl na nalezitostech",
            "odmitnuto pro vady",
            "odmitnuti pro vady",
            "odmitnuto pro formalni",
            "pro vady",
        ),
        "expansions": (
            "vadné podání",
            "formální vady podání",
            "nesplnění náležitostí ústavní stížnosti",
        ),
    },
    {
        "name": "missing_or_inadequate_reasoning",
        "label": "nedostatečné odůvodnění",
        "patterns": (
            "chybi oduvodneni",
            "chybelo oduvodneni",
            "chybi poradné oduvodneni",
            "bez radneho oduvodneni",
            "nedostatecne oduvodneni",
            "nedostatecne vysvetlil",
            "neuvedl jaka prava",
            "chaotick",
        ),
        "expansions": (
            "nedostatečné odůvodnění ústavní stížnosti",
            "chybějící odůvodnění porušení práv",
        ),
    },
    {
        "name": "failure_to_cure_filing_defects",
        "label": "neodstranění vad podání",
        "patterns": (
            "vady nebyly odstraneny",
            "nedoplnil podani",
            "neodstranil vady",
            "neodstraneni vad",
            "na vyzvu nereagoval",
            "nereagoval na vyzvu",
        ),
        "expansions": (
            "neodstranění vad podání",
            "nedoplnění podání ve lhůtě",
        ),
    },
    {
        "name": "maintenance",
        "label": "výživné",
        "patterns": ("vyzivne", "vyzivovaci povinnost", "oduvodnene potreby"),
        "expansions": ("výživné", "vyživovací povinnost", "odůvodněné potřeby dítěte"),
    },
    {
        "name": "paternity",
        "label": "určení otcovství",
        "patterns": ("otcovstvi", "popreni otcovstvi", "urceni otcovstvi"),
        "expansions": ("určení otcovství", "popření otcovství", "statusové řízení"),
    },
    {
        "name": "citizenship",
        "label": "státní občanství",
        "patterns": ("obcanstvi", "statni obcanstvi", "ceske obcanstvi", "zadost o obcanstvi"),
        "expansions": ("státní občanství České republiky", "žádost o české občanství", "správní uvážení při udělení občanství"),
    },
    {
        "name": "service_of_documents",
        "label": "doručování soudních písemností",
        "patterns": ("doruc", "nedorucil rozsudek", "fikce doruceni", "soudni pisemnost"),
        "expansions": ("doručování soudního rozhodnutí", "vadné doručení", "fikce doručení"),
    },
    {
        "name": "restoration_of_deadline",
        "label": "zmeškání procesní lhůty",
        "patterns": ("zmeskal", "zmeskani lhuty", "navraceni lhuty", "prominuti zmeskani"),
        "expansions": ("zmeškání lhůty", "navrácení lhůty", "prominutí zmeškání lhůty", "právo na přístup k soudu"),
    },
    {
        "name": "civil_procedure",
        "label": "civilní řízení",
        "patterns": ("o. s. r", "obcansky soudni rad", "civilni", "pripustnost dovolani", "dovolani"),
        "expansions": ("civilní řízení", "přípustnost dovolání", "§ 237 o. s. ř.", "občanský soudní řád"),
    },
    {
        "name": "criminal_procedure",
        "label": "trestní řízení",
        "patterns": ("trestni", "obvinen", "obhajob", "trestni dovolani"),
        "expansions": ("trestní řízení", "právo na obhajobu", "obviněný", "trestní dovolání"),
    },
    {
        "name": "constitutional_admissibility",
        "label": "přípustnost ústavní stížnosti",
        "patterns": ("ustavni stiznost", "nepripustn", "vycerpani procesnich prostredku", "odmitl ustavni stiznost"),
        "expansions": ("přípustnost ústavní stížnosti", "vyčerpání procesních prostředků", "odmítnutí ústavní stížnosti"),
    },
    {
        "name": "fair_trial",
        "label": "právo na spravedlivý proces",
        "patterns": ("spravedlivy proces", "extremni nesoulad", "dokazovani", "oduvodneni rozhodnuti"),
        "expansions": ("právo na spravedlivý proces", "dokazování", "odůvodnění rozhodnutí", "extrémní nesoulad"),
    },
    {
        "name": "omitted_evidence",
        "label": "opomenutý důkaz",
        "patterns": ("opomenut", "neprovedl dukaz", "opomenute dukazy"),
        "expansions": ("opomenutý důkaz", "neprovedení navrženého důkazu", "právo na spravedlivý proces"),
    },
    {
        "name": "property",
        "label": "vlastnické právo",
        "patterns": ("vlastnick", "nemovit", "pozem"),
        "expansions": ("vlastnické právo", "nemovitá věc", "pozemek"),
    },
    {
        "name": "contract",
        "label": "smluvní spor",
        "patterns": (
            "smlouv",
            "platnost smlouvy",
            "neplatnost smlouvy",
            "platnost pravniho jednani",
        ),
        "expansions": ("smlouva", "neplatnost smlouvy", "platnost právního jednání"),
        "merits_expansions": ("neplatnost smlouvy", "platnost právního jednání"),
    },
    {
        "name": "damages",
        "label": "náhrada škody",
        "patterns": (
            "nahrada skody",
            "vyse skody",
            "vysi skody",
            "odpovednost za skodu",
            "pricinna souvislost",
        ),
        "expansions": ("náhrada škody", "odpovědnost za škodu", "příčinná souvislost"),
        "merits_expansions": ("náhrada škody", "výše škody", "odpovědnost za škodu"),
    },
    {
        "name": "employment",
        "label": "pracovní právo",
        "patterns": ("pracovni", "zamestnan", "vypoved"),
        "expansions": ("pracovní právo", "zaměstnanec", "výpověď z pracovního poměru"),
    },
    {
        "name": "public_administration",
        "label": "výkon veřejné správy",
        "patterns": ("verejne spravy", "spravni organ", "spravni rozhodnuti", "spravni zaloba"),
        "expansions": ("výkon veřejné správy", "správní orgán", "soudní přezkum správního rozhodnutí"),
    },
    {
        "name": "jurisdiction_competence",
        "label": "pravomoc a příslušnost",
        "patterns": ("pravomoc", "prislusnost", "kompetenc"),
        "expansions": ("pravomoc soudu", "příslušnost soudu", "kompetenční spor"),
    },
    {
        "name": "extraordinary_remedy",
        "label": "mimořádný opravný prostředek",
        "patterns": ("mimoradny opravny prostredek", "obnova rizeni", "dovolani"),
        "expansions": ("mimořádný opravný prostředek", "dovolání", "obnova řízení"),
    },
    {
        "name": "right_to_interpreter",
        "label": "právo na tlumočníka",
        "patterns": ("tlumocnik", "tlumoceni", "nerozumi cesky", "cizinec nerozumi"),
        "expansions": ("právo na tlumočníka", "obviněný cizinec", "nerozumí česky", "prohlášení viny"),
    },
    {
        "name": "migration_or_asylum",
        "label": "mezinárodní ochrana a pobyt cizince",
        "patterns": ("mezinarodni ochrany", "azyl", "doplnkova ochrana", "pobyt cizince", "dlouhodobemu pobytu"),
        "expansions": ("mezinárodní ochrana", "azyl", "doplňková ochrana", "dlouhodobý pobyt cizince"),
    },
    {
        "name": "enforcement_proceedings",
        "label": "exekuční řízení",
        "patterns": ("exekuc", "vykon rozhodnuti", "exekucni titul", "odklad exekuce"),
        "expansions": ("exekuční řízení", "odklad exekuce", "exekuční titul", "povinný v exekuci"),
    },
    {
        "name": "court_costs",
        "label": "náklady řízení",
        "patterns": (
            "naklady rizeni",
            "nakladu rizeni",
            "nahrada nakladu",
            "nahradu nakladu",
            "soudni poplatek",
        ),
        "expansions": ("náklady řízení", "náhrada nákladů řízení", "soudní poplatek"),
    },
    {
        "name": "burden_of_proof",
        "label": "důkazní břemeno",
        "patterns": ("dukazni bremeno", "in dubio pro reo", "presumpce neviny"),
        "expansions": ("důkazní břemeno", "in dubio pro reo", "presumpce neviny", "hodnocení důkazů"),
    },
    {
        "name": "limitation_periods",
        "label": "promlčení",
        "patterns": ("promlc", "promlceci", "prekluz"),
        "expansions": ("promlčení", "promlčecí lhůta", "bezdůvodné obohacení"),
    },
    {
        "name": "public_law_sanctions",
        "label": "veřejnoprávní sankce",
        "patterns": ("pokuta", "prestup", "spravni delikt", "sankc"),
        "expansions": ("veřejnoprávní sankce", "pokuta", "přestupek", "správní delikt"),
    },
    {
        "name": "legal_standing",
        "label": "procesní legitimace",
        "patterns": ("aktivni legitimace", "vecna legitimace", "procesni legitimace", "pravni zajem"),
        "expansions": ("aktivní legitimace", "procesní legitimace", "návrh na zrušení zákona"),
    },
    {
        "name": "child_contact",
        "label": "styk rodiče s dítětem",
        "patterns": ("styk s nezletil", "kontakt s ditetem", "uprava styku", "nejlepsi zajem ditete"),
        "expansions": ("styk rodiče s dítětem", "úprava styku", "nejlepší zájem dítěte", "nezletilé dítě"),
    },
    {
        "name": "administrative_procedure",
        "label": "správní řízení",
        "patterns": ("spravni rizeni", "ucastenstvi", "stavebni rizeni", "uzemni rizeni"),
        "expansions": ("správní řízení", "účastenství ve správním řízení", "stavební řízení", "územní řízení"),
    },
    {
        "name": "court_competence",
        "label": "příslušnost soudu a zákonný soudce",
        "patterns": ("zakonny soudce", "vecna prislusnost", "mistni prislusnost", "prislusnost soudu"),
        "expansions": ("právo na zákonného soudce", "věcná příslušnost soudu", "místní příslušnost soudu"),
    },
    {
        "name": "tax_law",
        "label": "daňové řízení",
        "patterns": (
            "danove rizeni",
            "danovy",
            "danove",
            "platebni vymer",
            "financni urad",
            "dph",
            "dane z prijmu",
        ),
        "expansions": ("daňové řízení", "dodatečný platební výměr", "finanční úřad", "správní soudnictví"),
    },
    {
        "name": "validity_of_legal_acts",
        "label": "platnost právního jednání",
        "patterns": ("neplatnost pravniho jednani", "bezpravna vyhruzka", "natlak", "psychicke donuceni"),
        "expansions": ("neplatnost právního jednání", "bezprávná výhrůžka", "psychické donucení", "nátlak"),
    },
    {
        "name": "procedural_default",
        "label": "procesní zmeškání a zastavení řízení",
        "patterns": ("zastaveni dovolaciho rizeni", "nezaplaceni soudniho poplatku", "rozsudek pro zmeskani", "kontumacni rozsudek"),
        "expansions": ("zastavení dovolacího řízení", "nezaplacení soudního poplatku", "rozsudek pro zmeškání"),
    },
)
_CANDIDATE_RETRIEVAL_CONCEPT_NAMES = frozenset(
    {
        "international_child_removal",
        "domestic_custody",
        "maintenance",
        "paternity",
        "citizenship",
        "service_of_documents",
        "restoration_of_deadline",
        "civil_procedure",
        "criminal_procedure",
        "constitutional_admissibility",
        "fair_trial",
        "omitted_evidence",
        "property",
        "contract",
        "damages",
        "employment",
        "public_administration",
        "jurisdiction_competence",
        "extraordinary_remedy",
        "court_costs",
        "limitation_periods",
        "child_contact",
        "mandatory_lawyer_representation",
        "defective_filing",
        "missing_or_inadequate_reasoning",
        "failure_to_cure_filing_defects",
    }
)

_PROCEDURAL_PRIORITY_CONCEPT_NAMES = frozenset(
    {
        "constitutional_admissibility",
        "mandatory_lawyer_representation",
        "defective_filing",
        "missing_or_inadequate_reasoning",
        "failure_to_cure_filing_defects",
        "service_of_documents",
        "restoration_of_deadline",
        "extraordinary_remedy",
        "court_costs",
        "limitation_periods",
        "civil_procedure",
        "criminal_procedure",
    }
)

_MAX_RETRIEVAL_QUERIES = 8

# Folded-text hints used to bind a negated span to concept names.
_NEGATABLE_CONCEPT_HINTS: dict[str, tuple[str, ...]] = {
    "domestic_custody": (
        "meritorni spor o peci",
        "spor o peci",
        "peci o dite",
        "pece o dite",
        "pece o deti",
        "uprava styku",
        "opatrovnick",
        "styk rodice",
        "rozhodovani o peci",
    ),
    "child_contact": (
        "uprava styku",
        "styk rodice",
        "kontakt s ditetem",
        "styk s ditetem",
    ),
    "damages": (
        "vyse skody",
        "nahrada skody",
        "skody",
        "skoda",
    ),
    "contract": (
        "platnost smlouvy",
        "neplatnost smlouvy",
        "smlouvy",
        "smlouva",
    ),
    "criminal_guilt": (
        "obzalovany vinen",
        "byl vinen",
        "byla vinna",
        "vina",
        "vinen",
        "vinu",
    ),
}

_NEGATABLE_CONCEPT_LABELS: dict[str, str] = {
    "child_custody_merits": "meritorní spor o péči o dítě",
    "parent_contact_merits": "meritorní úprava styku s dítětem",
    "criminal_guilt": "meritorní posouzení viny",
}

_SCOPED_NEGATION_PREFIX_RE = re.compile(
    r"(?:"
    r"nehledam|"
    r"nechci|"
    r"neresim|"
    r"nejde\s+mi\s+o|"
    r"nejde\s+o|"
    r"nikoli|"
    r"ne\s+"
    r"),?\s+(?P<body>.{3,120}?)"
    r"(?=("
    r",?\s+ale\b|"
    r",?\s+nybrz\b|"
    r",?\s+nýbrž\b|"
    r"\s+spis\b|"
    r"\s+nez\b|"
    r"$|[.!?]"
    r"))",
    re.IGNORECASE,
)

# Explicit requested-issue focus: "hledám jen/pouze Y" demotes background domains.
_REQUESTED_FOCUS_RE = re.compile(
    r"(?:ale\s+)?hledam\s+(?:jen|pouze|ted|nyni)\s+(?P<body>.{3,160}?)(?=$|[.!?])",
    re.IGNORECASE,
)


def build_query_spec_v2(original_query: str) -> QuerySpecV2:
    normalized_original = normalize_legal_text(original_query)
    normalized_query = normalize_legal_text(original_query).lower()
    folded_query = _fold_text(normalized_query)

    entities: list[QueryEntity] = []
    locations: list[QueryEntity] = []
    events: list[QueryEvent] = []
    relations: list[QueryRelation] = []
    date_ranges: list[QueryDateRange] = []
    hard_constraints: list[QueryConstraint] = []
    soft_constraints: list[QueryConstraint] = []
    negative_constraints: list[QueryConstraint] = []

    origin: QueryEntity | None = None
    destination: QueryEntity | None = None
    movement_direction: str | None = None
    legal_provisions = _dedupe(_LEGAL_PROVISION_RE.findall(normalized_original))
    courts = _extract_courts(folded_query)
    document_types = _extract_document_types(folded_query)
    procedural_posture = _extract_procedural_posture(folded_query)
    decision_outcome = _extract_decision_outcome(folded_query)
    negations = _extract_negations(folded_query)
    modalities = _extract_modalities(folded_query)
    source_of_claims = _extract_source_of_claims(folded_query)
    durations = _extract_durations(folded_query)
    cited_cases, current_cases = _extract_case_references(
        normalized_original=normalized_original,
        folded_query=folded_query,
    )
    legal_concepts = _extract_legal_concepts(folded_query)
    scoped_negative_concepts = _extract_scoped_negative_concepts(folded_query)
    negated_requested_names = {item["name"] for item in scoped_negative_concepts}

    parent_entity = _extract_parent_role(folded_query, normalized_original, entities)
    child_entity = _extract_child_entity(folded_query, normalized_original, entities)

    location_match = _ORIGIN_DESTINATION_RE.search(normalized_original)
    if location_match:
        origin = _add_location_entity(
            locations,
            entities,
            text=_clean_location(location_match.group("origin")),
            role="origin",
        )
        destination = _add_location_entity(
            locations,
            entities,
            text=_clean_location(location_match.group("destination")),
            role="destination",
        )
        movement_direction = "origin_to_destination"
        # Origin/destination are retrieval/ranking hints, not all-or-nothing
        # verification requirements (lay "z X do Y" rarely appears as court_finding).
        _add_constraint(
            soft_constraints,
            ConstraintCategory.LOCATION,
            origin.text,
            ConstraintPolarity.SOFT,
            attribute="origin",
            source_ids=[origin.entity_id],
        )
        _add_constraint(
            soft_constraints,
            ConstraintCategory.LOCATION,
            destination.text,
            ConstraintPolarity.SOFT,
            attribute="destination",
            source_ids=[destination.entity_id],
        )

    action = _extract_action(folded_query)
    if action is not None:
        actor_id = parent_entity.entity_id if parent_entity is not None else None
        object_id = child_entity.entity_id if child_entity is not None else None
        event = QueryEvent(
            event_id=f"event_{len(events):03d}",
            event_type=action,
            trigger_text="unos" if action == "abduction" else action,
            normalized_trigger=action,
            actor_entity_id=actor_id,
            object_entity_id=object_id,
            location_entity_ids=[
                entity.entity_id for entity in (origin, destination) if entity is not None
            ],
            modality=modalities[0] if modalities else None,
            negated=bool(negations),
        )
        events.append(event)
        # Structural fact-pattern slots (actor/object/event/relation) stay soft.
        # Hard bar for abduction-like queries is the legal_concept constraint below.
        _add_constraint(
            soft_constraints,
            ConstraintCategory.EVENT,
            action,
            ConstraintPolarity.SOFT,
            attribute="action",
            source_ids=[event.event_id],
        )
        if parent_entity is not None:
            _add_constraint(
                soft_constraints,
                ConstraintCategory.ENTITY,
                parent_entity.role or parent_entity.normalized_text,
                ConstraintPolarity.SOFT,
                attribute="actor_role",
                source_ids=[parent_entity.entity_id],
            )
        if child_entity is not None:
            _add_constraint(
                soft_constraints,
                ConstraintCategory.ENTITY,
                child_entity.role or child_entity.normalized_text,
                ConstraintPolarity.SOFT,
                attribute="object_role",
                source_ids=[child_entity.entity_id],
            )
        if parent_entity is not None and child_entity is not None:
            relation = QueryRelation(
                relation_id=f"relation_{len(relations):03d}",
                subject_entity_id=parent_entity.entity_id,
                action=action,
                object_entity_id=child_entity.entity_id,
                qualifiers={
                    "movement_direction": movement_direction or "",
                    "origin": origin.normalized_text if origin is not None else "",
                    "destination": (
                        destination.normalized_text if destination is not None else ""
                    ),
                },
            )
            relations.append(relation)
            _add_constraint(
                soft_constraints,
                ConstraintCategory.RELATION,
                f"{parent_entity.role}:{action}:{child_entity.role}",
                ConstraintPolarity.SOFT,
                attribute="actor_action_object",
                source_ids=[relation.relation_id],
            )

    for value in legal_provisions:
        _add_constraint(
            soft_constraints,
            ConstraintCategory.LEGAL_PROVISION,
            value,
            ConstraintPolarity.SOFT,
        )
    candidate_retrieval_concepts = [
        concept
        for concept in legal_concepts
        if _concept_affects_candidate_retrieval(concept)
        and concept["name"] not in negated_requested_names
    ]
    focus_demoted_names = _focus_demoted_concept_names(
        folded_query=folded_query,
        candidate_concepts=candidate_retrieval_concepts,
    )
    if focus_demoted_names:
        candidate_retrieval_concepts = [
            concept
            for concept in candidate_retrieval_concepts
            if concept["name"] not in focus_demoted_names
        ]
    contextual_concepts = [
        concept
        for concept in legal_concepts
        if concept["name"] in negated_requested_names
        or concept["name"] in focus_demoted_names
        or (
            concept["name"] in {"domestic_custody", "child_contact"}
            and concept["name"] not in {item["name"] for item in candidate_retrieval_concepts}
            and any(token in folded_query for token in ("pece", "peci", "dite", "deti", "ospod", "social"))
        )
    ]
    # Prefer procedural / requested-issue concepts ahead of background domain.
    candidate_retrieval_concepts = _prioritize_retrieval_concepts(candidate_retrieval_concepts)
    for concept in candidate_retrieval_concepts:
        _add_constraint(
            hard_constraints,
            ConstraintCategory.LEGAL_PROVISION,
            _concept_constraint_value(concept),
            ConstraintPolarity.HARD,
            attribute=f"legal_concept:{concept['name']}",
        )
    for item in scoped_negative_concepts:
        _add_constraint(
            negative_constraints,
            ConstraintCategory.LEGAL_PROVISION,
            item["label"],
            ConstraintPolarity.NEGATIVE,
            attribute=f"legal_concept:{item['name']}",
        )
    for value in courts:
        _add_constraint(soft_constraints, ConstraintCategory.COURT, value, ConstraintPolarity.SOFT)
    for value in document_types:
        _add_constraint(
            soft_constraints,
            ConstraintCategory.DOCUMENT_TYPE,
            value,
            ConstraintPolarity.SOFT,
        )
    for value in procedural_posture:
        _add_constraint(
            soft_constraints,
            ConstraintCategory.PROCEDURAL_POSTURE,
            value,
            ConstraintPolarity.SOFT,
        )
    for value in decision_outcome:
        _add_constraint(
            soft_constraints,
            ConstraintCategory.DECISION_OUTCOME,
            value,
            ConstraintPolarity.SOFT,
        )
    for value in negations:
        _add_constraint(
            hard_constraints,
            ConstraintCategory.NEGATION,
            value,
            ConstraintPolarity.HARD,
        )
    for value in modalities:
        _add_constraint(
            soft_constraints,
            ConstraintCategory.MODALITY,
            value,
            ConstraintPolarity.SOFT,
        )
    for value in source_of_claims:
        _add_constraint(
            hard_constraints,
            ConstraintCategory.SOURCE_OF_CLAIM,
            value,
            ConstraintPolarity.HARD,
        )
    for value in cited_cases:
        _add_constraint(
            hard_constraints,
            ConstraintCategory.CITED_CASE,
            value,
            ConstraintPolarity.HARD,
        )
    for value in current_cases:
        _add_constraint(
            hard_constraints,
            ConstraintCategory.CURRENT_CASE,
            value,
            ConstraintPolarity.HARD,
        )

    for index, value in enumerate(_DATE_RE.findall(normalized_original)):
        date_range = QueryDateRange(
            date_range_id=f"date_{index:03d}",
            start=value,
            end=value,
            original_text=value,
        )
        date_ranges.append(date_range)
        _add_constraint(
            hard_constraints,
            ConstraintCategory.DATE_RANGE,
            value,
            ConstraintPolarity.HARD,
            source_ids=[date_range.date_range_id],
        )

    ambiguities = _extract_ambiguities(
        folded_query=folded_query,
        hard_constraints=hard_constraints,
        legal_concepts=legal_concepts,
        origin=origin,
        destination=destination,
    )
    retrieval_queries, suppressed_expansions = _build_retrieval_queries(
        original_query=original_query,
        normalized_query=normalized_query,
        origin=origin,
        destination=destination,
        parent_entity=parent_entity,
        child_entity=child_entity,
        action=action,
        legal_concepts=candidate_retrieval_concepts,
        contextual_concepts=contextual_concepts,
        negated_concept_names=negated_requested_names,
        procedural_posture=procedural_posture,
        decision_outcome=decision_outcome,
    )
    intent = _infer_intent(
        action=action,
        legal_provisions=legal_provisions,
        cited_cases=cited_cases,
        current_cases=current_cases,
        legal_concepts=legal_concepts,
    )
    structured_query = {
        "schema": "legal_query_spec_v2",
        "entity_count": len(entities),
        "event_count": len(events),
        "relation_count": len(relations),
        "hard_constraint_count": len(hard_constraints),
        "soft_constraint_count": len(soft_constraints),
        "negative_constraint_count": len(negative_constraints),
        "legal_concepts": [
            {"name": concept["name"], "label": concept["label"]}
            for concept in legal_concepts
        ],
        "candidate_retrieval_concepts": [
            {"name": concept["name"], "label": concept["label"]}
            for concept in candidate_retrieval_concepts
        ],
        "contextual_concepts": [
            {"name": concept["name"], "label": concept["label"]}
            for concept in contextual_concepts
        ],
        "negated_requested_concepts": [
            {"name": item["name"], "label": item["label"]}
            for item in scoped_negative_concepts
        ],
        "suppressed_expansions": suppressed_expansions,
    }
    return demote_structural_fact_slot_constraints(
        QuerySpecV2(
            original_query=original_query,
            normalized_query=normalized_query,
            structured_query=structured_query,
            retrieval_queries=retrieval_queries,
            intent=intent,
            entities=entities,
            events=events,
            relations=relations,
            locations=locations,
            origin=origin,
            destination=destination,
            movement_direction=movement_direction,
            date_ranges=date_ranges,
            durations=durations,
            legal_provisions=legal_provisions,
            courts=courts,
            document_types=document_types,
            procedural_posture=procedural_posture,
            decision_outcome=decision_outcome,
            negations=negations,
            modalities=modalities,
            source_of_claims=source_of_claims,
            cited_cases=cited_cases,
            current_case_identifiers=current_cases,
            hard_constraints=hard_constraints,
            soft_constraints=soft_constraints,
            negative_constraints=negative_constraints,
            ambiguities=ambiguities,
            requires_verification=bool(normalized_query),
        )
    )


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    return value


def _entity_from_dict(payload: dict[str, Any]) -> QueryEntity:
    return QueryEntity(
        entity_id=str(payload.get("entity_id") or ""),
        text=str(payload.get("text") or ""),
        normalized_text=str(payload.get("normalized_text") or ""),
        entity_type=str(payload.get("entity_type") or "generic"),
        role=_optional_str(payload.get("role")),
        source_start=_optional_int(payload.get("source_start")),
        source_end=_optional_int(payload.get("source_end")),
    )


def _intent_from_value(value: Any) -> QueryIntent:
    text = str(value or QueryIntent.UNKNOWN.value)
    aliases = {
        "legal_information_retrieval": QueryIntent.LEGAL_DOCUMENT_SEARCH,
        "legal_retrieval": QueryIntent.LEGAL_DOCUMENT_SEARCH,
        "case_law_search": QueryIntent.LEGAL_DOCUMENT_SEARCH,
    }
    if text in aliases:
        return aliases[text]
    try:
        return QueryIntent(text)
    except ValueError:
        return QueryIntent.UNKNOWN


def _event_from_dict(payload: dict[str, Any]) -> QueryEvent:
    return QueryEvent(
        event_id=str(payload.get("event_id") or ""),
        event_type=str(payload.get("event_type") or ""),
        trigger_text=str(payload.get("trigger_text") or ""),
        normalized_trigger=str(payload.get("normalized_trigger") or ""),
        actor_entity_id=_optional_str(payload.get("actor_entity_id")),
        object_entity_id=_optional_str(payload.get("object_entity_id")),
        location_entity_ids=[
            str(item) for item in payload.get("location_entity_ids") or []
        ],
        modality=_optional_str(payload.get("modality")),
        negated=bool(payload.get("negated", False)),
    )


def _relation_from_dict(payload: dict[str, Any]) -> QueryRelation:
    return QueryRelation(
        relation_id=str(payload.get("relation_id") or ""),
        subject_entity_id=str(payload.get("subject_entity_id") or ""),
        action=str(payload.get("action") or ""),
        object_entity_id=_optional_str(payload.get("object_entity_id")),
        qualifiers={str(key): str(value) for key, value in dict(payload.get("qualifiers") or {}).items()},
    )


def _date_range_from_dict(payload: dict[str, Any]) -> QueryDateRange:
    return QueryDateRange(
        date_range_id=str(payload.get("date_range_id") or ""),
        start=_optional_str(payload.get("start")),
        end=_optional_str(payload.get("end")),
        original_text=_optional_str(payload.get("original_text")),
    )


def _constraint_from_dict(payload: dict[str, Any]) -> QueryConstraint:
    value = str(payload.get("value") or "")
    normalized = str(payload.get("normalized_value") or "") or normalize_legal_text(value).lower()
    constraint_id = str(payload.get("constraint_id") or "").strip()
    if not constraint_id:
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
        constraint_id = f"constraint_llm_{digest}"
    return QueryConstraint(
        constraint_id=constraint_id,
        category=_category_from_value(payload.get("category")),
        value=value,
        normalized_value=normalized,
        polarity=_polarity_from_value(payload.get("polarity")),
        attribute=_optional_str(payload.get("attribute")),
        source_ids=[str(item) for item in payload.get("source_ids") or []],
    )


def _category_from_value(value: object) -> ConstraintCategory:
    try:
        return ConstraintCategory(str(value or "").strip().lower())
    except ValueError:
        return ConstraintCategory.ENTITY


def _polarity_from_value(value: object) -> ConstraintPolarity:
    try:
        return ConstraintPolarity(str(value or "").strip().lower())
    except ValueError:
        # Unknown / missing polarity must not silently become a fail-closed hard bar.
        return ConstraintPolarity.SOFT


def _optional_entity(payload: object) -> QueryEntity | None:
    return _entity_from_dict(payload) if isinstance(payload, dict) else None


def _optional_str(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _fold_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(char for char in decomposed if not unicodedata.combining(char)).lower()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = normalize_legal_text(value)
        key = _fold_text(normalized)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _add_location_entity(
    locations: list[QueryEntity],
    entities: list[QueryEntity],
    *,
    text: str,
    role: str,
) -> QueryEntity:
    entity = QueryEntity(
        entity_id=f"entity_{len(entities):03d}",
        text=text,
        normalized_text=normalize_legal_text(text).lower(),
        entity_type="location",
        role=role,
    )
    locations.append(entity)
    entities.append(entity)
    return entity


def _extract_parent_role(
    folded_query: str,
    original_query: str,
    entities: list[QueryEntity],
) -> QueryEntity | None:
    role: str | None = None
    text: str | None = None
    if re.search(r"\bmatk\w*", folded_query):
        role = "mother"
        text = "matka"
    elif re.search(r"\botc\w*", folded_query):
        role = "father"
        text = "otec"
    if role is None or text is None:
        return None
    entity = QueryEntity(
        entity_id=f"entity_{len(entities):03d}",
        text=text,
        normalized_text=text,
        entity_type="person",
        role=role,
        source_start=_fold_text(original_query).find(text),
    )
    entities.append(entity)
    return entity


def _extract_child_entity(
    folded_query: str,
    original_query: str,
    entities: list[QueryEntity],
) -> QueryEntity | None:
    if not re.search(r"\bdit\w*", folded_query):
        return None
    entity = QueryEntity(
        entity_id=f"entity_{len(entities):03d}",
        text="dítě",
        normalized_text="dítě",
        entity_type="person",
        role="child",
        source_start=_fold_text(original_query).find("dit"),
    )
    entities.append(entity)
    return entity


def _extract_action(folded_query: str) -> str | None:
    # Cover noun (únos) and verb forms (unést / unesla / unesl…).
    if re.search(r"\b(unos|unest|unesl\w*)\b", folded_query) or "unos" in folded_query:
        return "abduction"
    if "premist" in folded_query or "prevez" in folded_query or "odvez" in folded_query:
        return "movement"
    if "navracen" in folded_query or "navrat" in folded_query:
        return "return"
    return None


def _extract_legal_concepts(folded_query: str) -> list[dict[str, Any]]:
    concepts: list[dict[str, Any]] = []
    for rule in _LEGAL_CONCEPT_RULES:
        patterns = tuple(str(pattern) for pattern in rule["patterns"])
        if any(pattern in folded_query for pattern in patterns):
            concepts.append(rule)
    return concepts


def _concept_constraint_value(concept: dict[str, Any]) -> str:
    # Keep the human label only — concatenating every expansion made lexical/LLM
    # proof require dozens of tokens and fail-closed on relevant judgments.
    return str(concept["label"])


def _concept_affects_candidate_retrieval(concept: dict[str, Any]) -> bool:
    return str(concept["name"]) in _CANDIDATE_RETRIEVAL_CONCEPT_NAMES


def _add_constraint(
    constraints: list[QueryConstraint],
    category: ConstraintCategory,
    value: str,
    polarity: ConstraintPolarity,
    *,
    attribute: str | None = None,
    source_ids: list[str] | None = None,
) -> None:
    normalized_value = normalize_legal_text(value).lower()
    payload = "|".join(
        [category.value, attribute or "", normalized_value, str(len(constraints))]
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    constraints.append(
        QueryConstraint(
            constraint_id=f"constraint_{digest}",
            category=category,
            value=value,
            normalized_value=normalized_value,
            polarity=polarity,
            attribute=attribute,
            source_ids=list(source_ids or []),
        )
    )


def _clean_location(value: str) -> str:
    cleaned = normalize_legal_text(value).strip(" ,.;:")
    folded = _fold_text(cleaned)
    canonical = _LOCATION_CANONICAL.get(folded)
    return canonical or cleaned


_LOCATION_CANONICAL: dict[str, str] = {
    "ceska": "Česká republika",
    "cesko": "Česká republika",
    "ceskou": "Česká republika",
    "cr": "Česká republika",
    "ceske republiky": "Česká republika",
    "ceska republika": "Česká republika",
    "ruska": "Ruská federace",
    "rusko": "Ruská federace",
    "ruskem": "Ruská federace",
    "ruske federace": "Ruská federace",
    "ruska federace": "Ruská federace",
}


_STRUCTURAL_FACT_SLOT_ATTRIBUTES = frozenset(
    {
        "origin",
        "destination",
        "action",
        "actor_role",
        "object_role",
        "actor_action_object",
    }
)


def is_structural_fact_slot_constraint(constraint: QueryConstraint) -> bool:
    """Surface fact-pattern slots used for retrieval, not all-or-nothing proof.

    Legal-concept / negation / case-id constraints stay eligible as hard.
    """
    attribute = (constraint.attribute or "").strip()
    if attribute.startswith("legal_concept:"):
        return False
    if attribute in _STRUCTURAL_FACT_SLOT_ATTRIBUTES:
        return True
    if constraint.category == ConstraintCategory.LOCATION:
        return True
    if constraint.category in {
        ConstraintCategory.ENTITY,
        ConstraintCategory.EVENT,
        ConstraintCategory.RELATION,
    }:
        return True
    return False


def demote_structural_fact_slot_constraints(spec: QuerySpecV2) -> QuerySpecV2:
    """Move structural fact slots from hard → soft (idempotent)."""
    kept_hard: list[QueryConstraint] = []
    moved_soft: list[QueryConstraint] = []
    for constraint in spec.hard_constraints:
        if is_structural_fact_slot_constraint(constraint):
            moved_soft.append(
                QueryConstraint(
                    constraint_id=constraint.constraint_id,
                    category=constraint.category,
                    value=constraint.value,
                    normalized_value=constraint.normalized_value,
                    polarity=ConstraintPolarity.SOFT,
                    attribute=constraint.attribute,
                    source_ids=list(constraint.source_ids),
                )
            )
        else:
            kept_hard.append(constraint)
    if not moved_soft:
        return spec
    soft = _union_constraint_list(spec.soft_constraints, moved_soft)
    return replace(spec, hard_constraints=kept_hard, soft_constraints=soft)


def _union_constraint_list(
    primary: list[QueryConstraint], extra: list[QueryConstraint]
) -> list[QueryConstraint]:
    seen = {(item.category, item.attribute, item.normalized_value) for item in primary}
    result = list(primary)
    for constraint in extra:
        key = (constraint.category, constraint.attribute, constraint.normalized_value)
        if key in seen:
            continue
        seen.add(key)
        result.append(constraint)
    return result


def _extract_courts(folded_query: str) -> list[str]:
    courts: list[str] = []
    if "ustavni soud" in folded_query:
        courts.append("Ústavní soud")
    if "nejvyssi soud" in folded_query:
        courts.append("Nejvyšší soud")
    if "nejvyssi spravni soud" in folded_query:
        courts.append("Nejvyšší správní soud")
    return courts


def _extract_document_types(folded_query: str) -> list[str]:
    values: list[str] = []
    if "nalezu" in folded_query or "nalez" in folded_query:
        values.append("nález")
    if "usneseni" in folded_query:
        values.append("usnesení")
    if "rozsudek" in folded_query:
        values.append("rozsudek")
    return values


def _extract_procedural_posture(folded_query: str) -> list[str]:
    values: list[str] = []
    constitutional = (
        "ustavni stiznost" in folded_query
        or "ustavnimu soudu" in folded_query
        or "u ustavniho soudu" in folded_query
        or ("stiznost" in folded_query and "ustavn" in folded_query)
        # Mandatory counsel + formal filing defects is characteristic of ÚS complaints.
        or (
            "stiznost" in folded_query
            and "bez advokata" in folded_query
            and ("formalni" in folded_query or "nalezitost" in folded_query or "vadn" in folded_query)
        )
    )
    if constitutional:
        values.append("ústavní stížnost")
        values.append("constitutional_complaint")
    if "dovolani" in folded_query:
        values.append("dovolání")
    if "kasačni stiznost" in folded_query or "kasacni stiznost" in folded_query:
        values.append("kasační stížnost")
    return _dedupe(values)


def _extract_decision_outcome(folded_query: str) -> list[str]:
    values: list[str] = []
    if "vyhov" in folded_query:
        values.append("vyhověno")
    formal_rejection = (
        "formalni vad" in folded_query
        or "formalnich vad" in folded_query
        or "padne uz na formalnich" in folded_query
        or "padl na nalezitostech" in folded_query
        or ("odmit" in folded_query and ("vad" in folded_query or "nalezitost" in folded_query))
        or "bez meritorniho prezkumu" in folded_query
        or "nebyla vecne projednana" in folded_query
        or "nebyl vecne projednan" in folded_query
    )
    if formal_rejection:
        values.append("rejected_for_formal_defects")
        values.append("odmítnuto")
    elif "odmit" in folded_query:
        values.append("odmítnuto")
    if "zamít" in folded_query or "zamit" in folded_query:
        values.append("zamítnuto")
    return _dedupe(values)


def _extract_negations(folded_query: str) -> list[str]:
    values: list[str] = []
    if re.search(r"\b(ne|neni|nebyl\w*|nikoli|bez|nehledam|neresim|nechci)\b", folded_query):
        values.append("negation_present")
    consent_match = re.search(r"\bbez\s+souhlasu(?:\s+\w+)?", folded_query)
    if consent_match:
        values.append(consent_match.group(0))
    return _dedupe(values)


def _extract_scoped_negative_concepts(folded_query: str) -> list[dict[str, str]]:
    """Bind contrastive Czech negations to requested-case-type concept names."""
    found: list[dict[str, str]] = []
    seen: set[str] = set()

    def _register(concept_name: str) -> None:
        if concept_name in seen:
            return
        rule = next((item for item in _LEGAL_CONCEPT_RULES if item["name"] == concept_name), None)
        label = _NEGATABLE_CONCEPT_LABELS.get(concept_name) or (
            str(rule["label"]) if rule else concept_name
        )
        # Merits-scoped aliases for custody/contact when user rejects that case type.
        if concept_name == "domestic_custody":
            for alias in ("child_custody_merits", "parent_contact_merits"):
                if alias not in seen:
                    seen.add(alias)
                    found.append(
                        {
                            "name": alias,
                            "label": _NEGATABLE_CONCEPT_LABELS[alias],
                        }
                    )
        if concept_name == "child_contact":
            alias = "parent_contact_merits"
            if alias not in seen:
                seen.add(alias)
                found.append(
                    {
                        "name": alias,
                        "label": _NEGATABLE_CONCEPT_LABELS[alias],
                    }
                )
        seen.add(concept_name)
        found.append({"name": concept_name, "label": label})

    for match in _SCOPED_NEGATION_PREFIX_RE.finditer(folded_query):
        body = _fold_text(str(match.group("body") or ""))
        if not body.strip():
            continue
        for concept_name, hints in _NEGATABLE_CONCEPT_HINTS.items():
            if any(hint in body for hint in hints):
                _register(concept_name)

    # "spis X nez Y" / "pouze X, ne Y" — Y is de-emphasized / negated.
    for match in re.finditer(
        r"(?:spis|pouze|jen)\s+.{3,60}?\s+(?:nez|ne)\s+(?P<body>.{3,80}?)(?=$|[.!?])",
        folded_query,
        flags=re.IGNORECASE,
    ):
        body = _fold_text(str(match.group("body") or ""))
        for concept_name, hints in _NEGATABLE_CONCEPT_HINTS.items():
            if any(hint in body for hint in hints):
                _register(concept_name)

    return found


def _prioritize_retrieval_concepts(concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    procedural = [c for c in concepts if c["name"] in _PROCEDURAL_PRIORITY_CONCEPT_NAMES]
    other = [c for c in concepts if c["name"] not in _PROCEDURAL_PRIORITY_CONCEPT_NAMES]
    return procedural + other


def _focus_demoted_concept_names(
    *,
    folded_query: str,
    candidate_concepts: list[dict[str, Any]],
) -> set[str]:
    """When the user says 'hledám jen Y', keep Y (and procedural) and demote background domains."""
    match = None
    for candidate in _REQUESTED_FOCUS_RE.finditer(folded_query):
        match = candidate
    if match is None:
        return set()
    focus_body = _fold_text(str(match.group("body") or ""))
    if not focus_body.strip():
        return set()
    focus_names = {concept["name"] for concept in _extract_legal_concepts(focus_body)}
    demoted: set[str] = set()
    for concept in candidate_concepts:
        name = str(concept["name"])
        if name in _PROCEDURAL_PRIORITY_CONCEPT_NAMES:
            continue
        if name in focus_names:
            continue
        demoted.add(name)
    return demoted


def _concept_expansions_for_retrieval(
    concept: dict[str, Any],
    *,
    negated_concept_names: set[str],
) -> list[str]:
    expansions = [str(item) for item in concept.get("expansions") or []]
    merits = {str(item) for item in concept.get("merits_expansions") or []}
    if concept["name"] in negated_concept_names or any(
        alias in negated_concept_names
        for alias in ("child_custody_merits", "parent_contact_merits")
        if concept["name"] in {"domestic_custody", "child_contact"}
    ):
        return []
    if concept["name"] == "criminal_procedure" and "criminal_guilt" in negated_concept_names:
        expansions = [
            item
            for item in expansions
            if "vin" not in _fold_text(item)
        ]
    # Never emit merits expansions when custody merits are negatively scoped.
    if "child_custody_merits" in negated_concept_names or "parent_contact_merits" in negated_concept_names:
        expansions = [item for item in expansions if item not in merits]
    if concept["name"] in {"damages", "contract"} and concept["name"] in negated_concept_names:
        return []
    if merits and concept["name"] in negated_concept_names:
        expansions = [item for item in expansions if item not in merits]
    return expansions


def _expansion_contradicts_negatives(
    expansion: str,
    *,
    negated_concept_names: set[str],
) -> bool:
    folded = _fold_text(expansion)
    blocked_terms = {
        "uprava styku rodice s ditetem",
        "opatrovnicke rizeni",
        "meritorni spor o peci",
        "styk rodice s ditetem",
    }
    if negated_concept_names.intersection(
        {"domestic_custody", "child_contact", "child_custody_merits", "parent_contact_merits"}
    ):
        if any(term in folded for term in blocked_terms):
            return True
        if "opatrovnick" in folded or "uprava styku" in folded:
            return True
    if "criminal_guilt" in negated_concept_names and any(
        token in folded for token in ("vinen", "vina obzalovaneho", "prohlaseni viny")
    ):
        return True
    if "damages" in negated_concept_names and any(
        token in folded for token in ("nahrada skody", "vyse skody", "odpovednost za skodu")
    ):
        return True
    if "contract" in negated_concept_names and any(
        token in folded
        for token in ("neplatnost smlouvy", "platnost smlouvy", "platnost pravniho jednani")
    ):
        return True
    for concept_name in negated_concept_names:
        rule = next((item for item in _LEGAL_CONCEPT_RULES if item["name"] == concept_name), None)
        if rule is None:
            continue
        for pattern in rule.get("patterns") or ():
            if pattern and pattern in folded and len(pattern) >= 4:
                # Allow short contextual mentions only when explicitly marked context.
                if concept_name in {"domestic_custody", "child_contact"} and expansion.startswith("kontext:"):
                    continue
                if any(str(exp).lower() in expansion.lower() for exp in rule.get("merits_expansions") or ()):
                    return True
                if pattern in {"styk", "pece o", "opatrovnick"} and concept_name in {
                    "domestic_custody",
                    "child_contact",
                }:
                    return True
        for exp in rule.get("expansions") or ():
            if _fold_text(str(exp)) == folded:
                return True
            if concept_name in {"damages", "contract"} and _fold_text(str(exp)) in folded:
                return True
    return False


def _build_focused_procedural_query(
    *,
    procedural_posture: list[str],
    decision_outcome: list[str],
    legal_concepts: list[dict[str, Any]],
) -> str | None:
    parts: list[str] = []
    if any("ústavní stížnost" in item or item == "constitutional_complaint" for item in procedural_posture):
        parts.append("ústavní stížnost")
    elif procedural_posture:
        parts.append(procedural_posture[0])
    if "rejected_for_formal_defects" in decision_outcome:
        parts.append("odmítnutá pro formální vady")
    elif "odmítnuto" in decision_outcome:
        parts.append("odmítnutá")
    labels_by_name = {concept["name"]: concept["label"] for concept in legal_concepts}
    for name in (
        "mandatory_lawyer_representation",
        "missing_or_inadequate_reasoning",
        "defective_filing",
        "failure_to_cure_filing_defects",
    ):
        if name in labels_by_name:
            parts.append(str(labels_by_name[name]))
    if not parts:
        return None
    return ", ".join(parts)


def _build_retrieval_queries(
    *,
    original_query: str,
    normalized_query: str,
    origin: QueryEntity | None,
    destination: QueryEntity | None,
    parent_entity: QueryEntity | None,
    child_entity: QueryEntity | None,
    action: str | None,
    legal_concepts: list[dict[str, Any]],
    contextual_concepts: list[dict[str, Any]] | None = None,
    negated_concept_names: set[str] | None = None,
    procedural_posture: list[str] | None = None,
    decision_outcome: list[str] | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    negated = set(negated_concept_names or ())
    suppressed: list[dict[str, str]] = []
    queries: list[str] = []

    def _try_add(text: str, *, reason: str, allow_even_if_contradicts: bool = False) -> None:
        cleaned = " ".join(str(text or "").split())
        if not cleaned:
            return
        if cleaned in queries:
            return
        if not allow_even_if_contradicts and _expansion_contradicts_negatives(
            cleaned, negated_concept_names=negated
        ):
            suppressed.append({"expansion": cleaned, "reason": reason})
            return
        queries.append(cleaned)

    # Original user query is always preserved unchanged as the first retrieval query.
    _try_add(original_query, reason="original_query", allow_even_if_contradicts=True)
    focused = _build_focused_procedural_query(
        procedural_posture=list(procedural_posture or []),
        decision_outcome=list(decision_outcome or []),
        legal_concepts=legal_concepts,
    )
    if focused:
        _try_add(focused, reason="focused_procedural_contradiction")

    if action == "abduction" and child_entity is not None:
        parts = ["mezinárodní únos dítěte", "neoprávněné přemístění dítěte"]
        if parent_entity is not None:
            parts.append(parent_entity.text)
        if origin is not None and destination is not None:
            parts.append(f"z {origin.text} do {destination.text}")
        _try_add(" ".join(parts), reason="abduction_template_contradiction")

    reserve_context_slot = any(
        item["name"] in {"domestic_custody", "child_contact"}
        for item in (contextual_concepts or [])
    )
    expansion_cap = _MAX_RETRIEVAL_QUERIES - (1 if reserve_context_slot else 0)

    for concept in legal_concepts:
        expansions = _concept_expansions_for_retrieval(concept, negated_concept_names=negated)
        for expansion in expansions:
            _try_add(expansion, reason=f"blocked_by_negative:{concept['name']}")
            if len(queries) >= expansion_cap:
                break
        if len(queries) >= expansion_cap:
            break

    # Optional non-merits context after procedural expansions; original query already
    # preserves background wording when the expansion budget is exhausted.
    for concept in contextual_concepts or []:
        if concept["name"] not in {"domestic_custody", "child_contact"}:
            continue
        _try_add(
            "zásah do péče o děti",
            reason="context_family_background",
            allow_even_if_contradicts=True,
        )
        break

    # Keep normalized query only when it does not reintroduce negated merits terms.
    if normalized_query and normalized_query != original_query:
        _try_add(normalized_query, reason="normalized_query_contradiction")

    return queries[:_MAX_RETRIEVAL_QUERIES], suppressed


def _extract_modalities(folded_query: str) -> list[str]:
    values: list[str] = []
    if re.search(r"\b(musi|povinen|povinna)\b", folded_query):
        values.append("obligation")
    if re.search(r"\b(muze|mohlo|mela|mel)\b", folded_query):
        values.append("possibility")
    return values


def _extract_source_of_claims(folded_query: str) -> list[str]:
    values: list[str] = []
    for actor in ("matka", "otec", "stezovatel", "stezovatelka", "soud"):
        if re.search(rf"\b{actor}\b\s+(tvrdi|namita|uvadi)", folded_query):
            values.append(actor)
    if "podle stezovatele" in folded_query:
        values.append("stěžovatel")
    return _dedupe(values)


def _extract_durations(folded_query: str) -> list[str]:
    return _dedupe(re.findall(r"\b\d+\s+(?:dni|dnu|mesicu|let|roku)\b", folded_query))


def _extract_case_references(
    *,
    normalized_original: str,
    folded_query: str,
) -> tuple[list[str], list[str]]:
    cases = _dedupe(_CASE_ID_RE.findall(normalized_original))
    if not cases:
        return [], []
    cited_hint = any(
        hint in folded_query
        for hint in ("cituje", "citoval", "citovana", "srov.", "odkazuje na")
    )
    if cited_hint:
        return cases, []
    return [], cases


def _extract_ambiguities(
    *,
    folded_query: str,
    hard_constraints: list[QueryConstraint],
    legal_concepts: list[dict[str, Any]],
    origin: QueryEntity | None,
    destination: QueryEntity | None,
) -> list[str]:
    ambiguities: list[str] = []
    if ("do " in folded_query or "z " in folded_query) and (origin is None or destination is None):
        ambiguities.append("movement_direction_not_fully_bound")
    if folded_query and not hard_constraints:
        ambiguities.append("no_hard_constraints_extracted")
    broad_single_concept = len([token for token in folded_query.split() if len(token) >= 3]) <= 2
    if broad_single_concept and legal_concepts:
        ambiguities.append("single_broad_legal_concept_requires_clarification")
    return ambiguities


def _infer_intent(
    *,
    action: str | None,
    legal_provisions: list[str],
    cited_cases: list[str],
    current_cases: list[str],
    legal_concepts: list[dict[str, Any]],
) -> QueryIntent:
    if action is not None:
        return QueryIntent.FACT_PATTERN_MATCH
    if legal_provisions or legal_concepts:
        return QueryIntent.LEGAL_PROVISION_SEARCH
    if cited_cases or current_cases:
        return QueryIntent.CASE_CITATION_SEARCH
    return QueryIntent.LEGAL_DOCUMENT_SEARCH
