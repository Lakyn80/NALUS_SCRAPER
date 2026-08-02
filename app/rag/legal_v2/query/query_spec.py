from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass, field
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
        "patterns": ("unos ditete", "neopravnene premisteni ditete", "premistila dite", "odvezla dite", "haagsk", "obvykle bydliste"),
        "expansions": ("mezinárodní únos dítěte", "neoprávněné přemístění dítěte", "obvyklé bydliště dítěte", "Haagská úmluva"),
    },
    {
        "name": "domestic_custody",
        "label": "péče o nezletilé dítě",
        "patterns": ("pece", "styk", "opatrovnick", "svereni nezletileho", "kontakt s ditetem"),
        "expansions": ("péče o nezletilé dítě", "úprava styku rodiče s dítětem", "opatrovnické řízení"),
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
        "patterns": ("smlouv", "neplatnost smlouvy", "platnost pravniho jednani"),
        "expansions": ("smlouva", "neplatnost smlouvy", "platnost právního jednání"),
    },
    {
        "name": "damages",
        "label": "náhrada škody",
        "patterns": ("nahrada skody", "odpovednost za skodu", "pricinna souvislost"),
        "expansions": ("náhrada škody", "odpovědnost za škodu", "příčinná souvislost"),
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
        "patterns": ("naklady rizeni", "nahrada nakladu", "soudni poplatek"),
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
        "patterns": ("dan", "danove rizeni", "platebni vymer", "financni urad"),
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
    }
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
        _add_constraint(
            hard_constraints,
            ConstraintCategory.LOCATION,
            origin.text,
            ConstraintPolarity.HARD,
            attribute="origin",
            source_ids=[origin.entity_id],
        )
        _add_constraint(
            hard_constraints,
            ConstraintCategory.LOCATION,
            destination.text,
            ConstraintPolarity.HARD,
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
        _add_constraint(
            hard_constraints,
            ConstraintCategory.EVENT,
            action,
            ConstraintPolarity.HARD,
            attribute="action",
            source_ids=[event.event_id],
        )
        if parent_entity is not None:
            _add_constraint(
                hard_constraints,
                ConstraintCategory.ENTITY,
                parent_entity.role or parent_entity.normalized_text,
                ConstraintPolarity.HARD,
                attribute="actor_role",
                source_ids=[parent_entity.entity_id],
            )
        if child_entity is not None:
            _add_constraint(
                hard_constraints,
                ConstraintCategory.ENTITY,
                child_entity.role or child_entity.normalized_text,
                ConstraintPolarity.HARD,
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
                hard_constraints,
                ConstraintCategory.RELATION,
                f"{parent_entity.role}:{action}:{child_entity.role}",
                ConstraintPolarity.HARD,
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
    ]
    for concept in candidate_retrieval_concepts:
        _add_constraint(
            hard_constraints,
            ConstraintCategory.LEGAL_PROVISION,
            _concept_constraint_value(concept),
            ConstraintPolarity.HARD,
            attribute=f"legal_concept:{concept['name']}",
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
    retrieval_queries = _build_retrieval_queries(
        original_query=original_query,
        normalized_query=normalized_query,
        origin=origin,
        destination=destination,
        parent_entity=parent_entity,
        child_entity=child_entity,
        action=action,
        legal_concepts=candidate_retrieval_concepts,
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
    }
    return QuerySpecV2(
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
        return ConstraintPolarity.HARD


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
    if "unos" in folded_query or "unest" in folded_query:
        return "abduction"
    if "premist" in folded_query or "prevez" in folded_query:
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
    return " ".join([str(concept["label"]), *[str(item) for item in concept["expansions"]]])


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
    return normalize_legal_text(value).strip(" ,.;:")


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
    if "ustavni stiznost" in folded_query:
        values.append("ústavní stížnost")
    if "dovolani" in folded_query:
        values.append("dovolání")
    if "kasačni stiznost" in folded_query or "kasacni stiznost" in folded_query:
        values.append("kasační stížnost")
    return values


def _extract_decision_outcome(folded_query: str) -> list[str]:
    values: list[str] = []
    if "vyhov" in folded_query:
        values.append("vyhověno")
    if "odmit" in folded_query:
        values.append("odmítnuto")
    if "zamít" in folded_query or "zamit" in folded_query:
        values.append("zamítnuto")
    return values


def _extract_negations(folded_query: str) -> list[str]:
    values: list[str] = []
    if re.search(r"\b(ne|neni|nebyl\w*|nikoli|bez)\b", folded_query):
        values.append("negation_present")
    consent_match = re.search(r"\bbez\s+souhlasu(?:\s+\w+)?", folded_query)
    if consent_match:
        values.append(consent_match.group(0))
    return _dedupe(values)


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
) -> list[str]:
    queries = _dedupe([original_query, normalized_query])
    if action == "abduction" and child_entity is not None:
        parts = ["mezinárodní únos dítěte", "neoprávněné přemístění dítěte"]
        if parent_entity is not None:
            parts.append(parent_entity.text)
        if origin is not None and destination is not None:
            parts.append(f"z {origin.text} do {destination.text}")
        queries.append(" ".join(parts))
    for concept in legal_concepts:
        queries.extend(str(item) for item in concept["expansions"])
        queries.append(" ".join([original_query, *[str(item) for item in concept["expansions"][:2]]]))
    return queries


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
