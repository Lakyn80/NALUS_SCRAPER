"""Deterministic structured query interpretation for legal retrieval.

This interpreter is intentionally conservative. It extracts only constraints
that can be represented and verified deterministically in the first rollout.
Later LLM-based interpretation can be added behind the same typed model.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.rag.retrieval.constraint_models import (
    ConstraintCategory,
    ConstraintRequirement,
    InterpretationStatus,
    RelationPredicate,
    StructuredConstraint,
    StructuredEntity,
    StructuredQuery,
    StructuredRelation,
)


@dataclass(frozen=True)
class CountryAlias:
    code: str
    canonical_name: str
    aliases: tuple[str, ...]


COUNTRIES: tuple[CountryAlias, ...] = (
    CountryAlias(
        code="CZ",
        canonical_name="Česká republika",
        aliases=(
            "ceska republika",
            "ceske republice",
            "cesko",
            "ceska",
            "cesky",
            "ceske",
            "ceskeho",
        ),
    ),
    CountryAlias(
        code="RU",
        canonical_name="Ruská federace",
        aliases=(
            "rusko",
            "ruska federace",
            "ruske federace",
            "ruskemu",
            "ruskeho",
            "rusky",
            "ruska",
            "ruskym",
            "ruskemu obcanu",
        ),
    ),
    CountryAlias(
        code="UA",
        canonical_name="Ukrajina",
        aliases=("ukrajina", "ukrajine", "ukrajinsky", "ukrajinskeho", "ukrajinskemu"),
    ),
    CountryAlias(
        code="DE",
        canonical_name="Německo",
        aliases=("nemecko", "nemecka", "nemeckeho", "nemeckemu", "spolkova republika nemecko"),
    ),
    CountryAlias(
        code="AT",
        canonical_name="Rakousko",
        aliases=("rakousko", "rakouska", "rakouskeho", "rakouskemu"),
    ),
    CountryAlias(
        code="US",
        canonical_name="Spojené státy americké",
        aliases=("usa", "spojene staty", "spojenych statu", "americky", "americkeho"),
    ),
)


def interpret_structured_query(query: str) -> StructuredQuery:
    normalized = _normalize(query)
    if not normalized:
        return StructuredQuery(
            intent="empty_query",
            status=InterpretationStatus.UNAVAILABLE,
            constraints=[],
            ambiguities=["Query is empty."],
        )

    entities: list[StructuredEntity] = []
    relations: list[StructuredRelation] = []
    constraints: list[StructuredConstraint] = []
    ambiguities: list[str] = []
    expansions: list[str] = []

    _add_court_constraints(normalized, constraints)
    citizenship_event = _contains_any(normalized, ("obcanstvi", "statni obcanstvi")) and (
        _contains_any(normalized, ("udeleni", "udelit", "neudeleni", "zadost", "pozadal", "priznani"))
    )
    if citizenship_event:
        applicant = StructuredEntity(id="person:applicant", entity_type="person", role="applicant")
        citizenship = StructuredEntity(
            id="legal_status:czech_citizenship",
            entity_type="legal_status",
            attributes={"country_code": "CZ"},
        )
        entities.extend([applicant, citizenship])
        relation = StructuredRelation(
            subject=applicant.id,
            predicate=RelationPredicate.APPLIED_FOR,
            object=citizenship.id,
            requirement=ConstraintRequirement.HARD,
        )
        relations.append(relation)
        constraints.append(
            StructuredConstraint(
                id="legal_event:czech_citizenship_application_or_grant",
                category=ConstraintCategory.LEGAL_EVENT,
                value="czech_citizenship_application_or_grant",
                requirement=ConstraintRequirement.HARD,
                relation=relation,
                description="Document must concern application/grant/refusal of Czech citizenship.",
            )
        )
        expansions.extend(
            [
                "žádost o udělení státního občanství",
                "neudělení státního občanství",
                "státní občanství České republiky",
            ]
        )

    child_abduction_event = _contains_any(normalized, ("unos ditete", "mezinarodni unos", "haagska umluva")) or (
        _contains_any(normalized, ("neopravnene premisteni", "neopravnene zadrzeni", "navraceni ditete"))
        and _contains_any(normalized, ("dite", "ditete", "nezletile"))
    )
    if child_abduction_event:
        child = StructuredEntity(id="person:child", entity_type="person", role="affected_child")
        actor = StructuredEntity(id="person:actor", entity_type="person", role="removal_or_retention_actor")
        entities.extend([child, actor])
        relation = StructuredRelation(
            subject=actor.id,
            predicate=RelationPredicate.WRONGFULLY_REMOVED_OR_RETAINED,
            object=child.id,
            requirement=ConstraintRequirement.HARD,
        )
        relations.append(relation)
        constraints.append(
            StructuredConstraint(
                id="legal_event:international_child_abduction",
                category=ConstraintCategory.LEGAL_EVENT,
                value="international_child_abduction",
                requirement=ConstraintRequirement.HARD,
                relation=relation,
                description="Document must concern international child abduction or wrongful removal/retention.",
            )
        )
        expansions.extend(
            [
                "mezinárodní únos dítěte",
                "neoprávněné přemístění dítěte",
                "navrácení dítěte podle Haagské úmluvy",
            ]
        )

    detected_countries = _detected_countries(normalized)
    if citizenship_event:
        nationality_codes = _nationality_countries(normalized, detected_countries)
        for code in nationality_codes:
            relation = StructuredRelation(
                subject="person:applicant",
                predicate=RelationPredicate.HAS_NATIONALITY,
                object=f"country:{code}",
                requirement=ConstraintRequirement.HARD,
            )
            relations.append(relation)
            constraints.append(
                StructuredConstraint(
                    id=f"nationality:applicant:{code.lower()}",
                    category=ConstraintCategory.NATIONALITY,
                    value=code,
                    requirement=ConstraintRequirement.HARD,
                    relation=relation,
                    description="Applicant/person nationality must be proven for the requested country.",
                )
            )
    elif detected_countries:
        ambiguities.append(
            "Country mention detected, but the requested legal relation to the country is unresolved."
        )

    if child_abduction_event:
        destination_codes = _destination_countries(normalized, detected_countries)
        for code in destination_codes:
            relation = StructuredRelation(
                subject="person:actor",
                predicate=RelationPredicate.DESTINATION_COUNTRY,
                object=f"country:{code}",
                requirement=ConstraintRequirement.HARD,
            )
            relations.append(relation)
            constraints.append(
                StructuredConstraint(
                    id=f"country_relation:destination:{code.lower()}",
                    category=ConstraintCategory.COUNTRY_RELATION,
                    value=code,
                    requirement=ConstraintRequirement.HARD,
                    relation=relation,
                    description="Destination/retention country must match the requested country.",
                )
            )
        if _contains_any(normalized, ("matk", "otc", "otec", "rodic")):
            constraints.append(
                StructuredConstraint(
                    id="actor_role:parent",
                    category=ConstraintCategory.ACTOR_ROLE,
                    value="parent",
                    requirement=ConstraintRequirement.HARD,
                    description="The removal/retention actor must be a parent.",
                )
            )

    status = InterpretationStatus.STRUCTURED if constraints else InterpretationStatus.PARTIAL
    if not constraints:
        constraints.append(
            StructuredConstraint(
                id="legal_topic:unstructured_text",
                category=ConstraintCategory.LEGAL_TOPIC,
                value=normalized[:120],
                requirement=ConstraintRequirement.SOFT,
                description="No deterministic hard constraint extracted; use lexical retrieval only.",
            )
        )

    return StructuredQuery(
        intent="constraint_aware_document_retrieval",
        status=status,
        constraints=_dedupe_constraints(constraints),
        entities=_dedupe_entities(entities),
        relations=_dedupe_relations(relations),
        ambiguities=ambiguities,
        retrieval_expansions=sorted(set(expansions)),
        interpreter="deterministic_v1",
    )


def _add_court_constraints(normalized: str, constraints: list[StructuredConstraint]) -> None:
    if _contains_any(normalized, ("ustavni soud", "usoud", "nalus")):
        constraints.append(
            StructuredConstraint(
                id="court:constitutional",
                category=ConstraintCategory.COURT,
                value="constitutional_court",
                requirement=ConstraintRequirement.HARD,
                description="Document must come from the Constitutional Court.",
            )
        )
    if _contains_any(normalized, ("nejvyssi soud", "nsoud")) and "nejvyssi spravni" not in normalized:
        constraints.append(
            StructuredConstraint(
                id="court:supreme",
                category=ConstraintCategory.COURT,
                value="supreme_court",
                requirement=ConstraintRequirement.HARD,
                description="Document must come from the Supreme Court.",
            )
        )


def _nationality_countries(
    normalized: str,
    detected_countries: list[CountryAlias],
) -> list[str]:
    codes: list[str] = []
    for country in detected_countries:
        if country.code == "CZ":
            continue
        country_pattern = "|".join(re.escape(alias) for alias in country.aliases)
        if re.search(rf"({country_pattern}).{{0,30}}obcan", normalized) or re.search(
            rf"obcan.{{0,30}}({country_pattern})",
            normalized,
        ):
            codes.append(country.code)
    return sorted(set(codes))


def _destination_countries(
    normalized: str,
    detected_countries: list[CountryAlias],
) -> list[str]:
    codes: list[str] = []
    for country in detected_countries:
        country_pattern = "|".join(re.escape(alias) for alias in country.aliases)
        if re.search(rf"\b(do|v|ve|na)\s+({country_pattern})\b", normalized):
            codes.append(country.code)
    return sorted(set(codes))


def _detected_countries(normalized: str) -> list[CountryAlias]:
    found: list[CountryAlias] = []
    for country in COUNTRIES:
        if any(re.search(rf"\b{re.escape(alias)}\b", normalized) for alias in country.aliases):
            found.append(country)
    return found


def _dedupe_constraints(
    constraints: list[StructuredConstraint],
) -> list[StructuredConstraint]:
    seen: set[str] = set()
    result: list[StructuredConstraint] = []
    for constraint in constraints:
        if constraint.id in seen:
            continue
        seen.add(constraint.id)
        result.append(constraint)
    return result


def _dedupe_entities(entities: list[StructuredEntity]) -> list[StructuredEntity]:
    seen: set[str] = set()
    result: list[StructuredEntity] = []
    for entity in entities:
        if entity.id in seen:
            continue
        seen.add(entity.id)
        result.append(entity)
    return result


def _dedupe_relations(relations: list[StructuredRelation]) -> list[StructuredRelation]:
    seen: set[tuple[str, str, str]] = set()
    result: list[StructuredRelation] = []
    for relation in relations:
        key = (relation.subject, relation.predicate.value, relation.object)
        if key in seen:
            continue
        seen.add(key)
        result.append(relation)
    return result


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", without_marks.lower()).strip()


def _contains_any(normalized: str, needles: tuple[str, ...]) -> bool:
    return any(needle in normalized for needle in needles)
