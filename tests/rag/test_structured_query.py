from app.rag.retrieval.constraint_models import ConstraintCategory, ConstraintRequirement
from app.rag.retrieval.structured_query import interpret_structured_query


def _constraint_values(query: str) -> set[tuple[str, str, str]]:
    structured = interpret_structured_query(query)
    return {
        (constraint.category.value, constraint.value, constraint.requirement.value)
        for constraint in structured.constraints
    }


def test_interprets_citizenship_query_with_russian_nationality_constraint() -> None:
    values = _constraint_values("udělení českého občanství ruskému občanu")

    assert (
        ConstraintCategory.LEGAL_EVENT.value,
        "czech_citizenship_application_or_grant",
        ConstraintRequirement.HARD.value,
    ) in values
    assert (
        ConstraintCategory.NATIONALITY.value,
        "RU",
        ConstraintRequirement.HARD.value,
    ) in values


def test_interprets_child_abduction_destination_and_parent_role() -> None:
    values = _constraint_values("mezinárodní únos dítěte matkou do Ruska")

    assert (
        ConstraintCategory.LEGAL_EVENT.value,
        "international_child_abduction",
        ConstraintRequirement.HARD.value,
    ) in values
    assert (
        ConstraintCategory.COUNTRY_RELATION.value,
        "RU",
        ConstraintRequirement.HARD.value,
    ) in values
    assert (
        ConstraintCategory.ACTOR_ROLE.value,
        "parent",
        ConstraintRequirement.HARD.value,
    ) in values


def test_unstructured_query_stays_partial_with_soft_topic() -> None:
    structured = interpret_structured_query("náhrada škody podle občanského zákoníku")

    assert structured.status.value == "partial"
    assert structured.hard_constraints == []
    assert structured.soft_constraints[0].category == ConstraintCategory.LEGAL_TOPIC
