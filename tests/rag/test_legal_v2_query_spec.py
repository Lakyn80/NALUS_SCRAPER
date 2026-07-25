from __future__ import annotations

from app.rag.legal_v2.interpreter import (
    DeterministicQuerySpecProvider,
    interpret_query_spec_v2,
)
from app.rag.legal_v2.query_spec import (
    ConstraintCategory,
    QuerySpecV2,
    build_query_spec_v2,
)


def test_universal_query_spec_serializes_and_preserves_original_query() -> None:
    query = "únos dítěte matkou z Česka do Ruska"

    spec = build_query_spec_v2(query)
    restored = QuerySpecV2.from_dict(spec.to_dict())

    assert restored.original_query == query
    assert restored.retrieval_queries[0] == query
    assert restored.origin is not None
    assert restored.destination is not None
    assert restored.origin.text == "Česka"
    assert restored.destination.text == "Ruska"
    assert restored.movement_direction == "origin_to_destination"
    assert any(entity.role == "mother" for entity in restored.entities)
    assert any(entity.role == "child" for entity in restored.entities)
    assert any(
        constraint.attribute == "actor_action_object"
        and constraint.category == ConstraintCategory.RELATION
        for constraint in restored.hard_constraints
    )


def test_query_spec_distinguishes_mother_and_father_roles() -> None:
    mother = build_query_spec_v2("únos dítěte matkou do Ruska")
    father = build_query_spec_v2("únos dítěte otcem do Ruska")

    assert {entity.role for entity in mother.entities} >= {"mother", "child"}
    assert {entity.role for entity in father.entities} >= {"father", "child"}
    assert {
        constraint.normalized_value for constraint in mother.hard_constraints
    } != {constraint.normalized_value for constraint in father.hard_constraints}


def test_query_spec_preserves_negation_and_source_of_claim() -> None:
    spec = build_query_spec_v2(
        "matka tvrdí, že otec nebyl informován bez souhlasu"
    )

    assert spec.negations
    assert spec.source_of_claims == ["matka"]
    assert any(
        constraint.category == ConstraintCategory.NEGATION
        for constraint in spec.hard_constraints
    )
    assert any(
        constraint.category == ConstraintCategory.SOURCE_OF_CLAIM
        for constraint in spec.hard_constraints
    )


def test_query_spec_distinguishes_cited_case_from_current_case() -> None:
    cited = build_query_spec_v2("nález cituje sp. zn. II. ÚS 859/23")
    current = build_query_spec_v2("věc sp. zn. IV. ÚS 851/26")

    assert cited.cited_cases == ["II. ÚS 859/23"]
    assert cited.current_case_identifiers == []
    assert current.current_case_identifiers == ["IV. ÚS 851/26"]
    assert current.cited_cases == []
    assert any(
        constraint.category == ConstraintCategory.CITED_CASE
        for constraint in cited.hard_constraints
    )
    assert any(
        constraint.category == ConstraintCategory.CURRENT_CASE
        for constraint in current.hard_constraints
    )


def test_provider_schema_drift_is_repaired_to_local_contract() -> None:
    provider = DeterministicQuerySpecProvider(
        {
            "original_query": "únos dítěte matkou z Česka do Ruska",
            "normalized_query": "child abduction by mother from Czech Republic to Russia",
            "retrieval_queries": ["child abduction Czech Republic Russia"],
            "intent": "child abduction legal inquiry",
            "entities": ["dítě", "matka", "Česko", "Rusko"],
            "hard_constraints": ["child abduction by mother"],
            "requires_verification": False,
        }
    )

    result = interpret_query_spec_v2(
        "únos dítěte matkou z Česka do Ruska",
        provider=provider,
        allow_deterministic_fallback=False,
    )

    assert result.status == "ok"
    assert result.reason == "query_interpreter_schema_repaired:ValueError"
    assert result.query_spec is not None
    assert result.query_spec.retrieval_queries[0] == "únos dítěte matkou z Česka do Ruska"
    assert result.query_spec.requires_verification is True
    assert result.query_spec.hard_constraints
    assert result.query_spec.structured_query["provider_schema_repaired"] is True
