from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx

from app.rag.legal_v2.interpreter import (
    DeepSeekQuerySpecProvider,
    DeterministicQuerySpecProvider,
    interpret_query_spec_v2,
)
from app.rag.legal_v2.query_spec import (
    ConstraintCategory,
    QuerySpecV2,
    build_query_spec_v2,
)


def _openai_envelope(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _mock_client(json_data: dict):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.text = json.dumps(json_data)
    mock_resp.json.return_value = json_data
    mock_resp.headers = {}
    mock_resp.raise_for_status.return_value = None
    mock_instance = MagicMock()
    mock_instance.post.return_value = mock_resp
    mock_class = MagicMock(return_value=mock_instance)
    return mock_class, mock_instance


def test_universal_query_spec_serializes_and_preserves_original_query() -> None:
    query = "únos dítěte matkou z Česka do Ruska"

    spec = build_query_spec_v2(query)
    restored = QuerySpecV2.from_dict(spec.to_dict())

    assert restored.original_query == query
    assert restored.retrieval_queries[0] == query
    assert restored.origin is not None
    assert restored.destination is not None
    assert restored.origin.text == "Česká republika"
    assert restored.destination.text == "Ruská federace"
    assert restored.movement_direction == "origin_to_destination"
    assert any(entity.role == "mother" for entity in restored.entities)
    assert any(entity.role == "child" for entity in restored.entities)
    assert any(
        constraint.attribute == "actor_action_object"
        and constraint.category == ConstraintCategory.RELATION
        and constraint.polarity.value == "soft"
        for constraint in restored.soft_constraints
    )
    assert any(
        (constraint.attribute or "").startswith("legal_concept:")
        for constraint in restored.hard_constraints
    )
    assert not any(
        constraint.attribute
        in {"origin", "destination", "actor_role", "object_role", "action", "actor_action_object"}
        for constraint in restored.hard_constraints
    )


def test_deepseek_query_spec_request_uses_quality_first_thinking_default(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LLM_MODEL_DEEPSEEK", "deepseek-v4-flash")
    monkeypatch.setenv("LLM_TIMEOUT", "30")
    monkeypatch.setenv("LLM_RETRY", "0")
    monkeypatch.delenv("NALUS_LEGAL_V2_QUERYSPEC_TIMEOUT_SECONDS", raising=False)
    query = "únos dítěte matkou z Česka do Ruska"
    mock_class, mock_instance = _mock_client(
        _openai_envelope(json.dumps(build_query_spec_v2(query).to_dict(), ensure_ascii=False))
    )

    with patch("httpx.Client", mock_class):
        provider = DeepSeekQuerySpecProvider(api_key="k")
        result = provider.interpret(query)

    request_payload = mock_instance.post.call_args.kwargs["json"]
    assert isinstance(result, str)
    assert request_payload["model"] == "deepseek-v4-flash"
    assert request_payload["response_format"] == {"type": "json_object"}
    assert request_payload["max_tokens"] == 8000
    assert request_payload["temperature"] == 0.0
    assert request_payload["thinking"] == {"type": "enabled"}
    assert list(request_payload.keys()).count("thinking") == 1
    assert "extra_body" not in request_payload
    assert mock_class.call_args.kwargs["timeout"].connect == 120


def test_query_spec_distinguishes_mother_and_father_roles() -> None:
    mother = build_query_spec_v2("únos dítěte matkou do Ruska")
    father = build_query_spec_v2("únos dítěte otcem do Ruska")

    assert {entity.role for entity in mother.entities} >= {"mother", "child"}
    assert {entity.role for entity in father.entities} >= {"father", "child"}
    assert {
        constraint.normalized_value for constraint in mother.soft_constraints
    } != {constraint.normalized_value for constraint in father.soft_constraints}


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
    assert result.reason is not None
    assert result.reason.startswith("query_interpreter_schema_repaired:")
    assert result.query_spec is not None
    assert result.query_spec.retrieval_queries[0] == "únos dítěte matkou z Česka do Ruska"
    assert result.query_spec.requires_verification is True
    assert result.query_spec.hard_constraints
    assert result.query_spec.structured_query["provider_schema_repaired"] is True


def test_query_provider_json_is_extracted_from_text_envelope() -> None:
    query = "únos dítěte matkou z Česka do Ruska"
    provider = DeterministicQuerySpecProvider(
        "Výstup:\n"
        + json.dumps(build_query_spec_v2(query).to_dict(), ensure_ascii=False)
        + "\nKonec."
    )

    result = interpret_query_spec_v2(
        query,
        provider=provider,
        allow_deterministic_fallback=False,
    )

    assert result.status == "ok"
    assert result.query_spec is not None
    assert result.query_spec.original_query == query


def test_query_spec_normalizes_lay_and_formal_child_removal_concepts() -> None:
    lay = build_query_spec_v2("matka odvezla dítě do zahraničí bez souhlasu otce")
    formal = build_query_spec_v2(
        "mezinárodní únos dítěte a určení obvyklého bydliště podle Haagské úmluvy"
    )

    lay_concepts = {item["name"] for item in lay.structured_query["legal_concepts"]}
    formal_concepts = {item["name"] for item in formal.structured_query["legal_concepts"]}

    assert "international_child_removal" in lay_concepts
    assert "international_child_removal" in formal_concepts
    assert any(
        constraint.attribute == "legal_concept:international_child_removal"
        for constraint in formal.hard_constraints
    )
    assert any("Haagská úmluva" in query for query in formal.retrieval_queries)


def test_query_spec_preserves_service_and_deadline_concepts() -> None:
    spec = build_query_spec_v2("soud mi nedoručil rozsudek a zmeškal jsem odvolání")
    concepts = {item["name"] for item in spec.structured_query["legal_concepts"]}

    assert {"service_of_documents", "restoration_of_deadline"}.issubset(concepts)
    assert any("vadné doručení" in query for query in spec.retrieval_queries)
    assert any("navrácení lhůty" in query for query in spec.retrieval_queries)


def test_newer_legal_concepts_do_not_change_candidate_ranking_inputs() -> None:
    spec = build_query_spec_v2("obviněný cizinec nerozumí česky a potřebuje tlumočníka")
    concepts = {item["name"] for item in spec.structured_query["legal_concepts"]}
    candidate_concepts = {
        item["name"]
        for item in spec.structured_query["candidate_retrieval_concepts"]
    }

    assert "right_to_interpreter" in concepts
    assert "right_to_interpreter" not in candidate_concepts
    assert not any(
        constraint.attribute == "legal_concept:right_to_interpreter"
        for constraint in spec.hard_constraints
    )
    assert not any("právo na tlumočníka" in query for query in spec.retrieval_queries)


def test_single_broad_legal_concept_requires_clarification() -> None:
    spec = build_query_spec_v2("výživné")

    assert "maintenance" in {
        item["name"] for item in spec.structured_query["legal_concepts"]
    }
    assert "single_broad_legal_concept_requires_clarification" in spec.ambiguities


def test_constraint_from_dict_tolerates_missing_polarity_and_category() -> None:
    query = "matka odvezla dítě do zahraničí bez souhlasu otce"
    payload = {
        "original_query": query,
        "normalized_query": query,
        "retrieval_queries": [query],
        "intent": "case_law_search",
        "entities": [],
        "hard_constraints": [
            {
                "value": "matka",
                "normalized_value": "matka",
            }
        ],
        "requires_verification": True,
    }

    spec = QuerySpecV2.from_dict(payload)
    assert len(spec.hard_constraints) == 1
    assert spec.hard_constraints[0].polarity.value == "soft"
    assert spec.hard_constraints[0].category == ConstraintCategory.ENTITY
    assert spec.hard_constraints[0].constraint_id.startswith("constraint_llm_")


def test_lay_abduction_route_keeps_locations_soft_and_legal_concept_hard() -> None:
    spec = build_query_spec_v2("Matka unesla dítě z česka do Ruska")

    hard_attrs = {constraint.attribute for constraint in spec.hard_constraints}
    soft_attrs = {constraint.attribute for constraint in spec.soft_constraints}
    assert soft_attrs >= {"origin", "destination", "actor_role", "object_role", "action"}
    assert "legal_concept:international_child_removal" in hard_attrs
    assert not hard_attrs.intersection(
        {"origin", "destination", "actor_role", "object_role", "action", "actor_action_object"}
    )
    assert spec.origin is not None and spec.origin.text == "Česká republika"
    assert spec.destination is not None and spec.destination.text == "Ruská federace"


def test_interpret_merges_hard_constraints_lost_by_provider() -> None:
    query = "matka odvezla dítě do zahraničí bez souhlasu otce"
    deterministic = build_query_spec_v2(query)
    assert deterministic.hard_constraints
    provider = DeterministicQuerySpecProvider(
        {
            "original_query": query,
            "normalized_query": query,
            "retrieval_queries": [query],
            "intent": "case_law_search",
            "entities": [
                {
                    "entity_id": "e1",
                    "text": "matka",
                    "normalized_text": "matka",
                    "entity_type": "person",
                    "role": "mother",
                },
                {
                    "entity_id": "e2",
                    "text": "dítě",
                    "normalized_text": "dite",
                    "entity_type": "person",
                    "role": "child",
                },
            ],
            "origin": {
                "entity_id": "o1",
                "text": "Česka",
                "normalized_text": deterministic.origin.normalized_text if deterministic.origin else "ceska",
                "entity_type": "location",
            },
            "destination": {
                "entity_id": "d1",
                "text": "zahraničí",
                "normalized_text": (
                    deterministic.destination.normalized_text
                    if deterministic.destination
                    else "zahranici"
                ),
                "entity_type": "location",
            },
            "hard_constraints": [],
            "requires_verification": True,
            "ambiguities": [],
            "negations": list(deterministic.negations),
        }
    )

    result = interpret_query_spec_v2(
        query,
        provider=provider,
        allow_deterministic_fallback=False,
    )

    assert result.status == "ok"
    assert result.query_spec is not None
    assert result.query_spec.hard_constraints
    assert result.reason is not None
    assert "query_interpreter_merged:hard_constraints" in result.reason


def test_query_spec_prompt_excludes_clarification_intent() -> None:
    from app.rag.legal_v2.interpreter import _query_spec_prompt

    prompt = _query_spec_prompt("výživné")
    assert "clarification" not in prompt
    assert '"polarity": "hard"' in prompt or '"polarity":"hard"' in prompt.replace(" ", "")
