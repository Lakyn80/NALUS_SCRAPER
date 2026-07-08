"""Tests for legal query clarification gate."""

from __future__ import annotations

import json
from pathlib import Path

from app.rag.clarification.cache import (
    InMemoryClarificationCache,
    build_exact_query_cache_key,
    build_pattern_cache_key,
)
from app.rag.clarification.llm import LLMClarificationGenerator, TemplateClarificationLLM
from app.rag.clarification.models import ClarificationDecision, RetrievalHitSummary
from app.rag.clarification.qdrant_patterns import InMemoryClarificationPatternIndex
from app.rag.clarification.rules import assess_query_rules, should_ask_clarification
from app.rag.clarification.service import LegalQueryClarificationService

RAG_EVAL_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "rag_eval"
LONGFORM_DATASET = RAG_EVAL_DIR / "nalus_client_longform_eval_v1.json"

CLEAR_CRIMINAL_QUERY = (
    "V trestní věci byl obviněný odsouzen a odvolací soud zamítl jeho odvolání. "
    "Chce podat dovolání podle trestního řádu a tvrdí, že chyby vznikly už v řízení "
    "před soudem prvního stupně."
)

SIMILAR_AMBIGUOUS_QUERY = (
    "Prohrál jsem odvolání a chci se obrátit na Nejvyšší soud kvůli chybám z prvního řízení."
)

UNRELATED_FAMILY_QUERY = (
    "Chci řešit výživné a nevím, jestli mám podat návrh nebo odvolání."
)


class _FakeTextLLM:
    def __init__(self) -> None:
        self.call_count = 0

    def generate_text(self, prompt: str) -> str:
        del prompt
        self.call_count += 1
        return "LLM-generated clarification question?"


def _load_client_longform_04_question() -> str:
    payload = json.loads(LONGFORM_DATASET.read_text(encoding="utf-8"))
    for case in payload["cases"]:
        if case["id"] == "client-longform-04":
            return case["question"]
    raise AssertionError("client-longform-04 not found in dataset")


def test_clear_criminal_query_proceeds_to_retrieval() -> None:
    service = LegalQueryClarificationService(cache=InMemoryClarificationCache())
    decision = service.evaluate(CLEAR_CRIMINAL_QUERY)

    assert decision.decision == "proceed_to_retrieval"
    assert decision.detected_legal_domain == "criminal"
    assert decision.detected_procedure_stage == "dovolani"
    assert "legal_domain_ambiguous" not in decision.ambiguity_types
    assert decision.clarification_question_cs == ""
    assert decision.llm_called is False


def test_ambiguous_dovolani_query_asks_clarification() -> None:
    query = _load_client_longform_04_question()
    assessment = assess_query_rules(query)

    assert should_ask_clarification(assessment) is True
    assert "legal_domain_ambiguous" in assessment.ambiguity_types
    assert assessment.query_signature == "appeal_dovolani_previous_proceeding_ambiguous"

    service = LegalQueryClarificationService(cache=InMemoryClarificationCache())
    decision = service.evaluate(query)

    assert decision.decision == "ask_clarifying_question"
    assert "legal_domain_ambiguous" in decision.ambiguity_types
    assert "legal_domain" in decision.missing_slots
    assert "trestní dovolání" in decision.clarification_question_cs.lower()
    assert "civilní dovolání" in decision.clarification_question_cs.lower()
    assert decision.llm_called is False


def test_exact_cache_prevents_repeated_llm_call() -> None:
    cache = InMemoryClarificationCache()
    fake_llm = _FakeTextLLM()
    llm = LLMClarificationGenerator(fake_llm)
    service = LegalQueryClarificationService(cache=cache, llm=llm)
    query = _load_client_longform_04_question()

    first = service.evaluate(query)
    assert first.decision == "ask_clarifying_question"
    assert first.cache_hit is False
    assert first.llm_called is False
    assert fake_llm.call_count == 0

    second = service.evaluate(query)
    assert second.cache_hit is True
    assert second.llm_called is False
    assert fake_llm.call_count == 0
    assert second.clarification_question_cs == first.clarification_question_cs


def test_similar_ambiguous_query_reuses_cached_clarification() -> None:
    cache = InMemoryClarificationCache()
    service = LegalQueryClarificationService(
        cache=cache,
        pattern_index=InMemoryClarificationPatternIndex(),
    )
    seed_query = _load_client_longform_04_question()
    first = service.evaluate(seed_query)
    assert first.decision == "ask_clarifying_question"

    second = service.evaluate(SIMILAR_AMBIGUOUS_QUERY)
    assert second.decision == "ask_clarifying_question"
    assert "trestní dovolání" in second.clarification_question_cs.lower()
    assert second.query_signature == "appeal_dovolani_previous_proceeding_ambiguous"
    assert second.llm_called is False


def test_unrelated_ambiguous_query_does_not_reuse_dovolani_clarification() -> None:
    cache = InMemoryClarificationCache()
    service = LegalQueryClarificationService(
        cache=cache,
        pattern_index=InMemoryClarificationPatternIndex(),
    )
    service.evaluate(_load_client_longform_04_question())

    decision = service.evaluate(UNRELATED_FAMILY_QUERY)
    assert decision.decision == "ask_clarifying_question"
    assert decision.detected_legal_domain == "family"
    assert "remedy_ambiguous" in decision.ambiguity_types
    assert "trestní dovolání" not in decision.clarification_question_cs.lower()
    assert "civilní dovolání" not in decision.clarification_question_cs.lower()


def test_domain_mismatch_after_retrieval_asks_clarification() -> None:
    query = _load_client_longform_04_question()
    hits = [
        RetrievalHitSummary(rank=1, document_id="ECLI:CZ:NS:2024:23.CDO.271.2024.1", score=0.0164),
        RetrievalHitSummary(rank=2, document_id="ECLI:CZ:NS:2024:8.TDO.1022.2024.1", score=0.0164),
        RetrievalHitSummary(rank=3, document_id="ECLI:CZ:NS:2025:4.TDO.1056.2024.1", score=0.0161),
    ]

    service = LegalQueryClarificationService(cache=InMemoryClarificationCache())
    decision = service.evaluate(query, retrieval_hits=hits)

    assert decision.decision == "ask_clarifying_question"
    assert "retrieval_domain_mismatch" in decision.ambiguity_types or "legal_domain_ambiguous" in decision.ambiguity_types
    assert decision.clarification_question_cs


def test_decision_schema_keys() -> None:
    service = LegalQueryClarificationService(cache=InMemoryClarificationCache())
    decision = service.evaluate(_load_client_longform_04_question())
    payload = decision.to_dict()

    required_keys = {
        "decision",
        "confidence",
        "ambiguity_types",
        "missing_slots",
        "detected_legal_domain",
        "detected_procedure_stage",
        "clarification_question_cs",
        "reason_cs",
        "cache_key",
        "query_signature",
        "cache_hit",
        "semantic_cache_hit",
        "llm_called",
    }
    assert required_keys.issubset(payload.keys())
    assert payload["cache_key"].startswith("legal_query_clarification:v1:")


def test_cache_keys_are_stable() -> None:
    exact_a = build_exact_query_cache_key("test query")
    exact_b = build_exact_query_cache_key("test query")
    pattern_a = build_pattern_cache_key(query_signature="appeal_dovolani_previous_proceeding_ambiguous")
    pattern_b = build_pattern_cache_key(query_signature="appeal_dovolani_previous_proceeding_ambiguous")
    assert exact_a == exact_b
    assert pattern_a == pattern_b
    assert exact_a != pattern_a


def test_decision_roundtrip_from_dict() -> None:
    original = ClarificationDecision(
        decision="ask_clarifying_question",
        confidence=0.84,
        ambiguity_types=["legal_domain_ambiguous"],
        missing_slots=["legal_domain"],
        detected_legal_domain="unknown",
        detected_procedure_stage="dovolani",
        clarification_question_cs="Jedná se o trestní dovolání?",
        reason_cs="test",
        cache_key="legal_query_clarification:v1:abc",
        query_signature="appeal_dovolani_previous_proceeding_ambiguous",
        recommended_next_action="ask_user",
        cache_hit=False,
        semantic_cache_hit=False,
        llm_called=False,
    )
    restored = ClarificationDecision.from_dict(original.to_dict())
    assert restored == original
