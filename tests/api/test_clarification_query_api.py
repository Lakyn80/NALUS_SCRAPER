from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.rag_router as rtr
from app.api.rag_router import get_orchestrator, router
from app.rag.clarification.cache import InMemoryClarificationCache
from app.rag.clarification.orchestrator import ClarifyingOrchestratorService
from app.rag.clarification.qdrant_patterns import InMemoryClarificationPatternIndex
from app.rag.clarification.service import LegalQueryClarificationService
from app.rag.orchestrator.orchestrator_service import OrchestratorResult
from app.rag.retrieval.models import RetrievedChunk

RAG_EVAL_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "rag_eval"
LONGFORM_DATASET = RAG_EVAL_DIR / "nalus_client_longform_eval_v1.json"

CLEAR_CRIMINAL_QUERY = (
    "V trestní věci byl obviněný odsouzen rozsudkem soudu prvního stupně a proti tomu podal "
    "odvolání. Odvolací soud jeho odvolání po věcném přezkoumání zamítl. Obviněný teď chce "
    "podat dovolání podle trestního řádu a tvrdí, že chyby vznikly už v řízení před soudem "
    "prvního stupně. Potřebuji najít trestní judikaturu Nejvyššího soudu k situaci, kdy se "
    "dovolatel dovolává vady předchozího řízení po zamítnutí odvolání."
)

CLEAR_CIVIL_QUERY = (
    "V civilním řízení žalobce prohrál spor, podal odvolání a odvolací soud rozsudek potvrdil. "
    "Teď chce podat dovolání k Nejvyššímu soudu podle občanského soudního řádu a tvrdí, že "
    "odvolací soud nesprávně posoudil právní otázku. Potřebuji najít civilní judikaturu "
    "k dovolání po rozhodnutí odvolacího soudu."
)

SIMILAR_AMBIGUOUS_QUERY = (
    "Prohrál jsem odvolání a chci se obrátit na Nejvyšší soud kvůli chybám, které vznikly "
    "už u soudu prvního stupně."
)

UNRELATED_FAMILY_QUERY = (
    "Řeším výživné a nevím, jestli mám podat nový návrh nebo odvolání. Potřebuji najít "
    "podobné případy, ale nevím přesně, jak to právně pojmenovat."
)

DOMAIN_QUESTION = (
    "Jedná se o trestní dovolání podle trestního řádu, nebo o civilní dovolání podle "
    "občanského soudního řádu?"
)


def _load_client_longform_04_question() -> str:
    payload = json.loads(LONGFORM_DATASET.read_text(encoding="utf-8"))
    for case in payload["cases"]:
        if case["id"] == "client-longform-04":
            return case["question"]
    raise AssertionError("client-longform-04 not found in dataset")


def _chunk(document_id: str, *, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        id=document_id,
        text=f"Relevant text for {document_id}",
        score=score,
        source="dense",
        metadata={
            "document_id": document_id,
            "case_reference": document_id,
            "source": "supreme",
        },
    )


class _FakeDelegateOrchestrator:
    def __init__(self, *, retrieve_map: dict[str, list[RetrievedChunk]]) -> None:
        self._retrieve_map = retrieve_map
        self.run_calls: list[str] = []
        self.retrieve_calls: list[tuple[str, int]] = []

    def run(self, query: str) -> OrchestratorResult:
        self.run_calls.append(query)
        return OrchestratorResult(
            answer=f"final:{query}",
            sources=[f"src:{query}"],
            plan_steps=["retrieve", "synthesize"],
        )

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        self.retrieve_calls.append((query, top_k))
        return list(self._retrieve_map.get(query, []))[:top_k]


def _make_client(orchestrator) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    return TestClient(app)


def _make_wrapped_orchestrator(*, retrieve_map: dict[str, list[RetrievedChunk]]):
    delegate = _FakeDelegateOrchestrator(retrieve_map=retrieve_map)
    clarification = LegalQueryClarificationService(
        cache=InMemoryClarificationCache(),
        pattern_index=InMemoryClarificationPatternIndex(),
    )
    wrapped = ClarifyingOrchestratorService(delegate, clarification_service=clarification)
    return wrapped, delegate


@pytest.fixture(autouse=True)
def _reset_router_cache_state() -> None:
    original_cache = rtr._query_cache
    original_backend = rtr._query_cache_backend
    original_error = rtr._query_cache_error
    original_corpus_version = rtr._corpus_version
    rtr._query_cache = None
    rtr._query_cache_backend = "none"
    rtr._query_cache_error = None
    rtr._corpus_version = "test-corpus"
    yield
    rtr._query_cache = original_cache
    rtr._query_cache_backend = original_backend
    rtr._query_cache_error = original_error
    rtr._corpus_version = original_corpus_version


def test_original_ambiguous_query_stops_before_retrieval() -> None:
    wrapped, delegate = _make_wrapped_orchestrator(retrieve_map={})
    client = _make_client(wrapped)

    response = client.post("/api/rag/query", json={"query": _load_client_longform_04_question()})

    assert response.status_code == 200
    assert response.json() == {
        "answer": DOMAIN_QUESTION,
        "sources": [],
        "plan_steps": [],
    }
    assert delegate.retrieve_calls == []
    assert delegate.run_calls == []
    assert wrapped.last_trace is not None
    assert wrapped.last_trace.final_decision.decision == "ask_clarifying_question"
    assert wrapped.last_trace.retrieval_ran is False
    assert wrapped.last_trace.llm_called is False


def test_clear_criminal_query_proceeds_to_retrieval() -> None:
    wrapped, delegate = _make_wrapped_orchestrator(
        retrieve_map={
            CLEAR_CRIMINAL_QUERY: [
                _chunk("ECLI:CZ:NS:2024:8.TDO.1022.2024.1"),
                _chunk("ECLI:CZ:NS:2025:4.TDO.1056.2024.1", score=0.88),
            ]
        }
    )
    client = _make_client(wrapped)

    response = client.post("/api/rag/query", json={"query": CLEAR_CRIMINAL_QUERY})

    assert response.status_code == 200
    assert response.json()["answer"] == f"final:{CLEAR_CRIMINAL_QUERY}"
    assert delegate.retrieve_calls == [(CLEAR_CRIMINAL_QUERY, 5)]
    assert delegate.run_calls == [CLEAR_CRIMINAL_QUERY]
    assert wrapped.last_trace is not None
    assert wrapped.last_trace.final_decision.decision == "proceed_to_retrieval"
    assert wrapped.last_trace.final_decision.detected_legal_domain == "criminal"
    assert wrapped.last_trace.final_decision.detected_procedure_stage == "dovolani"
    assert wrapped.last_trace.retrieval_ran is True
    assert all("TDO" in hit for hit in wrapped.last_trace.preview_hit_ids)


def test_clear_civil_query_proceeds_to_retrieval() -> None:
    wrapped, delegate = _make_wrapped_orchestrator(
        retrieve_map={
            CLEAR_CIVIL_QUERY: [
                _chunk("ECLI:CZ:NS:2024:23.CDO.271.2024.1"),
                _chunk("ECLI:CZ:NS:2024:30.CDO.1111.2024.1", score=0.86),
            ]
        }
    )
    client = _make_client(wrapped)

    response = client.post("/api/rag/query", json={"query": CLEAR_CIVIL_QUERY})

    assert response.status_code == 200
    assert response.json()["answer"] == f"final:{CLEAR_CIVIL_QUERY}"
    assert delegate.retrieve_calls == [(CLEAR_CIVIL_QUERY, 5)]
    assert delegate.run_calls == [CLEAR_CIVIL_QUERY]
    assert wrapped.last_trace is not None
    assert wrapped.last_trace.final_decision.decision == "proceed_to_retrieval"
    assert wrapped.last_trace.final_decision.detected_legal_domain == "civil"
    assert wrapped.last_trace.final_decision.detected_procedure_stage == "dovolani"
    assert wrapped.last_trace.retrieval_ran is True
    assert all("CDO" in hit for hit in wrapped.last_trace.preview_hit_ids)


def test_similar_ambiguous_query_reuses_semantic_clarification() -> None:
    wrapped, delegate = _make_wrapped_orchestrator(retrieve_map={})
    client = _make_client(wrapped)

    seed = client.post("/api/rag/query", json={"query": _load_client_longform_04_question()})
    response = client.post("/api/rag/query", json={"query": SIMILAR_AMBIGUOUS_QUERY})

    assert seed.status_code == 200
    assert response.status_code == 200
    assert response.json()["answer"] == DOMAIN_QUESTION
    assert delegate.retrieve_calls == []
    assert delegate.run_calls == []
    assert wrapped.last_trace is not None
    assert wrapped.last_trace.final_decision.decision == "ask_clarifying_question"
    assert wrapped.last_trace.semantic_reuse_hit is True
    assert wrapped.last_trace.llm_called is False
    assert wrapped.last_trace.retrieval_ran is False


def test_family_query_does_not_reuse_wrong_dovolani_clarification() -> None:
    wrapped, delegate = _make_wrapped_orchestrator(retrieve_map={})
    client = _make_client(wrapped)

    client.post("/api/rag/query", json={"query": _load_client_longform_04_question()})
    response = client.post("/api/rag/query", json={"query": UNRELATED_FAMILY_QUERY})

    assert response.status_code == 200
    assert DOMAIN_QUESTION not in response.json()["answer"]
    assert (
        "odvol" in response.json()["answer"].lower()
        or "dovol" in response.json()["answer"].lower()
        or "proces" in response.json()["answer"].lower()
    )
    assert delegate.retrieve_calls == []
    assert delegate.run_calls == []
    assert wrapped.last_trace is not None
    assert wrapped.last_trace.semantic_reuse_hit is False
    assert wrapped.last_trace.retrieval_ran is False


def test_post_retrieval_mismatch_guard_asks_clarification() -> None:
    wrapped, delegate = _make_wrapped_orchestrator(
        retrieve_map={
            CLEAR_CIVIL_QUERY: [
                _chunk("ECLI:CZ:NS:2024:23.CDO.271.2024.1"),
                _chunk("ECLI:CZ:NS:2024:8.TDO.1022.2024.1", score=0.89),
                _chunk("ECLI:CZ:NS:2025:4.TDO.1056.2024.1", score=0.87),
            ]
        }
    )
    client = _make_client(wrapped)

    response = client.post("/api/rag/query", json={"query": CLEAR_CIVIL_QUERY})

    assert response.status_code == 200
    assert response.json() == {
        "answer": DOMAIN_QUESTION,
        "sources": [],
        "plan_steps": [],
    }
    assert delegate.retrieve_calls == [(CLEAR_CIVIL_QUERY, 5)]
    assert delegate.run_calls == []
    assert wrapped.last_trace is not None
    assert wrapped.last_trace.final_decision.decision == "ask_clarifying_question"
    assert (
        "civilní" in wrapped.last_trace.final_decision.reason_cs.lower()
        or "trestní" in wrapped.last_trace.final_decision.reason_cs.lower()
        or "vyhledávání" in wrapped.last_trace.final_decision.reason_cs.lower()
    )
    assert wrapped.last_trace.retrieval_ran is True
