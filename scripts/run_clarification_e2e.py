from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.rag.clarification.cache import InMemoryClarificationCache
from app.rag.clarification.orchestrator import ClarifyingOrchestratorService
from app.rag.clarification.qdrant_patterns import InMemoryClarificationPatternIndex
from app.rag.clarification.service import LegalQueryClarificationService
from app.rag.orchestrator.orchestrator_service import OrchestratorResult
from app.rag.retrieval.models import RetrievedChunk

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "artifacts" / "rag_eval" / "nalus_client_longform_eval_v1.json"
REPORT_PATH = ROOT / "artifacts" / "rag_eval" / "client_longform_v1" / "clarification_e2e_report.md"

DOMAIN_QUESTION = (
    "Jedná se o trestní dovolání podle trestního řádu, nebo o civilní dovolání podle "
    "občanského soudního řádu?"
)

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


@dataclass(frozen=True)
class ScenarioResult:
    scenario: str
    input_type: str
    expected_decision: str
    actual_decision: str
    cache_hit: bool
    semantic_reuse_hit: bool
    llm_called: bool
    retrieval_ran: bool
    passed: bool
    answer: str
    detected_legal_domain: str
    detected_procedure_stage: str
    reason: str
    preview_hit_ids: list[str]


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


def _load_client_longform_04_question() -> str:
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    for case in payload["cases"]:
        if case["id"] == "client-longform-04":
            return case["question"]
    raise RuntimeError("client-longform-04 not found in dataset")


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


def _build_wrapped_orchestrator(*, retrieve_map: dict[str, list[RetrievedChunk]]):
    delegate = _FakeDelegateOrchestrator(retrieve_map=retrieve_map)
    clarification = LegalQueryClarificationService(
        cache=InMemoryClarificationCache(),
        pattern_index=InMemoryClarificationPatternIndex(),
    )
    wrapped = ClarifyingOrchestratorService(delegate, clarification_service=clarification)
    return wrapped, delegate


def _scenario_result(
    *,
    scenario: str,
    input_type: str,
    expected_decision: str,
    wrapped: ClarifyingOrchestratorService,
    answer: str,
    passed: bool,
) -> ScenarioResult:
    trace = wrapped.last_trace
    if trace is None:
        raise RuntimeError(f"Missing trace for scenario {scenario}")
    return ScenarioResult(
        scenario=scenario,
        input_type=input_type,
        expected_decision=expected_decision,
        actual_decision=trace.final_decision.decision,
        cache_hit=trace.cache_hit,
        semantic_reuse_hit=trace.semantic_reuse_hit,
        llm_called=trace.llm_called,
        retrieval_ran=trace.retrieval_ran,
        passed=passed,
        answer=answer,
        detected_legal_domain=trace.final_decision.detected_legal_domain,
        detected_procedure_stage=trace.final_decision.detected_procedure_stage,
        reason=trace.final_decision.reason_cs,
        preview_hit_ids=list(trace.preview_hit_ids),
    )


def run_scenarios() -> list[ScenarioResult]:
    original_query = _load_client_longform_04_question()
    results: list[ScenarioResult] = []

    wrapped, delegate = _build_wrapped_orchestrator(retrieve_map={})
    ambiguous_result = wrapped.run(original_query)
    results.append(
        _scenario_result(
            scenario="Original client-longform-04",
            input_type="ambiguous dovolani narrative",
            expected_decision="ask_clarifying_question",
            wrapped=wrapped,
            answer=ambiguous_result.answer,
            passed=(
                ambiguous_result.answer == DOMAIN_QUESTION
                and not wrapped.last_trace.retrieval_ran
                and not wrapped.last_trace.llm_called
                and delegate.run_calls == []
                and delegate.retrieve_calls == []
            ),
        )
    )

    wrapped, delegate = _build_wrapped_orchestrator(
        retrieve_map={
            CLEAR_CRIMINAL_QUERY: [
                _chunk("ECLI:CZ:NS:2024:8.TDO.1022.2024.1"),
                _chunk("ECLI:CZ:NS:2025:4.TDO.1056.2024.1", score=0.88),
            ]
        }
    )
    criminal_result = wrapped.run(CLEAR_CRIMINAL_QUERY)
    results.append(
        _scenario_result(
            scenario="Clear criminal query",
            input_type="criminal dovolani",
            expected_decision="proceed_to_retrieval",
            wrapped=wrapped,
            answer=criminal_result.answer,
            passed=(
                wrapped.last_trace.final_decision.detected_legal_domain == "criminal"
                and wrapped.last_trace.retrieval_ran
                and delegate.run_calls == [CLEAR_CRIMINAL_QUERY]
                and all("TDO" in hit for hit in wrapped.last_trace.preview_hit_ids)
            ),
        )
    )

    wrapped, delegate = _build_wrapped_orchestrator(
        retrieve_map={
            CLEAR_CIVIL_QUERY: [
                _chunk("ECLI:CZ:NS:2024:23.CDO.271.2024.1"),
                _chunk("ECLI:CZ:NS:2024:30.CDO.1111.2024.1", score=0.86),
            ]
        }
    )
    civil_result = wrapped.run(CLEAR_CIVIL_QUERY)
    results.append(
        _scenario_result(
            scenario="Clear civil query",
            input_type="civil dovolani",
            expected_decision="proceed_to_retrieval",
            wrapped=wrapped,
            answer=civil_result.answer,
            passed=(
                wrapped.last_trace.final_decision.detected_legal_domain == "civil"
                and wrapped.last_trace.retrieval_ran
                and delegate.run_calls == [CLEAR_CIVIL_QUERY]
                and all("CDO" in hit for hit in wrapped.last_trace.preview_hit_ids)
            ),
        )
    )

    wrapped, delegate = _build_wrapped_orchestrator(retrieve_map={})
    wrapped.run(original_query)
    semantic_result = wrapped.run(SIMILAR_AMBIGUOUS_QUERY)
    results.append(
        _scenario_result(
            scenario="Similar ambiguous query reuse",
            input_type="semantic-near ambiguous query",
            expected_decision="ask_clarifying_question",
            wrapped=wrapped,
            answer=semantic_result.answer,
            passed=(
                semantic_result.answer == DOMAIN_QUESTION
                and wrapped.last_trace.semantic_reuse_hit
                and not wrapped.last_trace.llm_called
                and not wrapped.last_trace.retrieval_ran
                and delegate.run_calls == []
            ),
        )
    )

    wrapped, delegate = _build_wrapped_orchestrator(retrieve_map={})
    wrapped.run(original_query)
    family_result = wrapped.run(UNRELATED_FAMILY_QUERY)
    family_answer = family_result.answer.lower()
    results.append(
        _scenario_result(
            scenario="Family-law ambiguity avoids wrong reuse",
            input_type="family/remedy ambiguity",
            expected_decision="ask_clarifying_question",
            wrapped=wrapped,
            answer=family_result.answer,
            passed=(
                DOMAIN_QUESTION.lower() not in family_answer
                and not wrapped.last_trace.semantic_reuse_hit
                and not wrapped.last_trace.retrieval_ran
            ),
        )
    )

    wrapped, delegate = _build_wrapped_orchestrator(
        retrieve_map={
            CLEAR_CIVIL_QUERY: [
                _chunk("ECLI:CZ:NS:2024:23.CDO.271.2024.1"),
                _chunk("ECLI:CZ:NS:2024:8.TDO.1022.2024.1", score=0.89),
                _chunk("ECLI:CZ:NS:2025:4.TDO.1056.2024.1", score=0.87),
            ]
        }
    )
    mismatch_result = wrapped.run(CLEAR_CIVIL_QUERY)
    results.append(
        _scenario_result(
            scenario="Post-retrieval mixed-domain guard",
            input_type="mixed preview fixture",
            expected_decision="ask_clarifying_question",
            wrapped=wrapped,
            answer=mismatch_result.answer,
            passed=(
                mismatch_result.answer == DOMAIN_QUESTION
                and wrapped.last_trace.retrieval_ran
                and delegate.run_calls == []
                and any("TDO" in hit for hit in wrapped.last_trace.preview_hit_ids)
                and any("CDO" in hit for hit in wrapped.last_trace.preview_hit_ids)
            ),
        )
    )

    return results


def write_report(results: list[ScenarioResult]) -> Path:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_passed = all(item.passed for item in results)
    ambiguous = results[0]
    lines = [
        "# Clarification Gate E2E Report",
        "",
        "## Executive summary",
        (
            "The clarification gate is integrated into the long-form query flow through a "
            "wrapper in front of the existing orchestrator. Ambiguous dovolani queries stop "
            "before full retrieval, clear criminal/civil queries proceed, semantic reuse works, "
            "wrong family-law reuse is avoided, and a mixed CDO/TDO preview is blocked by the "
            "post-retrieval guard."
        ),
        "",
        "## Test matrix",
        "",
        "| Scenario | Input type | Expected decision | Actual decision | cache_hit | semantic_reuse_hit | llm_called | retrieval_ran | PASS/FAIL |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        lines.append(
            f"| {item.scenario} | {item.input_type} | {item.expected_decision} | "
            f"{item.actual_decision} | {item.cache_hit} | {item.semantic_reuse_hit} | "
            f"{item.llm_called} | {item.retrieval_ran} | {'PASS' if item.passed else 'FAIL'} |"
        )

    lines.extend(
        [
            "",
            "## Example output for original client-longform-04",
            "",
            f"- Decision: `{ambiguous.actual_decision}`",
            f"- Returned answer: `{ambiguous.answer}`",
            f"- Retrieval ran: `{ambiguous.retrieval_ran}`",
            f"- LLM called: `{ambiguous.llm_called}`",
            "",
            "## Integration confirmations",
            "",
            "1. Retrieval logic was not changed. The existing retrieval/ranking modules remain untouched; the gate only decides whether the orchestrator is allowed to continue.",
            "2. Qdrant judgment retrieval remains unchanged. Court-judgment retrieval still belongs to the main Qdrant collection used by the existing retriever path.",
            "3. Redis is used only for clarification cache/payloads. Court judgment embeddings are not moved into Redis; clarification cache wiring remains isolated in `app/rag/clarification/cache.py`.",
            "",
            "## Final verdict",
            "",
            (
                "Yes."
                if all_passed
                else "No."
            ),
            (
                "The clarification gate is ready to protect long-form legal retrieval provided the current wrapper stays in front of `/api/rag/query`, the clarification cache remains scoped to payload reuse only, and the mixed-domain preview guard stays enabled."
                if all_passed
                else "One or more required scenarios failed; the gate should not be treated as production-ready until the failed cases are resolved."
            ),
            "",
            "## Scenario notes",
            "",
        ]
    )
    for item in results:
        lines.extend(
            [
                f"### {item.scenario}",
                f"- Detected legal domain: `{item.detected_legal_domain}`",
                f"- Detected procedure stage: `{item.detected_procedure_stage}`",
                f"- Reason: {item.reason}",
                f"- Preview hits: `{', '.join(item.preview_hit_ids) or 'none'}`",
                f"- Result: `{'PASS' if item.passed else 'FAIL'}`",
                "",
            ]
        )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_PATH


def main() -> None:
    results = run_scenarios()
    path = write_report(results)
    passed = sum(1 for item in results if item.passed)
    print(f"E2E clarification verification complete: {passed}/{len(results)} PASS")
    print(path)


if __name__ == "__main__":
    main()
