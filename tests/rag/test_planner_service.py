"""
Unit tests for app.rag.planner.planner_service.

Run:
    pytest tests/rag/test_planner_service.py -v
"""

import logging

import pytest

from app.rag.planner.planner_service import (
    MockPlannerLLM,
    Plan,
    PlanStep,
    PlannerService,
    _build_prompt,
    _parse,
)
from app.rag.rewrite.query_rewrite_service import BaseTextLLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FixedLLM(BaseTextLLM):
    def __init__(self, response: str) -> None:
        self._response = response

    def generate_text(self, prompt: str) -> str:
        return self._response


class _FailingLLM(BaseTextLLM):
    def generate_text(self, prompt: str) -> str:
        raise RuntimeError("unavailable")


class _SpyLLM(BaseTextLLM):
    def __init__(self, response: str = "1. sub-query | reason") -> None:
        self.calls: list[str] = []
        self._response = response

    def generate_text(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._response


# ---------------------------------------------------------------------------
# PlanStep
# ---------------------------------------------------------------------------


class TestPlanStep:
    def test_stores_query_and_reason(self) -> None:
        step = PlanStep("únos dítěte", "mezinárodní aspekt")
        assert step.query == "únos dítěte"
        assert step.reason == "mezinárodní aspekt"

    def test_equality(self) -> None:
        a = PlanStep("q", "r")
        b = PlanStep("q", "r")
        assert a == b

    def test_inequality(self) -> None:
        assert PlanStep("q1", "r") != PlanStep("q2", "r")

    def test_repr_contains_query(self) -> None:
        assert "únos" in repr(PlanStep("únos dítěte", "r"))


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


class TestPlan:
    def test_stores_steps(self) -> None:
        steps = [PlanStep("q1", "r1"), PlanStep("q2", "r2")]
        plan = Plan(steps)
        assert plan.steps == steps

    def test_len(self) -> None:
        plan = Plan([PlanStep("q", "r")] * 3)
        assert len(plan) == 3

    def test_empty_plan(self) -> None:
        assert len(Plan([])) == 0


# ---------------------------------------------------------------------------
# _parse (pure helper)
# ---------------------------------------------------------------------------


class TestParse:
    def test_parses_numbered_lines(self) -> None:
        raw = "1. únos dítěte Haagská úmluva | mezinárodní aspekt\n2. rodičovská odpovědnost | příslušnost soudu"
        steps = _parse(raw)
        assert len(steps) == 2

    def test_first_step_query(self) -> None:
        steps = _parse("1. únos dítěte | důvod")
        assert steps[0].query == "únos dítěte"

    def test_first_step_reason(self) -> None:
        steps = _parse("1. únos dítěte | mezinárodní aspekt")
        assert steps[0].reason == "mezinárodní aspekt"

    def test_skips_lines_without_pipe(self) -> None:
        raw = "úvod\n1. dotaz | důvod\nzávěr"
        steps = _parse(raw)
        assert len(steps) == 1

    def test_skips_empty_lines(self) -> None:
        raw = "\n1. dotaz | důvod\n\n2. dotaz2 | důvod2\n"
        assert len(_parse(raw)) == 2

    def test_caps_at_max_five_steps(self) -> None:
        raw = "\n".join(f"{i}. dotaz{i} | důvod{i}" for i in range(1, 10))
        assert len(_parse(raw)) == 5

    def test_empty_response_returns_empty(self) -> None:
        assert _parse("") == []

    def test_handles_extra_pipes_in_reason(self) -> None:
        steps = _parse("1. dotaz | důvod | extra")
        assert steps[0].reason == "důvod | extra"

    def test_strips_whitespace(self) -> None:
        steps = _parse("1.   dotaz s mezerami   |   důvod s mezerami   ")
        assert steps[0].query == "dotaz s mezerami"
        assert steps[0].reason == "důvod s mezerami"


# ---------------------------------------------------------------------------
# MockPlannerLLM
# ---------------------------------------------------------------------------


class TestMockPlannerLLM:
    def test_returns_string(self) -> None:
        result = MockPlannerLLM().generate_text("any prompt")
        assert isinstance(result, str)

    def test_contains_pipe_separators(self) -> None:
        result = MockPlannerLLM().generate_text(_build_prompt("únos dítěte"))
        assert "|" in result

    def test_contains_numbered_lines(self) -> None:
        result = MockPlannerLLM().generate_text(_build_prompt("únos dítěte"))
        lines = [ln.strip() for ln in result.splitlines() if ln.strip()]
        assert any(ln[0].isdigit() for ln in lines)

    def test_deterministic(self) -> None:
        llm = MockPlannerLLM()
        prompt = _build_prompt("test")
        assert llm.generate_text(prompt) == llm.generate_text(prompt)

    def test_original_query_in_output(self) -> None:
        query = "matka unesla dítě do Ruska"
        result = MockPlannerLLM().generate_text(_build_prompt(query))
        assert query in result


# ---------------------------------------------------------------------------
# PlannerService — return type and structure
# ---------------------------------------------------------------------------


class TestPlannerServiceReturnType:
    def test_returns_plan_instance(self) -> None:
        result = PlannerService(MockPlannerLLM()).plan("únos dítěte")
        assert isinstance(result, Plan)

    def test_steps_is_list(self) -> None:
        result = PlannerService(MockPlannerLLM()).plan("q")
        assert isinstance(result.steps, list)

    def test_each_step_is_plan_step(self) -> None:
        result = PlannerService(MockPlannerLLM()).plan("q")
        for step in result.steps:
            assert isinstance(step, PlanStep)

    def test_step_query_is_str(self) -> None:
        result = PlannerService(MockPlannerLLM()).plan("q")
        assert all(isinstance(s.query, str) for s in result.steps)

    def test_step_reason_is_str(self) -> None:
        result = PlannerService(MockPlannerLLM()).plan("q")
        assert all(isinstance(s.reason, str) for s in result.steps)


# ---------------------------------------------------------------------------
# PlannerService — step count
# ---------------------------------------------------------------------------


class TestPlannerServiceStepCount:
    def test_mock_returns_steps(self) -> None:
        result = PlannerService(MockPlannerLLM()).plan("únos dítěte Rusko")
        assert len(result.steps) > 0

    def test_mock_returns_at_most_five_steps(self) -> None:
        result = PlannerService(MockPlannerLLM()).plan("q")
        assert len(result.steps) <= 5

    def test_three_line_response_gives_three_steps(self) -> None:
        raw = "1. dotaz1 | důvod1\n2. dotaz2 | důvod2\n3. dotaz3 | důvod3"
        result = PlannerService(_FixedLLM(raw)).plan("q")
        assert len(result.steps) == 3

    def test_two_line_response_gives_two_steps(self) -> None:
        raw = "1. dotaz1 | důvod1\n2. dotaz2 | důvod2"
        result = PlannerService(_FixedLLM(raw)).plan("q")
        assert len(result.steps) == 2


# ---------------------------------------------------------------------------
# PlannerService — Czech language
# ---------------------------------------------------------------------------


class TestPlannerServiceCzech:
    def test_step_queries_contain_czech_chars(self) -> None:
        result = PlannerService(MockPlannerLLM()).plan("matka unesla dítě do Ruska")
        all_text = " ".join(s.query for s in result.steps).lower()
        czech_chars = set("áčďéěíňóřšťúůýž")
        assert any(ch in all_text for ch in czech_chars)

    def test_prompt_is_in_czech(self) -> None:
        prompt = _build_prompt("únos dítěte")
        assert "dotaz" in prompt.lower() or "plán" in prompt.lower()

    def test_prompt_contains_original_query(self) -> None:
        query = "rodičovská odpovědnost po rozvodu"
        assert query in _build_prompt(query)


# ---------------------------------------------------------------------------
# PlannerService — fallback
# ---------------------------------------------------------------------------


class TestPlannerServiceFallback:
    def test_llm_exception_returns_single_step_fallback(self) -> None:
        result = PlannerService(_FailingLLM()).plan("únos dítěte")
        assert len(result.steps) == 1
        assert result.steps[0].query == "únos dítěte"

    def test_fallback_step_reason_mentions_fallback(self) -> None:
        result = PlannerService(_FailingLLM()).plan("q")
        assert "fallback" in result.steps[0].reason.lower()

    def test_empty_llm_response_returns_single_step_fallback(self) -> None:
        result = PlannerService(_FixedLLM("")).plan("únos dítěte")
        assert len(result.steps) == 1
        assert result.steps[0].query == "únos dítěte"

    def test_unparseable_response_returns_fallback(self) -> None:
        result = PlannerService(_FixedLLM("no pipes anywhere")).plan("q")
        assert len(result.steps) == 1

    def test_fallback_logs_warning_on_exception(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="app.rag.planner.planner_service"):
            PlannerService(_FailingLLM()).plan("q")
        assert any(
            "fallback" in r.getMessage().lower() or "falling back" in r.getMessage().lower()
            for r in caplog.records
        )


# ---------------------------------------------------------------------------
# PlannerService — logging
# ---------------------------------------------------------------------------


class TestPlannerServiceLogging:
    def test_planner_info_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.rag.planner.planner_service"):
            PlannerService(MockPlannerLLM()).plan("únos dítěte")
        msgs = [r.getMessage() for r in caplog.records]
        assert any("[planner]" in m for m in msgs)

    def test_log_contains_step_count(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.rag.planner.planner_service"):
            PlannerService(MockPlannerLLM()).plan("únos dítěte")
        planner_log = next(
            r.getMessage() for r in caplog.records if "[planner]" in r.getMessage()
        )
        assert "steps=" in planner_log

    def test_log_contains_query(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.rag.planner.planner_service"):
            PlannerService(MockPlannerLLM()).plan("délka řízení průtahy")
        planner_log = next(
            r.getMessage() for r in caplog.records if "[planner]" in r.getMessage()
        )
        assert "délka řízení průtahy" in planner_log


# ---------------------------------------------------------------------------
# PlannerService — LLM called correctly
# ---------------------------------------------------------------------------


class TestPlannerServiceLLMCall:
    def test_llm_called_once(self) -> None:
        spy = _SpyLLM()
        PlannerService(spy).plan("q")
        assert len(spy.calls) == 1

    def test_prompt_contains_query(self) -> None:
        spy = _SpyLLM()
        PlannerService(spy).plan("mezinárodní únos dítěte")
        assert "mezinárodní únos dítěte" in spy.calls[0]

    def test_prompt_specifies_step_range(self) -> None:
        spy = _SpyLLM()
        PlannerService(spy).plan("q")
        assert "2" in spy.calls[0] and "5" in spy.calls[0]
