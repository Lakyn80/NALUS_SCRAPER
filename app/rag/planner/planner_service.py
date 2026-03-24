"""
Planner Service — decomposes a complex legal query into ordered sub-queries.

Uses an injected BaseTextLLM (same interface as QueryRewriteService).
Returns a Plan with 2–5 PlanStep objects, each carrying a sub-query and
a reason.  Falls back to a single-step plan on parse failure so callers
never have to handle exceptions.

Usage:
    from app.rag.planner.planner_service import PlannerService, MockPlannerLLM

    planner = PlannerService(llm=MockPlannerLLM())
    plan = planner.plan("matka unesla dítě do Ruska, otec žádá návrat")
    for step in plan.steps:
        results = retrieval.search(step.query)
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.rag.rewrite.query_rewrite_service import BaseTextLLM

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_STEPS = 2
_MAX_STEPS = 5

_PROMPT_TEMPLATE = """\
Jsi právní systém pro analýzu dotazů.

Rozlož uživatelský dotaz na menší právní dílčí dotazy vhodné pro vyhledávání v judikatuře.

Pravidla:
- rozlož na {min_steps}–{max_steps} kroků
- každý krok musí být samostatný vyhledávací dotaz
- použij český jazyk a právní terminologii
- žádné vysvětlování mimo strukturu

Formát výstupu (každý řádek):
1. <dotaz> | <důvod>
2. <dotaz> | <důvod>

Uživatelský dotaz:
{query}

Plán:"""

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class PlanStep:
    def __init__(self, query: str, reason: str) -> None:
        self.query = query
        self.reason = reason

    def __repr__(self) -> str:
        return f"PlanStep(query={self.query!r}, reason={self.reason!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PlanStep):
            return NotImplemented
        return self.query == other.query and self.reason == other.reason


class Plan:
    def __init__(self, steps: list[PlanStep]) -> None:
        self.steps = steps

    def __repr__(self) -> str:
        return f"Plan(steps={self.steps!r})"

    def __len__(self) -> int:
        return len(self.steps)


# ---------------------------------------------------------------------------
# Mock LLM for tests and local development
# ---------------------------------------------------------------------------


class MockPlannerLLM(BaseTextLLM):
    """Deterministic stub that returns a fixed 3-step plan in Czech."""

    def generate_text(self, prompt: str) -> str:
        # Extract the user query from the prompt (last non-empty line before "Plán:")
        lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
        try:
            idx = lines.index("Plán:")
            original = lines[idx - 1]
        except (ValueError, IndexError):
            original = "právní dotaz"

        return (
            f"1. {original} Haagská úmluva | mezinárodní aspekt dotazu\n"
            f"2. rodičovská odpovědnost únos dítěte | určení příslušnosti soudu\n"
            f"3. návrat dítěte do země obvyklého bydliště | výkon rozhodnutí"
        )


# ---------------------------------------------------------------------------
# PlannerService
# ---------------------------------------------------------------------------


class PlannerService:
    """Decomposes a complex query into a Plan of legal sub-queries."""

    def __init__(self, llm: BaseTextLLM) -> None:
        self._llm = llm

    def plan(self, query: str) -> Plan:
        """Return a Plan, falling back to a single-step plan on any failure."""
        prompt = _build_prompt(query)
        try:
            response = self._llm.generate_text(prompt)
            steps = _parse(response)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[planner] LLM failed (%s); falling back to single step", exc)
            steps = [PlanStep(query=query, reason="fallback — LLM error")]

        if not steps:
            logger.warning("[planner] empty parse result; falling back to single step")
            steps = [PlanStep(query=query, reason="fallback — empty plan")]

        plan = Plan(steps=steps)
        logger.info("[planner] steps=%d query=%s", len(plan.steps), query)
        return plan


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_prompt(query: str) -> str:
    return _PROMPT_TEMPLATE.format(
        min_steps=_MIN_STEPS,
        max_steps=_MAX_STEPS,
        query=query,
    )


def _parse(response: str) -> list[PlanStep]:
    """Parse numbered lines of the form 'N. <query> | <reason>' into PlanSteps.

    Lines that do not match the expected format are silently skipped.
    Returns at most _MAX_STEPS steps.
    """
    steps: list[PlanStep] = []
    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue

        # Strip leading number and dot: "1. foo | bar" → "foo | bar"
        if line[0].isdigit():
            dot_pos = line.find(".")
            if dot_pos != -1:
                line = line[dot_pos + 1 :].strip()

        if "|" not in line:
            continue

        parts = line.split("|", maxsplit=1)
        query_part = parts[0].strip()
        reason_part = parts[1].strip() if len(parts) > 1 else ""

        if not query_part:
            continue

        steps.append(PlanStep(query=query_part, reason=reason_part))
        if len(steps) >= _MAX_STEPS:
            break

    return steps
