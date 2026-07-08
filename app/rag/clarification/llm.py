from __future__ import annotations

from abc import ABC, abstractmethod

from app.rag.clarification.models import AmbiguityType, RuleAssessment

_CLARIFICATION_PROMPT = """\
Vygeneruj jednu krátkou upřesňující otázku v češtině pro právní vyhledávání.

Pravidla:
- maximálně 2 věty
- žádné právní rady
- nevymýšlej skutkový stav
- ptej se jen na chybějící informace potřebné pro bezpečné vyhledávání
- jednoduchý jazyk pro klienta
- neptej se na ECLI
- neptej se na přesné paragrafy, pokud to není nutné

Typy nejasnosti: {ambiguity_types}
Chybějící informace: {missing_slots}
Původní dotaz:
{query}

Upřesňující otázka:"""

_DOMAIN_DOVOLANI_QUESTION = (
    "Jedná se o trestní dovolání podle trestního řádu, "
    "nebo o civilní dovolání podle občanského soudního řádu?"
)

_GENERIC_FALLBACK = (
    "Potřebuji upřesnit právní kontext dotazu, aby vyhledávání nenašlo podobné, "
    "ale špatně ukotvené rozhodnutí."
)

_TEMPLATE_QUESTIONS: dict[AmbiguityType, str] = {
    "legal_domain_ambiguous": _DOMAIN_DOVOLANI_QUESTION,
    "procedure_stage_ambiguous": (
        "V jakém stadiu řízení se věc nachází: první instance, odvolání, dovolání, nebo exekuce?"
    ),
    "role_ambiguous": (
        "Kdo je vaším klientem v řízení (např. obviněný, žalobce, povinný), abych mohl najít správnou judikaturu?"
    ),
    "remedy_ambiguous": (
        "Hledáte judikaturu k odvolání, dovolání, exekuci, nebo k jinému procesnímu kroku?"
    ),
    "jurisdiction_or_court_ambiguous": (
        "Potřebujete judikaturu Nejvyššího soudu v civilní, trestní, nebo jiné věci?"
    ),
    "retrieval_domain_mismatch": _DOMAIN_DOVOLANI_QUESTION,
}


class BaseClarificationLLM(ABC):
    @abstractmethod
    def generate_clarification(
        self,
        *,
        query: str,
        ambiguity_types: list[AmbiguityType],
        missing_slots: list[str],
    ) -> str: ...


class TemplateClarificationLLM(BaseClarificationLLM):
    """Deterministic templates — preferred over LLM for known ambiguity patterns."""

    def generate_clarification(
        self,
        *,
        query: str,
        ambiguity_types: list[AmbiguityType],
        missing_slots: list[str],
    ) -> str:
        del query, missing_slots
        for ambiguity_type in ambiguity_types:
            template = _TEMPLATE_QUESTIONS.get(ambiguity_type)
            if template:
                return template
        return _GENERIC_FALLBACK


class CountingClarificationLLM(BaseClarificationLLM):
    """Test helper that tracks whether an expensive LLM backend was invoked."""

    def __init__(self, inner: BaseClarificationLLM) -> None:
        self.inner = inner
        self.call_count = 0

    def generate_clarification(
        self,
        *,
        query: str,
        ambiguity_types: list[AmbiguityType],
        missing_slots: list[str],
    ) -> str:
        self.call_count += 1
        return self.inner.generate_clarification(
            query=query,
            ambiguity_types=ambiguity_types,
            missing_slots=missing_slots,
        )


class LLMClarificationGenerator(BaseClarificationLLM):
    """Calls external LLM only when templates do not cover the ambiguity."""

    def __init__(self, llm: object, *, fallback: BaseClarificationLLM | None = None) -> None:
        self._llm = llm
        self._fallback = fallback or TemplateClarificationLLM()

    def generate_clarification(
        self,
        *,
        query: str,
        ambiguity_types: list[AmbiguityType],
        missing_slots: list[str],
    ) -> str:
        template_answer = self._fallback.generate_clarification(
            query=query,
            ambiguity_types=ambiguity_types,
            missing_slots=missing_slots,
        )
        if template_answer != _GENERIC_FALLBACK:
            return template_answer

        prompt = _CLARIFICATION_PROMPT.format(
            ambiguity_types=", ".join(ambiguity_types) or "neznámé",
            missing_slots=", ".join(missing_slots) or "neznámé",
            query=query.strip(),
        )
        try:
            generated = " ".join(self._llm.generate_text(prompt).split())
        except Exception:
            return template_answer
        if not generated or len(generated) < 12:
            return template_answer
        return generated


def build_clarification_question(
    *,
    query: str,
    assessment: RuleAssessment,
    llm: BaseClarificationLLM | None = None,
) -> tuple[str, bool]:
    """Return (question, llm_called). Templates are always tried first."""
    generator = llm or TemplateClarificationLLM()
    if isinstance(generator, LLMClarificationGenerator):
        template_only = TemplateClarificationLLM()
        template_answer = template_only.generate_clarification(
            query=query,
            ambiguity_types=list(assessment.ambiguity_types),
            missing_slots=list(assessment.missing_slots),
        )
        if template_answer != _GENERIC_FALLBACK:
            return template_answer, False
        return (
            generator.generate_clarification(
                query=query,
                ambiguity_types=list(assessment.ambiguity_types),
                missing_slots=list(assessment.missing_slots),
            ),
            True,
        )

    if isinstance(generator, CountingClarificationLLM):
        before = generator.call_count
        answer = generator.generate_clarification(
            query=query,
            ambiguity_types=list(assessment.ambiguity_types),
            missing_slots=list(assessment.missing_slots),
        )
        return answer, generator.call_count > before

    answer = generator.generate_clarification(
        query=query,
        ambiguity_types=list(assessment.ambiguity_types),
        missing_slots=list(assessment.missing_slots),
    )
    return answer, False
