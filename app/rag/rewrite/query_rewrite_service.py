"""
Query Rewrite Service — rewrites user queries into precise legal search queries.

The injected llm must implement generate_text(prompt: str) -> str.
This is a separate interface from BaseLLM (which is structured RAG output).
Use MockTextLLM for tests, plug in any real provider via BaseTextLLM.

Usage:
    from app.rag.rewrite.query_rewrite_service import QueryRewriteService, MockTextLLM

    service = QueryRewriteService(llm=MockTextLLM())
    rewritten = service.rewrite("matka unesla dítě do Ruska")
    # → "mezinárodní únos dítěte Haagská úmluva návrat dítěte rodičovská odpovědnost"
"""

from abc import ABC, abstractmethod

from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Text LLM interface (text-in / text-out — separate from BaseLLM)
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
Přepiš uživatelský dotaz do přesnějšího právního vyhledávacího dotazu.

Pravidla:
- použij právní terminologii
- rozšiř vágní pojmy o synonyma a příbuzné koncepty
- zachovej český jazyk
- nevysvětluj, vypiš pouze přepsaný dotaz
- výstup musí být jednořádkový

Vstupní dotaz:
{query}

Přepsaný dotaz:"""


class BaseTextLLM(ABC):
    """Minimal contract for text-in / text-out LLM backends."""

    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """Return generated text for the given prompt."""


class MockTextLLM(BaseTextLLM):
    """Deterministic stub — expands query with common legal terms.

    Used in tests and local development.  No external calls.
    """

    # Legal expansions appended to make the rewrite visibly different from input.
    _EXPANSION = (
        "ústavní soud judikatura právní věta základní práva"
    )

    def generate_text(self, prompt: str) -> str:
        # Extract the original query from the prompt (last non-empty line before
        # "Přepsaný dotaz:" marker) and append legal expansion terms.
        lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
        # The original query is the line right before "Přepsaný dotaz:"
        try:
            idx = lines.index("Přepsaný dotaz:")
            original = lines[idx - 1]
        except (ValueError, IndexError):
            original = ""
        return f"{original} {self._EXPANSION}".strip()


# ---------------------------------------------------------------------------
# QueryRewriteService
# ---------------------------------------------------------------------------


class QueryRewriteService:
    """Rewrites a user query into a more precise legal search query.

    Falls back to the original query if the LLM raises an exception.
    Optional in the pipeline — omit to disable rewriting entirely.
    """

    def __init__(self, llm: BaseTextLLM) -> None:
        self._llm = llm

    def rewrite(self, query: str) -> str:
        """Return a rewritten query, or the original query on failure."""
        prompt = _build_prompt(query)
        try:
            rewritten = self._llm.generate_text(prompt).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[rewrite] LLM failed (%s); falling back to original query", exc
            )
            return query

        if not rewritten:
            logger.warning("[rewrite] LLM returned empty string; using original query")
            return query

        logger.info("[rewrite] original=%s rewritten=%s", query, rewritten)
        return rewritten


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_prompt(query: str) -> str:
    return _PROMPT_TEMPLATE.format(query=query)
