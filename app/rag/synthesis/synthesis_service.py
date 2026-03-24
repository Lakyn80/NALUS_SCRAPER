"""
Synthesis Service — combines multi-step ExecutionResult into a final structured answer.

Takes all retrieved chunks from an ExecutionResult, builds a prompt, calls an LLM,
and returns a structured SynthesisOutput with the answer text and source IDs.

Usage:
    from app.rag.synthesis.synthesis_service import SynthesisService, MockSynthesisLLM
    from app.rag.execution.execution_service import ExecutionResult

    service = SynthesisService(llm=MockSynthesisLLM())
    output = service.synthesize("únos dítěte do zahraničí", execution_result)

    print(output.answer)   # generated Czech legal answer
    print(output.sources)  # list of chunk IDs used
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.rag.execution.execution_service import ExecutionResult
from app.rag.rewrite.query_rewrite_service import BaseTextLLM

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a legal assistant specialized in constitutional law of the Czech Republic. "
    "Use only the provided case excerpts to answer. Do not hallucinate. "
    "If the excerpts do not contain sufficient information, say so explicitly."
)

_USER_TEMPLATE = """\
User question:
{query}

Relevant case excerpts:
{chunks_text}

Instructions:
- Answer only from the provided excerpts
- If information is insufficient, state it clearly
- Provide a structured legal explanation in Czech
- Keep it concise but precise
- Cite specific decisions by their ID (e.g. III.ÚS 255/22)
- Structure the answer: Shrnutí / Klíčové závěry / Relevantní rozhodnutí

Answer:"""

# Combined into a single prompt for BaseTextLLM (text-in / text-out interface)
_PROMPT_TEMPLATE = f"SYSTEM: {_SYSTEM_PROMPT}\n\nUSER:\n{{query_block}}"

# Resolved at call time via _build_prompt()
_PROMPT_TEMPLATE = """\
SYSTEM: {system}

USER:
{user}"""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class SynthesisOutput:
    answer: str
    sources: list[str]


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


class MockSynthesisLLM(BaseTextLLM):
    """Deterministic stub for tests — returns a fixed Czech legal answer."""

    def generate_text(self, prompt: str) -> str:
        return (
            "## Shrnutí\n"
            "Na základě dostupné judikatury Ústavního soudu lze konstatovat následující.\n\n"
            "## Klíčové závěry\n"
            "Ústavní soud opakovaně zdůraznil ochranu základních práv dotčených osob.\n\n"
            "## Relevantní rozhodnutí\n"
            "Viz citovaná rozhodnutí v úryvcích výše."
        )


# ---------------------------------------------------------------------------
# SynthesisService
# ---------------------------------------------------------------------------


class SynthesisService:
    """Synthesizes a final answer from multi-step ExecutionResult using an LLM."""

    def __init__(self, llm: BaseTextLLM) -> None:
        self._llm = llm

    def synthesize(self, query: str, execution_result: ExecutionResult) -> SynthesisOutput:
        chunks = execution_result.all_chunks()
        sources = list(dict.fromkeys(c.id for c in chunks))  # preserve order, deduplicated

        logger.info("[synthesis] sources=%d query=%s", len(sources), query)

        trace_event(
            logger,
            "synthesis.start",
            query=query,
            num_chunks=len(chunks),
            num_sources=len(sources),
        )

        chunks_text = _format_chunks(chunks)
        prompt = _PROMPT_TEMPLATE.format(
            system=_SYSTEM_PROMPT,
            user=_USER_TEMPLATE.format(query=query, chunks_text=chunks_text),
        )

        try:
            answer = self._llm.generate_text(prompt).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[synthesis] LLM failed (%s); returning empty answer", exc)
            answer = ""

        trace_event(logger, "synthesis.done", answer_length=len(answer))
        return SynthesisOutput(answer=answer, sources=sources)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_chunks(chunks: list) -> str:
    if not chunks:
        return "(žádné úryvky)"
    parts = []
    for chunk in chunks:
        parts.append(f"[{chunk.id}] (skóre: {chunk.score:.3f})\n{chunk.text}")
    return "\n\n---\n\n".join(parts)
