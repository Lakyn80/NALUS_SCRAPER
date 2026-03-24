"""
Strategy Service — decides how the pipeline should respond based on retrieval results.

Deterministic, pure, no LLM.  Designed for extension: add new modes by
adding branches to decide() without touching any other module.

Modes:
    no_results     — retrieval returned nothing
    direct_answer  — 1–2 chunks, single-source or default fallback
    llm_summary    — 3+ chunks, multi-source synthesis needed
"""

from app.rag.retrieval.models import RetrievedChunk


class StrategyDecision:
    def __init__(self, mode: str, reason: str) -> None:
        self.mode = mode
        self.reason = reason

    def __repr__(self) -> str:
        return f"StrategyDecision(mode={self.mode!r}, reason={self.reason!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StrategyDecision):
            return NotImplemented
        return self.mode == other.mode and self.reason == other.reason


class StrategyService:
    """Determines the response mode from retrieval results.

    Stateless — safe to share across requests.
    """

    def decide(self, query: str, results: list[RetrievedChunk]) -> StrategyDecision:
        n = len(results)

        if n == 0:
            return StrategyDecision("no_results", "empty retrieval")

        if n == 1:
            return StrategyDecision("direct_answer", "single chunk")

        if n >= 3:
            return StrategyDecision("llm_summary", "multi-source synthesis")

        return StrategyDecision("direct_answer", "default fallback")
