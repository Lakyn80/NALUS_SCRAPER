from __future__ import annotations

from dataclasses import replace

from app.rag.clarification.cache import (
    BaseClarificationCache,
    NullClarificationCache,
    build_exact_query_cache_key,
    build_pattern_cache_key,
    build_clarification_cache,
    cache_entry_from_decision,
    clarification_cache_ttl_seconds,
    decision_from_cache_entry,
)
from app.rag.clarification.llm import BaseClarificationLLM, TemplateClarificationLLM, build_clarification_question
from app.rag.clarification.models import ClarificationDecision, RetrievalHitSummary, RuleAssessment
from app.rag.clarification.qdrant_patterns import BaseClarificationPatternIndex, build_clarification_pattern_index
from app.rag.clarification.retrieval_feedback import detect_retrieval_domain_mismatch, is_cdo_tdo_mismatch
from app.rag.clarification.rules import assess_query_rules, should_ask_clarification


class LegalQueryClarificationService:
    """Detect under-specified legal queries and ask before misleading retrieval.

    Qdrant continues to retrieve court decisions. Redis caches clarification payloads.
    Optional Qdrant collection ``legal_query_clarification_patterns`` stores only
    clarification-pattern embeddings for similar ambiguous-query reuse.
    """

    def __init__(
        self,
        *,
        cache: BaseClarificationCache | None = None,
        llm: BaseClarificationLLM | None = None,
        pattern_index: BaseClarificationPatternIndex | None = None,
        enable_semantic_reuse: bool = True,
    ) -> None:
        self._cache = cache or NullClarificationCache()
        self._llm = llm or TemplateClarificationLLM()
        self._pattern_index = pattern_index or build_clarification_pattern_index()
        self._enable_semantic_reuse = enable_semantic_reuse

    @classmethod
    def from_env(cls, *, llm: BaseClarificationLLM | None = None) -> "LegalQueryClarificationService":
        cache, _ = build_clarification_cache()
        return cls(cache=cache, llm=llm)

    def evaluate(
        self,
        query: str,
        *,
        retrieval_hits: list[RetrievalHitSummary] | None = None,
    ) -> ClarificationDecision:
        assessment = assess_query_rules(query)

        if retrieval_hits:
            feedback = self._evaluate_retrieval_feedback(query, assessment, retrieval_hits)
            if feedback is not None:
                return feedback

        if not should_ask_clarification(assessment):
            return self._proceed_decision(assessment=assessment, query=query)

        exact_cache_key = build_exact_query_cache_key(query)
        cached_exact = self._cache.get(exact_cache_key)
        if cached_exact is not None:
            return decision_from_cache_entry(
                cached_exact,
                cache_key=exact_cache_key,
                detected_legal_domain=assessment.detected_legal_domain,
                detected_procedure_stage=assessment.detected_procedure_stage,
            )

        pattern_cache_key = build_pattern_cache_key(query_signature=assessment.query_signature)
        cached_pattern = self._cache.get(pattern_cache_key)
        if cached_pattern is not None:
            return decision_from_cache_entry(
                cached_pattern,
                cache_key=pattern_cache_key,
                detected_legal_domain=assessment.detected_legal_domain,
                detected_procedure_stage=assessment.detected_procedure_stage,
                semantic_cache_hit=True,
            )

        semantic_hit = self._try_semantic_reuse(query, assessment)
        if semantic_hit is not None:
            return semantic_hit

        return self._ask_decision(
            query=query,
            assessment=assessment,
            exact_cache_key=exact_cache_key,
            pattern_cache_key=pattern_cache_key,
        )

    def _try_semantic_reuse(self, query: str, assessment: RuleAssessment) -> ClarificationDecision | None:
        if not self._enable_semantic_reuse:
            return None

        similar_pattern = self._pattern_index.find_similar_pattern(query, assessment=assessment)
        if similar_pattern is None:
            return None

        pattern_cache_key = build_pattern_cache_key(query_signature=similar_pattern.query_signature)
        cached_pattern = self._cache.get(pattern_cache_key)
        if cached_pattern is not None:
            return decision_from_cache_entry(
                cached_pattern,
                cache_key=pattern_cache_key,
                detected_legal_domain=assessment.detected_legal_domain,
                detected_procedure_stage=assessment.detected_procedure_stage,
                semantic_cache_hit=True,
            )

        return self._ask_decision(
            query=query,
            assessment=replace(
                assessment,
                ambiguity_types=similar_pattern.ambiguity_types,
                missing_slots=similar_pattern.missing_slots,
                query_signature=similar_pattern.query_signature,
                reason_cs=(
                    "Dotaz je sémanticky podobný dříve známému nejednoznačnému vzoru; "
                    "je potřeba upřesnit právní doménu před vyhledáváním."
                ),
            ),
            exact_cache_key=build_exact_query_cache_key(query),
            pattern_cache_key=pattern_cache_key,
            clarification_override=similar_pattern.clarification_question_cs,
            semantic_cache_hit=True,
        )

    def _evaluate_retrieval_feedback(
        self,
        query: str,
        assessment: RuleAssessment,
        retrieval_hits: list[RetrievalHitSummary],
    ) -> ClarificationDecision | None:
        mismatch, _, reason = detect_retrieval_domain_mismatch(
            retrieval_hits,
            query_domain=assessment.detected_legal_domain,
        )
        cdo_tdo_mismatch = is_cdo_tdo_mismatch(retrieval_hits)
        if not mismatch and not cdo_tdo_mismatch:
            return None
        if assessment.detected_legal_domain != "unknown" and not cdo_tdo_mismatch:
            return None

        ambiguity_types = list(assessment.ambiguity_types)
        missing_slots = list(assessment.missing_slots)
        if "retrieval_domain_mismatch" not in ambiguity_types:
            ambiguity_types.append("retrieval_domain_mismatch")
        if "legal_domain" not in missing_slots:
            missing_slots.append("legal_domain")

        feedback_assessment = replace(
            assessment,
            ambiguity_types=tuple(ambiguity_types),
            missing_slots=tuple(missing_slots),
            query_signature="retrieval_domain_mismatch",
            reason_cs=reason or (
                "Po prvním vyhledávání se top výsledky míchají mezi civilní a trestní judikaturou."
            ),
        )
        return self._ask_decision(
            query=query,
            assessment=feedback_assessment,
            exact_cache_key=build_exact_query_cache_key(query),
            pattern_cache_key=build_pattern_cache_key(query_signature=feedback_assessment.query_signature),
        )

    def _ask_decision(
        self,
        *,
        query: str,
        assessment: RuleAssessment,
        exact_cache_key: str,
        pattern_cache_key: str,
        clarification_override: str | None = None,
        semantic_cache_hit: bool = False,
    ) -> ClarificationDecision:
        llm_called = False
        if clarification_override:
            clarification_question = clarification_override
        else:
            clarification_question, llm_called = build_clarification_question(
                query=query,
                assessment=assessment,
                llm=self._llm,
            )

        decision = ClarificationDecision(
            decision="ask_clarifying_question",
            confidence=self._clarification_confidence(assessment),
            ambiguity_types=list(assessment.ambiguity_types),
            missing_slots=list(assessment.missing_slots),
            detected_legal_domain=assessment.detected_legal_domain,
            detected_procedure_stage=assessment.detected_procedure_stage,
            clarification_question_cs=clarification_question,
            reason_cs=assessment.reason_cs,
            cache_key=exact_cache_key,
            query_signature=assessment.query_signature,
            recommended_next_action="ask_user",
            cache_hit=False,
            semantic_cache_hit=semantic_cache_hit,
            llm_called=llm_called,
        )
        entry = cache_entry_from_decision(decision)
        ttl = clarification_cache_ttl_seconds()
        self._cache.set(exact_cache_key, entry, ttl_seconds=ttl)
        self._cache.set(pattern_cache_key, entry, ttl_seconds=ttl)
        return decision

    def _proceed_decision(self, *, assessment: RuleAssessment, query: str) -> ClarificationDecision:
        confidence = min(
            0.95,
            (
                assessment.domain_confidence
                + assessment.procedure_confidence
                + assessment.remedy_confidence
            )
            / 3,
        )
        return ClarificationDecision(
            decision="proceed_to_retrieval",
            confidence=round(confidence, 3),
            ambiguity_types=[],
            missing_slots=[],
            detected_legal_domain=assessment.detected_legal_domain,
            detected_procedure_stage=assessment.detected_procedure_stage,
            clarification_question_cs="",
            reason_cs=assessment.reason_cs,
            cache_key=build_exact_query_cache_key(query),
            query_signature=assessment.query_signature,
            recommended_next_action="run_retrieval",
            cache_hit=False,
            semantic_cache_hit=False,
            llm_called=False,
        )

    @staticmethod
    def _clarification_confidence(assessment: RuleAssessment) -> float:
        penalties = len(assessment.ambiguity_types) * 0.08
        base = 0.9 - penalties
        if "legal_domain_ambiguous" in assessment.ambiguity_types:
            base = max(base, 0.82)
        if "retrieval_domain_mismatch" in assessment.ambiguity_types:
            base = max(base, 0.8)
        return round(max(0.55, min(0.95, base)), 3)
