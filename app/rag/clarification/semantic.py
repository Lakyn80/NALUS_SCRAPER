from __future__ import annotations

import math
from dataclasses import dataclass

from app.rag.clarification.models import CachedClarificationEntry, RuleAssessment
from app.rag.clarification.text_utils import simplify_text
from app.rag.retrieval.embedder import BaseEmbedder


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class SignatureEmbedder(BaseEmbedder):
    """Deterministic lightweight embedder for clarification-pattern similarity."""

    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

    def embed_query(self, query: str) -> list[float]:
        tokens = simplify_text(query).split()
        vector = [0.0] * self._dim
        if not tokens:
            return vector
        for token in tokens:
            bucket = hash(token) % self._dim
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


@dataclass(frozen=True)
class AmbiguityPattern:
    query_signature: str
    example_query: str
    ambiguity_types: tuple[str, ...]
    missing_slots: tuple[str, ...]
    clarification_question_cs: str


DEFAULT_AMBIGUITY_PATTERNS: tuple[AmbiguityPattern, ...] = (
    AmbiguityPattern(
        query_signature="appeal_dovolani_previous_proceeding_ambiguous",
        example_query=(
            "Klient se odvolal proti rozsudku, ale odvolací soud jeho odvolání zamítl. "
            "Teď chce podat dovolání a tvrdí, že chyby vznikly už před soudem prvního stupně."
        ),
        ambiguity_types=("legal_domain_ambiguous",),
        missing_slots=("legal_domain",),
        clarification_question_cs=(
            "Jedná se o trestní dovolání podle trestního řádu, "
            "nebo o civilní dovolání podle občanského soudního řádu?"
        ),
    ),
)


def _ambiguity_types_compatible(
    query_types: tuple[str, ...],
    pattern_types: tuple[str, ...],
) -> bool:
    return bool(set(query_types).intersection(pattern_types))


class ClarificationPatternStore:
    def __init__(
        self,
        *,
        embedder: BaseEmbedder | None = None,
        patterns: tuple[AmbiguityPattern, ...] = DEFAULT_AMBIGUITY_PATTERNS,
        similarity_threshold: float = 0.55,
    ) -> None:
        self._embedder = embedder or SignatureEmbedder()
        self._patterns = patterns
        self._similarity_threshold = similarity_threshold
        self._pattern_vectors = {
            pattern.query_signature: self._embedder.embed_query(pattern.example_query)
            for pattern in patterns
        }

    def find_similar_pattern(self, query: str, *, assessment: RuleAssessment) -> AmbiguityPattern | None:
        if not assessment.ambiguity_types:
            return None

        query_vector = self._embedder.embed_query(query)
        best_pattern: AmbiguityPattern | None = None
        best_score = 0.0
        for pattern in self._patterns:
            if not _ambiguity_types_compatible(assessment.ambiguity_types, pattern.ambiguity_types):
                continue
            pattern_vector = self._pattern_vectors[pattern.query_signature]
            score = cosine_similarity(query_vector, pattern_vector)
            if score > best_score:
                best_score = score
                best_pattern = pattern
        if best_pattern is None or best_score < self._similarity_threshold:
            return None
        return best_pattern

    def to_cache_entry(self, pattern: AmbiguityPattern) -> CachedClarificationEntry:
        return CachedClarificationEntry(
            query_signature=pattern.query_signature,
            ambiguity_types=list(pattern.ambiguity_types),
            missing_slots=list(pattern.missing_slots),
            clarification_question_cs=pattern.clarification_question_cs,
            detected_issue=pattern.ambiguity_types[0] if pattern.ambiguity_types else "none",
            recommended_next_action="ask_user",
        )
