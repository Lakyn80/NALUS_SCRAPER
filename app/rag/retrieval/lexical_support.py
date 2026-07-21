from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.rag.retrieval.models import RetrievedChunk

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_STOPWORDS = {
    "aby",
    "ale",
    "anebo",
    "ani",
    "bez",
    "byl",
    "byla",
    "bylo",
    "byt",
    "co",
    "do",
    "ho",
    "je",
    "jeho",
    "jej",
    "jejich",
    "jako",
    "jsou",
    "kdy",
    "ktera",
    "ktere",
    "ktery",
    "ma",
    "me",
    "mit",
    "mu",
    "na",
    "nad",
    "ne",
    "nebo",
    "neni",
    "od",
    "pod",
    "po",
    "pro",
    "pri",
    "pred",
    "se",
    "si",
    "sve",
    "svem",
    "svych",
    "tak",
    "to",
    "u",
    "uz",
    "ve",
    "vi",
    "vse",
    "za",
    "ze",
}


@dataclass(frozen=True)
class LexicalSupport:
    terms: list[str]
    matched_terms: list[str]
    required_matches: int

    @property
    def supported(self) -> bool:
        return len(self.matched_terms) >= self.required_matches

    @property
    def coverage(self) -> float:
        if not self.terms:
            return 1.0
        return len(self.matched_terms) / len(self.terms)


def significant_query_terms(query: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for token in _tokens(query):
        if token in _STOPWORDS or token.isdigit() or len(token) < 3:
            continue
        if token not in seen:
            seen.add(token)
            terms.append(token)
    return terms


def lexical_support(query_terms: list[str], chunk: RetrievedChunk) -> LexicalSupport:
    required_matches = _required_match_count(query_terms)
    if not query_terms:
        return LexicalSupport(terms=[], matched_terms=[], required_matches=0)

    chunk_tokens = _tokens(chunk.text)
    matched = [
        term
        for term in query_terms
        if any(_tokens_match(term, token) for token in chunk_tokens)
    ]
    return LexicalSupport(
        terms=query_terms,
        matched_terms=matched,
        required_matches=required_matches,
    )


def filter_supported_chunks(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    query_terms = significant_query_terms(query)
    if not query_terms:
        return chunks

    supported: list[RetrievedChunk] = []
    for chunk in chunks:
        support = lexical_support(query_terms, chunk)
        if not support.supported:
            continue
        metadata = dict(chunk.metadata)
        metadata["lexical_support"] = {
            "terms": support.terms,
            "matched_terms": support.matched_terms,
            "required_matches": support.required_matches,
            "coverage": support.coverage,
        }
        supported.append(
            RetrievedChunk(
                id=chunk.id,
                text=chunk.text,
                score=chunk.score,
                source=chunk.source,
                metadata=metadata,
            )
        )
    return supported


def _required_match_count(query_terms: list[str]) -> int:
    term_count = len(query_terms)
    if term_count <= 3:
        return term_count
    return max(2, (term_count * 3 + 4) // 5)


def _tokens(text: str) -> list[str]:
    return [_normalize(match.group(0)) for match in _TOKEN_RE.finditer(text)]


def _normalize(value: str) -> str:
    without_marks = "".join(
        char
        for char in unicodedata.normalize("NFKD", value.lower())
        if not unicodedata.combining(char)
    )
    return re.sub(r"[^0-9a-z_]", "", without_marks)


def _tokens_match(term: str, token: str) -> bool:
    if not term or not token:
        return False
    if _is_child_token(term) and _is_child_token(token):
        return True
    if term == token:
        return True
    if len(term) >= 4 and token.startswith(term):
        return True
    if len(token) >= 4 and term.startswith(token):
        return True
    if len(term) >= 5 and len(token) >= 5:
        return term[:5] == token[:5]
    return False


def _is_child_token(token: str) -> bool:
    return token.startswith(("dit", "det"))
