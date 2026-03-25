from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.data.corpus import CORPUS
from app.models.search_result import NalusResult
from app.rag.chunking.chunker import TextChunk, chunk_document
from app.rag.retrieval.keyword_retriever import CorpusEntry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_PATH = "results_rodinne_pravo_1000.json"
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 200


@dataclass(frozen=True)
class RuntimeCorpus:
    source_label: str
    document_count: int
    chunks: list[TextChunk]
    keyword_corpus: list[CorpusEntry]


def load_runtime_corpus(
    results_path: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RuntimeCorpus:
    source = _resolve_path(results_path or DEFAULT_RESULTS_PATH)
    if not source.exists():
        return build_seed_runtime_corpus()

    results = load_results_from_json(source)
    runtime_corpus = build_runtime_corpus(
        results,
        source_label=str(source),
        chunk_size=chunk_size,
        overlap=overlap,
    )
    if runtime_corpus.chunks:
        return runtime_corpus

    return build_seed_runtime_corpus()


def load_results_from_json(path: str | Path) -> list[NalusResult]:
    resolved = _resolve_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of decisions in {resolved}.")

    return [_result_from_dict(item) for item in data]


def build_runtime_corpus(
    results: list[NalusResult],
    source_label: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RuntimeCorpus:
    chunks: list[TextChunk] = []
    for result in results:
        chunks.extend(
            chunk_document(
                result,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

    _raise_on_duplicate_chunk_ids(chunks)

    return RuntimeCorpus(
        source_label=source_label,
        document_count=len(results),
        chunks=chunks,
        keyword_corpus=[(chunk.id, chunk.text) for chunk in chunks],
    )


def build_seed_runtime_corpus() -> RuntimeCorpus:
    chunks = [
        TextChunk(
            id=str_id,
            text=text,
            case_reference=str_id,
            ecli="",
            decision_date="",
            judge="",
            text_url="",
            chunk_index=0,
        )
        for str_id, text in CORPUS
    ]
    return RuntimeCorpus(
        source_label="seed-corpus",
        document_count=len(CORPUS),
        chunks=chunks,
        keyword_corpus=list(CORPUS),
    )


def _resolve_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return PROJECT_ROOT / path_obj


def _result_from_dict(item: object) -> NalusResult:
    if not isinstance(item, dict):
        raise ValueError("Each decision entry must be an object.")

    return NalusResult(
        result_id=item.get("result_id"),
        case_reference=item.get("case_reference"),
        ecli=item.get("ecli"),
        judge_rapporteur=item.get("judge_rapporteur"),
        petitioner=item.get("petitioner"),
        popular_name=item.get("popular_name"),
        decision_date=item.get("decision_date"),
        announcement_date=item.get("announcement_date"),
        filing_date=item.get("filing_date"),
        publication_date=item.get("publication_date"),
        related_regulations=list(item.get("related_regulations") or []),
        decision_form=item.get("decision_form"),
        importance=item.get("importance"),
        verdict=item.get("verdict"),
        topics_and_keywords=list(item.get("topics_and_keywords") or []),
        detail_url=item.get("detail_url"),
        text_url=item.get("text_url"),
        full_text=item.get("full_text"),
    )


def _raise_on_duplicate_chunk_ids(chunks: list[TextChunk]) -> None:
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.id in seen:
            raise ValueError(f"Duplicate chunk id detected: {chunk.id}")
        seen.add(chunk.id)
