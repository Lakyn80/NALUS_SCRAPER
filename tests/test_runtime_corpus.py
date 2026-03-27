from __future__ import annotations

import json
from pathlib import Path

from app.data.corpus import CORPUS
from app.data.runtime_corpus import (
    build_runtime_corpus,
    load_results_from_json,
    load_runtime_corpus,
)
from app.models.search_result import NalusResult


def _result_dict(
    case_reference: str = "I.ÚS 1/24",
    full_text: str | None = "Pravni veta a shrnuti rozhodnuti.",
) -> dict:
    return {
        "result_id": 1,
        "case_reference": case_reference,
        "ecli": None,
        "judge_rapporteur": None,
        "petitioner": None,
        "popular_name": None,
        "decision_date": None,
        "announcement_date": None,
        "filing_date": None,
        "publication_date": None,
        "related_regulations": [],
        "decision_form": None,
        "importance": None,
        "verdict": None,
        "topics_and_keywords": [],
        "detail_url": None,
        "text_url": "https://nalus.usoud.cz/Search/GetText.aspx?sz=1",
        "full_text": full_text,
    }


def _write_results(path: Path, payload: list[dict]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_load_results_from_json_maps_nalus_result(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    _write_results(path, [_result_dict(case_reference="I.ÚS 99/24")])

    results = load_results_from_json(path)

    assert len(results) == 1
    assert isinstance(results[0], NalusResult)
    assert results[0].case_reference == "I.ÚS 99/24"


def test_load_runtime_corpus_builds_chunks_and_keyword_corpus(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    _write_results(
        path,
        [
            _result_dict(case_reference="I.ÚS 1/24", full_text="A" * 100),
            _result_dict(case_reference="II.ÚS 2/24", full_text=None),
        ],
    )

    runtime_corpus = load_runtime_corpus(results_path=str(path), chunk_size=50, overlap=10)

    assert runtime_corpus.document_count == 2
    assert runtime_corpus.source_label == str(path)
    assert len(runtime_corpus.chunks) == 3
    assert runtime_corpus.keyword_corpus == [
        (chunk.id, chunk.text) for chunk in runtime_corpus.chunks
    ]


def test_load_runtime_corpus_falls_back_to_seed_when_file_missing(tmp_path: Path) -> None:
    runtime_corpus = load_runtime_corpus(results_path=str(tmp_path / "missing.json"))

    assert runtime_corpus.source_label == "seed-corpus"
    assert runtime_corpus.document_count == len(CORPUS)
    assert len(runtime_corpus.keyword_corpus) == len(CORPUS)


def test_build_runtime_corpus_deduplicates_overlapping_decisions() -> None:
    results = [
        NalusResult(
            result_id=1,
            case_reference="I.ÚS 1/24",
            ecli=None,
            judge_rapporteur=None,
            petitioner=None,
            popular_name=None,
            decision_date=None,
            announcement_date=None,
            filing_date=None,
            publication_date=None,
            text_url="https://nalus.usoud.cz/1",
            full_text="A" * 20,
        ),
        NalusResult(
            result_id=2,
            case_reference="I.ÚS 1/24",
            ecli=None,
            judge_rapporteur=None,
            petitioner=None,
            popular_name=None,
            decision_date=None,
            announcement_date=None,
            filing_date=None,
            publication_date=None,
            text_url="https://nalus.usoud.cz/2",
            full_text="B" * 80,
        ),
    ]

    runtime_corpus = build_runtime_corpus(results, source_label="test", chunk_size=50, overlap=10)

    assert runtime_corpus.document_count == 1
    assert len(runtime_corpus.chunks) == 2
    assert all(chunk.case_reference == "I.ÚS 1/24" for chunk in runtime_corpus.chunks)
    assert runtime_corpus.keyword_corpus == [
        (chunk.id, chunk.text) for chunk in runtime_corpus.chunks
    ]
