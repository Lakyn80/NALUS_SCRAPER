from __future__ import annotations

import copy
import json
import sqlite3
from pathlib import Path

from app.rag.eval.evidence_window import EvidenceWindowConfig, build_evidence_window
from app.rag.eval.legal_answer_eval import (
    aggregate_answer_metrics,
    evaluate_answer_item,
    load_gold_registry_from_dataset,
    run_answer_eval,
    write_answer_eval_outputs,
)
from app.rag.eval.legal_qa_benchmark import LegalQaItem, SourceConstraints


def _item(
    *,
    item_id: str = "nsoud-qa-test",
    ecli: str = "ECLI:CZ:NS:2025:1.TEST.1.1",
    keywords: list[str] | None = None,
) -> LegalQaItem:
    return LegalQaItem(
        id=item_id,
        corpus="nsoud",
        question="Je dovolání přípustné?",
        expected_answer_points=["Právní bod je podporován ověřeným zdrojem."],
        expected_source_constraints=SourceConstraints(source_document_id=ecli),
        expected_keywords=keywords or ["alpha", "beta", "gamma"],
        forbidden_answer_patterns=[],
        difficulty="medium",
        legal_topic="dovolání",
        evaluation_type="retrieval",
        source_pending=False,
        expected_target_corpus=None,
    )


def _hit(
    *,
    rank: int = 1,
    chunk_id: str = "11",
    chunk_index: int = 1,
    ecli: str = "ECLI:CZ:NS:2025:1.TEST.1.1",
    snippet: str = "alpha",
    text: str | None = None,
    score: float = 0.5,
) -> dict:
    metadata = {
        "document_id": ecli,
        "source_document_id": ecli,
        "ecli": ecli,
        "chunk_index": chunk_index,
    }
    if text is not None:
        metadata["text"] = text
    return {
        "rank": rank,
        "chunk_id": chunk_id,
        "text_snippet": snippet,
        "score": score,
        "source": "hybrid",
        "metadata": metadata,
    }


def _write_sidecar(path: Path, rows: list[tuple[str, str, str, str, str, int]]) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE bm25_chunks (
                chunk_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                document_id TEXT,
                source_document_id TEXT,
                ecli TEXT,
                chunk_index INTEGER
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO bm25_chunks
            (chunk_id, text, document_id, source_document_id, ecli, chunk_index)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def _write_sidecar_without_ecli(path: Path, rows: list[tuple[str, str, str, str, int]]) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE bm25_chunks (
                chunk_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                document_id TEXT,
                source_document_id TEXT,
                chunk_index INTEGER
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO bm25_chunks
            (chunk_id, text, document_id, source_document_id, chunk_index)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )


def _sidecar_with_document(path: Path, *, ecli: str = "ECLI:CZ:NS:2025:1.TEST.1.1") -> Path:
    _write_sidecar(
        path,
        [
            ("10", "previous alpha text with enough legal context for evaluation", ecli, ecli, ecli, 0),
            ("11", "anchor beta text with enough legal context for evaluation", ecli, ecli, ecli, 1),
            ("12", "next gamma text with enough legal context for evaluation", ecli, ecli, ecli, 2),
            ("13", "far delta", ecli, ecli, ecli, 3),
        ],
    )
    return path


def test_deterministic_ordering_of_adjacent_chunks(tmp_path: Path) -> None:
    sidecar = _sidecar_with_document(tmp_path / "sidecar.sqlite")
    window = build_evidence_window(
        anchor_hit=_hit(),
        hits=[],
        config=EvidenceWindowConfig(enabled=True),
        sidecar_path=sidecar,
    )
    assert window.ordered_chunk_ids == ["10", "11", "12"]
    assert window.ordered_chunk_indexes == [0, 1, 2]


def test_duplicate_chunk_index_uses_stable_chunk_id_tie_breaker(tmp_path: Path) -> None:
    ecli = "ECLI:CZ:NS:2025:1.TEST.1.1"
    _write_sidecar(
        tmp_path / "sidecar.sqlite",
        [
            ("10-b", "later duplicate alpha text with enough context", ecli, ecli, ecli, 0),
            ("10-a", "earlier duplicate alpha text with enough context", ecli, ecli, ecli, 0),
            ("11", "anchor beta text with enough legal context for evaluation", ecli, ecli, ecli, 1),
        ],
    )
    window = build_evidence_window(
        anchor_hit=_hit(chunk_id="11", chunk_index=1, ecli=ecli),
        hits=[],
        config=EvidenceWindowConfig(enabled=True, neighbor_chunks_before=1, neighbor_chunks_after=0),
        sidecar_path=tmp_path / "sidecar.sqlite",
    )
    assert window.ordered_chunk_ids == ["10-a", "11"]


def test_previous_anchor_next_selection(tmp_path: Path) -> None:
    sidecar = _sidecar_with_document(tmp_path / "sidecar.sqlite")
    window = build_evidence_window(
        anchor_hit=_hit(chunk_id="11", chunk_index=1),
        hits=[],
        config=EvidenceWindowConfig(enabled=True, neighbor_chunks_before=1, neighbor_chunks_after=1),
        sidecar_path=sidecar,
    )
    assert window.combined_text == (
        "previous alpha text with enough legal context for evaluation\n\n"
        "anchor beta text with enough legal context for evaluation\n\n"
        "next gamma text with enough legal context for evaluation"
    )


def test_sidecar_without_ecli_uses_source_document_identity(tmp_path: Path) -> None:
    ecli = "ECLI:CZ:US:2025:1.US.1.1"
    _write_sidecar_without_ecli(
        tmp_path / "sidecar.sqlite",
        [
            ("10", "previous alpha text with enough legal context", ecli, ecli, 0),
            ("11", "anchor beta text with enough legal context", ecli, ecli, 1),
            ("12", "next gamma text with enough legal context", ecli, ecli, 2),
        ],
    )
    window = build_evidence_window(
        anchor_hit=_hit(chunk_id="11", chunk_index=1, ecli=ecli),
        hits=[],
        config=EvidenceWindowConfig(enabled=True),
        sidecar_path=tmp_path / "sidecar.sqlite",
    )
    assert window.provenance_valid is True
    assert window.ordered_chunk_ids == ["10", "11", "12"]
    assert window.source == "bm25_sidecar"


def test_same_document_enforcement_accepts_matching_identity(tmp_path: Path) -> None:
    sidecar = _sidecar_with_document(tmp_path / "sidecar.sqlite")
    window = build_evidence_window(
        anchor_hit=_hit(ecli="ECLI:CZ:NS:2025:1.TEST.1.1"),
        hits=[],
        config=EvidenceWindowConfig(enabled=True),
        sidecar_path=sidecar,
    )
    assert window.provenance_valid is True
    assert window.document_id == "ECLI:CZ:NS:2025:1.TEST.1.1"


def test_different_document_neighbor_is_rejected(tmp_path: Path) -> None:
    ecli = "ECLI:CZ:NS:2025:1.TEST.1.1"
    other = "ECLI:CZ:NS:2025:2.TEST.2.1"
    hits = [
        _hit(chunk_id="11", chunk_index=1, ecli=ecli, text="anchor beta"),
        _hit(chunk_id="12", chunk_index=2, ecli=other, text="wrong gamma"),
    ]
    window = build_evidence_window(
        anchor_hit=hits[0],
        hits=hits,
        config=EvidenceWindowConfig(enabled=True, neighbor_chunks_before=0, neighbor_chunks_after=1),
    )
    assert window.ordered_chunk_ids == ["11"]
    assert window.missing_neighbors == [2]


def test_ecli_mismatch_is_rejected() -> None:
    ecli = "ECLI:CZ:NS:2025:1.TEST.1.1"
    hit = _hit(chunk_id="11", chunk_index=1, ecli=ecli, text="anchor beta")
    hit["metadata"]["ecli"] = "ECLI:CZ:NS:2025:2.TEST.2.1"
    window = build_evidence_window(
        anchor_hit=hit,
        hits=[hit],
        config=EvidenceWindowConfig(enabled=True),
    )
    assert window.provenance_valid is False
    assert window.construction_reason == "invalid_anchor_provenance"


def test_missing_provenance_falls_back_safely_to_anchor_only() -> None:
    hit = {"rank": 1, "chunk_id": "11", "text_snippet": "alpha beta gamma", "metadata": {}}
    window = build_evidence_window(
        anchor_hit=hit,
        hits=[hit],
        config=EvidenceWindowConfig(enabled=True),
    )
    assert window.provenance_valid is False
    assert window.ordered_chunk_ids == []
    assert "provenance" in str(window.failure_reason)


def test_character_bound_is_enforced(tmp_path: Path) -> None:
    ecli = "ECLI:CZ:NS:2025:1.TEST.1.1"
    _write_sidecar(
        tmp_path / "sidecar.sqlite",
        [
            ("10", "a" * 20, ecli, ecli, ecli, 0),
            ("11", "b" * 20, ecli, ecli, ecli, 1),
            ("12", "c" * 20, ecli, ecli, ecli, 2),
        ],
    )
    window = build_evidence_window(
        anchor_hit=_hit(chunk_id="11", chunk_index=1, ecli=ecli),
        hits=[],
        config=EvidenceWindowConfig(enabled=True, max_characters=25),
        sidecar_path=tmp_path / "sidecar.sqlite",
    )
    assert len(window.combined_text) == 25
    assert window.truncated is True


def test_chunk_count_bound_is_enforced(tmp_path: Path) -> None:
    sidecar = _sidecar_with_document(tmp_path / "sidecar.sqlite")
    window = build_evidence_window(
        anchor_hit=_hit(chunk_id="11", chunk_index=1),
        hits=[],
        config=EvidenceWindowConfig(
            enabled=True,
            neighbor_chunks_before=1,
            neighbor_chunks_after=2,
            max_chunks=2,
        ),
        sidecar_path=sidecar,
    )
    assert len(window.ordered_chunk_ids) == 2
    assert "11" in window.ordered_chunk_ids


def test_truncation_is_deterministic(tmp_path: Path) -> None:
    sidecar = _sidecar_with_document(tmp_path / "sidecar.sqlite")
    config = EvidenceWindowConfig(enabled=True, max_characters=17)
    first = build_evidence_window(anchor_hit=_hit(), hits=[], config=config, sidecar_path=sidecar)
    second = build_evidence_window(anchor_hit=_hit(), hits=[], config=config, sidecar_path=sidecar)
    assert first.combined_text == second.combined_text
    assert first.truncated is True


def test_evidence_window_disabled_preserves_historical_evaluator_behavior(tmp_path: Path) -> None:
    item = _item()
    registry = load_gold_registry_from_dataset([item])
    retrieval = {"hits": [_hit(snippet="alpha", text="alpha beta gamma")]}
    baseline = evaluate_answer_item(
        item=item,
        gold=registry[item.id],
        retrieval=retrieval,
        citation_required=True,
    )
    disabled = evaluate_answer_item(
        item=item,
        gold=registry[item.id],
        retrieval=retrieval,
        citation_required=True,
        evidence_window_config=EvidenceWindowConfig(enabled=False),
        evidence_sidecar_path=tmp_path / "unused.sqlite",
    )
    assert disabled.support_level == baseline.support_level
    assert disabled.support_keyword_coverage == baseline.support_keyword_coverage
    assert disabled.evidence_window_enabled is False


def test_keyword_coverage_can_use_combined_evidence_text(tmp_path: Path) -> None:
    sidecar = _sidecar_with_document(tmp_path / "sidecar.sqlite")
    item = _item()
    registry = load_gold_registry_from_dataset([item])
    result = evaluate_answer_item(
        item=item,
        gold=registry[item.id],
        retrieval={"hits": [_hit(snippet="alpha")]},
        citation_required=True,
        evidence_window_config=EvidenceWindowConfig(enabled=True),
        evidence_sidecar_path=sidecar,
    )
    assert result.support_keyword_coverage == 1.0
    assert result.support_level == "direct"


def test_citation_source_matching_remains_verified_document_provenance(tmp_path: Path) -> None:
    ecli = "ECLI:CZ:NS:2025:1.TEST.1.1"
    sidecar = _sidecar_with_document(tmp_path / "sidecar.sqlite", ecli=ecli)
    item = _item(ecli=ecli)
    registry = load_gold_registry_from_dataset([item])
    result = evaluate_answer_item(
        item=item,
        gold=registry[item.id],
        retrieval={"hits": [_hit(ecli=ecli, snippet="alpha")]},
        citation_required=True,
        evidence_window_config=EvidenceWindowConfig(enabled=True),
        evidence_sidecar_path=sidecar,
    )
    assert result.citation_available is True
    assert ecli in result.answer_skeleton


def test_mixed_corpus_only_skips_document_evidence_window(tmp_path: Path) -> None:
    item = LegalQaItem(
        id="mixed-qa-test",
        corpus="mixed",
        question="Porovnání korpusů?",
        expected_answer_points=["Smíšená otázka ověřuje jen směrování mezi korpusy."],
        expected_source_constraints=SourceConstraints(),
        expected_keywords=["ústavní", "nejvyšší"],
        forbidden_answer_patterns=[],
        difficulty="medium",
        legal_topic="mixed",
        evaluation_type="retrieval",
        source_pending=False,
        expected_target_corpus="both",
    )
    registry = load_gold_registry_from_dataset([item])
    results = run_answer_eval(
        items=[item],
        registry=registry,
        retrieval_by_id={
            item.id: {
                "hits": [
                    _hit(
                        rank=1,
                        chunk_id="11",
                        chunk_index=1,
                        ecli="ECLI:CZ:NS:2025:1.TEST.1.1",
                        snippet="document-level text must not become a mixed citation",
                    )
                ],
                "corpus_hit_at_3": True,
                "corpus_hit_at_5": True,
            }
        },
        citation_required=True,
        evidence_window_config=EvidenceWindowConfig(enabled=True),
        evidence_sidecar_path=tmp_path / "unused.sqlite",
    )
    metrics = aggregate_answer_metrics(results)
    result = results[0]
    assert result.support_level == "corpus_only"
    assert result.citation_available is False
    assert result.evidence_window_enabled is True
    assert result.evidence_window_chunk_ids == []
    assert result.evidence_window_provenance_valid is None
    assert metrics.corpus_only_count == 1
    assert metrics.corpus_routing_support_rate == 1.0
    assert metrics.usable_support_rate_gold == 1.0
    assert metrics.citation_available_rate_gold == 0.0
    assert metrics.evidence_window_used_count == 0
    assert metrics.evidence_window_failed_count == 0


def test_no_qdrant_writes_are_performed() -> None:
    source = Path("app/rag/eval/evidence_window.py").read_text(encoding="utf-8")
    forbidden_terms = ("upsert", "set_payload", "delete")
    assert not any(term in source for term in forbidden_terms)
    assert "qdrant" not in source.lower()


def test_no_retrieval_score_or_rank_modification(tmp_path: Path) -> None:
    sidecar = _sidecar_with_document(tmp_path / "sidecar.sqlite")
    retrieval = {"hits": [_hit(rank=4, score=0.123, snippet="alpha")]}
    original = copy.deepcopy(retrieval)
    item = _item()
    registry = load_gold_registry_from_dataset([item])
    evaluate_answer_item(
        item=item,
        gold=registry[item.id],
        retrieval=retrieval,
        citation_required=True,
        evidence_window_config=EvidenceWindowConfig(enabled=True),
        evidence_sidecar_path=sidecar,
    )
    assert retrieval == original


def test_no_cross_document_evidence_leakage(tmp_path: Path) -> None:
    ecli = "ECLI:CZ:NS:2025:1.TEST.1.1"
    other = "ECLI:CZ:NS:2025:2.TEST.2.1"
    _write_sidecar(
        tmp_path / "sidecar.sqlite",
        [
            ("11", "anchor beta", ecli, ecli, ecli, 1),
            ("12", "wrong gamma", other, other, other, 2),
        ],
    )
    window = build_evidence_window(
        anchor_hit=_hit(chunk_id="11", chunk_index=1, ecli=ecli),
        hits=[],
        config=EvidenceWindowConfig(enabled=True, neighbor_chunks_before=0, neighbor_chunks_after=1),
        sidecar_path=tmp_path / "sidecar.sqlite",
    )
    assert "wrong gamma" not in window.combined_text
    assert window.ordered_chunk_ids == ["11"]


def test_result_json_contains_evidence_window_diagnostics(tmp_path: Path) -> None:
    sidecar = _sidecar_with_document(tmp_path / "sidecar.sqlite")
    item = _item()
    registry = load_gold_registry_from_dataset([item])
    results = run_answer_eval(
        items=[item],
        registry=registry,
        retrieval_by_id={item.id: {"hits": [_hit(snippet="alpha")]}},
        citation_required=True,
        evidence_window_config=EvidenceWindowConfig(enabled=True),
        evidence_sidecar_path=sidecar,
    )
    metrics = aggregate_answer_metrics(results)
    output = tmp_path / "nsoud_evidence_window_candidate"
    write_answer_eval_outputs(
        output_dir=output,
        dataset_path=tmp_path / "dataset.jsonl",
        retrieval_results_path=tmp_path / "retrieval.jsonl",
        gold_review_path=tmp_path / "review.md",
        items=[item],
        registry=registry,
        retrieval_by_id={item.id: {"hits": [_hit(snippet="alpha")]}},
        results=results,
        metrics=metrics,
        no_llm=True,
        citation_required=True,
        corpus="nsoud",
        evidence_window_config=EvidenceWindowConfig(enabled=True),
        evidence_sidecar_path=sidecar,
    )
    payload = json.loads((output / "answer_eval_results.jsonl").read_text(encoding="utf-8"))
    assert payload["evidence_window_enabled"] is True
    assert payload["evidence_window_chunk_ids"] == ["10", "11", "12"]
    assert payload["combined_evidence_length"] > payload["original_snippet_length"]


def test_summary_contains_aggregate_evidence_window_counters(tmp_path: Path) -> None:
    sidecar = _sidecar_with_document(tmp_path / "sidecar.sqlite")
    item = _item()
    registry = load_gold_registry_from_dataset([item])
    results = run_answer_eval(
        items=[item],
        registry=registry,
        retrieval_by_id={item.id: {"hits": [_hit(snippet="alpha")]}},
        citation_required=True,
        evidence_window_config=EvidenceWindowConfig(enabled=True),
        evidence_sidecar_path=sidecar,
    )
    metrics = aggregate_answer_metrics(results)
    summary = json.loads(
        json.dumps(
            {
                "evidence_window_used_count": metrics.evidence_window_used_count,
                "evidence_window_failed_count": metrics.evidence_window_failed_count,
                "evidence_window_truncated_count": metrics.evidence_window_truncated_count,
                "same_document_neighbor_count": metrics.same_document_neighbor_count,
            }
        )
    )
    assert summary["evidence_window_used_count"] == 1
    assert summary["evidence_window_failed_count"] == 0
    assert summary["same_document_neighbor_count"] == 2


def test_nsoud_qa_010_style_fixture_recovers_support_from_adjacent_chunk(tmp_path: Path) -> None:
    ecli = "ECLI:CZ:NS:2025:29.NSCR.1.2025.1"
    _write_sidecar(
        tmp_path / "sidecar.sqlite",
        [
            ("1643", "takto: Dovolání se odmítá.", ecli, ecli, ecli, 1),
            (
                "1644",
                "Odvolací soud odmítl odvolání. Dovolání není objektivně přípustné. "
                "Správnost lze prověřit žalobou pro zmatečnost.",
                ecli,
                ecli,
                ecli,
                2,
            ),
            ("1645", "Další odůvodnění.", ecli, ecli, ecli, 3),
        ],
    )
    item = _item(
        item_id="nsoud-qa-010",
        ecli=ecli,
        keywords=["odmítl odvolání", "objektivně přípustné", "žalobou pro zmatečnost"],
    )
    registry = load_gold_registry_from_dataset([item])
    result = evaluate_answer_item(
        item=item,
        gold=registry[item.id],
        retrieval={"hits": [_hit(rank=4, chunk_id="1644", chunk_index=2, ecli=ecli, snippet="Odvolací soud odmítl")]},
        citation_required=True,
        evidence_window_config=EvidenceWindowConfig(enabled=True),
        evidence_sidecar_path=tmp_path / "sidecar.sqlite",
    )
    assert result.support_level == "partial"
    assert result.unsupported_answer_risk is False
    assert result.support_keyword_coverage == 1.0


def test_nsoud_qa_003_threshold_remains_unchanged() -> None:
    item = _item(
        item_id="nsoud-qa-003",
        keywords=["přípustnost", "dovolání", "občanský"],
    )
    registry = load_gold_registry_from_dataset([item])
    result = evaluate_answer_item(
        item=item,
        gold=registry[item.id],
        retrieval={
            "hits": [
                    _hit(
                    snippet="přípustnost dovolání a další dostatečně dlouhý právní kontext",
                    text=None,
                )
            ]
        },
        citation_required=True,
    )
    assert result.support_keyword_coverage == 2 / 3
    assert result.support_level == "partial"
    assert result.answer_eval_status != "pass"
