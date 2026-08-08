"""Unit tests for diversified_stage1_evidence_v1 CE passage selector."""

from __future__ import annotations

from app.rag.legal_v2.rerank.models import EvidenceChunkRecord, RerankCandidate
from app.rag.legal_v2.rerank.selectors.diversified_stage1_evidence_v1 import (
    DiversifiedStage1EvidenceSelectorV1,
)
from app.rag.legal_v2.rerank.selectors.names import DIVERSIFIED_STAGE1_EVIDENCE_V1
from app.rag.legal_v2.rerank.selectors.policy import get_evidence_passage_selector


def _chunk(
    chunk_id: str,
    text: str,
    *,
    rrf: int | None = None,
    dense: int | None = None,
    bm25: int | None = None,
    position: int | None = None,
    section: str | None = None,
) -> EvidenceChunkRecord:
    channels: list[str] = []
    if rrf is not None:
        channels.append("rrf")
    if dense is not None:
        channels.append("dense")
    if bm25 is not None:
        channels.append("bm25")
    return EvidenceChunkRecord(
        chunk_id=chunk_id,
        text=text,
        dense_rank=dense,
        bm25_rank=bm25,
        rrf_rank=rrf,
        retrieval_channels=tuple(channels),
        chunk_position=position,
        section=section,
    )


def _candidate(pool: list[EvidenceChunkRecord], *, ecli: str = "ECLI:CZ:US:1") -> RerankCandidate:
    return RerankCandidate(
        ecli=ecli,
        stage1_rank=4,
        stage1_score=0.42,
        passages=(),
        evidence_pool=tuple(pool),
        metadata={"benchmark_id": "should-be-ignored", "expected_ecli": "ECLI:NOPE"},
    )


def test_selector_policy_id() -> None:
    selector = get_evidence_passage_selector(DIVERSIFIED_STAGE1_EVIDENCE_V1)
    assert selector.policy_id == DIVERSIFIED_STAGE1_EVIDENCE_V1


def test_at_most_seven_no_duplicate_ids() -> None:
    pool = [
        _chunk("a", "rrf primary unique evidence about lease termination", rrf=1, position=1),
        _chunk("b", "dense primary unique evidence about landlord notice", dense=1, position=5),
        _chunk("c", "bm25 primary unique evidence about apartment rental", bm25=1, position=9),
        _chunk("d", "rrf secondary unique evidence about court reasoning", rrf=2, position=12),
        _chunk("e", "dense secondary unique evidence about procedural history", dense=2, position=20),
        _chunk("f", "bm25 secondary unique evidence about operative part", bm25=2, position=28),
        _chunk("g", "diversity support unique evidence from later section", rrf=3, dense=3, position=41),
        _chunk("h", "extra unique leftover evidence should not be needed", rrf=4, position=50),
    ]
    selected = DiversifiedStage1EvidenceSelectorV1().select(_candidate(pool), limit=7)
    assert len(selected) == 7
    ids = [p.chunk_id for p in selected]
    assert len(ids) == len(set(ids))


def test_deterministic_repeated_calls() -> None:
    pool = [
        _chunk("a", "alpha channel evidence text number one", rrf=1, dense=2, position=1),
        _chunk("b", "beta channel evidence text number two", dense=1, position=2),
        _chunk("c", "gamma channel evidence text number three", bm25=1, position=3),
        _chunk("d", "delta channel evidence text number four", rrf=2, position=10),
    ]
    selector = DiversifiedStage1EvidenceSelectorV1()
    cand = _candidate(pool)
    first = [p.chunk_id for p in selector.select(cand, limit=7)]
    second = [p.chunk_id for p in selector.select(cand, limit=7)]
    assert first == second


def test_channel_slots_and_provenance() -> None:
    pool = [
        _chunk("rrf1", "rrf one unique paragraph about constitutional complaint", rrf=1, position=1),
        _chunk("d1", "dense one unique paragraph about fundamental rights", dense=1, position=8),
        _chunk("b1", "bm25 one unique paragraph about statute interpretation", bm25=1, position=15),
        _chunk("rrf2", "rrf two unique paragraph about proportionality test", rrf=2, position=22),
        _chunk("d2", "dense two unique paragraph about public interest", dense=2, position=30),
        _chunk("b2", "bm25 two unique paragraph about legitimate aim", bm25=2, position=38),
        _chunk("div", "diversity unique paragraph about remedy and costs", rrf=5, dense=5, bm25=5, position=60),
    ]
    selected = DiversifiedStage1EvidenceSelectorV1().select(_candidate(pool), limit=7)
    by_reason = {p.selection_reason: p.chunk_id for p in selected}
    assert by_reason["rrf_primary"] == "rrf1"
    assert by_reason["dense_primary"] == "d1"
    assert by_reason["bm25_primary"] == "b1"
    assert by_reason["rrf_secondary"] == "rrf2"
    assert by_reason["dense_secondary"] == "d2"
    assert by_reason["bm25_secondary"] == "b2"
    assert by_reason["diversity_support"] == "div"
    assert selected[0].rrf_rank == 1
    assert selected[1].dense_rank == 1
    assert selected[2].bm25_rank == 1
    assert selected[0].selection_slot == 1
    assert all(p.requested_passages == 7 for p in selected)
    assert all(p.selected_passages == 7 for p in selected)


def test_near_duplicate_suppressed_diversity_prefers_distant() -> None:
    base = (
        "soud dospěl k závěru že výpověď z nájmu bytu byla neplatná protože "
        "pronajímatel neprokázal zákonný důvod podle občanského zákoníku"
    )
    pool = [
        _chunk("A", base, rrf=1, position=18),
        _chunk("B", "dense primary distinct facts about tenant family hardship", dense=1, position=5),
        _chunk("C", "bm25 primary distinct statute citation and procedural posture", bm25=1, position=9),
        _chunk("D", base + " a proto návrh zamítl", rrf=2, position=19),  # near-dupe of A
        _chunk("E", "rrf secondary distinct reasoning about proportionality", rrf=3, position=20),
        _chunk("F", "dense secondary distinct evidence about notice timing", dense=2, position=21),
        _chunk("G", "bm25 secondary distinct evidence about rent arrears", bm25=2, position=22),
        _chunk(
            "H",
            "strong distinct evidence from another section about damages and restitution",
            rrf=4,
            dense=3,
            bm25=3,
            position=41,
            section="court_reasoning",
        ),
    ]
    selected = DiversifiedStage1EvidenceSelectorV1().select(_candidate(pool), limit=7)
    ids = {p.chunk_id for p in selected}
    assert "D" not in ids
    assert "H" in ids
    assert "A" in ids


def test_fallback_fills_missing_channel_and_fewer_than_seven() -> None:
    pool = [
        _chunk("only1", "only one unique chunk of evidence available here", rrf=1, position=1),
        _chunk("only2", "second unique chunk without dense or bm25 ranks", rrf=2, position=2),
        _chunk("only3", "third unique chunk still no dense channel present", rrf=3, position=3),
    ]
    selected = DiversifiedStage1EvidenceSelectorV1().select(_candidate(pool), limit=7)
    assert len(selected) == 3
    assert {p.chunk_id for p in selected} == {"only1", "only2", "only3"}
    assert any((p.selection_reason or "").startswith("fallback_after_") for p in selected)


def test_stable_tie_break_by_chunk_id() -> None:
    pool = [
        _chunk("z", "tie evidence text shared strength profile aaa", rrf=1, dense=1, bm25=1, position=5),
        _chunk("a", "tie evidence text shared strength profile bbb", rrf=1, dense=1, bm25=1, position=5),
        _chunk("m", "tie evidence text shared strength profile ccc", rrf=1, dense=1, bm25=1, position=5),
    ]
    selected = DiversifiedStage1EvidenceSelectorV1().select(_candidate(pool), limit=1)
    assert selected[0].chunk_id == "a"


def test_selector_ignores_golden_labels_and_expected_ecli() -> None:
    pool = [
        _chunk("keep", "relevant unique evidence passage for selection", rrf=1, position=1),
        _chunk("other", "another unique evidence passage for selection", dense=1, position=2),
    ]
    cand_a = _candidate(pool, ecli="ECLI:CZ:US:AAA")
    cand_b = RerankCandidate(
        ecli="ECLI:CZ:US:BBB",
        stage1_rank=99,
        stage1_score=0.01,
        passages=(),
        evidence_pool=tuple(pool),
        metadata={
            "benchmark_id": "nalus-cs-pilot-004",
            "expected_primary_ecli": "ECLI:CZ:US:SHOULD_NOT_MATTER",
            "golden_labels": ["irrelevant"],
        },
    )
    selector = DiversifiedStage1EvidenceSelectorV1()
    ids_a = [p.chunk_id for p in selector.select(cand_a, limit=7)]
    ids_b = [p.chunk_id for p in selector.select(cand_b, limit=7)]
    assert ids_a == ids_b


def test_distinct_legal_passages_not_over_deduped() -> None:
    pool = [
        _chunk(
            "lease",
            "výpověď z nájmu bytu pro neplacení nájemného a porušení domovního řádu",
            rrf=1,
            position=1,
        ),
        _chunk(
            "labor",
            "výpověď z pracovního poměru pro porušení povinností zaměstnance",
            dense=1,
            position=2,
        ),
        _chunk(
            "admin",
            "zrušení správního rozhodnutí pro nepřezkoumatelnost odůvodnění",
            bm25=1,
            position=3,
        ),
    ]
    selected = DiversifiedStage1EvidenceSelectorV1().select(_candidate(pool), limit=7)
    assert {p.chunk_id for p in selected} == {"lease", "labor", "admin"}
