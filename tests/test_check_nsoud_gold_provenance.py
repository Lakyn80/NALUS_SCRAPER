from __future__ import annotations

import scripts.check_nsoud_gold_provenance as checker
from app.rag.eval.legal_qa_benchmark import LegalQaItem


def _item(**overrides) -> LegalQaItem:
    payload = {
        "id": "nsoud-qa-test-001",
        "corpus": "nsoud",
        "question": "Jak Nejvyssi soud posuzuje dovolaci duvod podle § 265b tr. r.?",
        "expected_answer_points": ["bod"],
        "expected_source_constraints": {
            "court": None,
            "source": None,
            "case_reference": None,
            "source_document_id": None,
            "decision_date": None,
        },
        "expected_keywords": ["265b", "trestní", "dovolání"],
        "forbidden_answer_patterns": [],
        "difficulty": "medium",
        "legal_topic": "trestni dovolani",
        "evaluation_type": "retrieval",
        "source_pending": True,
    }
    payload.update(overrides)
    return LegalQaItem.from_dict(payload)


def test_parse_chunk_metadata_from_json_string() -> None:
    payload = {
        "chunk_metadata": '{"source_document_id":"ECLI:CZ:NS:2025:5.TDO.1086.2024.1","case_number":"5 Tdo 1086/2024"}'
    }
    meta = checker._parse_chunk_metadata(payload)
    assert meta["source_document_id"] == "ECLI:CZ:NS:2025:5.TDO.1086.2024.1"
    assert meta["case_number"] == "5 Tdo 1086/2024"


def test_classify_support_level_detects_direct_anchor_match() -> None:
    item = _item()
    text = "Obvineny uplatnil dovolaci duvod podle § 265b odst. 1 písm. g) tr. r. a Nejvyssi soud jej posoudil."
    support, hit_count, ratio, anchor_hits = checker.classify_support_level(
        item=item,
        text=text,
        legal_area="criminal",
    )
    assert support == "direct"
    assert hit_count >= 2
    assert ratio >= 0.66
    assert anchor_hits >= 1


def test_classify_candidate_prefers_rank1_direct() -> None:
    candidate = checker.EnrichedHit(
        rank=1,
        chunk_id="735",
        document_id="ECLI:CZ:NS:2025:5.TDO.1086.2024.1",
        source_document_id="ECLI:CZ:NS:2025:5.TDO.1086.2024.1",
        ecli="ECLI:CZ:NS:2025:5.TDO.1086.2024.1",
        case_reference="5 Tdo 1086/2024",
        spisova_znacka="5 Tdo 1086/2024",
        decision_date=None,
        source="hybrid",
        text_snippet="uplatnil dovolaci duvod podle § 265b odst. 1 písm. g) tr. r.",
        metadata_keys_present=["chunk_id", "document_id"],
        provenance_sufficient_for_gold=True,
        section_type="reasoning",
        legal_area="criminal",
        keyword_hit_count=2,
        keyword_hit_ratio=0.66,
        anchor_hit_count=1,
        support_level="direct",
        baseline_provenance_present=False,
    )
    classification, reason, action = checker.classify_candidate(candidate, [candidate])
    assert classification == "gold_ready_direct"
    assert action == "annotate_gold"
    assert "rank-1" in reason


def test_enrich_hit_uses_qdrant_case_number_as_case_reference() -> None:
    item = _item()

    class FakeClient:
        pass

    payload = {
        "document_id": "ECLI:CZ:NS:2025:5.TDO.1086.2024.1",
        "source": "nsoud",
        "chunk_metadata": {
            "source_document_id": "ECLI:CZ:NS:2025:5.TDO.1086.2024.1",
            "case_number": "5 Tdo 1086/2024",
            "section_type": "reasoning",
            "legal_area": "criminal",
        },
    }

    original = checker._lookup_point_by_chunk_id
    checker._lookup_point_by_chunk_id = lambda client, collection_name, chunk_id: payload
    try:
        enriched = checker.enrich_hit(
            {"rank": 1, "chunk_id": "735", "source": "hybrid", "text_snippet": "dovolaci duvod podle § 265b"},
            item=item,
            client=FakeClient(),
            collection_name="collection",
        )
    finally:
        checker._lookup_point_by_chunk_id = original

    assert enriched.case_reference == "5 Tdo 1086/2024"
    assert enriched.spisova_znacka == "5 Tdo 1086/2024"
    assert enriched.source_document_id == "ECLI:CZ:NS:2025:5.TDO.1086.2024.1"
