"""Deterministic tests for scoped semantic negation and procedural-defect extraction."""

from __future__ import annotations

import json
from pathlib import Path

from app.rag.legal_v2.query_spec import ConstraintCategory, ConstraintPolarity, build_query_spec_v2

_GOLDEN_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "legal_v2"
    / "case_similarity_golden_v1_pilot.jsonl"
)

_BLOCKED_CUSTODY_MERITS = (
    "úprava styku rodiče s dítětem",
    "opatrovnické řízení",
)


def _negated_names(spec) -> set[str]:
    return {
        item["name"]
        for item in (spec.structured_query.get("negated_requested_concepts") or [])
    }


def _candidate_names(spec) -> list[str]:
    return [
        item["name"]
        for item in (spec.structured_query.get("candidate_retrieval_concepts") or [])
    ]


def _concept_names(spec) -> set[str]:
    return {item["name"] for item in (spec.structured_query.get("legal_concepts") or [])}


def _generated_queries(spec) -> list[str]:
    """Retrieval queries excluding the preserved original."""
    return list(spec.retrieval_queries[1:])


def test_case_a_family_law_merits_negation_and_procedural_focus() -> None:
    spec = build_query_spec_v2(
        "Nehledám meritorní spor o péči, ale odmítnutí ústavní stížnosti pro vady. "
        "Bez advokáta a chybí odůvodnění."
    )

    negated = _negated_names(spec)
    assert "child_custody_merits" in negated
    assert "parent_contact_merits" in negated
    assert "constitutional_complaint" in spec.procedural_posture
    assert "rejected_for_formal_defects" in spec.decision_outcome
    assert "mandatory_lawyer_representation" in _concept_names(spec)
    assert "missing_or_inadequate_reasoning" in _concept_names(spec)
    assert not any(term in " ".join(_generated_queries(spec)) for term in _BLOCKED_CUSTODY_MERITS)
    assert spec.retrieval_queries[0] == spec.original_query


def test_case_b_damages_versus_limitation() -> None:
    spec = build_query_spec_v2(
        "Nehledám rozhodnutí o výši škody, ale jen o promlčení nároku."
    )

    assert "damages" in _negated_names(spec)
    candidates = _candidate_names(spec)
    assert candidates
    assert candidates[0] == "limitation_periods"
    assert "damages" not in candidates
    assert not any("náhrada škody" in query for query in _generated_queries(spec))
    assert any("promlčení" in query for query in spec.retrieval_queries)


def test_case_c_contract_versus_costs() -> None:
    spec = build_query_spec_v2(
        "Nejde mi o platnost smlouvy, ale o náhradu nákladů řízení."
    )

    assert "contract" in _negated_names(spec)
    assert "court_costs" in _candidate_names(spec)
    assert "contract" not in _candidate_names(spec)
    assert any("náklad" in query.lower() for query in spec.retrieval_queries)
    assert not any("neplatnost smlouvy" in query for query in _generated_queries(spec))


def test_case_d_guilt_versus_procedural_admissibility() -> None:
    spec = build_query_spec_v2(
        "Neřeším, zda byl obžalovaný vinen, ale zda bylo dovolání odmítnuto pro vady."
    )

    assert "criminal_guilt" in _negated_names(spec)
    assert "defective_filing" in _concept_names(spec)
    assert "rejected_for_formal_defects" in spec.decision_outcome
    assert "dovolání" in spec.procedural_posture
    assert not any("viněn" in query.lower() or "vinen" in query.lower() for query in _generated_queries(spec))


def test_case_e_no_negation_keeps_custody_expansion() -> None:
    spec = build_query_spec_v2("Hledám rozhodnutí o úpravě styku rodiče s dítětem.")

    assert not _negated_names(spec)
    assert "domestic_custody" in _candidate_names(spec) or "child_contact" in _candidate_names(spec)
    assert any("úprava styku" in query for query in spec.retrieval_queries)


def test_case_f_background_context_demoted_by_requested_focus() -> None:
    spec = build_query_spec_v2(
        "Spor vznikl kvůli péči o dítě, ale hledám jen odmítnutí ústavní stížnosti pro vady."
    )

    candidates = _candidate_names(spec)
    contextual = {
        item["name"]
        for item in (spec.structured_query.get("contextual_concepts") or [])
    }
    assert "domestic_custody" in contextual
    assert "domestic_custody" not in candidates
    assert candidates
    assert candidates[0] in {
        "defective_filing",
        "constitutional_admissibility",
        "mandatory_lawyer_representation",
        "missing_or_inadequate_reasoning",
        "failure_to_cure_filing_defects",
    }
    assert "constitutional_complaint" in spec.procedural_posture
    assert not any(term in " ".join(_generated_queries(spec)) for term in _BLOCKED_CUSTODY_MERITS)


def test_procedural_phrase_variants_normalize_deterministically() -> None:
    cases = [
        ("bez advokáta", "mandatory_lawyer_representation"),
        ("nebyl zastoupen advokátem", "mandatory_lawyer_representation"),
        ("chybělo povinné právní zastoupení", "mandatory_lawyer_representation"),
        ("chybělo odůvodnění", "missing_or_inadequate_reasoning"),
        ("nedostatečně vysvětlil porušení práv", "missing_or_inadequate_reasoning"),
        ("neodstranil vady", "failure_to_cure_filing_defects"),
        ("návrh byl odmítnut pro formální vady", "defective_filing"),
        ("ústavní stížnost", None),
    ]
    for phrase, expected_concept in cases:
        spec = build_query_spec_v2(phrase)
        if expected_concept is not None:
            assert expected_concept in _concept_names(spec), phrase
        if phrase == "ústavní stížnost":
            assert "constitutional_complaint" in spec.procedural_posture
        if "odmítnut" in phrase and "vady" in phrase:
            assert "rejected_for_formal_defects" in spec.decision_outcome

    # Dedup + stable ordering across repeated concepts in one query.
    combined = build_query_spec_v2(
        "bez advokáta, nebyl zastoupen advokátem, chybělo povinné právní zastoupení"
    )
    lawyer = [
        item["name"]
        for item in combined.structured_query["legal_concepts"]
        if item["name"] == "mandatory_lawyer_representation"
    ]
    assert lawyer == ["mandatory_lawyer_representation"]
    assert _candidate_names(combined).count("mandatory_lawyer_representation") <= 1


def test_negative_constraints_are_scoped_and_recorded() -> None:
    spec = build_query_spec_v2(
        "Nehledám meritorní spor o péči, ale odmítnutí ústavní stížnosti pro vady."
    )
    negative_attrs = {
        constraint.attribute for constraint in spec.negative_constraints
    }
    assert "legal_concept:child_custody_merits" in negative_attrs
    assert all(
        constraint.polarity == ConstraintPolarity.NEGATIVE
        for constraint in spec.negative_constraints
    )
    assert all(
        constraint.category == ConstraintCategory.LEGAL_PROVISION
        for constraint in spec.negative_constraints
    )


def test_expansions_do_not_contradict_negative_constraints() -> None:
    spec = build_query_spec_v2(
        "Nehledám meritorní spor o péči, ale odmítnutí ústavní stížnosti pro vady."
    )
    negated = _negated_names(spec)
    assert negated
    joined = " ".join(_generated_queries(spec)).lower()
    for blocked in _BLOCKED_CUSTODY_MERITS:
        assert blocked.lower() not in joined
    suppressed = spec.structured_query.get("suppressed_expansions") or []
    assert isinstance(suppressed, list)


def _load_golden_query(benchmark_id: str) -> str:
    with _GOLDEN_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("benchmark_id") == benchmark_id:
                return str(row["query"])
    raise AssertionError(f"benchmark row not found: {benchmark_id}")


def test_case_similarity_pilot_004_query_spec_regression() -> None:
    """Load the exact stored query; production code must not reference the benchmark id."""
    query = _load_golden_query("nalus-cs-pilot-004")
    spec = build_query_spec_v2(query)

    assert spec.original_query == query
    assert spec.retrieval_queries[0] == query
    assert "constitutional_complaint" in spec.procedural_posture
    assert "rejected_for_formal_defects" in spec.decision_outcome
    assert "mandatory_lawyer_representation" in _concept_names(spec)
    assert "missing_or_inadequate_reasoning" in _concept_names(spec)
    assert "defective_filing" in _concept_names(spec)
    negated = _negated_names(spec)
    assert "child_custody_merits" in negated
    assert "parent_contact_merits" in negated
    assert not any(term in " ".join(_generated_queries(spec)) for term in _BLOCKED_CUSTODY_MERITS)
    assert "legal_concept:domestic_custody" not in {
        constraint.attribute for constraint in spec.hard_constraints
    }
    # Joined retrieval query used by the retriever (first three) must stay contradiction-safe.
    used = " ".join(spec.retrieval_queries[:3])
    assert not any(term in used for term in _BLOCKED_CUSTODY_MERITS)


def test_all_case_similarity_golden_queries_build_deterministically() -> None:
    rows: list[dict] = []
    with _GOLDEN_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    assert len(rows) == 20

    snapshots: list[dict] = []
    for row in rows:
        query = str(row["query"])
        spec = build_query_spec_v2(query)
        assert spec.retrieval_queries[0] == query
        snapshots.append(
            {
                "benchmark_id": row["benchmark_id"],
                "candidate_retrieval_concepts": _candidate_names(spec),
                "negated_requested_concepts": sorted(_negated_names(spec)),
                "procedural_posture": list(spec.procedural_posture),
                "decision_outcome": list(spec.decision_outcome),
                "retrieval_query_count": len(spec.retrieval_queries),
                "blocked_custody_merits_in_generated": any(
                    term in " ".join(_generated_queries(spec)) for term in _BLOCKED_CUSTODY_MERITS
                ),
            }
        )
        # No generated expansion may reintroduce explicitly negated custody merits.
        if "child_custody_merits" in _negated_names(spec):
            assert not snapshots[-1]["blocked_custody_merits_in_generated"]

    # Stable deterministic ordering for the pilot-004 snapshot fields.
    pilot = next(item for item in snapshots if item["benchmark_id"] == "nalus-cs-pilot-004")
    assert pilot["retrieval_query_count"] >= 2
    assert pilot["blocked_custody_merits_in_generated"] is False
