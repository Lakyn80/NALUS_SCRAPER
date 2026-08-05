from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from app.rag.legal_v2.benchmark.case_similarity_golden import (
    DEFAULT_PILOT_DATASET,
    EXPECTED_PILOT_COUNT,
    EXPECTED_QUERY_STYLE_COUNTS,
    CaseSimilarityGoldenItem,
    best_supporting_block_token_overlap,
    count_sentences,
    count_words,
    detect_query_leakage,
    load_case_similarity_golden_jsonl,
    longest_contiguous_normalized_token_overlap,
    longest_verbatim_sentence_overlap_tokens,
    validate_case_similarity_dataset,
)
from app.rag.legal_v2.benchmark.corpus import (
    CASE_SIMILARITY_SUPPLEMENTAL_CRIMINAL_SOURCES,
    load_case_similarity_corpus,
    load_case_similarity_primary_document_ids,
    load_reviewed_pool_corpus,
)
from scripts.legal_v2.export_case_similarity_golden_v1_manual_review import (
    EXPECTED_QUERY_COUNT,
    build_manual_review_markdown,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PILOT_PATH = PROJECT_ROOT / "benchmarks" / "legal_v2" / "case_similarity_golden_v1_pilot.jsonl"
STEP4A_PATH = PROJECT_ROOT / "benchmarks" / "legal_v2" / "retrieval_golden_v1_pilot.jsonl"
BUILDER = PROJECT_ROOT / "scripts" / "legal_v2" / "build_case_similarity_golden_v1_pilot.py"
FORBIDDEN_003_PHRASE = "škoda nevznikla v příčinné souvislosti s předběžným opatřením"


@pytest.fixture(scope="module")
def pilot_items() -> list[CaseSimilarityGoldenItem]:
    assert PILOT_PATH.exists(), f"missing pilot dataset: {PILOT_PATH}"
    return load_case_similarity_golden_jsonl(PILOT_PATH)


@pytest.fixture(scope="module")
def reviewed_corpus():
    return load_reviewed_pool_corpus()


@pytest.fixture(scope="module")
def case_similarity_corpus():
    return load_case_similarity_corpus()


@pytest.fixture(scope="module")
def primary_document_ids() -> list[str]:
    return load_case_similarity_primary_document_ids()


def test_default_pilot_path_points_at_tracked_dataset() -> None:
    assert DEFAULT_PILOT_DATASET.resolve() == PILOT_PATH.resolve()


def test_pilot_counts_and_unique_ids(pilot_items: list[CaseSimilarityGoldenItem]) -> None:
    assert len(pilot_items) == EXPECTED_PILOT_COUNT
    assert len({item.benchmark_id for item in pilot_items}) == EXPECTED_PILOT_COUNT
    assert all(item.split == "development" for item in pilot_items)
    assert all(item.benchmark_type == "case_similarity_document_retrieval" for item in pilot_items)
    assert all(item.human_review_status == "PENDING_HUMAN_REVIEW" for item in pilot_items)


def test_query_style_distribution(pilot_items: list[CaseSimilarityGoldenItem]) -> None:
    counts: dict[str, int] = {}
    for item in pilot_items:
        counts[item.query_style] = counts.get(item.query_style, 0) + 1
    assert counts == EXPECTED_QUERY_STYLE_COUNTS


def test_every_reviewed_document_used_exactly_once(
    pilot_items: list[CaseSimilarityGoldenItem],
    primary_document_ids: list[str],
) -> None:
    assert len(primary_document_ids) == EXPECTED_PILOT_COUNT
    sources = [item.source_document_id for item in pilot_items]
    assert sorted(sources) == sorted(primary_document_ids)
    assert len(set(sources)) == EXPECTED_PILOT_COUNT


def test_document_and_block_references(
    pilot_items: list[CaseSimilarityGoldenItem],
    case_similarity_corpus,
    primary_document_ids: list[str],
) -> None:
    report = validate_case_similarity_dataset(
        pilot_items,
        corpus_documents=case_similarity_corpus.documents,
        blocks_by_id=case_similarity_corpus.blocks_by_id,
        expected_document_ids=primary_document_ids,
        dataset_path=str(PILOT_PATH),
    )
    assert report.ok, report.model_dump()


def test_supporting_blocks_belong_to_expected_document(
    pilot_items: list[CaseSimilarityGoldenItem],
    case_similarity_corpus,
) -> None:
    for item in pilot_items:
        assert 2 <= len(item.supporting_block_ids) <= 5
        for block_id in item.supporting_block_ids:
            block = case_similarity_corpus.blocks_by_id[block_id]
            assert block.document_id == item.source_document_id


def test_hard_negatives_present_and_non_overlapping(
    pilot_items: list[CaseSimilarityGoldenItem],
) -> None:
    for item in pilot_items:
        assert 1 <= len(item.hard_negative_document_ids) <= 3
        expected = set(item.expected_document_ids)
        alts = set(item.accepted_alternative_document_ids)
        hards = set(item.hard_negative_document_ids)
        assert not (expected & alts)
        assert not (expected & hards)
        assert not (alts & hards)
        assert {row.document_id for row in item.hard_negative_rationales} == hards


def test_step4a_dataset_untouched() -> None:
    assert STEP4A_PATH.exists()
    text = STEP4A_PATH.read_text(encoding="utf-8")
    assert text.count("\n") >= 30
    assert "nalus-rg-pilot-001" in text
    assert "case_similarity_document_retrieval" not in text


def test_query_length_and_sentence_constraints(
    pilot_items: list[CaseSimilarityGoldenItem],
) -> None:
    for item in pilot_items:
        assert 60 <= count_words(item.query) <= 180
        assert 3 <= count_sentences(item.query) <= 8


def test_no_source_identifier_leakage(
    pilot_items: list[CaseSimilarityGoldenItem],
    case_similarity_corpus,
) -> None:
    refs = {ref.document_id: ref for ref in case_similarity_corpus.documents}
    for item in pilot_items:
        docs = (
            list(item.expected_document_ids)
            + list(item.accepted_alternative_document_ids)
            + list(item.hard_negative_document_ids)
        )
        leaks = detect_query_leakage(
            item.query,
            document_ids=docs,
            case_numbers=[refs[doc].case_number for doc in docs],
            source_ids=[refs[doc].source_id for doc in docs],
        )
        assert leaks == [], (item.benchmark_id, leaks)


def test_no_copied_supporting_sentence(
    pilot_items: list[CaseSimilarityGoldenItem],
    case_similarity_corpus,
) -> None:
    for item in pilot_items:
        for block_id in item.supporting_block_ids:
            block = case_similarity_corpus.blocks_by_id[block_id]
            assert longest_verbatim_sentence_overlap_tokens(item.query, block.raw_text) < 12


def test_contiguous_token_overlap_reports_three_tokens() -> None:
    count, text = longest_contiguous_normalized_token_overlap(
        "alpha beta gamma delta",
        "prefix alpha beta gamma suffix",
    )
    assert count == 3
    assert text == "alpha beta gamma"


def test_contiguous_token_overlap_reports_zero_for_non_overlap() -> None:
    count, text = longest_contiguous_normalized_token_overlap(
        "alpha beta gamma",
        "delta epsilon zeta",
    )
    assert count == 0
    assert text == ""


def test_complete_sentence_leakage_rule_still_fails_at_twelve_tokens() -> None:
    sentence = " ".join(f"token{i}" for i in range(12)) + "."
    assert longest_verbatim_sentence_overlap_tokens(sentence, sentence) == 12
    short = " ".join(f"token{i}" for i in range(11)) + "."
    assert longest_verbatim_sentence_overlap_tokens(short, short) == 0


def test_manual_review_overlap_diagnostics_are_not_all_zero(
    pilot_items: list[CaseSimilarityGoldenItem],
    case_similarity_corpus,
) -> None:
    markdown, _stats = build_manual_review_markdown(pilot_items, case_similarity_corpus)
    zero_lines = [
        line
        for line in markdown.splitlines()
        if line.startswith("- longest_verbatim_overlap_tokens: `0`")
    ]
    nonzero = [
        line
        for line in markdown.splitlines()
        if line.startswith("- longest_verbatim_overlap_tokens: `")
        and not line.endswith("`0`")
    ]
    assert nonzero, "expected at least one non-zero overlap diagnostic"
    assert len(zero_lines) < EXPECTED_PILOT_COUNT

    counts = []
    for item in pilot_items:
        count, _text, block_id = best_supporting_block_token_overlap(
            item.query,
            case_similarity_corpus.blocks_by_id,
            item.supporting_block_ids,
        )
        counts.append(count)
        if count > 0:
            assert block_id in item.supporting_block_ids
    assert max(counts) > 0


def test_manual_review_export_has_twenty_sections(
    pilot_items: list[CaseSimilarityGoldenItem],
    case_similarity_corpus,
) -> None:
    markdown, stats = build_manual_review_markdown(pilot_items, case_similarity_corpus)
    assert stats.query_sections == EXPECTED_QUERY_COUNT
    assert markdown.count("## Query: ") == EXPECTED_QUERY_COUNT


def test_entry_003_query_rewrite_and_primary(
    pilot_items: list[CaseSimilarityGoldenItem],
    case_similarity_corpus,
) -> None:
    item = next(row for row in pilot_items if row.benchmark_id == "nalus-cs-pilot-003")
    assert item.source_document_id == "doc-d513b3e81616439a"
    assert "gift_revocation" in item.factual_facets
    assert "preliminary_injunction" in item.factual_facets
    assert "causation_for_damages_from_interim_measure" in item.legal_issue_facets
    assert FORBIDDEN_003_PHRASE not in item.query
    assert "dostatečná souvislost" in item.query
    refs = {ref.document_id: ref for ref in case_similarity_corpus.documents}
    leaks = detect_query_leakage(
        item.query,
        document_ids=item.expected_document_ids + item.hard_negative_document_ids,
        case_numbers=[refs[doc].case_number for doc in item.expected_document_ids],
        source_ids=[refs[doc].source_id for doc in item.expected_document_ids],
    )
    assert leaks == []
    for block_id in item.supporting_block_ids:
        block = case_similarity_corpus.blocks_by_id[block_id]
        assert longest_verbatim_sentence_overlap_tokens(item.query, block.raw_text) < 12
    overlap, _, _ = best_supporting_block_token_overlap(
        item.query,
        case_similarity_corpus.blocks_by_id,
        item.supporting_block_ids,
    )
    assert overlap < 8


def test_entry_007_hard_negatives_exist_and_are_not_primary(
    pilot_items: list[CaseSimilarityGoldenItem],
    case_similarity_corpus,
) -> None:
    item = next(row for row in pilot_items if row.benchmark_id == "nalus-cs-pilot-007")
    assert item.source_document_id == "doc-af3c185ad674a7da"
    assert "CORPUS BLOCKER" in (item.notes or "")
    assert item.hard_negative_evaluable is False
    assert item.hard_negative_blocker == "insufficient_same_domain_corpus"
    corpus_ids = {ref.document_id for ref in case_similarity_corpus.documents}
    for document_id in item.hard_negative_document_ids:
        assert document_id in corpus_ids
        assert document_id != item.source_document_id
        assert document_id not in item.accepted_alternative_document_ids


def test_entry_016_hard_negatives_are_criminal(
    pilot_items: list[CaseSimilarityGoldenItem],
    case_similarity_corpus,
) -> None:
    item = next(row for row in pilot_items if row.benchmark_id == "nalus-cs-pilot-016")
    assert item.source_document_id == "doc-4f3c37d9c5a1afb7"
    supplemental_ids = {row["document_id"] for row in CASE_SIMILARITY_SUPPLEMENTAL_CRIMINAL_SOURCES}
    assert set(item.hard_negative_document_ids) == supplemental_ids
    refs = {ref.document_id: ref for ref in case_similarity_corpus.documents}
    for document_id in item.hard_negative_document_ids:
        assert document_id in refs
        assert document_id != item.source_document_id
        assert refs[document_id].decision_type == "criminal_appeal"
        assert refs[document_id].court == "high_court_olomouc"
        assert case_similarity_corpus.blocks_for_document(document_id)


def test_hard_negative_block_ids_resolve_to_declared_documents(
    pilot_items: list[CaseSimilarityGoldenItem],
    case_similarity_corpus,
) -> None:
    for item in pilot_items:
        for document_id in item.hard_negative_document_ids:
            blocks = case_similarity_corpus.blocks_for_document(document_id)
            assert blocks
            for block in blocks:
                assert block.document_id == document_id


def test_untouched_entries_keep_stable_fingerprints(
    pilot_items: list[CaseSimilarityGoldenItem],
) -> None:
    """Entries outside the 003/007/016 correction scope keep stable substance."""
    protected = {
        "nalus-cs-pilot-001": ("doc-0a90125eb71851b4", "client_narrative"),
        "nalus-cs-pilot-002": ("doc-b73cac9b3dfc8a42", "concise_case_description"),
        "nalus-cs-pilot-004": ("doc-16b9100a8b9122dd", "noisy_client_narrative"),
        "nalus-cs-pilot-005": ("doc-e5ac4b1fcd075062", "multi_issue_client_narrative"),
        "nalus-cs-pilot-006": ("doc-abd57ac0aa5dfe5b", "client_narrative"),
        "nalus-cs-pilot-008": ("doc-a5292901931de05a", "client_narrative"),
        "nalus-cs-pilot-009": ("doc-f2c776a1533521c3", "concise_case_description"),
        "nalus-cs-pilot-010": ("doc-976fafa1e2c6f093", "client_narrative"),
        "nalus-cs-pilot-011": ("doc-cfa470876b0d5ed7", "multi_issue_client_narrative"),
        "nalus-cs-pilot-012": ("doc-db9f10005638d155", "client_narrative"),
        "nalus-cs-pilot-013": ("doc-4af3171b4be427e9", "concise_case_description"),
        "nalus-cs-pilot-014": ("doc-e6af147081ae754f", "multi_issue_client_narrative"),
        "nalus-cs-pilot-015": ("doc-f4a701825747ed58", "client_narrative"),
        "nalus-cs-pilot-017": ("doc-84ae84698dfd0205", "client_narrative"),
        "nalus-cs-pilot-018": ("doc-dc644c7e6d827609", "noisy_client_narrative"),
        "nalus-cs-pilot-019": ("doc-c7b72b0d6121d7f3", "noisy_client_narrative"),
        "nalus-cs-pilot-020": ("doc-6cca3be81564e762", "concise_case_description"),
    }
    by_id = {item.benchmark_id: item for item in pilot_items}
    assert len(protected) == 17
    for benchmark_id, (document_id, style) in protected.items():
        item = by_id[benchmark_id]
        assert item.source_document_id == document_id, benchmark_id
        assert item.query_style == style, benchmark_id
        assert item.expected_document_ids == [document_id]


def test_deterministic_byte_identical_rebuild(tmp_path: Path) -> None:
    tracked = PILOT_PATH.read_bytes()
    out = tmp_path / "rebuild.jsonl"
    report = tmp_path / "report.json"
    completed = subprocess.run(
        [sys.executable, str(BUILDER), "--output", str(out), "--report", str(report)],
        cwd=str(PROJECT_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    rebuilt = out.read_bytes()
    assert rebuilt == tracked
    assert hashlib.sha256(rebuilt).hexdigest() == hashlib.sha256(tracked).hexdigest()
