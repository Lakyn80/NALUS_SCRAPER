"""
Unit tests for app.rag.query.query_processor.

Run:
    pytest tests/rag/test_query_processor.py -v
"""

import logging

import pytest

from app.rag.query.query_processor import (
    ProcessedQuery,
    _extract_keywords,
    _map_legal_concepts,
    _normalize,
    process_query,
)


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_lowercases_input(self) -> None:
        assert _normalize("ÚNOS DÍTĚTE") == "únos dítěte"

    def test_collapses_multiple_spaces(self) -> None:
        assert _normalize("žena   cizinka") == "žena cizinka"

    def test_strips_leading_trailing_whitespace(self) -> None:
        assert _normalize("  únos  ") == "únos"

    def test_collapses_tabs_and_newlines(self) -> None:
        assert _normalize("únos\t\ndítěte") == "únos dítěte"

    def test_empty_string_returns_empty(self) -> None:
        assert _normalize("") == ""

    def test_already_normalized_unchanged(self) -> None:
        assert _normalize("únos dítěte") == "únos dítěte"


# ---------------------------------------------------------------------------
# _extract_keywords
# ---------------------------------------------------------------------------


class TestExtractKeywords:
    def test_splits_on_spaces(self) -> None:
        result = _extract_keywords("únos dítěte ruska")
        assert "únos" in result
        assert "dítěte" in result
        assert "ruska" in result

    def test_filters_words_shorter_than_3(self) -> None:
        result = _extract_keywords("do na za únos")
        assert "do" not in result
        assert "na" not in result
        assert "za" not in result
        assert "únos" in result

    def test_exactly_3_chars_included(self) -> None:
        result = _extract_keywords("syn byl dán")
        assert "syn" in result
        assert "byl" in result
        assert "dán" in result

    def test_deduplicates_words(self) -> None:
        result = _extract_keywords("únos únos dítěte dítěte")
        assert result.count("únos") == 1
        assert result.count("dítěte") == 1

    def test_preserves_insertion_order(self) -> None:
        result = _extract_keywords("žena cizinka unesla dítě")
        assert result.index("žena") < result.index("cizinka")
        assert result.index("cizinka") < result.index("unesla")

    def test_empty_query_returns_empty_list(self) -> None:
        assert _extract_keywords("") == []

    def test_all_short_words_returns_empty_list(self) -> None:
        assert _extract_keywords("do na v z") == []


# ---------------------------------------------------------------------------
# _map_legal_concepts
# ---------------------------------------------------------------------------


class TestMapLegalConcepts:
    def test_exact_key_match(self) -> None:
        concepts = _map_legal_concepts(["únos"])
        assert "mezinárodní únos dítěte" in concepts

    def test_substring_key_match(self) -> None:
        # "unesla" contains key "unesl"
        concepts = _map_legal_concepts(["unesla"])
        assert "mezinárodní únos dítěte" in concepts

    def test_no_match_returns_empty(self) -> None:
        concepts = _map_legal_concepts(["soud", "rozhodl", "takto"])
        assert concepts == []

    def test_multiple_keywords_produce_multiple_concepts(self) -> None:
        concepts = _map_legal_concepts(["unesla", "dítě", "ruska"])
        assert "mezinárodní únos dítěte" in concepts
        assert "rodičovská odpovědnost" in concepts
        assert "mezinárodní prvek" in concepts

    def test_concepts_deduplicated(self) -> None:
        # Both "únos" and "unesl" map to the same concept
        concepts = _map_legal_concepts(["únos", "unesla"])
        assert concepts.count("mezinárodní únos dítěte") == 1

    def test_empty_keywords_returns_empty(self) -> None:
        assert _map_legal_concepts([]) == []

    def test_haag_maps_to_multiple_concepts(self) -> None:
        concepts = _map_legal_concepts(["haagská"])
        assert "mezinárodní únos dítěte" in concepts
        assert "Haagská úmluva" in concepts

    def test_alimenty_maps_to_vyživovaci(self) -> None:
        concepts = _map_legal_concepts(["alimenty"])
        assert "vyživovací povinnost" in concepts

    def test_cizinka_maps_to_mezinarodni_prvek(self) -> None:
        # "cizinka" contains key "cizin"
        concepts = _map_legal_concepts(["cizinka"])
        assert "mezinárodní prvek" in concepts


# ---------------------------------------------------------------------------
# process_query — integration
# ---------------------------------------------------------------------------


class TestProcessQuery:
    def test_returns_processed_query_instance(self) -> None:
        result = process_query("únos dítěte")
        assert isinstance(result, ProcessedQuery)

    def test_original_query_preserved_verbatim(self) -> None:
        query = "Žena Cizinka UNESLA Dítě"
        result = process_query(query)
        assert result.original_query == query

    def test_normalized_query_is_lowercase(self) -> None:
        result = process_query("ÚNOS DÍTĚTE")
        assert result.normalized_query == "únos dítěte"

    def test_keywords_extracted(self) -> None:
        result = process_query("žena cizinka unesla dítě do Ruska")
        assert "žena" in result.keywords
        assert "cizinka" in result.keywords
        assert "unesla" in result.keywords

    def test_short_words_excluded_from_keywords(self) -> None:
        result = process_query("únos dítěte do Ruska")
        assert "do" not in result.keywords

    def test_main_use_case_query(self) -> None:
        result = process_query("žena cizinka unesla dítě do Ruska")
        assert "mezinárodní únos dítěte" in result.legal_concepts
        assert "rodičovská odpovědnost" in result.legal_concepts
        assert "mezinárodní prvek" in result.legal_concepts

    def test_empty_query_returns_empty_fields(self) -> None:
        result = process_query("")
        assert result.original_query == ""
        assert result.normalized_query == ""
        assert result.keywords == []
        assert result.legal_concepts == []

    def test_query_with_only_short_words(self) -> None:
        result = process_query("do na za")
        assert result.keywords == []
        assert result.legal_concepts == []

    def test_duplicate_keywords_deduplicated(self) -> None:
        result = process_query("únos únos dítěte dítěte")
        assert result.keywords.count("únos") == 1
        assert result.keywords.count("dítěte") == 1

    def test_legal_concepts_deduplicated(self) -> None:
        # "únos" and "unesla" both map to the same concept
        result = process_query("únos unesla")
        assert result.legal_concepts.count("mezinárodní únos dítěte") == 1

    def test_alimony_query(self) -> None:
        result = process_query("neplacení alimenty výživa")
        assert "vyživovací povinnost" in result.legal_concepts

    def test_rozvod_query(self) -> None:
        result = process_query("rozvod manželů péče o dítě")
        assert "rodinné právo" in result.legal_concepts
        assert "svěření do péče" in result.legal_concepts


# ---------------------------------------------------------------------------
# process_query — trace events
# ---------------------------------------------------------------------------


class TestProcessQueryTrace:
    def test_all_four_trace_events_emitted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="app.rag.query.query_processor"):
            process_query("únos dítěte")

        messages = [r.getMessage() for r in caplog.records]
        assert any("TRACE query.start" in m for m in messages)
        assert any("TRACE query.normalized" in m for m in messages)
        assert any("TRACE query.keywords" in m for m in messages)
        assert any("TRACE query.concepts" in m for m in messages)

    def test_trace_start_contains_original_query(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="app.rag.query.query_processor"):
            process_query("únos dítěte")

        start = next(
            r.getMessage()
            for r in caplog.records
            if "query.start" in r.getMessage()
        )
        assert "únos dítěte" in start

    def test_trace_keywords_contains_keyword_list(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="app.rag.query.query_processor"):
            process_query("únos dítěte")

        kw_msg = next(
            r.getMessage()
            for r in caplog.records
            if "query.keywords" in r.getMessage()
        )
        assert "keywords=" in kw_msg

    def test_trace_concepts_contains_concepts_list(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="app.rag.query.query_processor"):
            process_query("únos dítěte")

        concepts_msg = next(
            r.getMessage()
            for r in caplog.records
            if "query.concepts" in r.getMessage()
        )
        assert "legal_concepts=" in concepts_msg
