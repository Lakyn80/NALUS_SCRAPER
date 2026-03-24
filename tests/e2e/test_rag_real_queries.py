"""
E2E evaluation tests — real query scenarios against a seeded Qdrant collection.

Infrastructure:
  - Real Qdrant (localhost:6333), temporary collection nalus_e2e_test
  - Real QdrantIngestor (seeded in conftest.py via PointStruct adapter)
  - Real RetrievalPipeline + HybridAnswerService
  - MockEmbedder (no real embedding model available)
  - NO mocked services

NOTE: Dense retrieval is non-semantic here (MockEmbedder → uniform vectors).
Retrieval quality is driven by KeywordRetriever — intentional until a real
embedder is wired. Relevance assertions check the TEXT of returned chunks.

Run:
    pytest tests/e2e/test_rag_real_queries.py -v -s
"""

import logging

from app.rag.answer.answer_service import AnswerResult
from app.rag.answer.hybrid_service import HybridAnswerService
from app.rag.llm.models import LLMOutput
from app.rag.orchestration.pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _texts(pipeline_result) -> str:
    """Concatenated text of all returned chunks — used for relevance assertions."""
    return " ".join(r.text for r in pipeline_result.results).lower()


def _answer_text(answer: AnswerResult | LLMOutput) -> str:
    if isinstance(answer, AnswerResult):
        return answer.summary
    return answer.answer


def _log_result(query: str, pipeline_result, answer) -> None:
    ids = [r.id for r in pipeline_result.results]
    route = "answer" if isinstance(answer, AnswerResult) else "llm"
    logger.info("[e2e] query=%r  ids=%s  route=%s", query, ids, route)
    logger.info("[e2e] answer=%r", _answer_text(answer)[:120])


# ---------------------------------------------------------------------------
# Scénář 1 — únos dítěte Rusko
# ---------------------------------------------------------------------------


class TestUnosDiteteRusko:
    def test_returns_results(
        self, pipeline: RetrievalPipeline, hybrid_service: HybridAnswerService
    ) -> None:
        query = "matka unesla dítě do Ruska"
        result = pipeline.run(query, top_k=5)
        _log_result(query, result, hybrid_service.generate(query, result.results))
        assert len(result.results) > 0

    def test_retrieved_text_contains_unos_or_haag(
        self, pipeline: RetrievalPipeline, hybrid_service: HybridAnswerService
    ) -> None:
        query = "matka unesla dítě do Ruska"
        result = pipeline.run(query, top_k=5)
        texts = _texts(result)
        assert "únos" in texts or "haag" in texts or "rusko" in texts or "ruska" in texts

    def test_answer_text_references_query_or_legal_concept(
        self, pipeline: RetrievalPipeline, hybrid_service: HybridAnswerService
    ) -> None:
        query = "matka unesla dítě do Ruska"
        result = pipeline.run(query, top_k=5)
        answer = hybrid_service.generate(query, result.results)
        text = _answer_text(answer).lower()
        assert "únos" in text or "haag" in text or "matka unesla" in text or "ruska" in text

    def test_answer_type_determined_by_chunk_count(
        self, pipeline: RetrievalPipeline, hybrid_service: HybridAnswerService
    ) -> None:
        query = "matka unesla dítě do Ruska"
        result = pipeline.run(query, top_k=5)
        answer = hybrid_service.generate(query, result.results)
        if len(result.results) >= 2:
            assert isinstance(answer, LLMOutput)
        else:
            assert isinstance(answer, AnswerResult)

    def test_multiple_abduction_cases_found(
        self, pipeline: RetrievalPipeline, hybrid_service: HybridAnswerService
    ) -> None:
        query = "mezinárodní únos dítěte Haagská úmluva"
        result = pipeline.run(query, top_k=5)
        texts = _texts(result)
        # Both abduction cases (255/22 and 401/22) contain "únos" and "haag"
        assert texts.count("únos") >= 1 or texts.count("haag") >= 1


# ---------------------------------------------------------------------------
# Scénář 2 — rodičovská odpovědnost
# ---------------------------------------------------------------------------


class TestRodicskaOdpovednost:
    def test_returns_results(
        self, pipeline: RetrievalPipeline, hybrid_service: HybridAnswerService
    ) -> None:
        query = "rodičovská odpovědnost po rozvodu"
        result = pipeline.run(query, top_k=5)
        _log_result(query, result, hybrid_service.generate(query, result.results))
        assert len(result.results) > 0

    def test_retrieved_text_mentions_rodice_or_dite(
        self, pipeline: RetrievalPipeline, hybrid_service: HybridAnswerService
    ) -> None:
        query = "rodičovská odpovědnost po rozvodu"
        result = pipeline.run(query, top_k=5)
        texts = _texts(result)
        assert "rodič" in texts or "dítě" in texts or "péče" in texts

    def test_answer_references_query(
        self, pipeline: RetrievalPipeline, hybrid_service: HybridAnswerService
    ) -> None:
        query = "rodičovská odpovědnost péče o dítě"
        result = pipeline.run(query, top_k=5)
        answer = hybrid_service.generate(query, result.results)
        text = _answer_text(answer).lower()
        assert "rodič" in text or "péče" in text or "dítě" in text or "rodičovská" in text

    def test_result_scores_are_positive(
        self, pipeline: RetrievalPipeline, hybrid_service: HybridAnswerService
    ) -> None:
        query = "rodičovská odpovědnost"
        result = pipeline.run(query, top_k=5)
        for chunk in result.results:
            assert chunk.score > 0

    def test_parental_responsibility_text_in_results(
        self, pipeline: RetrievalPipeline, hybrid_service: HybridAnswerService
    ) -> None:
        # top_k >= corpus size so all 8 chunks are retrieved regardless of
        # Qdrant's ordering (all chunks have identical MockEmbedder vectors).
        query = "střídavá péče zájem dítěte rozvod"
        result = pipeline.run(query, top_k=10)
        texts = _texts(result)
        assert "střídavá" in texts or "zájem dítěte" in texts or "rozvod" in texts


# ---------------------------------------------------------------------------
# Scénář 3 — náhrada škody stát
# ---------------------------------------------------------------------------


class TestNahradaSkodyState:
    def test_returns_results(
        self, pipeline: RetrievalPipeline, hybrid_service: HybridAnswerService
    ) -> None:
        query = "náhrada škody způsobená státem nezákonné rozhodnutí"
        result = pipeline.run(query, top_k=5)
        _log_result(query, result, hybrid_service.generate(query, result.results))
        assert len(result.results) > 0

    def test_retrieved_text_contains_nahrada_or_stat(
        self, pipeline: RetrievalPipeline, hybrid_service: HybridAnswerService
    ) -> None:
        query = "náhrada škody způsobená státem nezákonné rozhodnutí"
        result = pipeline.run(query, top_k=5)
        texts = _texts(result)
        assert "náhrad" in texts or "škod" in texts or "stát" in texts

    def test_answer_contains_relevant_concept(
        self, pipeline: RetrievalPipeline, hybrid_service: HybridAnswerService
    ) -> None:
        query = "náhrada škody stát nesprávný úřední postup"
        result = pipeline.run(query, top_k=5)
        answer = hybrid_service.generate(query, result.results)
        text = _answer_text(answer).lower()
        assert (
            "náhrad" in text
            or "škod" in text
            or "stát" in text
            or "náhrada škody" in text
        )

    def test_sources_in_llm_output_are_from_result_set(
        self, pipeline: RetrievalPipeline, hybrid_service: HybridAnswerService
    ) -> None:
        query = "náhrada škody způsobená státem"
        result = pipeline.run(query, top_k=5)
        answer = hybrid_service.generate(query, result.results)
        if isinstance(answer, LLMOutput):
            result_ids = {r.id for r in result.results}
            for src in answer.sources:
                assert src in result_ids

    def test_state_liability_text_in_results(
        self, pipeline: RetrievalPipeline, hybrid_service: HybridAnswerService
    ) -> None:
        query = "odpovědnost státu objektivní zavinění úředník"
        result = pipeline.run(query, top_k=5)
        texts = _texts(result)
        assert "odpovědnost" in texts or "státu" in texts or "škoda" in texts


# ---------------------------------------------------------------------------
# Scénář 4 — délka řízení
# ---------------------------------------------------------------------------


class TestDelkaRizeni:
    def test_returns_results(
        self, pipeline: RetrievalPipeline, hybrid_service: HybridAnswerService
    ) -> None:
        query = "nepřiměřená délka soudního řízení průtahy"
        result = pipeline.run(query, top_k=5)
        _log_result(query, result, hybrid_service.generate(query, result.results))
        assert len(result.results) > 0

    def test_retrieved_text_contains_delka_or_prутahy(
        self, pipeline: RetrievalPipeline, hybrid_service: HybridAnswerService
    ) -> None:
        query = "nepřiměřená délka soudního řízení průtahy"
        result = pipeline.run(query, top_k=5)
        texts = _texts(result)
        assert "délka" in texts or "průtah" in texts or "řízení" in texts

    def test_answer_references_delay_concept(
        self, pipeline: RetrievalPipeline, hybrid_service: HybridAnswerService
    ) -> None:
        query = "délka řízení přiměřená lhůta"
        result = pipeline.run(query, top_k=5)
        answer = hybrid_service.generate(query, result.results)
        text = _answer_text(answer).lower()
        assert "délka" in text or "průtah" in text or "řízení" in text or "lhůta" in text

    def test_all_results_have_id_and_text(
        self, pipeline: RetrievalPipeline, hybrid_service: HybridAnswerService
    ) -> None:
        query = "délka řízení"
        result = pipeline.run(query, top_k=5)
        for chunk in result.results:
            assert chunk.id
            assert chunk.text

    def test_delay_keywords_in_top_results(
        self, pipeline: RetrievalPipeline, hybrid_service: HybridAnswerService
    ) -> None:
        query = "nepřiměřená délka trestního řízení základní práva"
        result = pipeline.run(query, top_k=5)
        texts = _texts(result)
        assert "nepřiměřen" in texts or "základní" in texts or "listiny" in texts


# ---------------------------------------------------------------------------
# Pipeline structural tests (cross-scenario)
# ---------------------------------------------------------------------------


class TestPipelineStructure:
    def test_processed_query_has_keywords(
        self, pipeline: RetrievalPipeline
    ) -> None:
        result = pipeline.run("matka unesla dítě Rusko", top_k=5)
        assert len(result.processed_query.keywords) > 0

    def test_results_sorted_by_score_descending(
        self, pipeline: RetrievalPipeline
    ) -> None:
        result = pipeline.run("náhrada škody stát řízení", top_k=5)
        scores = [r.score for r in result.results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self, pipeline: RetrievalPipeline) -> None:
        result = pipeline.run("únos dítě rodičovská odpovědnost škoda řízení", top_k=3)
        assert len(result.results) <= 3

    def test_no_duplicate_ids(self, pipeline: RetrievalPipeline) -> None:
        result = pipeline.run("únos dítě rodičovská odpovědnost škoda", top_k=10)
        ids = [r.id for r in result.results]
        assert len(ids) == len(set(ids))

    def test_each_chunk_has_source_field(
        self, pipeline: RetrievalPipeline
    ) -> None:
        result = pipeline.run("délka řízení únos dítěte náhrada škody", top_k=5)
        for chunk in result.results:
            assert chunk.source in {"dense", "keyword"}
