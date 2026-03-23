import unittest
from unittest.mock import patch

from app.models.search_result import NalusResult
from app.services.decision_service import enrich_results_with_text


class TestDecisionService(unittest.TestCase):
    @patch("app.services.decision_service.extract_plain_text")
    @patch("app.services.decision_service.fetch_decision_html")
    def test_enrich_results_with_text_sets_full_text(
        self,
        mock_fetch_decision_html,
        mock_extract_plain_text,
    ) -> None:
        result = NalusResult(
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
            text_url="https://nalus.usoud.cz/Search/GetText.aspx?sz=1-1-24",
        )
        mock_fetch_decision_html.return_value = "<html>full decision</html>"
        mock_extract_plain_text.return_value = "Full decision text"

        enriched_results = enrich_results_with_text([result])

        self.assertEqual(len(enriched_results), 1)
        self.assertEqual(enriched_results[0].full_text, "Full decision text")
        self.assertIsNone(result.full_text)
        mock_fetch_decision_html.assert_called_once_with(result.text_url)
        mock_extract_plain_text.assert_called_once_with("<html>full decision</html>")

    def test_enrich_results_with_text_keeps_result_without_text_url(self) -> None:
        result = NalusResult(
            result_id=2,
            case_reference="I.ÚS 2/24",
            ecli=None,
            judge_rapporteur=None,
            petitioner=None,
            popular_name=None,
            decision_date=None,
            announcement_date=None,
            filing_date=None,
            publication_date=None,
            text_url=None,
        )

        enriched_results = enrich_results_with_text([result])

        self.assertEqual(len(enriched_results), 1)
        self.assertIsNone(enriched_results[0].full_text)
        self.assertIsNot(enriched_results[0], result)

    @patch("app.services.decision_service.fetch_decision_html")
    def test_enrich_results_with_text_skips_failed_download(
        self,
        mock_fetch_decision_html,
    ) -> None:
        result = NalusResult(
            result_id=3,
            case_reference="I.ÚS 3/24",
            ecli=None,
            judge_rapporteur=None,
            petitioner=None,
            popular_name=None,
            decision_date=None,
            announcement_date=None,
            filing_date=None,
            publication_date=None,
            text_url="https://nalus.usoud.cz/Search/GetText.aspx?sz=1-3-24",
        )
        mock_fetch_decision_html.side_effect = RuntimeError("network error")

        enriched_results = enrich_results_with_text([result])

        self.assertEqual(len(enriched_results), 1)
        self.assertIsNone(enriched_results[0].full_text)


if __name__ == "__main__":
    unittest.main()
