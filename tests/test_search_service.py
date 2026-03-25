import unittest
from dataclasses import replace
from unittest.mock import patch

from app.models.search_result import NalusResult, NalusSearchPage
from app.services.search_service import collect_results


def _make_result(result_id: int) -> NalusResult:
    return NalusResult(
        result_id=result_id,
        case_reference=f"I.ÚS {result_id}/24",
        ecli=None,
        judge_rapporteur=None,
        petitioner=None,
        popular_name=None,
        decision_date=None,
        announcement_date=None,
        filing_date=None,
        publication_date=None,
        text_url=f"https://nalus.usoud.cz/Search/GetText.aspx?sz={result_id}",
    )


class TestSearchService(unittest.TestCase):
    @patch("app.services.search_service.enrich_results_with_text")
    @patch("app.services.search_service.extract_search_page")
    @patch("app.services.search_service.fetch_page_html")
    def test_collect_results_collects_multiple_pages(
        self,
        mock_fetch_page_html,
        mock_extract_search_page,
        mock_enrich_results_with_text,
    ) -> None:
        page_one_results = [_make_result(1), _make_result(2)]
        page_two_results = [_make_result(3), _make_result(4)]

        mock_fetch_page_html.side_effect = ["page-1-html", "page-2-html"]
        mock_extract_search_page.side_effect = [
            NalusSearchPage(
                query="rodinné právo",
                page_index=0,
                page_number=1,
                page_size=2,
                start_result=1,
                end_result=2,
                total_results=4,
                total_pages=2,
                results=page_one_results,
            ),
            NalusSearchPage(
                query="rodinné právo",
                page_index=1,
                page_number=2,
                page_size=2,
                start_result=3,
                end_result=4,
                total_results=4,
                total_pages=2,
                results=page_two_results,
            ),
        ]
        mock_enrich_results_with_text.side_effect = lambda results: [
            replace(result, full_text="full text") for result in results
        ]

        results = collect_results(query="rodinné právo", page_start=1, page_end=2)

        self.assertEqual([result.result_id for result in results], [1, 2, 3, 4])
        self.assertEqual(mock_fetch_page_html.call_count, 2)
        mock_fetch_page_html.assert_any_call(query="rodinné právo", page=0)
        mock_fetch_page_html.assert_any_call(query="rodinné právo", page=1)
        self.assertEqual(mock_enrich_results_with_text.call_count, 2)

    @patch("app.services.search_service.enrich_results_with_text")
    @patch("app.services.search_service.extract_search_page")
    @patch("app.services.search_service.fetch_page_html")
    def test_collect_results_respects_max_results_before_enrichment(
        self,
        mock_fetch_page_html,
        mock_extract_search_page,
        mock_enrich_results_with_text,
    ) -> None:
        page_results = [_make_result(1), _make_result(2), _make_result(3)]

        mock_fetch_page_html.return_value = "page-1-html"
        mock_extract_search_page.return_value = NalusSearchPage(
            query="rodinné právo",
            page_index=0,
            page_number=1,
            page_size=3,
            start_result=1,
            end_result=3,
            total_results=3,
            total_pages=1,
            results=page_results,
        )
        mock_enrich_results_with_text.side_effect = lambda results: [
            replace(result, full_text="full text") for result in results
        ]

        results = collect_results(
            query="rodinné právo",
            page_start=1,
            page_end=1,
            fetch_full_text=True,
            max_results=2,
        )

        self.assertEqual([result.result_id for result in results], [1, 2])
        enriched_input = mock_enrich_results_with_text.call_args.args[0]
        self.assertEqual([result.result_id for result in enriched_input], [1, 2])

    @patch("app.services.search_service.time.sleep")
    @patch("app.services.search_service.enrich_results_with_text")
    @patch("app.services.search_service.extract_search_page")
    @patch("app.services.search_service.fetch_page_html")
    def test_collect_results_auto_stops_on_last_page_when_page_end_missing(
        self,
        mock_fetch_page_html,
        mock_extract_search_page,
        mock_enrich_results_with_text,
        mock_sleep,
    ) -> None:
        mock_fetch_page_html.side_effect = ["page-1-html", "page-2-html"]
        mock_extract_search_page.side_effect = [
            NalusSearchPage(
                query="rodinné právo",
                page_index=0,
                page_number=1,
                page_size=2,
                start_result=1,
                end_result=2,
                total_results=4,
                total_pages=2,
                results=[_make_result(1), _make_result(2)],
            ),
            NalusSearchPage(
                query="rodinné právo",
                page_index=1,
                page_number=2,
                page_size=2,
                start_result=3,
                end_result=4,
                total_results=4,
                total_pages=2,
                results=[_make_result(3), _make_result(4)],
            ),
        ]
        mock_enrich_results_with_text.side_effect = lambda results: results

        results = collect_results(
            query="rodinné právo",
            page_start=1,
            page_end=None,
            fetch_full_text=False,
            max_results=10,
            batch_pages=10,
            batch_sleep_seconds=0,
        )

        self.assertEqual([result.result_id for result in results], [1, 2, 3, 4])
        self.assertEqual(mock_fetch_page_html.call_count, 2)
        mock_sleep.assert_not_called()

    @patch("app.services.search_service.time.sleep")
    @patch("app.services.search_service.enrich_results_with_text")
    @patch("app.services.search_service.extract_search_page")
    @patch("app.services.search_service.fetch_page_html")
    def test_collect_results_sleeps_between_batches(
        self,
        mock_fetch_page_html,
        mock_extract_search_page,
        mock_enrich_results_with_text,
        mock_sleep,
    ) -> None:
        mock_fetch_page_html.side_effect = ["page-1-html", "page-2-html", "page-3-html"]
        mock_extract_search_page.side_effect = [
            NalusSearchPage(
                query="rodinné právo",
                page_index=0,
                page_number=1,
                page_size=1,
                start_result=1,
                end_result=1,
                total_results=3,
                total_pages=3,
                results=[_make_result(1)],
            ),
            NalusSearchPage(
                query="rodinné právo",
                page_index=1,
                page_number=2,
                page_size=1,
                start_result=2,
                end_result=2,
                total_results=3,
                total_pages=3,
                results=[_make_result(2)],
            ),
            NalusSearchPage(
                query="rodinné právo",
                page_index=2,
                page_number=3,
                page_size=1,
                start_result=3,
                end_result=3,
                total_results=3,
                total_pages=3,
                results=[_make_result(3)],
            ),
        ]
        mock_enrich_results_with_text.side_effect = lambda results: results

        results = collect_results(
            query="rodinné právo",
            page_start=1,
            page_end=None,
            fetch_full_text=False,
            max_results=10,
            batch_pages=2,
            batch_sleep_seconds=1.5,
        )

        self.assertEqual([result.result_id for result in results], [1, 2, 3])
        mock_sleep.assert_called_once_with(1.5)

    def test_collect_results_validates_page_range(self) -> None:
        with self.assertRaises(ValueError):
            collect_results(query="rodinné právo", page_start=2, page_end=1)

    @patch("app.services.search_service.enrich_results_with_text")
    @patch("app.services.search_service.extract_search_page")
    @patch("app.services.search_service.fetch_page_html")
    def test_collect_results_stops_on_duplicate(
        self,
        mock_fetch_page_html,
        mock_extract_search_page,
        mock_enrich_results_with_text,
    ) -> None:
        duplicate = _make_result(1)

        mock_fetch_page_html.return_value = "page-1-html"
        mock_extract_search_page.return_value = NalusSearchPage(
            query="rodinné právo",
            page_index=0,
            page_number=1,
            page_size=2,
            start_result=1,
            end_result=2,
            total_results=2,
            total_pages=1,
            results=[duplicate, duplicate],
        )
        mock_enrich_results_with_text.side_effect = lambda results: results

        with self.assertRaises(RuntimeError):
            collect_results(query="rodinné právo", page_start=1, page_end=1)

    @patch("app.services.search_service.enrich_results_with_text")
    @patch("app.services.search_service.extract_search_page")
    @patch("app.services.search_service.fetch_page_html")
    def test_collect_results_stops_on_missing_full_text(
        self,
        mock_fetch_page_html,
        mock_extract_search_page,
        mock_enrich_results_with_text,
    ) -> None:
        result = _make_result(1)

        mock_fetch_page_html.return_value = "page-1-html"
        mock_extract_search_page.return_value = NalusSearchPage(
            query="rodinné právo",
            page_index=0,
            page_number=1,
            page_size=1,
            start_result=1,
            end_result=1,
            total_results=1,
            total_pages=1,
            results=[result],
        )
        mock_enrich_results_with_text.return_value = [result]

        with self.assertRaises(RuntimeError):
            collect_results(query="rodinné právo", page_start=1, page_end=1, fetch_full_text=True)

    @patch("app.services.search_service.fetch_page_html")
    @patch("app.services.search_service.extract_search_page")
    def test_collect_results_stops_on_empty_page(
        self,
        mock_extract_search_page,
        mock_fetch_page_html,
    ) -> None:
        mock_fetch_page_html.return_value = "page-1-html"
        mock_extract_search_page.return_value = NalusSearchPage(
            query="rodinné právo",
            page_index=0,
            page_number=1,
            page_size=0,
            start_result=0,
            end_result=0,
            total_results=0,
            total_pages=1,
            results=[],
        )

        with self.assertRaises(RuntimeError):
            collect_results(query="rodinné právo", page_start=1, page_end=1, fetch_full_text=False)


if __name__ == "__main__":
    unittest.main()
