import os
import unittest

from app.crawler.extractor import extract_search_page
from app.crawler.playwright_crawler import fetch_page_html
from app.services.decision_service import enrich_results_with_text


@unittest.skipUnless(
    os.getenv("NALUS_LIVE_TEST") == "1",
    "Set NALUS_LIVE_TEST=1 to run live integration test.",
)
class TestLiveFullText(unittest.TestCase):
    def test_downloads_real_full_decision_text(self) -> None:
        html = fetch_page_html()
        search_page = extract_search_page(html)
        result = next(item for item in search_page.results if item.text_url)

        enriched_results = enrich_results_with_text([result])
        full_text = enriched_results[0].full_text

        self.assertIsNotNone(full_text)
        self.assertIn("Ústavní soud", full_text)
        self.assertGreater(len(full_text), 1000)


if __name__ == "__main__":
    unittest.main()
