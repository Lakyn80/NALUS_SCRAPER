import unittest
from unittest.mock import Mock, patch

from app.crawler.text_fetcher import extract_plain_text, fetch_decision_html


class TestTextFetcher(unittest.TestCase):
    @patch("app.crawler.text_fetcher.requests.get")
    def test_fetch_decision_html_returns_response_text(self, mock_get: Mock) -> None:
        response = Mock()
        response.text = "<html>decision</html>"
        response.raise_for_status = Mock()
        response.encoding = None
        mock_get.return_value = response

        html = fetch_decision_html("https://nalus.usoud.cz/Search/GetText.aspx?sz=1-1-01")

        self.assertEqual(html, "<html>decision</html>")
        self.assertEqual(response.encoding, "utf-8")
        response.raise_for_status.assert_called_once_with()
        mock_get.assert_called_once_with(
            "https://nalus.usoud.cz/Search/GetText.aspx?sz=1-1-01",
            timeout=60,
        )

    def test_extract_plain_text_removes_script_and_style(self) -> None:
        html = """
        <html>
          <head>
            <style>.hidden { display: none; }</style>
            <script>console.log('ignore')</script>
          </head>
          <body>
            <h1>Rozsudek</h1>
            <p>První odstavec.</p>
            <p>Druhý odstavec.</p>
          </body>
        </html>
        """

        text = extract_plain_text(html)

        self.assertIn("Rozsudek", text)
        self.assertIn("První odstavec.", text)
        self.assertIn("Druhý odstavec.", text)
        self.assertNotIn("console.log", text)
        self.assertNotIn("display: none", text)


if __name__ == "__main__":
    unittest.main()
