import unittest
from unittest.mock import Mock, patch

from app.crawler.text_fetcher import extract_plain_text, fetch_decision_html


class TestTextFetcher(unittest.TestCase):
    def test_fetch_decision_html_uses_provided_session(self) -> None:
        session = Mock()
        response = Mock()
        response.text = "<html>decision</html>"
        response.raise_for_status = Mock()
        response.encoding = None
        session.get.return_value = response

        html = fetch_decision_html("https://nalus.usoud.cz/Search/GetText.aspx?sz=1-1-01", session=session)

        self.assertEqual(html, "<html>decision</html>")
        self.assertEqual(response.encoding, "utf-8")
        response.raise_for_status.assert_called_once_with()
        response.close.assert_called_once_with()
        session.get.assert_called_once_with(
            "https://nalus.usoud.cz/Search/GetText.aspx?sz=1-1-01",
            timeout=(10, 30),
        )

    @patch("app.crawler.text_fetcher.create_text_session")
    def test_fetch_decision_html_closes_owned_session(self, mock_get: Mock) -> None:
        session = Mock()
        response = Mock()
        response.text = "<html>decision</html>"
        response.raise_for_status = Mock()
        response.encoding = None
        session.get.return_value = response
        mock_get.return_value = session

        html = fetch_decision_html("https://nalus.usoud.cz/Search/GetText.aspx?sz=1-1-01")

        self.assertEqual(html, "<html>decision</html>")
        self.assertEqual(response.encoding, "utf-8")
        response.raise_for_status.assert_called_once_with()
        response.close.assert_called_once_with()
        session.get.assert_called_once_with(
            "https://nalus.usoud.cz/Search/GetText.aspx?sz=1-1-01",
            timeout=(10, 30),
        )
        session.close.assert_called_once_with()

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
