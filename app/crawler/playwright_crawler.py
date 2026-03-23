import html
import re

import requests

BASE_URL = "https://nalus.usoud.cz"
SEARCH_URL = "https://nalus.usoud.cz/Search/Search.aspx"
RESULTS_URL = "https://nalus.usoud.cz/Search/Results.aspx"
RESULTS_URL_SUFFIX = "/Search/Results.aspx"

_HIDDEN_INPUT_RE = re.compile(
    r'<input[^>]+type=["\']hidden["\'][^>]+name=["\'](?P<name>[^"\']+)["\'][^>]+value=["\'](?P<value>.*?)["\']',
    re.IGNORECASE | re.DOTALL,
)


def _extract_hidden_fields(html_text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in _HIDDEN_INPUT_RE.finditer(html_text):
        fields[match.group("name")] = html.unescape(match.group("value"))

    required = {"__VIEWSTATE", "__VIEWSTATEGENERATOR", "__EVENTVALIDATION"}
    missing = required - fields.keys()
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise RuntimeError(f"Missing WebForms hidden fields: {missing_list}")

    return fields


def _default_headers() -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "cs,en-US;q=0.9,en;q=0.8",
    }


def _run_search(session: requests.Session, query: str) -> requests.Response:
    headers = _default_headers()
    search_response = session.get(SEARCH_URL, headers=headers, timeout=60)
    search_response.raise_for_status()

    hidden_fields = _extract_hidden_fields(search_response.text)
    payload = {
        **hidden_fields,
        "__EVENTTARGET": "",
        "__EVENTARGUMENT": "",
        "ctl00$MainContent$nalezy": "on",
        "ctl00$MainContent$usneseni": "on",
        "ctl00$MainContent$stanoviska_plena": "on",
        "ctl00$MainContent$naveti": "on",
        "ctl00$MainContent$vyrok": "on",
        "ctl00$MainContent$oduvodneni": "on",
        "ctl00$MainContent$odlisne_stanovisko": "on",
        "ctl00$MainContent$text": query,
        "ctl00$MainContent$but_search": "Vyhledat",
    }

    result_response = session.post(
        SEARCH_URL,
        data=payload,
        headers={
            **headers,
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": SEARCH_URL,
            "Origin": BASE_URL,
        },
        timeout=60,
        allow_redirects=True,
    )
    result_response.raise_for_status()

    if not result_response.url.endswith(RESULTS_URL_SUFFIX):
        raise RuntimeError(
            f"Search did not redirect to Results.aspx. Final URL: {result_response.url}"
        )
    if "Výsledky" not in result_response.text:
        raise RuntimeError("Results page loaded but 'Výsledky' marker was not found.")

    return result_response


def fetch_page_html(query: str = "rodinné právo", page: int = 0) -> str:
    if page < 0:
        raise ValueError("Page index must be zero or greater.")

    session = requests.Session()
    result_response = _run_search(session, query)
    if page == 0:
        return result_response.text

    paged_response = session.get(
        RESULTS_URL,
        params={"page": page},
        headers={
            **_default_headers(),
            "Referer": result_response.url,
        },
        timeout=60,
    )
    paged_response.raise_for_status()

    if "ResultDetail.aspx" not in paged_response.text:
        raise RuntimeError(f"Failed to load results page {page + 1}.")

    return paged_response.text
