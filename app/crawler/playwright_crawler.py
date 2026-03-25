import html
import re
import time

import requests

BASE_URL = "https://nalus.usoud.cz"
SEARCH_URL = "https://nalus.usoud.cz/Search/Search.aspx"
RESULTS_URL = "https://nalus.usoud.cz/Search/Results.aspx"
RESULTS_URL_SUFFIX = "/Search/Results.aspx"
REQUEST_TIMEOUT_SECONDS = 15
REQUEST_MAX_RETRIES = 3

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


def _safe_request(request_callable, url: str, max_retries: int = REQUEST_MAX_RETRIES, **kwargs) -> requests.Response | None:
    request_kwargs = {**kwargs, "timeout": REQUEST_TIMEOUT_SECONDS}

    for attempt in range(max_retries):
        try:
            response = request_callable(url, **request_kwargs)
            response.raise_for_status()
            return response
        except Exception as exc:
            print(f"[WARN] request failed attempt={attempt + 1} url={url} error={exc}")
            if attempt < max_retries - 1:
                time.sleep(1)

    print(f"[ERROR] request failed permanently url={url}")
    return None


def _run_search(session: requests.Session, query: str) -> requests.Response:
    headers = _default_headers()
    search_response = _safe_request(session.get, SEARCH_URL, headers=headers)
    if search_response is None:
        raise RuntimeError(f"Failed to load search page: {SEARCH_URL}")

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

    result_response = _safe_request(
        session.post,
        SEARCH_URL,
        data=payload,
        headers={
            **headers,
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": SEARCH_URL,
            "Origin": BASE_URL,
        },
        allow_redirects=True,
    )
    if result_response is None:
        raise RuntimeError(f"Failed to submit search request: {SEARCH_URL}")

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

    paged_response = _safe_request(
        session.get,
        RESULTS_URL,
        params={"page": page},
        headers={
            **_default_headers(),
            "Referer": result_response.url,
        },
    )
    if paged_response is None:
        raise RuntimeError(f"Failed to load results page {page + 1}.")

    if "ResultDetail.aspx" not in paged_response.text:
        raise RuntimeError(f"Failed to load results page {page + 1}.")

    return paged_response.text
