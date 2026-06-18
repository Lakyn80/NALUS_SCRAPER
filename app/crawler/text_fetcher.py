import os
import time

import requests
import urllib3
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup

REQUEST_CONNECT_TIMEOUT_SECONDS = 10
REQUEST_READ_TIMEOUT_SECONDS = 30
REQUEST_MAX_RETRIES = 3
REQUEST_RETRY_BACKOFF_SECONDS = 2


def _ssl_verify_enabled() -> bool:
    value = os.getenv("NALUS_SSL_VERIFY", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def create_text_session() -> requests.Session:
    session = requests.Session()
    session.verify = _ssl_verify_enabled()
    if not session.verify:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "cs,en-US;q=0.9,en;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )
    adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _safe_get(
    url: str,
    *,
    session: requests.Session | None = None,
    max_retries: int = REQUEST_MAX_RETRIES,
) -> requests.Response | None:
    request_session = session or create_text_session()

    for attempt in range(max_retries):
        try:
            response = request_session.get(
                url,
                timeout=(REQUEST_CONNECT_TIMEOUT_SECONDS, REQUEST_READ_TIMEOUT_SECONDS),
            )
            response.raise_for_status()
            return response
        except Exception as exc:
            print(f"[WARN] request failed attempt={attempt + 1} url={url} error={exc}")
            if attempt < max_retries - 1:
                time.sleep(REQUEST_RETRY_BACKOFF_SECONDS * (attempt + 1))

    print(f"[ERROR] request failed permanently url={url}")
    return None


def fetch_decision_html(text_url: str, session: requests.Session | None = None) -> str:
    owns_session = session is None
    request_session = session or create_text_session()

    try:
        response = _safe_get(text_url, session=request_session)
        if response is None:
            raise RuntimeError(f"Failed to fetch decision HTML: {text_url}")

        response.encoding = "utf-8"
        html = response.text
        response.close()
        return html
    finally:
        if owns_session:
            request_session.close()


def extract_plain_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style"]):
        tag.decompose()

    lines = []
    for line in soup.get_text("\n").splitlines():
        cleaned = " ".join(line.split()).strip()
        if cleaned:
            lines.append(cleaned)

    return "\n".join(lines)
