import time

import requests
from bs4 import BeautifulSoup
REQUEST_TIMEOUT_SECONDS = 15
REQUEST_MAX_RETRIES = 3


def _safe_get(url: str, max_retries: int = REQUEST_MAX_RETRIES) -> requests.Response | None:
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response
        except Exception as exc:
            print(f"[WARN] request failed attempt={attempt + 1} url={url} error={exc}")
            if attempt < max_retries - 1:
                time.sleep(1)

    print(f"[ERROR] request failed permanently url={url}")
    return None


def fetch_decision_html(text_url: str) -> str:
    response = _safe_get(text_url)
    if response is None:
        raise RuntimeError(f"Failed to fetch decision HTML: {text_url}")

    response.encoding = "utf-8"
    return response.text


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
