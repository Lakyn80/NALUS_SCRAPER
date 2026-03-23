import requests
from bs4 import BeautifulSoup


def fetch_decision_html(text_url: str) -> str:
    response = requests.get(text_url, timeout=60)
    response.raise_for_status()
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
