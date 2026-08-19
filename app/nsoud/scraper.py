from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None


BASE_URL = "https://rozhodnuti.nsoud.cz"
SEARCH_HOME_URL = f"{BASE_URL}/"
SEARCH_POST_URL = f"{BASE_URL}/judikatura/judikatura_ns.nsf/searchRozhodnuti2?createdocument"
SOURCE_ATTRIBUTION = "Nejvyšší soud České republiky, rozhodnuti.nsoud.cz"
USER_AGENT = "nalus-scraper/nsoud-sample (+https://rozhodnuti.nsoud.cz/)"
REQUEST_TIMEOUT_SECONDS = 45
DEFAULT_REQUEST_DELAY_SECONDS = 1.0
RETRY_LIMIT = 3
INITIAL_WINDOW_DAYS = 7
MAX_WINDOW_DAYS = 3650
MAX_SEARCH_EXPANSIONS = 8
MAX_PAGE_SIZE = 60
DEFAULT_MAX_PAGES = 10

CASE_NUMBER_LABELS = ("Spisová značka", "Senátní značka")
STANDARD_CITATION_PREFIX = "Citace rozhodnutí Nejvyššího soudu by měla obsahovat"

logger = logging.getLogger("nsoud_scraper")


@dataclass(frozen=True)
class ResponseData:
    url: str
    html: str
    headers: dict[str, str]


@dataclass(frozen=True)
class SearchResultLink:
    case_number: str | None
    detail_url: str


@dataclass(frozen=True)
class SearchPageData:
    url: str
    html: str
    total_results: int
    links: list[SearchResultLink]


@dataclass
class DiscoveryResult:
    provider_name: str
    transport_name: str
    remote_date_filter_supported: bool = False
    remote_date_filter_field: str | None = None
    form_actions: list[str] = field(default_factory=list)
    search_url: str | None = None
    query_params: list[tuple[str, str]] = field(default_factory=list)
    pagination_urls: list[str] = field(default_factory=list)
    detail_urls: list[str] = field(default_factory=list)
    attempted_urls: list[str] = field(default_factory=list)
    total_results: int | None = None


@dataclass
class ScrapeStats:
    provider_name: str = ""
    transport_name: str = ""
    records_written: int = 0
    records_updated: int = 0
    duplicates_skipped: int = 0
    parse_failures: int = 0
    pages_visited: int = 0
    records_discovered: int = 0
    unique_candidates: int = 0
    site_total_results: int | None = None
    locally_filtered_out: int = 0
    attempted_urls: list[str] = field(default_factory=list)
    first_record_keys: list[str] = field(default_factory=list)
    all_written_have_full_text: bool = True
    failure_reasons: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ScrapeConfig:
    limit: int
    date_from: date | None
    date_to: date | None
    delay_seconds: float
    max_pages: int
    out_path: Path
    debug_dir: Path | None
    exhaust: bool = False


def parse_cli_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date '{value}', expected YYYY-MM-DD.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a small sample of Czech Supreme Court decisions."
    )
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of decisions to write.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--date-from", type=parse_cli_date, help="Publication date lower bound in YYYY-MM-DD.")
    parser.add_argument("--date-to", type=parse_cli_date, help="Publication date upper bound in YYYY-MM-DD.")
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY_SECONDS,
        help=f"Delay in seconds between page/detail requests (default: {DEFAULT_REQUEST_DELAY_SECONDS}).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help=f"Hard maximum number of results pages to visit (default: {DEFAULT_MAX_PAGES}).",
    )
    parser.add_argument(
        "--exhaust",
        action="store_true",
        help="Paginate until search results are exhausted (limit becomes a soft cap only if set high).",
    )
    parser.add_argument(
        "--debug-save-html",
        action="store_true",
        help="Save raw HTML snapshots for debugging under the output directory.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def normalize_text(value: str | None) -> str:
    if not value:
        return ""

    text = value.replace("\xa0", " ").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compute_content_hash(full_text: str, url: str) -> str:
    normalized_text = normalize_text(full_text)
    normalized_url = normalize_text(url)
    payload = f"{normalized_url}\n{normalized_text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_existing_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    if not path.exists():
        return hashes

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue

            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSONL line %s in %s: %s", line_number, path, exc)
                continue

            content_hash = payload.get("content_hash")
            if isinstance(content_hash, str) and content_hash:
                hashes.add(content_hash)

    return hashes


def write_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")


def parse_czech_date(raw_value: str | None) -> str | None:
    text = normalize_text(raw_value)
    if not text:
        return None

    match = re.search(r"(\d{1,2})\.\s*(\d{1,2})\.\s*(\d{4})", text)
    if not match:
        return None

    day, month, year = (int(part) for part in match.groups())
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        logger.warning("Invalid Czech date value: %s", raw_value)
        return None


def polite_delay(delay_seconds: float) -> None:
    if delay_seconds > 0:
        time.sleep(delay_seconds)


def build_page_url(search_url: str, *, start: int, count: int) -> str:
    parsed = urlparse(search_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["Start"] = str(start)
    query["Count"] = str(count)
    updated_query = urlencode(query)
    return urlunparse(parsed._replace(query=updated_query))


def extract_total_results(soup: BeautifulSoup) -> int:
    summary = soup.find("h3")
    summary_text = normalize_text(summary.get_text(" ", strip=True) if summary else "")
    if "Nebyly nalezeny" in summary_text:
        return 0

    match = re.search(r"z\s+(\d+)\s+zobrazovaných dokumentů", summary_text)
    if match:
        return int(match.group(1))

    match = re.search(r"Výsledky\s+\d+\s*-\s*\d+\s+z\s+(\d+)", summary_text)
    if match:
        return int(match.group(1))

    return 0


def parse_search_results(html: str) -> tuple[list[SearchResultLink], int]:
    soup = BeautifulSoup(html, "lxml")
    total_results = extract_total_results(soup)
    rows = soup.select("table#tabl tbody tr") or soup.select("table#tabl tr")
    links: list[SearchResultLink] = []

    for row in rows:
        cells = row.find_all("td")
        if not cells:
            continue

        court_text = ""
        if len(cells) >= 2:
            court_text = normalize_text(cells[1].get_text(" ", strip=True))
        if court_text and court_text != "Nejvyšší soud":
            continue

        detail_anchor = row.select_one('a.odk, a[href*="/WebSearch/"]')
        if not detail_anchor:
            continue

        href = detail_anchor.get("href")
        if not href:
            continue

        case_number = normalize_text(detail_anchor.get_text(" ", strip=True)) or None
        detail_url = urljoin(BASE_URL, href)
        links.append(SearchResultLink(case_number=case_number, detail_url=detail_url))

    return links, total_results


def extract_form_actions(home_html: str) -> list[str]:
    soup = BeautifulSoup(home_html, "lxml")
    actions: list[str] = []
    for form in soup.find_all("form"):
        action = form.get("action")
        if not action:
            continue
        actions.append(urljoin(BASE_URL, action))
    return actions


def save_debug_snapshot(debug_dir: Path | None, name: str, html: str) -> None:
    if debug_dir is None:
        return

    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / name
    path.write_text(html, encoding="utf-8")


class BaseProvider(ABC):
    name = "base"
    transport_name = "base"

    def __init__(self, debug_dir: Path | None = None) -> None:
        self.debug_dir = debug_dir

    @abstractmethod
    def fetch_home(self) -> ResponseData:
        raise NotImplementedError

    @abstractmethod
    def submit_search(self, start_date: date, end_date: date, page_size: int) -> SearchPageData:
        raise NotImplementedError

    @abstractmethod
    def fetch_detail(self, url: str) -> ResponseData:
        raise NotImplementedError

    def fetch_results_page(self, url: str) -> SearchPageData:
        response = self.fetch_detail(url)
        links, total_results = parse_search_results(response.html)
        return SearchPageData(
            url=response.url,
            html=response.html,
            total_results=total_results,
            links=links,
        )


class DirectHttpProvider(BaseProvider):
    name = "direct_http"

    def __init__(self, debug_dir: Path | None = None) -> None:
        super().__init__(debug_dir=debug_dir)
        self.transport_name = "requests"
        self._requests_session = requests.Session() if requests is not None else None

    def _request_via_requests(
        self,
        method: str,
        url: str,
        *,
        data: dict[str, str] | None = None,
        allow_redirects: bool = True,
    ) -> ResponseData:
        if self._requests_session is None:
            raise RuntimeError("requests is unavailable.")

        response = self._requests_session.request(
            method=method,
            url=url,
            data=data,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "cs,en;q=0.8"},
            timeout=REQUEST_TIMEOUT_SECONDS,
            allow_redirects=allow_redirects,
        )
        if response.status_code >= 400:
            response.raise_for_status()
        self.transport_name = "requests"
        return ResponseData(url=response.url, html=response.text, headers=dict(response.headers))

    def _request_via_httpx(
        self,
        method: str,
        url: str,
        *,
        data: dict[str, str] | None = None,
        allow_redirects: bool = True,
    ) -> ResponseData:
        if httpx is None:
            raise RuntimeError("httpx is unavailable for direct HTTP fallback.")

        with httpx.Client(
            follow_redirects=allow_redirects,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "cs,en;q=0.8"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as client:
            response = client.request(method=method, url=url, data=data)
            if response.status_code >= 400:
                response.raise_for_status()
            if not allow_redirects and 300 <= response.status_code < 400:
                self.transport_name = "httpx_fallback"
                return ResponseData(url=str(response.url), html=response.text, headers=dict(response.headers))
            self.transport_name = "httpx_fallback"
            return ResponseData(url=str(response.url), html=response.text, headers=dict(response.headers))

    def request(
        self,
        method: str,
        url: str,
        *,
        data: dict[str, str] | None = None,
        allow_redirects: bool = True,
    ) -> ResponseData:
        last_error: Exception | None = None

        for attempt in range(1, RETRY_LIMIT + 1):
            try:
                return self._request_via_requests(
                    method,
                    url,
                    data=data,
                    allow_redirects=allow_redirects,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Direct HTTP requests transport failed for %s %s on attempt %s/%s: %s",
                    method,
                    url,
                    attempt,
                    RETRY_LIMIT,
                    exc,
                )
                try:
                    return self._request_via_httpx(
                        method,
                        url,
                        data=data,
                        allow_redirects=allow_redirects,
                    )
                except Exception as fallback_exc:
                    last_error = fallback_exc
                    logger.warning(
                        "Direct HTTP fallback transport failed for %s %s on attempt %s/%s: %s",
                        method,
                        url,
                        attempt,
                        RETRY_LIMIT,
                        fallback_exc,
                    )
                    if attempt < RETRY_LIMIT:
                        time.sleep(attempt)

        raise RuntimeError(f"Direct HTTP request failed for {method} {url}: {last_error}")

    def fetch_home(self) -> ResponseData:
        response = self.request("GET", SEARCH_HOME_URL)
        save_debug_snapshot(self.debug_dir, "direct_http_home.html", response.html)
        return response

    def submit_search(self, start_date: date, end_date: date, page_size: int) -> SearchPageData:
        post_response = self.request(
            "POST",
            SEARCH_POST_URL,
            data={
                "od": start_date.isoformat(),
                "do": end_date.isoformat(),
                "soud": "Nejvyšší soud",
                "pocet_vysledku": str(page_size),
            },
            allow_redirects=False,
        )

        location = post_response.headers.get("location") or post_response.headers.get("Location")
        if not location:
            raise RuntimeError("Search POST did not return a redirect target.")

        search_url = urljoin(BASE_URL, location)
        response = self.request("GET", search_url)
        save_debug_snapshot(self.debug_dir, "direct_http_search.html", response.html)
        links, total_results = parse_search_results(response.html)
        return SearchPageData(
            url=response.url,
            html=response.html,
            total_results=total_results,
            links=links,
        )

    def fetch_detail(self, url: str) -> ResponseData:
        response = self.request("GET", url)
        return response


class PlaywrightProvider(BaseProvider):
    name = "playwright"
    transport_name = "playwright"

    def _render_page(
        self,
        action: str,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        page_size: int = 20,
    ) -> ResponseData:
        if sync_playwright is None:
            raise RuntimeError("Playwright is unavailable.")

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(user_agent=USER_AGENT)

            if action == "home":
                page.goto(SEARCH_HOME_URL, wait_until="networkidle", timeout=REQUEST_TIMEOUT_SECONDS * 1000)
            elif action == "search":
                if start_date is None or end_date is None:
                    raise RuntimeError("Playwright search requires both start_date and end_date.")
                page.goto(SEARCH_HOME_URL, wait_until="networkidle", timeout=REQUEST_TIMEOUT_SECONDS * 1000)
                page.fill('input[name="od"]', start_date.isoformat())
                page.fill('input[name="do"]', end_date.isoformat())
                page.select_option('select[name="soud"]', label="Nejvyšší soud")
                page_size_label = "20"
                if page_size >= 60:
                    page_size_label = "60"
                elif page_size >= 40:
                    page_size_label = "40"
                page.select_option('select[name="pocet_vysledku"]', label=page_size_label)
                with page.expect_navigation(wait_until="networkidle", timeout=REQUEST_TIMEOUT_SECONDS * 1000):
                    page.click('button[type="submit"]')
                # The NS search results table is required by parse_search_results()
                # (it selects table#tabl and anchor rows). On some runs, networkidle can
                # be reached before the table is fully rendered, so we wait explicitly.
                try:
                    page.wait_for_selector("table#tabl", timeout=REQUEST_TIMEOUT_SECONDS * 1000)
                except Exception:
                    # Preserve fail-closed behavior in run_discovery(); if the table never appears,
                    # parse_search_results will still return empty links and raise.
                    pass
            else:
                page.goto(action, wait_until="networkidle", timeout=REQUEST_TIMEOUT_SECONDS * 1000)

            response = ResponseData(url=page.url, html=page.content(), headers={})
            browser.close()
            return response

    def fetch_home(self) -> ResponseData:
        response = self._render_page("home")
        save_debug_snapshot(self.debug_dir, "playwright_home.html", response.html)
        return response

    def submit_search(self, start_date: date, end_date: date, page_size: int) -> SearchPageData:
        response = self._render_page("search", start_date=start_date, end_date=end_date, page_size=page_size)
        save_debug_snapshot(self.debug_dir, "playwright_search.html", response.html)
        links, total_results = parse_search_results(response.html)
        page_url = build_page_url(response.url, start=0, count=min(page_size, MAX_PAGE_SIZE))
        return SearchPageData(
            url=page_url,
            html=response.html,
            total_results=total_results,
            links=links[:page_size],
        )

    def fetch_detail(self, url: str) -> ResponseData:
        return self._render_page(url)


def run_discovery(
    provider: BaseProvider,
    *,
    window_start: date,
    window_end: date,
    page_size: int,
) -> DiscoveryResult:
    discovery = DiscoveryResult(
        provider_name=provider.name,
        transport_name=provider.transport_name,
    )

    home_response = provider.fetch_home()
    discovery.attempted_urls.append(home_response.url)
    discovery.form_actions = extract_form_actions(home_response.html)
    logger.info("Discovery form actions: %s", discovery.form_actions)

    search_page = provider.submit_search(window_start, window_end, page_size)
    discovery.transport_name = provider.transport_name
    discovery.search_url = search_page.url
    discovery.total_results = search_page.total_results
    discovery.attempted_urls.append(search_page.url)
    discovery.query_params = list(parse_qsl(urlparse(search_page.url).query, keep_blank_values=True))
    remote_supported, remote_field = extract_remote_date_filter_support(discovery.query_params)
    discovery.remote_date_filter_supported = remote_supported
    discovery.remote_date_filter_field = remote_field
    logger.info("Discovery search URL: %s", search_page.url)
    logger.info("Discovery query params: %s", discovery.query_params)
    logger.info(
        "Discovery remote date filter supported=%s field=%s",
        discovery.remote_date_filter_supported,
        discovery.remote_date_filter_field,
    )

    if search_page.total_results > page_size:
        pagination_url = build_page_url(search_page.url, start=page_size, count=page_size)
        discovery.pagination_urls.append(pagination_url)
        logger.info("Discovery pagination URL: %s", pagination_url)

    for link in search_page.links[: min(3, len(search_page.links))]:
        discovery.detail_urls.append(link.detail_url)

    logger.info("Discovery detail URLs: %s", discovery.detail_urls)

    # Legitimately empty month: query executed successfully, but no rows exist.
    if not discovery.detail_urls and (search_page.total_results or 0) == 0:
        logger.info("Discovery found zero site results for requested month/window.")
        return discovery

    if not discovery.detail_urls:
        raise RuntimeError("Discovery did not yield any detail URLs.")

    detail_response = provider.fetch_detail(discovery.detail_urls[0])
    discovery.attempted_urls.append(detail_response.url)
    save_debug_snapshot(
        provider.debug_dir,
        f"{provider.name}_detail_sample.html",
        detail_response.html,
    )
    _ = parse_decision_detail(detail_response.html, detail_response.url)
    return discovery


def discover_provider(config: ScrapeConfig) -> tuple[BaseProvider, DiscoveryResult]:
    page_size = min(max(config.limit, 20), MAX_PAGE_SIZE)
    window_end = config.date_to or datetime.now().date()
    window_start = config.date_from or (window_end - timedelta(days=INITIAL_WINDOW_DAYS - 1))
    providers: list[BaseProvider] = [
        DirectHttpProvider(debug_dir=config.debug_dir),
        PlaywrightProvider(debug_dir=config.debug_dir),
    ]
    failures: list[str] = []

    for provider in providers:
        try:
            logger.info("Running discovery with provider=%s", provider.name)
            discovery = run_discovery(
                provider,
                window_start=window_start,
                window_end=window_end,
                page_size=page_size,
            )
            logger.info(
                "Discovery succeeded with provider=%s transport=%s",
                discovery.provider_name,
                discovery.transport_name,
            )
            return provider, discovery
        except Exception as exc:
            failures.append(f"{provider.name}: {exc}")
            logger.warning("Discovery failed for provider=%s: %s", provider.name, exc)

    raise RuntimeError("All scraper providers failed discovery: " + " | ".join(failures))


def extract_remote_date_filter_support(query_params: list[tuple[str, str]]) -> tuple[bool, str | None]:
    query_value = ""
    for key, value in query_params:
        if key == "Query":
            query_value = value.lower()
            break

    if "[datum_predani_na_web]" in query_value:
        return True, "publication_date"
    if "[datum_rozhodnuti]" in query_value:
        return True, "decision_date"
    return False, None


def infer_legal_area(case_number: str | None, metadata: dict[str, str]) -> str | None:
    dotcene_predpisy = metadata.get("Dotčené předpisy", "").lower()
    case_number_upper = (case_number or "").upper()

    if "tr. ř." in dotcene_predpisy or "tr. zákoníku" in dotcene_predpisy or "Důvod dovolání" in metadata:
        return "criminal"

    civil_markers = ("o. s. ř.", "obč.", "insolven", "iz", "obch.")
    if any(marker in dotcene_predpisy for marker in civil_markers):
        return "civil"

    if re.search(r"\bT[A-Z]{0,4}\b", case_number_upper):
        return "criminal"

    if re.search(r"\b(CDO|ICDO|ODO|ND|NSCR|O)\b", case_number_upper):
        return "civil"

    return None


def extract_metadata_map(detail_soup: BeautifulSoup) -> tuple[dict[str, str], str | None]:
    metadata_table = detail_soup.select_one("div.main_detail table#tabl")
    if metadata_table is None:
        raise ValueError("Decision detail metadata table not found.")

    metadata: dict[str, str] = {}
    first_case_number: str | None = None

    for row in metadata_table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        raw_label = normalize_text(cells[0].get_text(" ", strip=True))
        raw_value = normalize_text(cells[1].get_text(" ", strip=True))
        label = raw_label.rstrip(":").strip()

        if label in CASE_NUMBER_LABELS and raw_value:
            metadata[label] = raw_value
            continue

        if ":" in raw_label:
            if raw_value:
                metadata[label] = raw_value
            continue

        if first_case_number is None and raw_label:
            first_case_number = raw_label

    return metadata, first_case_number


def extract_full_text(detail_soup: BeautifulSoup) -> str:
    main_detail = detail_soup.select_one("div.main_detail")
    if main_detail is None:
        raise ValueError("Decision detail container not found.")

    candidates: list[str] = []
    for element in main_detail.find_all("div"):
        style = (element.get("style") or "").lower()
        if "text-align:justify" not in style:
            continue

        text = normalize_text(element.get_text(" ", strip=True))
        if len(text) >= 250:
            candidates.append(text)

    if candidates:
        return max(candidates, key=len)

    metadata_table = main_detail.select_one("table#tabl")
    if metadata_table is None:
        return ""

    fallback_parts: list[str] = []
    for sibling in metadata_table.next_siblings:
        if not hasattr(sibling, "get_text"):
            continue

        text = normalize_text(sibling.get_text(" ", strip=True))
        if not text or text.startswith(STANDARD_CITATION_PREFIX):
            continue
        fallback_parts.append(text)

    return normalize_text("\n\n".join(fallback_parts))


def parse_decision_detail(html: str, url: str) -> dict[str, Any] | None:
    soup = BeautifulSoup(html, "lxml")
    metadata, first_case_number = extract_metadata_map(soup)

    case_number = next((metadata.get(label) for label in CASE_NUMBER_LABELS if metadata.get(label)), None)
    if not case_number:
        case_number = first_case_number

    document_type = metadata.get("Typ rozhodnutí")
    full_text = extract_full_text(soup)
    if not full_text:
        logger.warning("Decision detail has empty full_text: %s", url)
        return None

    publication_date = parse_czech_date(metadata.get("Zveřejněno na webu"))
    decision_date = parse_czech_date(metadata.get("Datum rozhodnutí"))
    ecli = metadata.get("ECLI")
    legal_area = infer_legal_area(case_number, metadata)
    title_parts = [part for part in (case_number, document_type.title() if document_type else None) if part]
    title = " ".join(title_parts) if title_parts else case_number

    record = {
        "source": "nsoud",
        "court": "Nejvyšší soud",
        "authority_level": "supreme",
        "case_number": case_number,
        "ecli": ecli,
        "decision_date": decision_date,
        "publication_date": publication_date,
        "document_type": document_type,
        "legal_area": legal_area,
        "title": title,
        "url": url,
        "full_text": full_text,
        "source_attribution": SOURCE_ATTRIBUTION,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }
    # content_hash kept for change detection; canonical_id is document identity.
    try:
        from app.court_staging.identity import enrich_record_identity

        record = enrich_record_identity(record, source="nsoud")
    except Exception:  # pragma: no cover - keep scraper usable if staging pkg missing
        record["content_hash"] = compute_content_hash(record["full_text"], record["url"])
    return record


def record_matches_requested_dates(
    record: dict[str, Any],
    *,
    date_from: date | None,
    date_to: date | None,
    remote_date_filter_supported: bool,
) -> bool:
    if date_from is None or date_to is None:
        return True

    for field_name in ("publication_date", "decision_date"):
        raw_value = normalize_text(record.get(field_name))
        if not raw_value:
            continue
        try:
            record_date = date.fromisoformat(raw_value)
        except ValueError:
            continue
        return date_from <= record_date <= date_to

    return remote_date_filter_supported


def collect_candidate_links(
    provider: BaseProvider,
    config: ScrapeConfig,
    stats: ScrapeStats,
) -> list[SearchResultLink]:
    page_size = min(max(config.limit if not config.exhaust else MAX_PAGE_SIZE, 20), MAX_PAGE_SIZE)
    if config.exhaust:
        candidate_target = page_size * max(config.max_pages, 1)
    else:
        candidate_target = max(config.limit * 3, page_size * max(config.max_pages, 1))
    seen_urls: set[str] = set()
    candidates: list[SearchResultLink] = []

    if config.date_from is not None and config.date_to is not None:
        search_windows = [(config.date_from, config.date_to)]
    else:
        end_date = datetime.now().date()
        window_days = INITIAL_WINDOW_DAYS
        search_windows = []
        for _ in range(MAX_SEARCH_EXPANSIONS):
            start_date = end_date - timedelta(days=window_days - 1)
            search_windows.append((start_date, end_date))
            window_days = min(window_days * 2, MAX_WINDOW_DAYS)

    for attempt, (start_date, end_date) in enumerate(search_windows, start=1):
        logger.info(
            "Scrape discovery window %s/%s using publication dates %s..%s",
            attempt,
            len(search_windows),
            start_date.isoformat(),
            end_date.isoformat(),
        )

        search_page = provider.submit_search(start_date, end_date, page_size)
        stats.attempted_urls.append(search_page.url)
        stats.pages_visited += 1
        logger.info(
            "Window search URL=%s total=%s rows=%s pages_visited=%s/%s",
            search_page.url,
            search_page.total_results,
            len(search_page.links),
            stats.pages_visited,
            config.max_pages,
        )

        for link in search_page.links:
            if link.detail_url in seen_urls:
                continue
            seen_urls.add(link.detail_url)
            candidates.append(link)
            logger.info("Discovered detail URL: %s", link.detail_url)
        stats.records_discovered = len(candidates)
        stats.site_total_results = search_page.total_results or stats.site_total_results

        next_start = page_size
        while (
            next_start < max(search_page.total_results, 1)
            and stats.pages_visited < config.max_pages
            and (
                config.exhaust
                or len(candidates) < min(search_page.total_results, candidate_target)
            )
        ):
            if not config.exhaust and len(candidates) >= config.limit:
                break
            polite_delay(config.delay_seconds)
            page_url = build_page_url(search_page.url, start=next_start, count=page_size)
            stats.attempted_urls.append(page_url)
            logger.info("Following pagination URL: %s", page_url)
            page = provider.fetch_results_page(page_url)
            stats.pages_visited += 1
            if page.total_results:
                stats.site_total_results = page.total_results
            if not page.links:
                logger.warning("Pagination returned no rows for %s", page_url)
                break

            for link in page.links:
                if link.detail_url in seen_urls:
                    continue
                seen_urls.add(link.detail_url)
                candidates.append(link)
                logger.info("Discovered detail URL: %s", link.detail_url)
            stats.records_discovered = len(candidates)

            next_start += page_size

        if stats.pages_visited >= config.max_pages:
            logger.info("Reached hard max-pages=%s, stopping pagination.", config.max_pages)
            return candidates

        if not config.exhaust and len(candidates) >= config.limit:
            return candidates

        if config.date_from is None or config.date_to is None:
            polite_delay(config.delay_seconds)

    return candidates


def scrape_sample(config: ScrapeConfig) -> ScrapeStats:
    if config.limit <= 0:
        raise ValueError("--limit must be a positive integer.")
    if config.max_pages <= 0:
        raise ValueError("--max-pages must be a positive integer.")
    if config.delay_seconds < 0:
        raise ValueError("--delay must be non-negative.")
    if (config.date_from is None) != (config.date_to is None):
        raise ValueError("--date-from and --date-to must be provided together.")
    if config.date_from is not None and config.date_to is not None and config.date_from > config.date_to:
        raise ValueError("--date-from must be earlier than or equal to --date-to.")

    logger.info(
        "Scrape configuration: date_from=%s date_to=%s limit=%s max_pages=%s exhaust=%s delay=%.3f out=%s",
        config.date_from.isoformat() if config.date_from else None,
        config.date_to.isoformat() if config.date_to else None,
        config.limit,
        config.max_pages,
        config.exhaust,
        config.delay_seconds,
        config.out_path,
    )

    provider, discovery = discover_provider(config)
    stats = ScrapeStats(
        provider_name=discovery.provider_name,
        transport_name=discovery.transport_name,
        attempted_urls=list(discovery.attempted_urls),
    )

    from app.court_staging.identity import ChangeKind
    from app.court_staging.jsonl_store import load_canonical_index, rewrite_jsonl_upsert

    known = load_canonical_index([config.out_path])
    logger.info("Loaded %s existing canonical ids from %s", len(known), config.out_path)

    if config.date_from is not None and config.date_to is not None:
        if discovery.remote_date_filter_supported:
            logger.info(
                "Remote date filtering is available via %s for requested range %s..%s.",
                discovery.remote_date_filter_field,
                config.date_from.isoformat(),
                config.date_to.isoformat(),
            )
        else:
            logger.warning(
                "Direct date filtering is unavailable; scraping safely bounded pages and filtering records locally."
            )

    candidate_links = collect_candidate_links(provider, config, stats)
    stats.unique_candidates = len(candidate_links)
    logger.info(
        "Collected %s candidate detail URLs across %s visited pages (site_total=%s).",
        len(candidate_links),
        stats.pages_visited,
        stats.site_total_results,
    )

    for index, candidate in enumerate(candidate_links, start=1):
        if not config.exhaust and (stats.records_written + stats.records_updated) >= config.limit:
            break

        polite_delay(config.delay_seconds)
        logger.info("Fetching detail %s/%s: %s", index, len(candidate_links), candidate.detail_url)
        stats.attempted_urls.append(candidate.detail_url)

        try:
            response = provider.fetch_detail(candidate.detail_url)
            record = parse_decision_detail(response.html, response.url)
        except Exception as exc:  # pragma: no cover - defensive scraper path
            stats.parse_failures += 1
            stats.failure_reasons["detail_fetch_or_parse"] = (
                stats.failure_reasons.get("detail_fetch_or_parse", 0) + 1
            )
            logger.warning("Failed to parse decision detail %s: %s", candidate.detail_url, exc)
            continue

        if record is None:
            stats.parse_failures += 1
            stats.failure_reasons["empty_or_invalid_detail"] = (
                stats.failure_reasons.get("empty_or_invalid_detail", 0) + 1
            )
            continue

        if not record_matches_requested_dates(
            record,
            date_from=config.date_from,
            date_to=config.date_to,
            remote_date_filter_supported=discovery.remote_date_filter_supported,
        ):
            stats.locally_filtered_out += 1
            logger.info(
                "Skipping out-of-range record after local date check: %s publication_date=%s decision_date=%s",
                record.get("case_number"),
                record.get("publication_date"),
                record.get("decision_date"),
            )
            continue

        if config.debug_dir is not None and stats.records_written == 0 and stats.records_updated == 0:
            save_debug_snapshot(config.debug_dir, "first_written_detail.html", response.html)

        change = rewrite_jsonl_upsert(config.out_path, record, known=known, source="nsoud")
        if change is ChangeKind.UNCHANGED:
            stats.duplicates_skipped += 1
            logger.info("Skipping unchanged record %s", record.get("canonical_id") or candidate.detail_url)
            continue
        if change is ChangeKind.UPDATED:
            stats.records_updated += 1
        else:
            stats.records_written += 1
        stats.all_written_have_full_text = stats.all_written_have_full_text and bool(record.get("full_text"))

        if not stats.first_record_keys:
            stats.first_record_keys = list(record.keys())

        logger.info(
            "Upserted record kind=%s written=%s updated=%s id=%s",
            change.value,
            stats.records_written,
            stats.records_updated,
            record.get("canonical_id") or record.get("case_number") or response.url,
        )

    logger.info(
        "Scrape finished: pages_visited=%s records_discovered=%s unique_candidates=%s "
        "records_written=%s records_updated=%s duplicates_skipped=%s local_date_filtered=%s failed=%s",
        stats.pages_visited,
        stats.records_discovered,
        stats.unique_candidates,
        stats.records_written,
        stats.records_updated,
        stats.duplicates_skipped,
        stats.locally_filtered_out,
        stats.parse_failures,
    )
    return stats


def main() -> int:
    configure_logging()
    args = parse_args()
    debug_dir = args.out.parent / "_debug_html" if args.debug_save_html else None
    config = ScrapeConfig(
        limit=args.limit,
        date_from=args.date_from,
        date_to=args.date_to,
        delay_seconds=args.delay,
        max_pages=args.max_pages,
        out_path=args.out,
        debug_dir=debug_dir,
        exhaust=bool(getattr(args, "exhaust", False)),
    )

    try:
        stats = scrape_sample(config)
    except Exception as exc:
        logger.exception("Sample scrape failed: %s", exc)
        print("number of records written: 0")
        print("number of duplicates skipped: 0")
        print("pages visited: 0")
        print(f"output path: {args.out}")
        print("first record keys: []")
        print("whether full_text is non-empty for all written records: False")
        print("failed detail fetches: 0")
        return 1

    print(f"number of records written: {stats.records_written}")
    print(f"number of duplicates skipped: {stats.duplicates_skipped}")
    print(f"pages visited: {stats.pages_visited}")
    print(f"output path: {args.out}")
    print(f"first record keys: {stats.first_record_keys}")
    print(
        "whether full_text is non-empty for all written records: "
        f"{bool(stats.records_written and stats.all_written_have_full_text)}"
    )
    print(f"failed detail fetches: {stats.parse_failures}")

    if stats.records_written == 0:
        print("attempted urls:")
        for attempted_url in stats.attempted_urls:
            print(attempted_url)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
