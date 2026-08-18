"""Nejvyšší správní soud scraper targeting vyhledavac.nssoud.cz (staging only)."""

from __future__ import annotations

import argparse
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from app.court_staging.identity import ChangeKind, enrich_record_identity
from app.court_staging.jsonl_store import load_canonical_index, rewrite_jsonl_upsert
from app.court_staging.paths import assert_safe_staging_path

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

BASE_URL = "https://vyhledavac.nssoud.cz"
SEARCH_URL = f"{BASE_URL}/Home/Index?formular=4"
SOURCE_ATTRIBUTION = "Nejvyšší správní soud České republiky, vyhledavac.nssoud.cz"
USER_AGENT = "nalus-scraper/nssoud (+https://vyhledavac.nssoud.cz/)"
DEFAULT_DELAY = 1.0

logger = logging.getLogger("nssoud_scraper")

_ECLI_FIND = re.compile(r"ECLI:CZ:NSS:[^\s<>\"']+", re.I)
_DATE_FIND = re.compile(r"(\d{1,2})\.\s*(\d{1,2})\.\s*(\d{4})")


@dataclass
class ScrapeStats:
    records_written: int = 0
    records_updated: int = 0
    duplicates_skipped: int = 0
    parse_failures: int = 0
    pages_visited: int = 0
    records_discovered: int = 0
    unique_candidates: int = 0
    site_total_results: int | None = None
    failure_reasons: dict[str, int] = field(default_factory=dict)
    first_record_keys: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScrapeConfig:
    limit: int
    date_from: date | None
    date_to: date | None
    delay_seconds: float
    max_pages: int
    out_path: Path
    exhaust: bool = False
    query: str = "*"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_cli_date(value: str) -> date:
    return date.fromisoformat(value)


def polite_delay(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def normalize_space(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def parse_czech_date(raw: str | None) -> str | None:
    text = normalize_space(raw)
    match = _DATE_FIND.search(text or "")
    if not match:
        return None
    day, month, year = (int(p) for p in match.groups())
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _client() -> Any:
    if httpx is None:
        raise RuntimeError("httpx is required for nssoud scraper")
    return httpx.Client(
        timeout=60.0,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )


def extract_result_links(html: str, base_url: str = BASE_URL) -> tuple[list[dict[str, str]], int | None]:
    """Best-effort extraction of detail links from search HTML."""
    soup = BeautifulSoup(html, "lxml")
    links: list[dict[str, str]] = []
    seen: set[str] = set()

    total: int | None = None
    body_text = soup.get_text(" ", strip=True)
    total_match = re.search(r"(\d+)\s+(?:výsledk|dokument)", body_text, flags=re.I)
    if total_match:
        total = int(total_match.group(1))

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        text = normalize_space(anchor.get_text(" ", strip=True))
        href_l = href.lower()
        if not any(token in href_l for token in ("document", "detail", "dokument", "soubor", "id=")):
            # Keep anchors that look like case numbers (e.g. 1 As 12/2020)
            if not re.search(r"\b\d+\s+[A-Za-z]{1,6}\s+\d+/\d{4}\b", text):
                continue
        url = urljoin(base_url, href)
        if url in seen:
            continue
        seen.add(url)
        links.append({"url": url, "label": text})

    return links, total


def parse_decision_detail(html: str, url: str) -> dict[str, Any] | None:
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text("\n", strip=True)
    full_text = normalize_space(text)
    # Prefer larger main/content blocks when present.
    for selector in ("main", "article", "#content", ".document", ".dokument"):
        node = soup.select_one(selector)
        if node:
            candidate = normalize_space(node.get_text("\n", strip=True))
            if len(candidate) > 200:
                full_text = candidate
                break

    if len(full_text) < 80:
        logger.warning("NSS detail too short: %s", url)
        return None

    ecli_match = _ECLI_FIND.search(html) or _ECLI_FIND.search(full_text)
    ecli = ecli_match.group(0).upper() if ecli_match else None

    case_number = None
    case_match = re.search(
        r"\b(\d+\s+[A-Za-zÁ-ž]{1,8}\s+\d+/\d{4}(?:\s*[-–]\s*\d+)?)\b",
        full_text,
    )
    if case_match:
        case_number = normalize_space(case_match.group(1))

    decision_date = parse_czech_date(full_text[:2000])

    record = {
        "source": "nssoud",
        "court": "Nejvyšší správní soud",
        "authority_level": "supreme_administrative",
        "case_number": case_number,
        "spisova_znacka": case_number,
        "ecli": ecli,
        "decision_date": decision_date,
        "publication_date": None,
        "url": url,
        "source_url": url,
        "full_text": full_text,
        "source_attribution": SOURCE_ATTRIBUTION,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }
    return enrich_record_identity(record, source="nssoud")


def _search_get(client: Any, *, query: str, page: int) -> str:
    # Public UI is ASP.NET-like; GET with query params is a best-effort probe path.
    # Probe report may refine endpoints; scraper remains resilient to empty pages.
    params = {
        "formular": "4",
        "q": query,
        "page": str(page),
    }
    response = client.get(f"{BASE_URL}/Home/Index", params=params)
    response.raise_for_status()
    return response.text


def scrape(config: ScrapeConfig) -> ScrapeStats:
    assert_safe_staging_path(config.out_path)
    if config.limit <= 0:
        raise ValueError("limit must be positive")
    if config.max_pages <= 0:
        raise ValueError("max_pages must be positive")

    stats = ScrapeStats()
    known = load_canonical_index([config.out_path])
    candidates: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    with _client() as client:
        for page in range(1, config.max_pages + 1):
            polite_delay(config.delay_seconds)
            try:
                html = _search_get(client, query=config.query, page=page)
            except Exception as exc:
                stats.failure_reasons["search_page"] = stats.failure_reasons.get("search_page", 0) + 1
                stats.notes.append(f"search_page_failed:{page}:{exc}")
                break
            stats.pages_visited += 1
            page_links, total = extract_result_links(html)
            if total is not None:
                stats.site_total_results = total
            if not page_links:
                stats.notes.append(f"no_links_on_page_{page}")
                # Still try homepage once for pilot discovery.
                if page == 1:
                    home = client.get(SEARCH_URL)
                    home.raise_for_status()
                    page_links, total = extract_result_links(home.text)
                    if total is not None:
                        stats.site_total_results = total
                if not page_links:
                    break

            new_on_page = 0
            for link in page_links:
                url = link["url"]
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                candidates.append(link)
                new_on_page += 1
            stats.records_discovered = len(candidates)
            if new_on_page == 0:
                break
            if not config.exhaust and len(candidates) >= config.limit * 3:
                break

        stats.unique_candidates = len(candidates)
        for index, candidate in enumerate(candidates, start=1):
            if not config.exhaust and (stats.records_written + stats.records_updated) >= config.limit:
                break
            polite_delay(config.delay_seconds)
            url = candidate["url"]
            try:
                response = client.get(url)
                response.raise_for_status()
                record = parse_decision_detail(response.text, str(response.url))
            except Exception as exc:
                stats.parse_failures += 1
                stats.failure_reasons["detail_fetch_or_parse"] = (
                    stats.failure_reasons.get("detail_fetch_or_parse", 0) + 1
                )
                logger.warning("NSS detail failed %s: %s", url, exc)
                continue
            if record is None:
                stats.parse_failures += 1
                stats.failure_reasons["empty_or_invalid_detail"] = (
                    stats.failure_reasons.get("empty_or_invalid_detail", 0) + 1
                )
                continue

            # Optional local date filter when dates available.
            if config.date_from and config.date_to and record.get("decision_date"):
                try:
                    d = date.fromisoformat(str(record["decision_date"]))
                    if d < config.date_from or d > config.date_to:
                        continue
                except ValueError:
                    pass

            change = rewrite_jsonl_upsert(config.out_path, record, known=known, source="nssoud")
            if change is ChangeKind.UNCHANGED:
                stats.duplicates_skipped += 1
                continue
            if change is ChangeKind.UPDATED:
                stats.records_updated += 1
            else:
                stats.records_written += 1
            if not stats.first_record_keys:
                stats.first_record_keys = list(record.keys())
            logger.info(
                "NSS upsert kind=%s id=%s (%s/%s)",
                change.value,
                record.get("canonical_id"),
                index,
                len(candidates),
            )

    if stats.records_written + stats.records_updated == 0:
        stats.notes.append(
            "No records written. Run probe_source and refine search/detail selectors for DXCFTS UI."
        )
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pilot/sample scrape of NSS decisions into court_staging.")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--date-from", type=parse_cli_date)
    p.add_argument("--date-to", type=parse_cli_date)
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    p.add_argument("--max-pages", type=int, default=5)
    p.add_argument("--exhaust", action="store_true")
    p.add_argument("--query", default="*")
    return p.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    out = assert_safe_staging_path(args.out)
    config = ScrapeConfig(
        limit=args.limit,
        date_from=args.date_from,
        date_to=args.date_to,
        delay_seconds=args.delay,
        max_pages=args.max_pages,
        out_path=out,
        exhaust=bool(args.exhaust),
        query=args.query,
    )
    try:
        stats = scrape(config)
    except Exception as exc:
        logger.exception("NSS scrape failed: %s", exc)
        print(f"error: {exc}")
        return 1
    print(f"records_written: {stats.records_written}")
    print(f"records_updated: {stats.records_updated}")
    print(f"duplicates_skipped: {stats.duplicates_skipped}")
    print(f"pages_visited: {stats.pages_visited}")
    print(f"unique_candidates: {stats.unique_candidates}")
    print(f"site_total_results: {stats.site_total_results}")
    print(f"parse_failures: {stats.parse_failures}")
    print(f"out: {out}")
    if stats.notes:
        print(f"notes: {stats.notes}")
    return 0 if (stats.records_written + stats.records_updated) > 0 or stats.unique_candidates == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
