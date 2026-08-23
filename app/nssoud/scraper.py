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
from app.nssoud.form import (
    ECLI_CONDITION_TECH,
    FULLTEXT_CONDITION_TECH,
    apply_decision_date_window,
    apply_named_text,
    date_filter_was_applied,
    detail_field,
    encode_form_body,
    extract_result_links,
    is_search_or_nav_url,
    parse_infinite_scroll_state,
    serialize_findform,
    summarize_findform,
)

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

BASE_URL = "https://vyhledavac.nssoud.cz"
SEARCH_URL = f"{BASE_URL}/Home/Index?formular=4"
MORE_ROWS_URL = f"{BASE_URL}/Home/MyResTRowsCont"
SOURCE_ATTRIBUTION = "Nejvyšší správní soud České republiky, vyhledavac.nssoud.cz"
USER_AGENT = "nalus-scraper/nssoud (+https://vyhledavac.nssoud.cz/)"
DEFAULT_DELAY = 1.5

logger = logging.getLogger("nssoud_scraper")

_ECLI_FIND = re.compile(r"ECLI:CZ:NSS:[^\s<>\"']+", re.I)
_DATE_FIND = re.compile(r"(\d{1,2})\.\s*(\d{1,2})\.\s*(\d{4})")


@dataclass
class ScrapeStats:
    records_written: int = 0
    records_updated: int = 0
    duplicates_skipped: int = 0
    parse_failures: int = 0
    skipped_unavailable: int = 0
    pages_visited: int = 0
    records_discovered: int = 0
    unique_candidates: int = 0
    site_total_results: int | None = None
    remote_date_filter_applied: bool = False
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


_RETRYABLE_STATUS = {500, 502, 503, 504}
_GET_MAX_ATTEMPTS = 5


def _client() -> Any:
    if httpx is None:
        raise RuntimeError("httpx is required for nssoud scraper")
    return httpx.Client(
        timeout=90.0,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )


def _get_with_retries(client: Any, url: str) -> Any:
    """GET with bounded backoff for transient NSS origin 5xx."""
    last_exc: Exception | None = None
    for attempt in range(1, _GET_MAX_ATTEMPTS + 1):
        try:
            response = client.get(url)
            if response.status_code in _RETRYABLE_STATUS and attempt < _GET_MAX_ATTEMPTS:
                wait = min(60.0, 2.0 ** attempt)
                logger.warning(
                    "NSS GET %s status=%s attempt=%s/%s; sleeping %.1fs",
                    url,
                    response.status_code,
                    attempt,
                    _GET_MAX_ATTEMPTS,
                    wait,
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response
        except Exception as exc:
            last_exc = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = status in _RETRYABLE_STATUS or status is None
            if attempt >= _GET_MAX_ATTEMPTS or not retryable:
                raise
            wait = min(60.0, 2.0 ** attempt)
            logger.warning(
                "NSS GET %s error attempt=%s/%s: %s; sleeping %.1fs",
                url,
                attempt,
                _GET_MAX_ATTEMPTS,
                exc,
                wait,
            )
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc


def _post_form(client: Any, url: str, pairs: list[tuple[str, str]], *, referer: str) -> Any:
    response = client.post(
        url,
        content=encode_form_body(pairs),
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": BASE_URL,
            "Referer": referer,
        },
    )
    response.raise_for_status()
    return response


def parse_decision_detail(
    html: str,
    url: str,
    *,
    full_text_html: str | None = None,
) -> dict[str, Any] | None:
    soup = BeautifulSoup(html, "lxml")
    case_number = normalize_space(detail_field(html, "oznacenivecivcelku"))
    ecli_raw = normalize_space(detail_field(html, "ecli"))
    decision_date = parse_czech_date(detail_field(html, "datumvydanirozhodnuti"))
    ecli_match = _ECLI_FIND.search(ecli_raw or "") or _ECLI_FIND.search(html)
    ecli = ecli_match.group(0).upper() if ecli_match else None

    full_text = ""
    if full_text_html:
        body = BeautifulSoup(full_text_html, "lxml")
        full_text = normalize_space(body.get_text("\n", strip=True))
    if len(full_text) < 80:
        text = soup.get_text("\n", strip=True)
        full_text = normalize_space(text)

    if len(full_text) < 80:
        logger.warning("NSS detail too short: %s", url)
        return None
    if is_search_or_nav_url(url):
        logger.warning("NSS refused search/nav URL as decision: %s", url)
        return None

    if not case_number:
        case_match = re.search(
            r"\b(\d+\s+[A-Za-zÁ-ž]{1,8}\s+\d+/\d{4}(?:\s*[-–]\s*\d+)?)\b",
            full_text,
        )
        if case_match:
            case_number = normalize_space(case_match.group(1))
    if not decision_date:
        decision_date = parse_czech_date(full_text[:2000])

    record = {
        "source": "nssoud",
        "court": "Nejvyšší správní soud",
        "authority_level": "supreme_administrative",
        "case_number": case_number or None,
        "spisova_znacka": case_number or None,
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


def _search_pages(client: Any, config: ScrapeConfig, stats: ScrapeStats) -> list[dict[str, str]]:
    polite_delay(config.delay_seconds)
    form_response = client.get(SEARCH_URL)
    form_response.raise_for_status()
    summary = summarize_findform(form_response.text)
    if not summary.get("present"):
        raise RuntimeError("NSS findform missing on GET")
    pairs = serialize_findform(form_response.text)
    if config.date_from or config.date_to:
        if not apply_decision_date_window(pairs, date_from=config.date_from, date_to=config.date_to):
            stats.notes.append("remote_date_fields_missing")
    query = (config.query or "").strip()
    if query and query != "*":
        if query.upper().startswith("ECLI:"):
            apply_named_text(pairs, tech=ECLI_CONDITION_TECH, text=query)
        else:
            apply_named_text(pairs, tech=FULLTEXT_CONDITION_TECH, text=query)

    polite_delay(config.delay_seconds)
    search_response = _post_form(client, SEARCH_URL, pairs, referer=SEARCH_URL)
    stats.pages_visited += 1
    html = search_response.text
    state = parse_infinite_scroll_state(html)
    stats.remote_date_filter_applied = date_filter_was_applied(state)
    if config.date_from or config.date_to:
        if stats.remote_date_filter_applied:
            stats.notes.append("remote_date_filter_applied:datumvydanirozhodnuti")
        else:
            stats.notes.append("remote_date_filter_not_confirmed_in_currParams")

    candidates: list[dict[str, str]] = []
    seen: set[str] = set()

    def _absorb(page_html: str) -> int:
        page_links, total = extract_result_links(page_html, BASE_URL)
        if total is not None:
            stats.site_total_results = total
        added = 0
        for link in page_links:
            url = link["url"]
            if url in seen or is_search_or_nav_url(url):
                continue
            seen.add(url)
            candidates.append(link)
            added += 1
        return added

    added = _absorb(html)
    if added == 0:
        stats.notes.append("no_links_on_page_1")
        return candidates

    more_path = state.get("more_rows_url") or "/Home/MyResTRowsCont"
    more_url = urljoin(BASE_URL, str(more_path))
    page_num = 1
    while stats.pages_visited < config.max_pages:
        if not config.exhaust and len(candidates) >= config.limit:
            break
        if not state.get("vyhledavaci_podminky"):
            stats.notes.append("pagination_state_missing")
            break
        polite_delay(config.delay_seconds)
        more = client.post(
            more_url,
            content=encode_form_body(
                [
                    ("vyhledavaciPodminky", str(state["vyhledavaci_podminky"])),
                    ("zobrazeniVysledkuId", str(state.get("zobrazeni_vysledku_id") or "1")),
                    ("pageNum", str(page_num)),
                    ("resultOrder", str(state.get("result_order") or "")),
                ]
            ),
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Origin": BASE_URL,
                "Referer": SEARCH_URL,
            },
        )
        more.raise_for_status()
        stats.pages_visited += 1
        fragment = more.text or ""
        if len(fragment.strip()) <= 5:
            stats.notes.append(f"pagination_end_page_{page_num}")
            break
        added = _absorb(fragment)
        if added == 0:
            stats.notes.append(f"no_links_on_page_{page_num + 1}")
            break
        page_num += 1
    return candidates


def scrape(config: ScrapeConfig) -> ScrapeStats:
    assert_safe_staging_path(config.out_path)
    if config.limit <= 0:
        raise ValueError("limit must be positive")
    if config.max_pages <= 0:
        raise ValueError("max_pages must be positive")

    stats = ScrapeStats()
    known = load_canonical_index([config.out_path])

    with _client() as client:
        try:
            candidates = _search_pages(client, config, stats)
        except Exception as exc:
            stats.failure_reasons["search_page"] = stats.failure_reasons.get("search_page", 0) + 1
            stats.notes.append(f"search_page_failed:{exc}")
            logger.exception("NSS search failed")
            return stats

        stats.records_discovered = len(candidates)
        stats.unique_candidates = len(candidates)

        for index, candidate in enumerate(candidates, start=1):
            if not config.exhaust and (stats.records_written + stats.records_updated) >= config.limit:
                break
            url = candidate["url"]
            html_url = candidate.get("html_url") or ""
            polite_delay(config.delay_seconds)
            try:
                detail_response = _get_with_retries(client, url)
                full_text_html = None
                if html_url:
                    polite_delay(config.delay_seconds)
                    html_response = _get_with_retries(client, html_url)
                    full_text_html = html_response.text
                record = parse_decision_detail(
                    detail_response.text,
                    str(detail_response.url),
                    full_text_html=full_text_html,
                )
            except Exception as exc:
                stats.parse_failures += 1
                stats.failure_reasons["detail_fetch_or_parse"] = (
                    stats.failure_reasons.get("detail_fetch_or_parse", 0) + 1
                )
                logger.warning("NSS detail failed %s: %s", url, exc)
                continue
            if record is None:
                stats.skipped_unavailable += 1
                stats.failure_reasons["empty_or_invalid_detail"] = (
                    stats.failure_reasons.get("empty_or_invalid_detail", 0) + 1
                )
                continue

            if not record.get("decision_date") and candidate.get("decision_date_raw"):
                record["decision_date"] = parse_czech_date(candidate.get("decision_date_raw"))
            if not record.get("case_number") and candidate.get("case_number"):
                record["case_number"] = candidate["case_number"]
                record["spisova_znacka"] = candidate["case_number"]
                record = enrich_record_identity(record, source="nssoud")

            if config.date_from and config.date_to and record.get("decision_date"):
                try:
                    decision_day = date.fromisoformat(str(record["decision_date"]))
                    if decision_day < config.date_from or decision_day > config.date_to:
                        stats.notes.append(f"local_date_filter_skip:{record.get('canonical_id')}")
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
                "NSS upsert kind=%s id=%s date=%s (%s/%s)",
                change.value,
                record.get("canonical_id"),
                record.get("decision_date"),
                index,
                len(candidates),
            )

    if stats.records_written + stats.records_updated == 0:
        stats.notes.append(
            "No records written. NSS search uses POST form#findform; check CSRF/session and date fields."
        )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pilot/sample scrape of NSS decisions into court_staging.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--date-from", type=parse_cli_date)
    parser.add_argument("--date-to", type=parse_cli_date)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--exhaust", action="store_true")
    parser.add_argument("--query", default="*")
    return parser.parse_args()


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
    print(f"remote_date_filter_applied: {stats.remote_date_filter_applied}")
    print(f"parse_failures: {stats.parse_failures}")
    print(f"out: {out}")
    if stats.notes:
        print(f"notes: {stats.notes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
