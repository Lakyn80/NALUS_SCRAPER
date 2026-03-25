import time

from app.crawler.extractor import extract_search_page
from app.crawler.playwright_crawler import fetch_page_html
from app.models.search_result import NalusResult
from app.services.decision_service import enrich_results_with_text


def collect_results(
    query: str,
    page_start: int = 1,
    page_end: int | None = 1,
    fetch_full_text: bool = True,
    max_results: int | None = None,
    batch_pages: int = 10,
    batch_sleep_seconds: float = 2.0,
) -> list[NalusResult]:
    if page_start < 1:
        raise ValueError("page_start must be greater than or equal to 1.")
    if page_end is not None and page_end < page_start:
        raise ValueError("page_end must be greater than or equal to page_start.")
    if max_results is not None and max_results < 1:
        raise ValueError("max_results must be greater than or equal to 1.")
    if batch_pages < 1:
        raise ValueError("batch_pages must be greater than or equal to 1.")
    if batch_sleep_seconds < 0:
        raise ValueError("batch_sleep_seconds must be greater than or equal to 0.")

    collected_results: list[NalusResult] = []
    seen_ids: set[str] = set()
    current_page = page_start
    batch_start = page_start
    pages_in_batch = 0
    validated_count = 0
    validated_batches = 0

    while True:
        if page_end is not None and current_page > page_end:
            break

        try:
            html = fetch_page_html(query=query, page=current_page - 1)
            search_page = extract_search_page(html, query=query)
        except Exception as exc:
            print(f"[SCRAPER] page_failed page={current_page} error={exc}")
            raise RuntimeError(f"Failed to fetch or parse page {current_page}.") from exc

        if not search_page.results:
            print(f"[ERROR] Empty page at page={current_page}")
            raise RuntimeError(f"Empty page at page={current_page}")

        unique_page_results: list[NalusResult] = []
        for result in search_page.results:
            unique_id = result.ecli or result.case_reference
            if not unique_id:
                raise RuntimeError("Decision is missing both ecli and case_reference.")
            if unique_id in seen_ids:
                print(f"[ERROR] Duplicate decision detected: {result.case_reference or unique_id}")
                raise RuntimeError(f"Duplicate decision detected: {result.case_reference or unique_id}")
            seen_ids.add(unique_id)
            unique_page_results.append(result)

        if max_results is not None:
            remaining_slots = max_results - len(collected_results)
            if remaining_slots <= 0:
                break
            if len(unique_page_results) > remaining_slots:
                dropped_results = unique_page_results[remaining_slots:]
                unique_page_results = unique_page_results[:remaining_slots]
                for dropped_result in dropped_results:
                    dropped_unique_id = dropped_result.ecli or dropped_result.case_reference
                    if dropped_unique_id:
                        seen_ids.discard(dropped_unique_id)

        if fetch_full_text:
            unique_page_results = enrich_results_with_text(unique_page_results)

        for result in unique_page_results:
            if not result.case_reference:
                raise RuntimeError("Collected decision is missing case_reference.")
            if not result.text_url:
                raise RuntimeError(f"Collected decision is missing text_url: {result.case_reference}")
            if fetch_full_text and not result.full_text:
                print(f"[ERROR] Missing full_text for: {result.case_reference}")
                raise RuntimeError(f"Missing full_text for: {result.case_reference}")

            collected_results.append(result)
            print(f"[SCRAPER] decision_added case={result.case_reference}")

            while len(collected_results) >= validated_count + 10:
                if len(collected_results) != len(seen_ids):
                    raise RuntimeError("Collected decision count does not match unique identifiers.")
                for item in collected_results[: validated_count + 10]:
                    if not item.case_reference:
                        raise RuntimeError("Collected decision is missing case_reference.")
                    if not item.text_url:
                        raise RuntimeError(f"Collected decision is missing text_url: {item.case_reference}")
                    if fetch_full_text and not item.full_text:
                        print(f"[ERROR] Missing full_text for: {item.case_reference}")
                        raise RuntimeError(f"Missing full_text for: {item.case_reference}")
                validated_count += 10
                validated_batches += 1
                print(f"[SCRAPER] validation_pass batch={validated_batches} size=10")

            if max_results is not None and len(collected_results) >= max_results:
                collected_results = collected_results[:max_results]
                seen_ids = {
                    item.ecli or item.case_reference
                    for item in collected_results
                    if item.ecli or item.case_reference
                }
                break

        pages_in_batch += 1
        print(f"[SCRAPER] page={current_page} total_collected={len(collected_results)}")

        reached_max_results = max_results is not None and len(collected_results) >= max_results
        reached_last_page = current_page >= search_page.total_pages
        reached_requested_end = page_end is not None and current_page >= page_end
        batch_complete = pages_in_batch >= batch_pages

        if batch_complete or reached_max_results or reached_last_page or reached_requested_end:
            print(f"[SCRAPER] batch_done pages={batch_start}-{current_page}")
            if not (reached_max_results or reached_last_page or reached_requested_end):
                time.sleep(batch_sleep_seconds)
            batch_start = current_page + 1
            pages_in_batch = 0

        if reached_max_results or reached_last_page or reached_requested_end:
            break

        current_page += 1

    if len(collected_results) != len(seen_ids):
        raise RuntimeError("Collected decision count does not match unique identifiers.")

    for result in collected_results[validated_count:]:
        if not result.case_reference:
            raise RuntimeError("Collected decision is missing case_reference.")
        if not result.text_url:
            raise RuntimeError(f"Collected decision is missing text_url: {result.case_reference}")
        if fetch_full_text and not result.full_text:
            print(f"[ERROR] Missing full_text for: {result.case_reference}")
            raise RuntimeError(f"Missing full_text for: {result.case_reference}")

    return collected_results
