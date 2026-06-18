import time
from dataclasses import replace

from app.crawler.text_fetcher import create_text_session, extract_plain_text, fetch_decision_html
from app.models.search_result import NalusResult


def enrich_results_with_text(results: list[NalusResult]) -> list[NalusResult]:
    enriched_results: list[NalusResult] = []
    session = create_text_session()

    try:
        for result in results:
            if not result.text_url:
                enriched_results.append(replace(result))
                continue

            try:
                decision_html = fetch_decision_html(result.text_url, session=session)
                full_text = extract_plain_text(decision_html)
                time.sleep(0.5)
            except Exception:
                enriched_results.append(replace(result, full_text=None))
                session.close()
                session = create_text_session()
                continue

            enriched_results.append(replace(result, full_text=full_text))
    finally:
        session.close()

    return enriched_results
