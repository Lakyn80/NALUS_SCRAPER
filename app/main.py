import json
from dataclasses import asdict

from app.crawler.extractor import extract_search_page
from app.crawler.playwright_crawler import fetch_page_html
from app.services.decision_service import enrich_results_with_text


def main():
    html = fetch_page_html()
    search_page = extract_search_page(html)
    results = enrich_results_with_text(search_page.results)

    print(f"PARSED RESULTS: {len(results)}")

    # uložení do JSON
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    print("Saved to results.json")


if __name__ == "__main__":
    main()
