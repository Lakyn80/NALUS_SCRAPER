from app.crawler.playwright_crawler import fetch_page_html


def fetch_page(page: int, query: str = "rodinné právo") -> str:
    return fetch_page_html(query=query, page=page)
