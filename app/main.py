import json
import os
import re
import unicodedata
from dataclasses import asdict

from dotenv import load_dotenv

from app.services.search_service import collect_results


def _read_positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer. Received: {raw_value}") from exc

    if value < 1:
        raise SystemExit(f"{name} must be greater than or equal to 1.")

    return value


def _read_optional_positive_int(name: str) -> int | None:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return None

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer. Received: {raw_value}") from exc

    if value < 1:
        raise SystemExit(f"{name} must be greater than or equal to 1.")

    return value


def _read_non_negative_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a number. Received: {raw_value}") from exc

    if value < 0:
        raise SystemExit(f"{name} must be greater than or equal to 0.")

    return value


def _read_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise SystemExit(f"{name} must be a boolean value.")


def _slugify_query(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", normalized).strip("_").lower()
    return slug or "results"


def _default_output_path(
    query: str,
    max_results: int | None,
    page_start: int,
    page_end: int | None,
) -> str:
    if max_results is None and page_end == page_start:
        return "results.json"

    query_slug = _slugify_query(query)
    if max_results is not None:
        return f"results_{query_slug}_{max_results}.json"
    if page_end is None:
        return f"results_{query_slug}_bulk.json"
    return f"results_{query_slug}_pages_{page_start}_{page_end}.json"


def main():
    load_dotenv()

    query = os.getenv("NALUS_QUERY", "rodinné právo").strip() or "rodinné právo"
    single_page = _read_positive_int("NALUS_PAGE", 1)
    page_start = _read_positive_int("NALUS_PAGE_START", single_page)
    max_results = _read_optional_positive_int("NALUS_MAX_RESULTS")
    explicit_page_end = _read_optional_positive_int("NALUS_PAGE_END")
    fetch_full_text = _read_bool("NALUS_FETCH_FULL_TEXT", True)
    batch_pages = _read_positive_int("NALUS_BATCH_PAGES", 10)
    batch_sleep_seconds = _read_non_negative_float("NALUS_BATCH_SLEEP_SECONDS", 2.0)

    page_end = explicit_page_end
    if page_end is None and max_results is None:
        page_end = page_start

    output_path = (
        os.getenv("NALUS_OUTPUT_PATH", "").strip()
        or _default_output_path(query, max_results, page_start, page_end)
    )

    if page_end is not None and page_end < page_start:
        raise SystemExit("NALUS_PAGE_END must be greater than or equal to NALUS_PAGE_START.")

    results = collect_results(
        query=query,
        page_start=page_start,
        page_end=page_end,
        fetch_full_text=fetch_full_text,
        max_results=max_results,
        batch_pages=batch_pages,
        batch_sleep_seconds=batch_sleep_seconds,
    )

    print(f"QUERY: {query}")
    print(f"PAGES: {page_start}-{page_end if page_end is not None else 'auto'}")
    print(f"FULL TEXT: {fetch_full_text}")
    print(f"PARSED RESULTS: {len(results)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
