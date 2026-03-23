import html
import re
from math import ceil
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from app.models.search_result import NalusResult, NalusSearchPage

BASE_SEARCH_URL = "https://nalus.usoud.cz/Search/"
_HEADER_RE = re.compile(
    r"Výsledky\s+(?P<start>\d+)\s*-\s*(?P<end>\d+)\s+z celkem\s+(?P<total>\d+)",
    re.IGNORECASE,
)
_RESULT_ID_RE = re.compile(r"sel_(?P<id>\d+)")
_TEXT_URL_RE = re.compile(r"https://nalus\.usoud\.cz(?::443)?/Search/GetText\.aspx\?sz=[^\"']+")
_TEXT_ID_RE = re.compile(r"sz=([A-Za-z0-9\-_]+)")


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = " ".join(value.replace("\xa0", " ").split()).strip()
    if not normalized or normalized == "(-)":
        return None

    return normalized


def _cell_lines(cell: Tag) -> list[str]:
    lines = []
    for line in cell.get_text("\n", strip=True).split("\n"):
        cleaned = _clean_text(line)
        if cleaned:
            lines.append(cleaned)
    return lines


def _collapse_consecutive_duplicates(lines: list[str]) -> list[str]:
    collapsed: list[str] = []
    for line in lines:
        if collapsed and collapsed[-1] == line:
            continue
        collapsed.append(line)
    return collapsed


def _extract_case_cell_metadata(cell: Tag) -> tuple[str | None, str | None, str | None]:
    detail_link = cell.find("a", href=True)
    case_reference = _clean_text(detail_link.get_text()) if detail_link else None
    metadata_lines = _cell_lines(cell)
    if case_reference and metadata_lines and metadata_lines[0] == case_reference:
        metadata_lines = metadata_lines[1:]

    ecli = next((line for line in metadata_lines if line.startswith("ECLI:")), None)
    judge_rapporteur = None
    if metadata_lines:
        last_line = metadata_lines[-1]
        if last_line != ecli and not last_line.startswith("ECLI:"):
            judge_rapporteur = last_line

    return case_reference, ecli, judge_rapporteur


def _extract_result_id(cell: Tag) -> int | None:
    checkbox = cell.find("input", id=_RESULT_ID_RE)
    if checkbox and checkbox.get("id"):
        match = _RESULT_ID_RE.fullmatch(checkbox["id"])
        if match:
            return int(match.group("id"))
    return None


def _extract_text_url(action_row: Tag) -> str | None:
    match = _TEXT_URL_RE.search(html.unescape(action_row.decode()))
    if not match:
        return None
    return match.group(0)


def _extract_page_counts(soup: BeautifulSoup) -> tuple[int, int, int, int]:
    header = soup.find("tr", class_="resultHeaderCount")
    if header is None:
        raise RuntimeError("Results header was not found in the NALUS response.")

    header_text = " ".join(header.get_text(" ", strip=True).split())
    match = _HEADER_RE.search(header_text)
    if not match:
        raise RuntimeError("Could not parse results counts from the NALUS header.")

    start_result = int(match.group("start"))
    end_result = int(match.group("end"))
    total_results = int(match.group("total"))

    page_input = header.find("input", attrs={"name": "pageNumber"})
    if page_input is None or not page_input.get("value"):
        raise RuntimeError("Could not determine current NALUS page number.")
    page_number = int(page_input["value"])

    page_links = []
    for link in header.find_all("a", href=True):
        text = _clean_text(link.get_text())
        if text and text.isdigit():
            page_links.append(int(text))

    total_pages = max([page_number, *page_links], default=page_number)

    return start_result, end_result, total_results, total_pages


def _find_results_table(soup: BeautifulSoup) -> Tag:
    for table in soup.find_all("table"):
        if any(
            row.get("id") == "headerRowFirst"
            for row in table.find_all("tr", recursive=False)
        ):
            return table
    raise RuntimeError("Could not locate the NALUS results table.")


def extract_search_page(html_text: str, query: str = "") -> NalusSearchPage:
    soup = BeautifulSoup(html_text, "lxml")
    start_result, end_result, total_results, total_pages = _extract_page_counts(soup)
    results_table = _find_results_table(soup)

    rows = results_table.find_all("tr", recursive=False)
    results: list[NalusResult] = []

    for index, row in enumerate(rows):
        cells = row.find_all("td", recursive=False)
        if len(cells) != 10:
            continue
        row_classes = set(row.get("class", []))
        if not row_classes.intersection({"resultData0", "resultData1"}):
            continue

        action_row = rows[index + 1] if index + 1 < len(rows) else None
        case_reference, ecli, judge_rapporteur = _extract_case_cell_metadata(cells[1])
        petitioner_lines = _collapse_consecutive_duplicates(_cell_lines(cells[2]))
        date_lines = _cell_lines(cells[3])
        form_lines = _cell_lines(cells[5])

        detail_link = cells[1].find("a", href=True)
        detail_url = (
            urljoin(BASE_SEARCH_URL, detail_link["href"])
            if detail_link and detail_link.get("href")
            else None
        )

        results.append(
            NalusResult(
                result_id=_extract_result_id(cells[0]),
                case_reference=case_reference,
                ecli=ecli,
                judge_rapporteur=judge_rapporteur,
                petitioner=petitioner_lines[0] if petitioner_lines else None,
                popular_name=" ".join(petitioner_lines[1:]) if len(petitioner_lines) > 1 else None,
                decision_date=date_lines[0] if len(date_lines) > 0 else None,
                announcement_date=date_lines[1] if len(date_lines) > 1 else None,
                filing_date=date_lines[2] if len(date_lines) > 2 else None,
                publication_date=date_lines[3] if len(date_lines) > 3 else None,
                related_regulations=_cell_lines(cells[4]),
                decision_form=form_lines[0] if len(form_lines) > 0 else None,
                importance=form_lines[1] if len(form_lines) > 1 else None,
                verdict=_clean_text(cells[6].get_text(" ", strip=True)),
                topics_and_keywords=_cell_lines(cells[8]),
                detail_url=detail_url,
                text_url=_extract_text_url(action_row) if action_row is not None else None,
            )
        )

    current_page_number = max(ceil(end_result / max(len(results), 1)), 1)
    page_input = soup.find("input", attrs={"name": "pageNumber"})
    if page_input and page_input.get("value", "").isdigit():
        current_page_number = int(page_input["value"])

    return NalusSearchPage(
        query=query,
        page_index=max(current_page_number - 1, 0),
        page_number=current_page_number,
        page_size=len(results),
        start_result=start_result,
        end_result=end_result,
        total_results=total_results,
        total_pages=total_pages,
        results=results,
    )


def extract_ids(html_text: str) -> list[str]:
    page = extract_search_page(html_text)
    identifiers: list[str] = []

    for result in page.results:
        if not result.text_url:
            continue
        match = _TEXT_ID_RE.search(result.text_url)
        if match:
            identifiers.append(match.group(1))

    return identifiers
