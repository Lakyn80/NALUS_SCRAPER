from dataclasses import dataclass, field


@dataclass(slots=True)
class NalusResult:
    result_id: int | None
    case_reference: str | None
    ecli: str | None
    judge_rapporteur: str | None
    petitioner: str | None
    popular_name: str | None
    decision_date: str | None
    announcement_date: str | None
    filing_date: str | None
    publication_date: str | None
    related_regulations: list[str] = field(default_factory=list)
    decision_form: str | None = None
    importance: str | None = None
    verdict: str | None = None
    topics_and_keywords: list[str] = field(default_factory=list)
    detail_url: str | None = None
    text_url: str | None = None
    full_text: str | None = None


@dataclass(slots=True)
class NalusSearchPage:
    query: str
    page_index: int
    page_number: int
    page_size: int
    start_result: int
    end_result: int
    total_results: int
    total_pages: int
    results: list[NalusResult] = field(default_factory=list)
