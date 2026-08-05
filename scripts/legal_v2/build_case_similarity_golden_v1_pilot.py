#!/usr/bin/env python3
"""Build the tracked case-similarity golden v1 pilot JSONL.

Document-level, source-grounded, offline only. No LLM / provider calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.benchmark.case_similarity_golden import (  # noqa: E402
    AnswerEvidenceItem,
    AlternativeRationale,
    CaseSimilarityGoldenItem,
    CaseSimilarityProvenance,
    HardNegativeRationale,
    count_sentences,
    count_words,
    evidence_excerpt_in_block,
    validate_case_similarity_dataset,
    write_case_similarity_jsonl,
)
from app.rag.legal_v2.benchmark.corpus import (  # noqa: E402
    load_case_similarity_corpus,
    load_case_similarity_primary_document_ids,
)

DEFAULT_OUT = PROJECT_ROOT / "benchmarks" / "legal_v2" / "case_similarity_golden_v1_pilot.jsonl"
DEFAULT_REPORT = (
    PROJECT_ROOT / "artifacts" / "legal_v2" / "case_similarity_golden_v1_pilot" / "build_report.json"
)
BUILDER_NAME = "build_case_similarity_golden_v1_pilot.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    corpus = load_case_similarity_corpus()
    primary_document_ids = load_case_similarity_primary_document_ids()
    by_doc_index = {
        (block.document_id, block.block_index): block for block in corpus.blocks_by_id.values()
    }
    refs_by_id = {ref.document_id: ref for ref in corpus.documents}

    def block(document_id: str, index: int):
        key = (document_id, index)
        if key not in by_doc_index:
            raise KeyError(f"missing block {document_id}#{index}")
        return by_doc_index[key]

    def resolve_evidence(document_id: str, specs: list[tuple[int | None, str]]) -> list[AnswerEvidenceItem]:
        items: list[AnswerEvidenceItem] = []
        seen: set[str] = set()
        for index, excerpt in specs:
            if index is None:
                found = None
                for candidate in corpus.blocks_for_document(document_id):
                    if evidence_excerpt_in_block(excerpt, candidate.raw_text):
                        found = candidate
                        break
                if found is None:
                    raise KeyError(f"no block in {document_id} contains excerpt: {excerpt[:80]}")
                target = found
            else:
                target = block(document_id, index)
                if not evidence_excerpt_in_block(excerpt, target.raw_text):
                    raise ValueError(
                        f"excerpt not in {document_id}#{index}: {excerpt[:80]}"
                    )
            if target.block_id in seen:
                raise ValueError(f"duplicate supporting block {target.block_id}")
            seen.add(target.block_id)
            items.append(AnswerEvidenceItem(block_id=target.block_id, excerpt=excerpt))
        return items

    raw_specs: list[dict[str, Any]] = _curated_specs()
    items: list[CaseSimilarityGoldenItem] = []
    for spec in raw_specs:
        document_id = spec["document_id"]
        ref = refs_by_id[document_id]
        evidence = resolve_evidence(document_id, spec["evidence"])
        hard_rationales = [
            HardNegativeRationale(
                document_id=row["document_id"],
                looks_similar_because=row["looks_similar_because"],
                materially_incorrect_because=row["materially_incorrect_because"],
            )
            for row in spec["hard_negatives"]
        ]
        alt_rationales = [
            AlternativeRationale(document_id=row["document_id"], rationale=row["rationale"])
            for row in spec.get("alternatives", [])
        ]
        item = CaseSimilarityGoldenItem(
            benchmark_id=spec["benchmark_id"],
            query=spec["query"],
            query_style=spec["query_style"],
            difficulty=spec["difficulty"],
            source_document_id=document_id,
            expected_document_ids=[document_id],
            accepted_alternative_document_ids=[row["document_id"] for row in spec.get("alternatives", [])],
            hard_negative_document_ids=[row["document_id"] for row in spec["hard_negatives"]],
            supporting_block_ids=[row.block_id for row in evidence],
            answer_evidence=evidence,
            factual_facets=list(spec["factual_facets"]),
            legal_issue_facets=list(spec["legal_issue_facets"]),
            procedural_facets=list(spec["procedural_facets"]),
            similarity_rationale=spec["similarity_rationale"],
            hard_negative_rationales=hard_rationales,
            accepted_alternative_rationales=alt_rationales,
            hard_negative_evaluable=bool(spec.get("hard_negative_evaluable", True)),
            hard_negative_blocker=spec.get("hard_negative_blocker"),
            provenance=CaseSimilarityProvenance(
                builder=BUILDER_NAME,
                corpus_role="reviewed_pool",
                review_number=ref.review_number,
                source_case_number=ref.case_number,
                source_court=ref.court,
                notes=spec.get("provenance_notes"),
            ),
            notes=spec.get("notes"),
        )
        items.append(item)

    items.sort(key=lambda row: row.benchmark_id)
    write_case_similarity_jsonl(args.output, items)

    tracked_bytes = args.output.read_bytes()
    rebuild_path = args.report.parent / "_rebuild_check.jsonl"
    write_case_similarity_jsonl(rebuild_path, items)
    rebuild_bytes = rebuild_path.read_bytes()

    report = validate_case_similarity_dataset(
        items,
        corpus_documents=corpus.documents,
        blocks_by_id=corpus.blocks_by_id,
        expected_document_ids=primary_document_ids,
        dataset_path=str(args.output),
        rebuild_bytes=rebuild_bytes,
        tracked_bytes=tracked_bytes,
    )
    payload = {
        "ok": report.ok,
        "item_count": report.item_count,
        "output": str(args.output),
        "word_counts": {item.benchmark_id: count_words(item.query) for item in items},
        "sentence_counts": {item.benchmark_id: count_sentences(item.query) for item in items},
        "issues": [issue.model_dump() for issue in report.issues],
        "warnings": [issue.model_dump() for issue in report.warnings],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rebuild_path.unlink(missing_ok=True)

    print(
        "\n".join(
            [
                f"output={args.output}",
                f"items={len(items)}",
                f"ok={report.ok}",
                f"issues={len(report.issues)}",
                f"warnings={len(report.warnings)}",
                f"report={args.report}",
            ]
        )
    )
    if not report.ok:
        for issue in report.issues:
            print(f"ERROR [{issue.code}] {issue.benchmark_id or '-'}: {issue.message}")
        return 1
    return 0


def _curated_specs() -> list[dict[str, Any]]:
    """Explicitly curated source-first case descriptions (deterministic)."""
    return [
        {
            "benchmark_id": "nalus-cs-pilot-001",
            "document_id": "doc-0a90125eb71851b4",
            "query_style": "client_narrative",
            "difficulty": "medium",
            "query": (
                "Měl jsem civilní spor, který se táhl skoro třináct let, a teď po státu chci "
                "náhradu nemajetkové újmy za nepřiměřenou délku řízení. Nižší soudy uznaly, "
                "že průtahy byly, ale peněžitou náhradu mi buď nepřiznaly, nebo ji výrazně "
                "snížily. Dovolání pak odmítli jako nepřípustné a já mám pocit, že mi tím "
                "odepřeli přístup k soudu. Hledám podobné rozsudky, kde se řeší náhrada za "
                "dlouhé řízení a přezkum odmítnutí dovolání."
            ),
            "evidence": [
                (8, "domáhal zaplacení částky 27 000 000 Kč s příslušenstvím jako náhrady nemajetkové újmy"),
                (9, "v uvedeném řízení došlo k nesprávnému úřednímu postupu v podobě průtahů v řízení"),
                (11, "Dovolání stěžovatele odmítl rovněž napadeným usnesením Nejvyšší soud jako nepřípustné"),
                (18, "ústavní stížnost představuje zjevně neopodstatněný návrh"),
            ],
            "factual_facets": ["excessive_length_of_proceedings", "non_pecuniary_damage_claim"],
            "legal_issue_facets": ["compensation_for_unreasonable_delay", "inadmissible_appeal_on_points_of_law"],
            "procedural_facets": ["constitutional_complaint"],
            "similarity_rationale": (
                "Primary match is the judgment about compensation for excessive civil-case length "
                "and a rejected appeal on points of law."
            ),
            "hard_negatives": [
                {
                    "document_id": "doc-c7b72b0d6121d7f3",
                    "looks_similar_because": "Also concerns a claim for non-pecuniary harm after a harmful incident.",
                    "materially_incorrect_because": "It is about personality rights against a security guard, not state liability for court delay.",
                },
                {
                    "document_id": "doc-d513b3e81616439a",
                    "looks_similar_because": "Also a monetary damages claim after civil litigation with causation and higher-court review issues.",
                    "materially_incorrect_because": "Damages are tied to an interim measure and gift/sale facts, not compensation for excessive court delay.",
                },
            ],
            "alternatives": [],
            "notes": "Reviewed pool #1.",
        },
        {
            "benchmark_id": "nalus-cs-pilot-002",
            "document_id": "doc-b73cac9b3dfc8a42",
            "query_style": "concise_case_description",
            "difficulty": "easy",
            "query": (
                "Podali jsme návrh na zrušení soudních rozhodnutí, ale bez advokáta a bez "
                "řádných náležitostí podání. Už dříve nás soudy upozorňovaly, jak má návrh "
                "vypadat a co doplnit, a přesto jsme znovu poslali vadné podání. Zajímá mě, "
                "jestli se v podobných věcech návrh odmítá hned bez další výzvy k opravě. "
                "Hledám rozsudky o odmítnutí vadného návrhu po opakovaném poučení a o tom, "
                "kdy už soud neopakuje výzvu k odstranění vad."
            ),
            "evidence": [
                (4, "bez právního zastoupení"),
                (7, "podáním, které nesplňovalo náležitosti řádného návrhu na zahájení řízení"),
            ],
            "factual_facets": ["defective_constitutional_filing", "repeated_prior_instructions"],
            "legal_issue_facets": ["rejection_without_further_request_to_cure"],
            "procedural_facets": ["constitutional_complaint"],
            "similarity_rationale": (
                "Matches the short order rejecting a formally defective filing after repeated prior instructions."
            ),
            "hard_negatives": [
                {
                    "document_id": "doc-f2c776a1533521c3",
                    "looks_similar_because": "Also a short order refusing a filing for missing lawyer representation.",
                    "materially_incorrect_because": "There the applicant got a cure deadline and failed it; here the court skips another cure call after repeated prior warnings.",
                },
                {
                    "document_id": "doc-16b9100a8b9122dd",
                    "looks_similar_because": "Also rejected for missing formal requirements including lawyer representation.",
                    "materially_incorrect_because": "Underlying family-protection dispute differs and the refusal is mainly for missing counsel and reasoning defects.",
                },
            ],
            "alternatives": [],
            "notes": "Reviewed pool #2.",
        },
        {
            "benchmark_id": "nalus-cs-pilot-003",
            "document_id": "doc-d513b3e81616439a",
            "query_style": "client_narrative",
            "difficulty": "hard",
            "query": (
                "Chtěl jsem zrušit dar a pak nemovitost prodat, ale mezitím přišlo předběžné "
                "opatření, které mi podle mého názoru bránilo v dispozici s věcí. Později jsem "
                "žaloval o náhradu škody ve výši stovek tisíc, protože obchod nevyšel a přišel "
                "jsem o dohodnutou cenu. Soudy podle mě nesprávně uzavřely, že mezi omezením "
                "nakládání s nemovitostí a neuskutečněným prodejem nebyla dostatečná souvislost, "
                "a dovolání odmítly. Hledám podobné případy o škodě způsobené předběžným "
                "opatřením a o příčinné souvislosti."
            ),
            "evidence": [
                (9, "zamítnuta žaloba, kterou se stěžovatel domáhal zaplacení 300 000 Kč s příslušenstvím"),
                (10, "tvrzená škoda nevznikla v příčinné souvislosti se sporným předběžným opatřením"),
                (11, "Nejvyšší soud následné dovolání stěžovatele odmítl jako nepřípustné"),
                (13, "závěr soudů, že tvrzená škoda nevznikla v příčinné souvislosti s předběžným opatřením"),
            ],
            "factual_facets": ["gift_revocation", "preliminary_injunction", "failed_sale_damage_claim"],
            "legal_issue_facets": ["causation_for_damages_from_interim_measure"],
            "procedural_facets": ["constitutional_complaint"],
            "similarity_rationale": (
                "Targets the damage claim linked to an interim measure after gift revocation and a blocked sale."
            ),
            "hard_negatives": [
                {
                    "document_id": "doc-0a90125eb71851b4",
                    "looks_similar_because": "Also seeks money after unsuccessful civil litigation and complains about higher-court review.",
                    "materially_incorrect_because": "That case is about delay damages against the state, not causation after a preliminary injunction.",
                },
                {
                    "document_id": "doc-e5ac4b1fcd075062",
                    "looks_similar_because": "Also a property/transfer dispute later reviewed for constitutional fairness.",
                    "materially_incorrect_because": "It concerns pre-emption and phased litigation costs, not interim-measure damages.",
                },
            ],
            "alternatives": [],
            "notes": "Reviewed pool #3.",
        },
        {
            "benchmark_id": "nalus-cs-pilot-004",
            "document_id": "doc-16b9100a8b9122dd",
            "query_style": "noisy_client_narrative",
            "difficulty": "easy",
            "query": (
                "Sociálka a soudy řeší odebrání dětí a já jsem proti tomu hned podal stížnost, "
                "ale bez advokáta, spíš tak naléhavě a chaoticky. V podání mi chybí pořádné "
                "odůvodnění a zastoupení, takže to asi stejně shodí. Potřebuju najít podobné "
                "věci, kde rodič napadá zásah do péče o děti, ale návrh padne už na formálních "
                "vadách. Nehledám teď meritorní spor o péči, spíš jak se takové vadné stížnosti "
                "odmítají."
            ),
            "evidence": [
                (7, "domáhá zrušení v záhlaví označených rozhodnutí"),
                (8, "zkoumá, zda ústavní stížnost splňuje požadované náležitosti"),
                (9, "Stěžovatel není zastoupena advokátem"),
                (10, "pro nenaplnění formálních podmínek pro podání ústavní stížnosti odmítl"),
            ],
            "factual_facets": ["child_protection_intervention", "unrepresented_parent_filing"],
            "legal_issue_facets": ["formal_rejection_of_constitutional_complaint"],
            "procedural_facets": ["constitutional_complaint"],
            "similarity_rationale": (
                "Best match is the short refusal of a parent complaint about child removal for missing formal requirements."
            ),
            "hard_negatives": [
                {
                    "document_id": "doc-a5292901931de05a",
                    "looks_similar_because": "Also a family-law custody/contact dispute brought by a parent.",
                    "materially_incorrect_because": "That judgment reaches the merits of a provisional care request; it is not a pure formal rejection for missing counsel.",
                },
                {
                    "document_id": "doc-b73cac9b3dfc8a42",
                    "looks_similar_because": "Also rejected for defective constitutional filing without proper form.",
                    "materially_incorrect_because": "Underlying dispute is not child-protection removal and the factual pattern differs.",
                },
            ],
            "alternatives": [],
            "notes": "Reviewed pool #4.",
        },
        {
            "benchmark_id": "nalus-cs-pilot-005",
            "document_id": "doc-e5ac4b1fcd075062",
            "query_style": "multi_issue_client_narrative",
            "difficulty": "hard",
            "query": (
                "Kupovali jsme pozemky a tvrdíme, že nám někdo porušil předkupní právo. Spor se "
                "měnil: nejdřív určení vlastnictví a nahrazení projevu vůle, pak změna žaloby na "
                "jiný oddělený pozemek a otázku, jestli je na něm samostatná stavba. Ve finální "
                "fázi po zúžení nároku jsme byli věcně úspěšní, ale soudy špatně rozložily "
                "náklady, protože největší výdaje vznikly až v této pozdější fázi. Hledám "
                "rozsudky o předkupním právu k pozemku a o nákladech řízení dělených do fází."
            ),
            "evidence": [
                (10, "povinnosti hradit náklady řízení děleného do samostatných fází"),
                (11, "dvě žaloby týkající se tvrzeného porušení předkupního práva k pozemku"),
                (14, "návrhem ze dne 5. 6. 2023 na změnu žaloby"),
                (15, "zda je požární nádrž samostatnou stavbou netvořící součást pozemku"),
            ],
            "factual_facets": ["preemption_right_over_land", "changed_claim_after_survey_plan", "phased_litigation_costs"],
            "legal_issue_facets": ["preemption_enforcement", "cost_allocation_across_procedural_phases"],
            "procedural_facets": ["constitutional_complaint"],
            "similarity_rationale": (
                "Matches the multi-phase land pre-emption dispute with a later cost-allocation constitutional issue."
            ),
            "hard_negatives": [
                {
                    "document_id": "doc-84ae84698dfd0205",
                    "looks_similar_because": "Also concerns land registration and ownership after marital property changes.",
                    "materially_incorrect_because": "It is a Part Five cadastre/SJM fiction case, not pre-emption or phased costs.",
                },
                {
                    "document_id": "doc-dc644c7e6d827609",
                    "looks_similar_because": "Also a land-return / ownership legitimacy dispute.",
                    "materially_incorrect_because": "Restitution under agricultural land rules, not contractual pre-emption.",
                },
            ],
            "alternatives": [],
            "notes": "Reviewed pool #5.",
        },
        {
            "benchmark_id": "nalus-cs-pilot-006",
            "document_id": "doc-abd57ac0aa5dfe5b",
            "query_style": "client_narrative",
            "difficulty": "medium",
            "query": (
                "Jsme spolek a podali jsme kasační stížnost ve správním sporu proti odmítnutí "
                "naší žaloby. Soud nás vyzval k zaplacení soudního poplatku, ale dal nám jen "
                "pět dní místo obvyklých patnácti. Nestihli jsme to zaplatit včas a řízení "
                "skončilo. Tvrdíme, že zkrácení lhůty zasáhlo do přístupu k soudu. Hledám "
                "podobná rozhodnutí o zkrácené lhůtě k zaplacení soudního poplatku u kasační "
                "stížnosti a o tom, kdy je takový postup nepřípustný."
            ),
            "evidence": [
                (11, "odmítl žalobu stěžovatele na ochranu před nezákonným zásahem"),
                (12, "vyzval k zaplacení soudního poplatku za kasační stížnost ve výši 5 000 Kč"),
                (14, "zkrácením zákonné patnáctidenní lhůty k zaplacení soudního poplatku"),
                (16, "Ústavní soud posoudil splnění procesních předpokladů řízení"),
            ],
            "factual_facets": ["shortened_court_fee_deadline", "ngo_cassation_complaint"],
            "legal_issue_facets": ["access_to_court_via_court_fee_deadline"],
            "procedural_facets": ["constitutional_complaint", "administrative_cassation"],
            "similarity_rationale": (
                "Primary judgment concerns a shortened court-fee payment deadline for a cassation complaint."
            ),
            "hard_negatives": [
                {
                    "document_id": "doc-db9f10005638d155",
                    "looks_similar_because": "Also turns on unpaid appeal court fees and consequences of fee non-payment.",
                    "materially_incorrect_because": "Commercial appeal fee/stoppage and costs dispute, not a shortened statutory fee deadline in administrative cassation.",
                },
                {
                    "document_id": "doc-f2c776a1533521c3",
                    "looks_similar_because": "Also a missed cure deadline leading to rejection.",
                    "materially_incorrect_because": "Deadline concerned appointing a lawyer, not paying a court fee.",
                },
            ],
            "alternatives": [],
            "notes": "Reviewed pool #6.",
        },
        {
            "benchmark_id": "nalus-cs-pilot-007",
            "document_id": "doc-af3c185ad674a7da",
            "query_style": "multi_issue_client_narrative",
            "difficulty": "hard",
            "query": (
                "Jsem advokát a kárný senát mě potrestal za to, že jsem dřív zastupoval firmu "
                "a později vystupoval v sporech, které se týkaly její insolvence a majetkové "
                "podstaty. Namítal jsem podjatost člena senátu, zánik plné moci prohlášením "
                "konkurzu i to, že nestačí pouhá souvislost věcí bez konkrétních důvěrných "
                "informací. Správní soud to ale potvrdil. Hledám podobné rozsudky o kárném "
                "provinění advokáta a konfliktu zájmů po insolvenci klienta."
            ),
            "evidence": [
                (9, "dopustil kárného provinění tím, že ačkoliv od roku 2013"),
                (12, "námitky podjatosti vznesené stěžovatelem vůči členovi odvolacího kárného senátu"),
                (14, "plná moc udělená stěžovateli společností"),
                (15, "v souvislosti s insolvenčním řízením společnosti"),
            ],
            "factual_facets": ["lawyer_discipline", "former_client_insolvency", "alleged_bias_of_panel_member"],
            "legal_issue_facets": ["conflict_of_interest_after_prior_representation", "power_of_attorney_after_bankruptcy"],
            "procedural_facets": ["constitutional_complaint", "disciplinary_review"],
            "similarity_rationale": (
                "Matches the lawyer-discipline conflict-of-interest dispute tied to prior representation and insolvency."
            ),
            "hard_negatives": [
                {
                    "document_id": "doc-abd57ac0aa5dfe5b",
                    "looks_similar_because": "Also a constitutional review after another adjudicatory proceeding involving procedural access and professional/organisational responsibility.",
                    "materially_incorrect_because": "Concerns a shortened court-fee deadline in administrative cassation, not advocate disciplinary conflict of interest after insolvency representation.",
                },
                {
                    "document_id": "doc-0a90125eb71851b4",
                    "looks_similar_because": "Also a constitutional complaint after unsuccessful ordinary-court review.",
                    "materially_incorrect_because": "Subject matter is delay damages, not lawyer discipline.",
                },
            ],
            "alternatives": [],
            "hard_negative_evaluable": False,
            "hard_negative_blocker": "insufficient_same_domain_corpus",
            "notes": (
                "Reviewed pool #7. CORPUS BLOCKER for honest same-domain hard negatives: "
                "local raw_sources (93) + NSoud dumps contain no second lawyer-discipline / "
                "former-client-conflict / insolvency-representation peer judgment. Existing "
                "cross-domain hard negatives are retained only to satisfy schema min=1; they "
                "are not claimed as strong hard negatives. Corpus expansion required."
            ),
        },
        {
            "benchmark_id": "nalus-cs-pilot-008",
            "document_id": "doc-a5292901931de05a",
            "query_style": "client_narrative",
            "difficulty": "medium",
            "query": (
                "S bývalou partnerkou máme syna; chlapec je v péči matky a já mám styk každý "
                "druhý víkend od pátku do neděle. Meritorní řízení o péči se táhne. Požádal "
                "jsem o prozatímní úpravu, aby syn jezdil už od čtvrtka a aby byl jasný režim "
                "na letní prázdniny. Soud to zamítl spíš podle písemností a já mám pocit, že "
                "zaměnil dočasnou úpravu s konečným rozhodnutím. Hledám podobné rozsudky o "
                "prozatímní úpravě péče a styku s dítětem."
            ),
            "evidence": [
                (8, "Chlapec byl svěřen do péče matce a otci byl upraven styk s ním na každý druhý víkend"),
                (10, "návrh na vydání prozatímního rozhodnutí, jímž by soud upravil péči otce od čtvrtka do neděle"),
                (13, "soud popřel povahu prozatímního rozhodnutí"),
                (15, "u prozatímních rozhodnutí vychází z toho, že poměry dítěte"),
            ],
            "factual_facets": ["weekend_contact_arrangement", "request_to_extend_provisional_care", "summer_holiday_schedule"],
            "legal_issue_facets": ["provisional_child_care_order", "temporary_versus_final_custody_decision"],
            "procedural_facets": ["constitutional_complaint", "family_interim_measure"],
            "similarity_rationale": (
                "Best match is the provisional child-care expansion dispute including holiday contact."
            ),
            "hard_negatives": [
                {
                    "document_id": "doc-16b9100a8b9122dd",
                    "looks_similar_because": "Also a parent challenging state/court interference with children.",
                    "materially_incorrect_because": "Formal rejection about child-protection removal, not a provisional contact schedule.",
                },
                {
                    "document_id": "doc-976fafa1e2c6f093",
                    "looks_similar_because": "Also a parent-child financial/family dispute.",
                    "materially_incorrect_because": "Concerns termination of maintenance for an adult child student, not provisional custody contact.",
                },
            ],
            "alternatives": [],
            "notes": "Reviewed pool #8.",
        },
        {
            "benchmark_id": "nalus-cs-pilot-009",
            "document_id": "doc-f2c776a1533521c3",
            "query_style": "concise_case_description",
            "difficulty": "easy",
            "query": (
                "Jsem ve výkonu trestu a napadl jsem rozhodnutí soudu ústavní stížností. "
                "Dostal jsem výzvu, abych si ve stanovené lhůtě zajistil advokáta a doložil "
                "zastoupení, ale nestihl jsem to a lhůta uplynula. Stížnost proto odmítli bez "
                "meritorního přezkumu. Hledám podobné případy, kde vězeňský stěžovatel nedodrží "
                "lhůtu k doplnění právního zastoupení a podání padne na formální vadě, nikoli "
                "na obsahu samotného sporu."
            ),
            "evidence": [
                (4, "t. č. věznice"),
                (7, "vyzval jej soudce zpravodaj k odstranění této vady ve lhůtě 30 dnů"),
            ],
            "factual_facets": ["prisoner_constitutional_complaint", "missed_lawyer_cure_deadline"],
            "legal_issue_facets": ["rejection_for_missing_legal_representation"],
            "procedural_facets": ["constitutional_complaint"],
            "similarity_rationale": (
                "Matches the short order rejecting a prisoner complaint after a missed lawyer-cure deadline."
            ),
            "hard_negatives": [
                {
                    "document_id": "doc-b73cac9b3dfc8a42",
                    "looks_similar_because": "Also a short formal rejection of a defective constitutional filing.",
                    "materially_incorrect_because": "No cure deadline after a fresh warning; repeated prior instructions and immediate refusal.",
                },
                {
                    "document_id": "doc-16b9100a8b9122dd",
                    "looks_similar_because": "Also rejected for missing lawyer representation.",
                    "materially_incorrect_because": "Underlying facts concern child-protection measures, not a prisoner complaint.",
                },
            ],
            "alternatives": [],
            "notes": "Reviewed pool #9.",
        },
        {
            "benchmark_id": "nalus-cs-pilot-010",
            "document_id": "doc-976fafa1e2c6f093",
            "query_style": "client_narrative",
            "difficulty": "medium",
            "query": (
                "Platil jsem výživné zletilé dceři, která studovala, a domáhal jsem se zrušení "
                "vyživovací povinnosti, protože studium podle mě už nesplňovalo podmínky další "
                "přípravy na povolání. Prvostupňový soud povinnost zrušil dříve, odvolací soud "
                "ale vázal konec až na pozdější úspěšné ukončení studia. Rozhodnutí mi přišlo "
                "překvapivé a krátce odůvodněné. Hledám podobné rozsudky o zániku výživného ke "
                "zletilému dítěti po studiu a o tom, k jakému datu povinnost končí."
            ),
            "evidence": [
                (8, "zrušení vyživovací povinnosti stěžovatele (otce) vůči jeho dceři"),
                (11, "domáhal zrušení vyživovací povinnosti za dobu od 17. 9. 2018"),
                (13, "zrušil vyživovací povinnost stěžovatele k vedlejší účastnici řízení až s účinností ode dne 18. 9. 2021"),
                (15, "rozhodnutí městského soudu pro stěžovatele překvapivé"),
            ],
            "factual_facets": ["adult_child_maintenance", "study_continuation_dispute"],
            "legal_issue_facets": ["termination_date_of_maintenance_obligation", "surprise_appellate_decision"],
            "procedural_facets": ["constitutional_complaint", "civil_appeal_review"],
            "similarity_rationale": (
                "Targets the maintenance-termination dispute tied to an adult child's professional studies."
            ),
            "hard_negatives": [
                {
                    "document_id": "doc-a5292901931de05a",
                    "looks_similar_because": "Also a parent-child family dispute about care arrangements.",
                    "materially_incorrect_because": "Provisional custody/contact, not maintenance for an adult student.",
                },
                {
                    "document_id": "doc-0a90125eb71851b4",
                    "looks_similar_because": "Also complains about appellate/higher-court fairness after a money claim.",
                    "materially_incorrect_because": "Delay damages against the state, not child support.",
                },
            ],
            "alternatives": [],
            "notes": "Reviewed pool #10.",
        },
        {
            "benchmark_id": "nalus-cs-pilot-011",
            "document_id": "doc-cfa470876b0d5ed7",
            "query_style": "multi_issue_client_narrative",
            "difficulty": "hard",
            "query": (
                "Po mně chtějí zaplacení směnky na stovky tisíc i s úroky. Firma, která směnku "
                "vystavila, měla mít na listině podpis jednatele, ale já tvrdím, že ten člověk "
                "jednatelem nebyl a podpis není jeho. Dále namítám, že označení remitenta je "
                "rozporné a že jsem směnku podepsal jen jako rukojmí. Nižší soud směnečný platební "
                "rozkaz potvrdil. Hledám podobné obchodní spory o platnost směnky a obranu avalisty."
            ),
            "evidence": [
                (6, "zaplacení směnečného peníze ve výši 260 000 Kč"),
                (8, "směnka je neplatná, když jménem výstavce směnky měl jednat"),
                (10, "Žalovaná směnku podepsala k doložce směnečný rukojmí"),
                (11, "Směnku vyhodnotil soud jako směnku platnou"),
            ],
            "factual_facets": ["promissory_note_claim", "disputed_issuer_signature", "aval_defence"],
            "legal_issue_facets": ["bill_of_exchange_validity", "avalist_liability"],
            "procedural_facets": ["commercial_appeal"],
            "similarity_rationale": (
                "Matches the commercial appeal about promissory-note validity and avalist objections."
            ),
            "hard_negatives": [
                {
                    "document_id": "doc-e6af147081ae754f",
                    "looks_similar_because": "Also a commercial payment dispute after a business contract breakdown.",
                    "materially_incorrect_because": "Software licence/fees dispute, not a bill of exchange.",
                },
                {
                    "document_id": "doc-db9f10005638d155",
                    "looks_similar_because": "Also a Prague commercial appeal about money and procedural costs.",
                    "materially_incorrect_because": "About unpaid appeal fees and cost awards, not promissory-note validity.",
                },
            ],
            "alternatives": [],
            "notes": "Reviewed pool #11.",
        },
        {
            "benchmark_id": "nalus-cs-pilot-012",
            "document_id": "doc-db9f10005638d155",
            "query_style": "client_narrative",
            "difficulty": "medium",
            "query": (
                "Prohráli jsme obchodní spor a podali odvolání, ale soudní poplatek jsme "
                "nezaplatili ani v dodatečné lhůtě, takže odvolací řízení zastavili. Teď se "
                "hádáme hlavně o náhradě nákladů: protistrana chce odměnu za vyjádření k "
                "odvolání včetně DPH a já říkám, že výzva k poplatku mi nebyla řádně a včas "
                "doručena, dopis prý nešel doporučeně a platbu jsem udělal podle skutečného "
                "doručení. Hledám podobné rozsudky o zastavení odvolání pro neuhrazený "
                "poplatek a o nákladech takového řízení."
            ),
            "evidence": [
                (6, "neuhrazení soudního poplatku z odvolání ani v soudem určené dodatečné lhůtě"),
                (7, "náhradu nákladů právního zastoupení, a to za jeden hlavní úkon právní služby"),
                (8, "napadá jen výrok usnesení soudu prvního stupně o náhradě nákladů odvolacího řízení"),
                (10, "vyzváni usnesením"),
            ],
            "factual_facets": ["unpaid_appeal_court_fee", "appeal_stopped", "costs_of_appeal_dispute"],
            "legal_issue_facets": ["fee_nonpayment_stoppage", "appeal_cost_award"],
            "procedural_facets": ["commercial_appeal"],
            "similarity_rationale": (
                "Primary match is the commercial appeal stopped for unpaid fees with a follow-on costs fight."
            ),
            "hard_negatives": [
                {
                    "document_id": "doc-abd57ac0aa5dfe5b",
                    "looks_similar_because": "Also centres on court-fee payment timing and access consequences.",
                    "materially_incorrect_because": "Shortened fee deadline in administrative cassation, not commercial appeal stoppage/costs.",
                },
                {
                    "document_id": "doc-4af3171b4be427e9",
                    "looks_similar_because": "Also a Prague commercial/civil appeal ending without full merits.",
                    "materially_incorrect_because": "Appeal was withdrawn; co-ownership of a cooperative share, not unpaid fees.",
                },
            ],
            "alternatives": [],
            "notes": "Reviewed pool #12.",
        },
        {
            "benchmark_id": "nalus-cs-pilot-013",
            "document_id": "doc-4af3171b4be427e9",
            "query_style": "concise_case_description",
            "difficulty": "easy",
            "query": (
                "Soud zrušil podílové spoluvlastnictví k družstevnímu podílu spojenému s nájmem "
                "bytu v bytovém družstvu. Já jsem se proti rozsudku odvolal, později odvolání "
                "částečně a nakonec úplně vzal zpět a navrhl zastavení odvolacího řízení. "
                "Protistrana náhradu nákladů odvolacího řízení nenavrhovala. Potřebuju podobná "
                "rozhodnutí, kde odvolací soud zastaví řízení po zpětvzetí odvolání ve sporu "
                "o družstevní podíl a rozhodne o nákladech bez meritorního přezkumu."
            ),
            "evidence": [
                (5, "rozhodl o zrušení podílového spoluvlastnictví"),
                (7, "vzal žalovaný své odvolání zpět v celém rozsahu"),
                (8, "odvolací soud odvolací řízení podle citovaného ustanovení"),
                (9, "žalobkyně náhradu nákladů odvolacího řízení v souvislosti se zpětvzetím odvolání žalovaného nenavrhovala"),
            ],
            "factual_facets": ["cooperative_share_coownership", "appeal_withdrawal"],
            "legal_issue_facets": ["stoppage_after_appeal_withdrawal"],
            "procedural_facets": ["commercial_appeal"],
            "similarity_rationale": (
                "Matches the appeal stopped after full withdrawal in a cooperative-share co-ownership case."
            ),
            "hard_negatives": [
                {
                    "document_id": "doc-db9f10005638d155",
                    "looks_similar_because": "Also an appeal that ends without full merits and turns on costs.",
                    "materially_incorrect_because": "Stopped for unpaid fees, not voluntary withdrawal; different commercial subject.",
                },
                {
                    "document_id": "doc-84ae84698dfd0205",
                    "looks_similar_because": "Also concerns shared ownership interests in property.",
                    "materially_incorrect_because": "Cadastre/SJM land registration under Part Five, not cooperative share appeal withdrawal.",
                },
            ],
            "alternatives": [],
            "notes": "Reviewed pool #13.",
        },
        {
            "benchmark_id": "nalus-cs-pilot-014",
            "document_id": "doc-e6af147081ae754f",
            "query_style": "multi_issue_client_narrative",
            "difficulty": "hard",
            "query": (
                "Najali jsme firmu na informační systém a licenci. Aplikace byla pomalá, na "
                "mobilech nefungovala jak měla a implementace nebyla hotová podle dohody. "
                "Přestali jsme platit faktury a od smlouvy jsme odstoupili. Dodavatel teď chce "
                "doplatek licence, startup fee i další vyrovnání a tvrdí, že vady programu samy "
                "o sobě neopravňují neplatit. My navíc namítali i mezinárodní příslušnost. "
                "Hledám podobné spory o vadné dodávce softwaru a licenčních platbách."
            ),
            "evidence": [
                (6, "smlouvu o poskytování licence informačního systému"),
                (7, "namítla nedostatek mezinárodní pravomoci (příslušnosti) českých soudů"),
                (9, "poskytnout žalované licenci k programu, tento program implementovat"),
                (12, "po své implementaci program vykazoval nedostatky"),
            ],
            "factual_facets": ["defective_software_delivery", "licence_fee_claim", "withdrawal_from_contract"],
            "legal_issue_facets": ["payment_despite_software_defects", "international_jurisdiction_objection"],
            "procedural_facets": ["civil_appeal"],
            "similarity_rationale": (
                "Primary match is the software-licence implementation dispute with unpaid fees and jurisdiction objection."
            ),
            "hard_negatives": [
                {
                    "document_id": "doc-cfa470876b0d5ed7",
                    "looks_similar_because": "Also a Prague commercial money claim after a failed business relationship.",
                    "materially_incorrect_because": "Promissory note/aval dispute, not software delivery and licence fees.",
                },
                {
                    "document_id": "doc-f4a701825747ed58",
                    "looks_similar_because": "Also a commercial/registry compliance dispute involving business entities.",
                    "materially_incorrect_because": "Trust-fund document filing penalty, not IT contract performance.",
                },
            ],
            "alternatives": [],
            "notes": "Reviewed pool #14.",
        },
        {
            "benchmark_id": "nalus-cs-pilot-015",
            "document_id": "doc-f4a701825747ed58",
            "query_style": "client_narrative",
            "difficulty": "medium",
            "query": (
                "Jsme správci svěřenského fondu a rejstříkový soud nám dal pořádkovou pokutu, "
                "protože jsme prý neuložili účetní závěrky a přehledy majetku do sbírky listin. "
                "My namítáme, že evidence svěřenských fondů není klasický veřejný rejstřík a že "
                "po nás chtějí něco, co zákon takto neukládá. Odvolací soud pokutu zrušil. "
                "Hledám podobná rozhodnutí o povinnostech svěřenských fondů ukládat listiny a "
                "o přezkumu pořádkových pokut."
            ),
            "evidence": [
                (2, "se neukládá pořádková pokuta ve výši 2 000 Kč"),
                (5, "jakožto svěřenským správcům"),
                (6, "do sbírky listin nezakládá zákonem vyžadované dokumenty"),
                (7, "není evidence svěřenských fondů veřejným rejstříkem"),
            ],
            "factual_facets": ["trust_fund_filing_duty", "registry_penalty"],
            "legal_issue_facets": ["application_of_public_register_rules_to_trust_funds", "order_penalty_review"],
            "procedural_facets": ["commercial_appeal"],
            "similarity_rationale": (
                "Matches the trust-fund registry penalty dispute about document filing duties."
            ),
            "hard_negatives": [
                {
                    "document_id": "doc-db9f10005638d155",
                    "looks_similar_because": "Also a commercial appeal about a monetary procedural sanction/consequence.",
                    "materially_incorrect_because": "Unpaid appeal fee stoppage and costs, not trust-fund document filing.",
                },
                {
                    "document_id": "doc-abd57ac0aa5dfe5b",
                    "looks_similar_because": "Also about a court-imposed payment deadline with access consequences.",
                    "materially_incorrect_because": "Court fee for cassation, not registry order penalty for a trust fund.",
                },
            ],
            "alternatives": [],
            "notes": "Reviewed pool #15.",
        },
        {
            "benchmark_id": "nalus-cs-pilot-016",
            "document_id": "doc-4f3c37d9c5a1afb7",
            "query_style": "noisy_client_narrative",
            "difficulty": "hard",
            "query": (
                "Jde o velkou trestní věc kolem fiktivních faktur, DPH a údajného uplácení. "
                "Obžalovaní z firem prý dávali do účetnictví doklady o dodávkách, které nebyly "
                "skutečné, a snažili se získat nadměrný odpočet. Část skutků řeší i spolupachatelství "
                "a podplácení. Odvolací soud rozsudek ruší nebo mění a část věci vrací soudu "
                "prvního stupně k novému rozhodnutí. Hledám podobné trestní odvolací rozsudky "
                "o daňových podvodech a úplatcích."
            ),
            "evidence": [
                (5, "uznán vinným zločinem podplácení"),
                (12, "s úmyslem zkrátit daň a vylákat nadměrný odpočet DPH"),
                (13, "fiktivní faktury a jiné účetní doklady"),
                (18, "vrací soudu prvního stupně, aby učinil nové rozhodnutí"),
            ],
            "factual_facets": ["fictitious_invoices", "vat_fraud", "bribery_allegations"],
            "legal_issue_facets": ["tax_fraud_liability", "bribery_co_perpetration"],
            "procedural_facets": ["criminal_appeal"],
            "similarity_rationale": (
                "Only reviewed criminal-appeal judgment covering VAT fraud via fictitious invoices and bribery."
            ),
            "hard_negatives": [
                {
                    "document_id": "doc-4fbdc1db957f44e7",
                    "looks_similar_because": (
                        "Criminal appellate judgment about bribery/corruption tied to public "
                        "contracts, with annulment and a fresh appellate decision "
                        "(preview blocks p:00002, p:00008, p:00010)."
                    ),
                    "materially_incorrect_because": (
                        "It is public-procurement advantage and bribery without the fictitious "
                        "VAT-invoice / tax-fraud scheme that defines the primary judgment."
                    ),
                },
                {
                    "document_id": "doc-68c126d146c84fa1",
                    "looks_similar_because": (
                        "Multi-defendant criminal appeal involving bribery connected to public "
                        "contracts and appellate annulment "
                        "(preview blocks p:00003, p:00012, p:00019, p:00022)."
                    ),
                    "materially_incorrect_because": (
                        "Facts centre on monthly kickbacks for awarded public contracts, not "
                        "fictitious invoices used to extract VAT over-refunds."
                    ),
                },
            ],
            "alternatives": [],
            "notes": (
                "Reviewed pool #16. Hard negatives are supplemental local criminal appeals "
                "from court_format_study raw_sources (outside the 20 reviewed primaries)."
            ),
        },
        {
            "benchmark_id": "nalus-cs-pilot-017",
            "document_id": "doc-84ae84698dfd0205",
            "query_style": "client_narrative",
            "difficulty": "hard",
            "query": (
                "Po rozvodu jsme nestihli dohodu o vypořádání společného jmění a já tvrdím, že "
                "podle zákonné fikce vzniklo podílové spoluvlastnictví k pozemkům. Katastr ale "
                "zápis neprovedl a soud žalobu na nahrazení rozhodnutí katastru zamítl. Namítám, "
                "že souhlasné prohlášení bylo jen deklarací faktického stavu, ne převodem. "
                "Hledám podobné spory o vklad do katastru po fikci vypořádání společného jmění "
                "a o tom, jak se má zapsat ideální polovina."
            ),
            "evidence": [
                (6, "nahrazeno rozhodnutí Katastrálního úřadu"),
                (7, "nastala uplynutím této doby fikce podle ustanovení § 741"),
                (8, "Souhlasné prohlášení žalobkyně a účastníka řízení 1/ nelze považovat za právní jednání"),
                (9, "existuje nesoulad mezi zapsaným a skutečným stavem"),
            ],
            "factual_facets": ["post_divorce_joint_property", "land_cadastre_registration"],
            "legal_issue_facets": ["section_741_fiction", "replacement_of_cadastre_decision"],
            "procedural_facets": ["civil_appeal", "part_five_civil_procedure"],
            "similarity_rationale": (
                "Matches the Part Five cadastre case about SJM fiction and ideal co-ownership of land."
            ),
            "hard_negatives": [
                {
                    "document_id": "doc-e5ac4b1fcd075062",
                    "looks_similar_because": "Also a land-ownership transfer dispute with survey/parcel issues.",
                    "materially_incorrect_because": "Pre-emption and phased costs, not SJM fiction/cadastre replacement.",
                },
                {
                    "document_id": "doc-dc644c7e6d827609",
                    "looks_similar_because": "Also seeks court replacement of an administrative land decision.",
                    "materially_incorrect_because": "Agricultural restitution legitimacy, not divorce SJM fiction.",
                },
            ],
            "alternatives": [],
            "notes": "Reviewed pool #17.",
        },
        {
            "benchmark_id": "nalus-cs-pilot-018",
            "document_id": "doc-dc644c7e6d827609",
            "query_style": "noisy_client_narrative",
            "difficulty": "hard",
            "query": (
                "Jsme řeholní řád a po státu chceme vydání pozemků podle zákona o majetkovém "
                "vyrovnání s církvemi a náboženskými společnostmi. Spor se táhne přes několik "
                "zrušujících rozhodnutí a klíčové je, jestli jsme oprávněnou osobou. Soudy řeší, "
                "jak nás po válce klasifikovaly orgány státu při konfiskaci a jestli to nárok "
                "na vydání blokuje. Potřebuju podobné rozsudky o církevních restitucích, "
                "oprávnění řeholního řádu a nahrazení rozhodnutí pozemkového úřadu."
            ),
            "evidence": [
                (5, "rozhodnutí Státního pozemkového úřadu"),
                (9, "zda je žalobce osobou aktivně legitimovanou"),
                (10, "v období bezprostředně po skončení 2. světové války"),
                (31, "registrovaná církev a náboženská společnost"),
            ],
            "factual_facets": [
                "church_property_restitution",
                "religious_order_entitlement",
                "postwar_confiscation_classification",
            ],
            "legal_issue_facets": [
                "church_property_settlement_entitlement",
                "replacement_of_land_office_decision",
            ],
            "procedural_facets": ["civil_appeal", "part_five_civil_procedure"],
            "similarity_rationale": (
                "Primary match is the church/religious-order property settlement dispute about "
                "entitled-person status, postwar confiscation classification and land-office decision replacement."
            ),
            "hard_negatives": [
                {
                    "document_id": "doc-84ae84698dfd0205",
                    "looks_similar_because": "Also Part Five litigation seeking replacement of an administrative land decision.",
                    "materially_incorrect_because": "Divorce SJM/cadastre fiction, not church property settlement entitlement.",
                },
                {
                    "document_id": "doc-e5ac4b1fcd075062",
                    "looks_similar_because": "Also a complex land-ownership dispute with repeated appellate history.",
                    "materially_incorrect_because": "Private pre-emption dispute, not church/religious-order restitution against the land office.",
                },
            ],
            "alternatives": [],
            "notes": "Reviewed pool #18.",
        },
        {
            "benchmark_id": "nalus-cs-pilot-019",
            "document_id": "doc-c7b72b0d6121d7f3",
            "query_style": "noisy_client_narrative",
            "difficulty": "medium",
            "query": (
                "V provozovně mě napadl muž, který tam brigádně zajišťoval ostrahu a pořádek; "
                "už za to byl i odsouzený. Já po něm chci odškodnění za zásah do osobnosti, ne "
                "jen klasickou škodu. Soudy ale říkají, že jednal v rámci práce pro provozovatele "
                "a že to nebyl exces, takže odpovědnost má jít jinam. Já s tím nesouhlasím. "
                "Hledám podobné spory o nemajetkovou újmu po napadení pracovníkem ostrahy."
            ),
            "evidence": [
                (5, "domáhal zaplacení nemajetkové újmy"),
                (6, "fyzicky napadl žalobce"),
                (7, "sledoval ochranu zaměstnankyně"),
                (8, "nárok na zaplacení náhrady nemajetkové újmy je odlišný od práva na náhradu škody"),
            ],
            "factual_facets": ["security_guard_assault", "non_pecuniary_personality_claim"],
            "legal_issue_facets": ["employee_excess_doctrine", "personality_rights_compensation"],
            "procedural_facets": ["civil_appeal"],
            "similarity_rationale": (
                "Matches the personality-rights claim after an on-duty security assault and excess analysis."
            ),
            "hard_negatives": [
                {
                    "document_id": "doc-0a90125eb71851b4",
                    "looks_similar_because": "Also a non-pecuniary damage claim for personal harm.",
                    "materially_incorrect_because": "State liability for court delay, not assault by a security worker.",
                },
                {
                    "document_id": "doc-4f3c37d9c5a1afb7",
                    "looks_similar_because": "Also involves harmful conduct and criminal wrongdoing vocabulary.",
                    "materially_incorrect_because": "Criminal tax/bribery appeal, not civil personality rights against a guard.",
                },
            ],
            "alternatives": [],
            "notes": "Reviewed pool #19.",
        },
        {
            "benchmark_id": "nalus-cs-pilot-020",
            "document_id": "doc-6cca3be81564e762",
            "query_style": "concise_case_description",
            "difficulty": "medium",
            "query": (
                "Jako zaměstnanec jsem pořizoval fotografie, které zaměstnavatel dál používá "
                "v knize, na webu a na sítích bez uvedení autorství. Domáhám se zákazu užití a "
                "omluvy. Soud žalobu zamítl s tím, že šlo o zaměstnanecké dílo. Já namítám, že "
                "focení nebylo v pracovní náplni, nebylo sjednáno ve smlouvě a nedostal jsem za "
                "to přiměřenou odměnu. Hledám podobné autorskoprávní spory o fotografie vytvořené "
                "zaměstnancem."
            ),
            "evidence": [
                (5, "zdržet se neoprávněného užívání fotografií"),
                (6, "napadl v odstavcích I. a II. výroku"),
                (8, "focení do pracovní náplně žalobce nespadalo"),
                (9, "za fotografie obdržel jakoukoli přiměřenou odměnu"),
            ],
            "factual_facets": ["employee_photographs", "uncredited_online_and_book_use"],
            "legal_issue_facets": ["employee_work_under_copyright_act", "adequate_remuneration_for_employee_work"],
            "procedural_facets": ["civil_appeal"],
            "similarity_rationale": (
                "Matches the employee-copyright dispute over photographs used without credit or remuneration."
            ),
            "hard_negatives": [
                {
                    "document_id": "doc-e6af147081ae754f",
                    "looks_similar_because": "Also a commercial dispute about digital content/system use and unpaid consideration.",
                    "materially_incorrect_because": "Software licence implementation fees, not employee photograph copyright.",
                },
                {
                    "document_id": "doc-c7b72b0d6121d7f3",
                    "looks_similar_because": "Also a civil claim about personal rights and non-pecuniary remedies.",
                    "materially_incorrect_because": "Personality rights after assault, not copyright authorship of photos.",
                },
            ],
            "alternatives": [],
            "notes": "Reviewed pool #20.",
        },
    ]


if __name__ == "__main__":
    raise SystemExit(main())
