#!/usr/bin/env python3
"""Build the tracked retrieval-golden v1 pilot JSONL from development corpus blocks.

Evidence-first: every positive query is authored against a concrete canonical block.
No LLM / provider calls. Offline only.
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

from app.rag.legal_v2.benchmark.corpus import (  # noqa: E402
    load_development_corpus,
    rank_blocks_by_token_overlap,
)
from app.rag.legal_v2.benchmark.retrieval_golden import (  # noqa: E402
    InspectedNegativeCandidate,
    RetrievalGoldenItem,
    evidence_excerpt_in_block,
    validate_retrieval_golden_dataset,
    write_jsonl,
)

DEFAULT_OUT = PROJECT_ROOT / "benchmarks" / "legal_v2" / "retrieval_golden_v1_pilot.jsonl"
DEFAULT_REPORT = (
    PROJECT_ROOT / "artifacts" / "legal_v2" / "retrieval_golden_v1_pilot" / "build_report.json"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    corpus = load_development_corpus()
    by_doc_index = {
        (block.document_id, block.block_index): block for block in corpus.blocks_by_id.values()
    }

    def block(document_id: str, index: int):
        key = (document_id, index)
        if key not in by_doc_index:
            raise KeyError(f"missing block {document_id}#{index}")
        return by_doc_index[key]

    def find_containing(document_id: str, needle: str):
        for candidate in corpus.blocks_for_document(document_id):
            if evidence_excerpt_in_block(needle, candidate.raw_text):
                return candidate
        raise KeyError(f"no block in {document_id} contains excerpt")

    # --- Positive query specs: (query_id, doc_id, block_index_or_None, excerpt, query, ...)
    # When block_index is None, locate by excerpt.
    positives_raw: list[dict[str, Any]] = [
        # doc-b73 short CC — 3
        {
            "query_id": "nalus-rg-pilot-001",
            "document_id": "doc-b73cac9b3dfc8a42",
            "block_index": 7,
            "excerpt": (
                "podáním, které nesplňovalo náležitosti řádného návrhu na zahájení řízení "
                "před Ústavním soudem dané zákonem č. 182/1993 Sb., o Ústavním soudu, ve znění "
                "pozdějších předpisů (dále jen \"zákon o Ústavním soudu\"). Jelikož je z lustra "
                "patrné, že v minulosti byli navrhovatelé opakovaně informováni o zákonných "
                "náležitostech návrhu a postupu jak jeho nedostatky odstranit, považuje Ústavní "
                "soud za nadbytečné činit tak opakovaně a vyzývat je k odstranění vad"
            ),
            "query": (
                "Proč Ústavní soud odmítl návrh manželů Houžvičkových bez výzvy k odstranění vad?"
            ),
            "query_type": "court_reasoning",
            "difficulty": "medium",
            "legal_area": "constitutional_procedure",
            "court": "constitutional_court",
            "hard_negative_indexes": [4],
            "grounding_note": "Grounded in short-order reasoning on formal defects of the filing.",
        },
        {
            "query_id": "nalus-rg-pilot-002",
            "document_id": "doc-b73cac9b3dfc8a42",
            "block_index": 7,
            "excerpt": "jejich podání bez dalšího odmítá podle § 43 odst. 1 písm. a) zákona o Ústavním soudu.",
            "query": (
                "Podle kterého ustanovení zákona o Ústavním soudu lze odmítnout vadný návrh "
                "bez další výzvy, pokud navrhovatelé už byli opakovaně poučeni?"
            ),
            "query_type": "legal_rule",
            "difficulty": "easy",
            "legal_area": "constitutional_procedure",
            "court": "constitutional_court",
            "hard_negative_indexes": [4],
            "grounding_note": "Same block, distinct statutory-focus question; not a near duplicate of 001.",
        },
        {
            "query_id": "nalus-rg-pilot-003",
            "document_id": "doc-b73cac9b3dfc8a42",
            "block_index": None,
            "excerpt": "Poučení: Proti rozhodnutí Ústavního soudu není odvolání přípustné.",
            "query": "Mohu se odvolat proti rozhodnutí Ústavního soudu v této věci?",
            "query_type": "operative_outcome",
            "difficulty": "easy",
            "legal_area": "constitutional_procedure",
            "court": "constitutional_court",
            "hard_negative_indexes": [7],
            "grounding_note": "Instruction/poučení block stating appeal is inadmissible.",
        },
        # doc-a529 roman CC — 5
        {
            "query_id": "nalus-rg-pilot-004",
            "document_id": "doc-a5292901931de05a",
            "block_index": 8,
            "excerpt": (
                "Chlapec byl svěřen do péče matce a otci byl upraven styk s ním na každý druhý "
                "víkend od pátku do neděle."
            ),
            "query": (
                "Jaká byla předchozí úprava péče a styku schválená Okresním soudem v Opavě "
                "v říjnu 2023?"
            ),
            "query_type": "fact_specific",
            "difficulty": "easy",
            "legal_area": "family_law",
            "court": "constitutional_court",
            "hard_negative_indexes": [10],
            "grounding_note": "Fact block describing prior custody/contact arrangement.",
        },
        {
            "query_id": "nalus-rg-pilot-005",
            "document_id": "doc-a5292901931de05a",
            "block_index": 10,
            "excerpt": (
                "Otec v dubnu 2026 podal návrh na vydání prozatímního rozhodnutí, jímž by soud "
                "upravil péči otce od čtvrtka do neděle každý druhý týden a dále upravil režim "
                "péče o letních prázdninách."
            ),
            "query": (
                "O jakou prozatímní úpravu péče požádal otec v dubnu 2026 v opatrovnické věci?"
            ),
            "query_type": "procedural",
            "difficulty": "easy",
            "legal_area": "family_law",
            "court": "constitutional_court",
            "hard_negative_indexes": [8],
            "grounding_note": "Procedural fact: content of the provisional-decision motion.",
        },
        {
            "query_id": "nalus-rg-pilot-006",
            "document_id": "doc-a5292901931de05a",
            "block_index": 15,
            "excerpt": (
                "Dočasná úprava je vždy určitou výjimkou z pravidla, která se uplatní pouze "
                "v nezbytných případech - je-li zde naléhavá potřeba upravit poměry před "
                "rozhodnutím ve věci samé."
            ),
            "query": (
                "Kdy podle § 465a a násl. z. ř. s. může soud upravit poměry dítěte jen "
                "prozatímně, a ne meritorně?"
            ),
            "query_type": "legal_rule",
            "difficulty": "hard",
            "legal_area": "family_law",
            "court": "constitutional_court",
            "hard_negative_indexes": [17],
            "grounding_note": "Legal framework for provisional decisions after 1 Jan 2026.",
        },
        {
            "query_id": "nalus-rg-pilot-007",
            "document_id": "doc-a5292901931de05a",
            "block_index": 18,
            "excerpt": (
                "V zamítnutí návrhu na vydání prozatímního rozhodnutí nelze spatřovat ani "
                "zásah do rovnosti rodičů a práva otce a chlapce na vzájemný kontakt."
            ),
            "query": (
                "Znamená zamítnutí návrhu na prozatímní rozhodnutí automaticky zásah do "
                "rovnosti rodičů a práva na kontakt s dítětem?"
            ),
            "query_type": "concept_distinction",
            "difficulty": "medium",
            "legal_area": "family_law",
            "court": "constitutional_court",
            "hard_negative_indexes": [15],
            "grounding_note": "Court distinguishes denial of provisional relief from equality/contact rights.",
        },
        {
            "query_id": "nalus-rg-pilot-008",
            "document_id": "doc-a5292901931de05a",
            "block_index": 19,
            "excerpt": (
                "Ústavní stížnost tak odmítl mimo ústní jednání bez přítomnosti účastníků "
                "jako návrh zjevně neopodstatněný [§ 43 odst. 2 písm. a) zákona o Ústavním soudu]."
            ),
            "query": (
                "Jak Ústavní soud naložil s ústavní stížností otce proti usnesení o "
                "prozatímním rozhodnutí?"
            ),
            "query_type": "operative_outcome",
            "difficulty": "easy",
            "legal_area": "constitutional_procedure",
            "court": "constitutional_court",
            "accepted_alternative_indexes": [5],
            "hard_negative_indexes": [10],
            "grounding_note": (
                "Primary: detailed rejection reasoning under § 43(2)(a). "
                "Accepted alternative: operative disposition block stating "
                "'Ústavní stížnost se odmítá.' "
                "Hard negative: father's provisional-decision motion content, not the disposition."
            ),
        },
        # doc-abd57 judgment — 5
        {
            "query_id": "nalus-rg-pilot-009",
            "document_id": "doc-abd57ac0aa5dfe5b",
            "block_index": 12,
            "excerpt": (
                "Nejvyšší správní soud usnesením ze dne 3. 2. 2026 č. j. 9 As 13/2026-11 "
                "stěžovatele vyzval k zaplacení soudního poplatku za kasační stížnost ve výši "
                "5 000 Kč, a to ve lhůtě 5 dnů od doručení usnesení."
            ),
            "query": (
                "Na jakou lhůtu Nejvyšší správní soud zkrátil výzvu k zaplacení soudního "
                "poplatku za kasační stížnost ve věci shromáždění?"
            ),
            "query_type": "fact_specific",
            "difficulty": "easy",
            "legal_area": "administrative_procedure",
            "court": "constitutional_court",
            "hard_negative_indexes": [14],
            "grounding_note": "Concrete fee-deadline fact from the procedural history.",
        },
        {
            "query_id": "nalus-rg-pilot-010",
            "document_id": "doc-abd57ac0aa5dfe5b",
            "block_index": 14,
            "excerpt": (
                "podle nějž může soud lhůtu k zaplacení soudního poplatku ve smyslu § 9 odst. 1 "
                "zákona o soudních poplatcích stanovit kratší pouze ve výjimečných případech"
            ),
            "query": (
                "Jak stěžovatel s odkazem na nález Pl. ÚS 9/20 vyložil možnost zkrátit "
                "zákonnou lhůtu k zaplacení soudního poplatku?"
            ),
            "query_type": "legal_rule",
            "difficulty": "medium",
            "legal_area": "court_fees",
            "court": "constitutional_court",
            "hard_negative_indexes": [12],
            "grounding_note": (
                "Grounded in the complainant's argument section: this is the complainant's "
                "characterization of Pl. ÚS 9/20, not an unattributed holding of the current "
                "decision."
            ),
        },
        {
            "query_id": "nalus-rg-pilot-011",
            "document_id": "doc-abd57ac0aa5dfe5b",
            "block_index": 24,
            "excerpt": (
                "Ústavní soud není součástí soudní soustavy a nepřísluší mu proto ani právo "
                "vykonávat dohled nad rozhodovací činností obecných soudů. Do rozhodovací "
                "činnosti obecných soudů je oprávněn zasáhnout pouze tehdy, došlo-li jejich "
                "pravomocným rozhodnutím v řízení, jehož byl stěžovatel účastníkem, k porušení "
                "jeho základních práv či svobod chráněných ústavním pořádkem."
            ),
            "query": (
                "Kdy může Ústavní soud zasáhnout do rozhodování obecných soudů a kdy nikoli?"
            ),
            "query_type": "court_reasoning",
            "difficulty": "medium",
            "legal_area": "constitutional_procedure",
            "court": "constitutional_court",
            "hard_negative_indexes": [33],
            "grounding_note": "Classic US role/limit reasoning block.",
        },
        {
            "query_id": "nalus-rg-pilot-012",
            "document_id": "doc-abd57ac0aa5dfe5b",
            "block_index": 33,
            "excerpt": (
                "Vlastní naplnění práva shromažďovacího přitom nelze chápat jen jako "
                "individuální právo člověka, ale také jako konstitutivní prvek demokratického "
                "státního uspořádání"
            ),
            "query": (
                "Proč je podle Ústavního soudu právo shromažďovací nejen individuálním právem?"
            ),
            "query_type": "court_reasoning",
            "difficulty": "medium",
            "legal_area": "assembly_rights",
            "court": "constitutional_court",
            "hard_negative_indexes": [24],
            "grounding_note": "Constitutional characterization of assembly rights.",
        },
        {
            "query_id": "nalus-rg-pilot-013",
            "document_id": "doc-abd57ac0aa5dfe5b",
            "block_index": 36,
            "excerpt": (
                "Přestože měl k posouzení ať již předpokladů řízení o kasační stížnosti či "
                "samotnému věcnému posouzení stížnosti časový prostor v řádu týdnů, Nejvyšší "
                "správní soud uložil stěžovateli splnění poplatkové povinnosti na samé hranici "
                "ústavnosti. K tomu poskytnuté poměrně lakonické a obecné odůvodnění směřující "
                "v podstatě pouze k naléhavosti posuzované věci přitom přímo odporuje požadavkům "
                "obsaženým ve výše citované vlastní judikatuře Nejvyššího správního soudu, které "
                "vycházejí z ústavně zaručeného práva na přístup k soudu a soudní ochranu. "
                "Navíc nelze přehlížet okolnost plynoucí ze samotného vyjádření Nejvyššího "
                "správního soudu k ústavní stížnosti, že totiž stěžovatel se svojí správní "
                "žalobou domáhal určení, že - zjednodušeně řečeno - postup žalovaného v reakci "
                "na pozdější oznámení \"konkurujícího\" shromáždění znamenal zásah do práv "
                "stěžovatele. Stěžovatel se domáhal pouze deklaratorního rozhodnutí a věcné "
                "rozhodnutí Nejvyššího správního soudu by tudíž ke kolizi dvou shromáždění na "
                "stejném místě a ve stejném čase vést nemohlo."
            ),
            "query": (
                "Proč Ústavní soud kritizoval zkrácení lhůty k zaplacení poplatku Nejvyšším "
                "správním soudem jako na hranici ústavnosti?"
            ),
            "query_type": "client_paraphrase",
            "difficulty": "hard",
            "legal_area": "court_fees",
            "court": "constitutional_court",
            "hard_negative_indexes": [12],
            "grounding_note": "Holding-style critique of fee-deadline shortening.",
        },
        # Prague Co — 4
        {
            "query_id": "nalus-rg-pilot-014",
            "document_id": "doc-e6af147081ae754f",
            "block_index": 5,
            "excerpt": (
                "uložil žalované povinnost zaplatit žalobkyni 527 750 Kč spolu se smluvním "
                "úrokem z prodlení ve výši 0,1 % denně"
            ),
            "query": (
                "Kolik měla žalovaná podle rozsudku soudu prvního stupně zaplatit žalobkyni "
                "a s jakým smluvním úrokem z prodlení?"
            ),
            "query_type": "operative_outcome",
            "difficulty": "easy",
            "legal_area": "civil_commercial",
            "court": "high_court_prague",
            "hard_negative_indexes": [6],
            "grounding_note": "First-instance operative award amounts.",
        },
        {
            "query_id": "nalus-rg-pilot-015",
            "document_id": "doc-e6af147081ae754f",
            "block_index": 6,
            "excerpt": (
                "uzavřela se žalovanou smlouvu o poskytování licence informačního systému "
                "Anonymizováno (dále jen „program“) a analytických služeb"
            ),
            "query": (
                "O jakou smlouvu se v řízení před Vrchním soudem v Praze vedl spor mezi "
                "žalobkyní a žalovanou?"
            ),
            "query_type": "fact_specific",
            "difficulty": "easy",
            "legal_area": "civil_commercial",
            "court": "high_court_prague",
            "hard_negative_indexes": [5],
            "grounding_note": "Contract subject-matter facts.",
        },
        {
            "query_id": "nalus-rg-pilot-016",
            "document_id": "doc-e6af147081ae754f",
            "block_index": 7,
            "excerpt": (
                "Žalovaná ve své obraně předně namítla nedostatek mezinárodní pravomoci "
                "(příslušnosti) českých soudů"
            ),
            "query": "Jakou procesní námitku žalovaná vznesla ohledně příslušnosti českých soudů?",
            "query_type": "procedural",
            "difficulty": "easy",
            "legal_area": "civil_procedure",
            "court": "high_court_prague",
            "hard_negative_indexes": [10],
            "grounding_note": "International jurisdiction objection.",
        },
        {
            "query_id": "nalus-rg-pilot-017",
            "document_id": "doc-e6af147081ae754f",
            "block_index": 15,
            "excerpt": (
                "soud dovodil, že žalobkyni vzniklo právo na zaplacení této částky z titulu "
                "smluvní pokuty, byť předmětné ujednání nebylo jako smluvní pokuta výslovně označeno"
            ),
            "query": (
                "Může být ujednání o finančním vypořádání za licenci posouzeno jako smluvní "
                "pokuta, i když tak není výslovně nazváno?"
            ),
            "query_type": "concept_distinction",
            "difficulty": "hard",
            "legal_area": "contract_law",
            "court": "high_court_prague",
            "hard_negative_indexes": [10],
            "grounding_note": "Characterization of contractual settlement as contractual penalty.",
        },
        # Prague Cmo — 4
        {
            "query_id": "nalus-rg-pilot-018",
            "document_id": "doc-db9f10005638d155",
            "block_index": 5,
            "excerpt": (
                "Napadeným usnesením zastavil soud prvního stupně řízení o odvoláních žalovaných"
            ),
            "query": (
                "Co rozhodl soud prvního stupně o odvoláních žalovaných v obchodní věci "
                "2 Cmo 32/2026?"
            ),
            "query_type": "operative_outcome",
            "difficulty": "easy",
            "legal_area": "commercial_procedure",
            "court": "high_court_prague",
            "hard_negative_indexes": [7],
            "grounding_note": "First-instance order stopping appeals.",
        },
        {
            "query_id": "nalus-rg-pilot-019",
            "document_id": "doc-db9f10005638d155",
            "block_index": 13,
            "excerpt": (
                "Ohledně výzvy k zaplacení soudního poplatku v dodatečné lhůtě podle § 9 odst. 1 "
                "zákona o soudních poplatcích není zákonem stanoveno, že má být doručována do "
                "vlastních rukou adresátů."
            ),
            "query": (
                "Musí být výzva k zaplacení soudního poplatku v dodatečné lhůtě doručována "
                "do vlastních rukou podle § 49 o. s. ř.?"
            ),
            "query_type": "legal_rule",
            "difficulty": "medium",
            "legal_area": "court_fees",
            "court": "high_court_prague",
            "hard_negative_indexes": [10],
            "grounding_note": "Delivery-rule distinction for fee payment notices.",
        },
        {
            "query_id": "nalus-rg-pilot-020",
            "document_id": "doc-db9f10005638d155",
            "block_index": 7,
            "excerpt": (
                "Soud žalobkyni přiznal náhradu nákladů právního zastoupení, a to za jeden "
                "hlavní úkon právní služby (vyjádření k odvolání) ve výši 12 460 Kč, jednu "
                "paušální náhradu výdajů 300 Kč a daň z přidané hodnoty z těchto částek ve výši "
                "2 679,60 Kč. To je celkem soudem přiznaných 15 439,60 Kč."
            ),
            "query": (
                "Jak soud prvního stupně vyčíslil náhradu nákladů právního zastoupení "
                "žalobkyně v odvolacím řízení?"
            ),
            "query_type": "fact_specific",
            "difficulty": "easy",
            "legal_area": "costs",
            "court": "high_court_prague",
            "hard_negative_indexes": [5],
            "grounding_note": "Cost award breakdown.",
        },
        {
            "query_id": "nalus-rg-pilot-021",
            "document_id": "doc-db9f10005638d155",
            "block_index": 16,
            "excerpt": (
                "platby, které soudu prvního stupně došly, se uskutečnily až po uplynutí soudem "
                "stanovené lhůty k úhradě, to je 15 dnů od doručení výzvy k zaplacení, a tedy "
                "v případě druhého žalovaného do 28. 5. 2025, a proto k nim nelze podle § 9 "
                "odst. 1, poslední věta, zákona o soudních poplatcích přihlížet. Významné je i "
                "to, že žádnou z plateb, které došly soudu prvního stupně dne 6. 6. 2025 nelze "
                "vůbec nijak přičíst druhému žalovanému."
            ),
            "query": (
                "Proč odvolací soud dovodil, že soudní poplatek druhého žalovaného nebyl "
                "řádně uhrazen?"
            ),
            "query_type": "court_reasoning",
            "difficulty": "medium",
            "legal_area": "court_fees",
            "court": "high_court_prague",
            "hard_negative_indexes": [13],
            "grounding_note": "Reasoning that late/unattributable payments do not cure the fee defect.",
        },
        # Olomouc civil — 4
        {
            "query_id": "nalus-rg-pilot-022",
            "document_id": "doc-84ae84698dfd0205",
            "block_index": 6,
            "excerpt": (
                "zamítl Krajský soud v Ostravě výrokem I. žalobu, podle níž se žalobkyně "
                "domáhala vydání rozhodnutí, jímž bude nahrazeno rozhodnutí Katastrálního úřadu"
            ),
            "query": (
                "Jak rozhodl krajský soud o žalobě na nahrazení rozhodnutí katastrálního úřadu "
                "o vkladu vlastnického práva k pozemkům?"
            ),
            "query_type": "operative_outcome",
            "difficulty": "easy",
            "legal_area": "property_cadastre",
            "court": "high_court_olomouc",
            "hard_negative_indexes": [7],
            "grounding_note": "First-instance dismissal of cadastre-replacement claim.",
        },
        {
            "query_id": "nalus-rg-pilot-023",
            "document_id": "doc-84ae84698dfd0205",
            "block_index": 7,
            "excerpt": (
                "Protože do tří let po rozvodu manželství nedošlo k uzavření dohody o "
                "vypořádání SJM, nastala uplynutím této doby fikce podle ustanovení § 741 "
                "písm. b/ z.č. 89/2012 Sb., občanského zákoníku. Pozemky se tedy staly "
                "podílovým spoluvlastnictvím a žalobkyně je vlastnicí jedné jejich ideální "
                "poloviny"
            ),
            "query": (
                "Co podle žalobkyně nastalo ohledně SJM a pozemků uplynutím tří let po rozvodu "
                "bez dohody o vypořádání?"
            ),
            "query_type": "client_paraphrase",
            "difficulty": "medium",
            "legal_area": "family_property",
            "court": "high_court_olomouc",
            "hard_negative_indexes": [24],
            "grounding_note": "Plaintiff theory of § 741 fiction; hard negative is court dictum on same theme.",
        },
        {
            "query_id": "nalus-rg-pilot-024",
            "document_id": "doc-84ae84698dfd0205",
            "block_index": 19,
            "excerpt": (
                "soud v řízení podle páté části o.s.ř. nepřezkoumává správnost rozhodnutí "
                "správního orgánu, ale rozhoduje v plné jurisdikci o návrhu uplatněném "
                "v řízení před správním orgánem"
            ),
            "query": (
                "V čem spočívá zvláštnost řízení podle páté části o. s. ř. oproti přezkumu "
                "správního rozhodnutí?"
            ),
            "query_type": "legal_rule",
            "difficulty": "hard",
            "legal_area": "civil_procedure",
            "court": "high_court_olomouc",
            "hard_negative_indexes": [6],
            "grounding_note": (
                "Part-five OSŘ full-jurisdiction principle from p:00019 only; "
                "case-application block p:00020 is not a sufficient alternative."
            ),
        },
        {
            "query_id": "nalus-rg-pilot-025",
            "document_id": "doc-84ae84698dfd0205",
            "block_index": 24,
            "excerpt": (
                "vypořádání společného jmění se nepovažuje za nakládání s nemovitostí, jak "
                "vyplývá např. z rozhodnutí vydaného pod sp.zn. 22 Cdo 2526/2016"
            ),
            "query": (
                "Považuje se vypořádání společného jmění manželů za nakládání s nemovitostí "
                "podle judikatury Nejvyššího soudu citované odvolacím soudem?"
            ),
            "query_type": "concept_distinction",
            "difficulty": "medium",
            "legal_area": "family_property",
            "court": "high_court_olomouc",
            "hard_negative_indexes": [7],
            "grounding_note": "NS doctrine: SJM settlement is not disposition of real estate.",
        },
        # Olomouc criminal — 4
        {
            "query_id": "nalus-rg-pilot-026",
            "document_id": "doc-4f3c37d9c5a1afb7",
            "block_index": 12,
            "excerpt": (
                "tím, že zahrnul do účetnictví společnosti právnická osoba právnická osoba "
                "fiktivní faktury a daňové doklady, vystavené právnická osoba s. r. o. za nákup "
                "hovězích usní, vzápětí nechal vystavit pro právnická osoba s. r. o. a právnická "
                "osoba se sídlem ve Slovenské republice faktury na stejné komodity hovězích usní "
                "a podílel se tak na simulaci fiktivních obchodních vztahů v řetězcích obchodních "
                "společností, ale také na simulaci údajných přeprav těchto komodit na Slovensko "
                "a v úmyslu vyvolat dojem skutečného provádění těchto obchodů se podílel na "
                "simulování plateb, formou údajných vkladů za právnická osoba s. r. o. a "
                "právnická osoba, a to v hotovosti na účet společnosti právnická osoba a rovněž "
                "se podílel na simulaci dopravy zboží, údajně dodaného těmto společnostem na "
                "Slovensko, ačkoliv věděl, že společnosti právnická osoba zboží fakticky dodáno "
                "nebylo"
            ),
            "query": (
                "Čím se podle skutkové věty měl předseda představenstva podílet na zkrácení "
                "DPH a vylákání nadměrného odpočtu?"
            ),
            "query_type": "fact_specific",
            "difficulty": "medium",
            "legal_area": "tax_crime",
            "court": "high_court_olomouc",
            "hard_negative_indexes": [14],
            "grounding_note": "Count-1 factual description of VAT carousel/fictitious invoices.",
        },
        {
            "query_id": "nalus-rg-pilot-027",
            "document_id": "doc-4f3c37d9c5a1afb7",
            "block_index": 20,
            "excerpt": (
                "Napadeným rozsudkem Krajského soudu v Brně ze dne 17.6.2021, č.j. 40 T 6/2017-4200, "
                "byli uznáni vinnými"
            ),
            "query": (
                "Jaký rozsudek krajského soudu je napaden v trestním odvolacím řízení "
                "6 TO 80/2021?"
            ),
            "query_type": "procedural",
            "difficulty": "easy",
            "legal_area": "criminal_procedure",
            "court": "high_court_olomouc",
            "hard_negative_indexes": [26],
            "grounding_note": "Identifies the challenged first-instance criminal judgment.",
        },
        {
            "query_id": "nalus-rg-pilot-028",
            "document_id": "doc-4f3c37d9c5a1afb7",
            "block_index": 26,
            "excerpt": (
                "Na druhé straně se však státní zástupce neztotožňuje se závěry soudu prvního "
                "stupně, pokud jde o právní kvalifikaci obžalovaných příjmení, příjmení a "
                "příjmení, jejichž jednání nalézací soud kvalifikoval jako pomoc dle § 24 "
                "odst. 1 tr. zákoníku a nikoliv jako spolupachatelství dle § 23 tr. zákoníku. "
                "V dané souvislosti poukázal státní zástupce na popis jednání obžalovaných "
                "příjmení, příjmení a příjmení, jímž byli uznání vinnými. V souvislosti s "
                "jednáním posledně jmenovaných obžalovaných zdůraznil, že ve smyslu § 23 tr. "
                "zákoníku platí, že pokud byl trestný čin spáchán úmyslným společným jednáním "
                "dvou nebo více osob, odpovídá každá z nich, jako by trestný čin spáchala sama "
                "(spolupachatelé). Za spolupachatelství, jež ze své podstaty předpokládá "
                "společné úmyslné jednání více osob, je třeba považovat i variantu, kdy "
                "spolupachatelé vykonávají určitou činnost, která teprve jako celek tvoří "
                "jednání vyžadované příslušnou skutkovou podstatou. Jinými slovy spolupachatelé "
                "si rozdělí jednání předvídané příslušnou skutkovou podstatou tak, že teprve "
                "jednáním ostatních může dojít ke spáchání trestného činu."
            ),
            "query": (
                "Proč státní zástupce napadl právní kvalifikaci části obžalovaných jako "
                "pouhou pomoc místo spolupachatelství?"
            ),
            "query_type": "court_reasoning",
            "difficulty": "hard",
            "legal_area": "criminal_law",
            "court": "high_court_olomouc",
            "hard_negative_indexes": [12],
            "grounding_note": "Prosecutor appeal distinguishing aid vs co-perpetration.",
        },
        {
            "query_id": "nalus-rg-pilot-029",
            "document_id": "doc-4f3c37d9c5a1afb7",
            "block_index": 14,
            "excerpt": (
                "vystavoval za tuto společnost fiktivní faktury na dodávky hovězích usní "
                "společnosti anonymizováno & anonymizováno s. r. o., ačkoliv věděl, že "
                "právnická osoba s. r. o. fakticky žádnou činnost nevykonávala, neboť její "
                "jednatel jméno příjmení byl pouze nastrčenou osobou, která neměla o činnosti "
                "této firmy žádné povědomí, obžalovaný sám podával za právnická osoba s. r. o. "
                "daňová přiznání, do nichž zahrnul pouze část dokladů vystavených pro "
                "společnost anonymizováno & anonymizováno s. r. o. a kromě toho se podílel na "
                "vystavování fiktivních dokladů za dopravu hovězích usní do skladů právnická "
                "osoba s. r. o., vybíral finanční prostředky v hotovosti z účtů právnická osoba "
                "a celkově tak poskytl obžalovaným jméno příjmení a jméno příjmení pomoc k "
                "tomu, aby společnost právnická osoba v měsíčních přiznáních k DPH za "
                "zdaňovací období leden až prosince 2011 neoprávněně uplatnila nárok na "
                "nadměrný odpočet DPH"
            ),
            "query": (
                "Jakou roli měl obžalovaný, který fakticky ovládal společnost vystavující "
                "fiktivní faktury na hovězí usně?"
            ),
            "query_type": "client_paraphrase",
            "difficulty": "medium",
            "legal_area": "tax_crime",
            "court": "high_court_olomouc",
            "hard_negative_indexes": [12],
            "grounding_note": "Distinct defendant role in the invoice chain (count 3).",
        },
    ]

    items: list[RetrievalGoldenItem] = []
    for spec in positives_raw:
        document_id = spec["document_id"]
        excerpt = spec["excerpt"]
        if spec["block_index"] is None:
            primary = find_containing(document_id, excerpt)
        else:
            primary = block(document_id, int(spec["block_index"]))
        if not evidence_excerpt_in_block(excerpt, primary.raw_text):
            raise AssertionError(f"{spec['query_id']}: excerpt not in primary block {primary.block_id}")

        alt_ids: list[str] = []
        for idx in spec.get("accepted_alternative_indexes") or []:
            alt = block(document_id, int(idx))
            alt_ids.append(alt.block_id)
        if spec["query_id"] == "nalus-rg-pilot-008":
            # Resolve operative disposition by exact text, not by caption/header.
            operative = find_containing(document_id, "Ústavní stížnost se odmítá.")
            if "takto:" in operative.raw_text and operative.raw_text.strip().endswith("takto:"):
                raise AssertionError("008 alternative resolved to caption/header, not disposition")
            if "Ústavní stížnost se odmítá." not in operative.raw_text:
                raise AssertionError("008 alternative missing operative disposition text")
            alt_ids = [operative.block_id]
        hard_ids: list[str] = []
        for idx in spec.get("hard_negative_indexes") or []:
            hard = block(document_id, int(idx))
            if hard.block_id == primary.block_id:
                raise AssertionError(f"{spec['query_id']}: hard negative equals primary")
            hard_ids.append(hard.block_id)

        items.append(
            RetrievalGoldenItem(
                query_id=spec["query_id"],
                query=spec["query"],
                split="development",
                is_negative=False,
                query_type=spec["query_type"],
                difficulty=spec["difficulty"],
                court=spec["court"],
                jurisdiction="CZ",
                legal_area=spec["legal_area"],
                source_document_id=document_id,
                expected_document_ids=[document_id],
                primary_expected_block_id=primary.block_id,
                expected_block_ids=[primary.block_id],
                accepted_alternative_block_ids=alt_ids,
                hard_negative_block_ids=hard_ids,
                evidence_excerpt=excerpt,
                grounding_note=spec["grounding_note"],
                negative_rationale=None,
                inspected_negative_candidates=[],
            )
        )

    negative_query = (
        "Za jakých podmínek lze podle nařízení Dublin III předat žadatele o mezinárodní "
        "ochranu z České republiky do Maďarska, pokud tvrdí systémové nedostatky azylového "
        "řízení a přijímacích podmínek?"
    )
    ranked = rank_blocks_by_token_overlap(
        negative_query,
        list(corpus.blocks_by_id.values()),
        top_k=8,
    )
    # Manual rejection reasons after inspecting top candidates in this session.
    # Ranked blocks were mostly long criminal-appeal passages (token overlap on
    # generic Czech procedural words), plus one family provisional-decision block.
    inspected: list[InspectedNegativeCandidate] = []
    for rank, (candidate, score) in enumerate(ranked[:5], start=1):
        preview = " ".join(candidate.raw_text.split())[:160]
        if "Dublin" in candidate.raw_text or "azyl" in candidate.raw_text.casefold():
            reason = "Unexpected asylum-related text; re-check before accepting as negative."
        elif candidate.document_id.endswith("afb7"):
            reason = (
                f"Criminal tax-fraud appeal block ({candidate.primary_class}); discusses "
                f"domestic Czech criminal liability/procedure, not Dublin III transfer to Hungary. "
                f"Preview: {preview}"
            )
        elif candidate.document_id.endswith("de05a"):
            reason = (
                "Family/provisional-decision constitutional reasoning under z. ř. s.; "
                "no asylum transfer criteria or Hungary reception conditions."
            )
        else:
            reason = (
                f"Development-corpus block ({candidate.primary_class}) lacks any Dublin III / "
                f"international-protection transfer holding. Preview: {preview}"
            )
        inspected.append(
            InspectedNegativeCandidate(
                document_id=candidate.document_id,
                block_id=candidate.block_id,
                rank=rank,
                overlap_score=round(float(score), 4),
                rejection_reason=reason,
            )
        )

    items.append(
        RetrievalGoldenItem(
            query_id="nalus-rg-pilot-030",
            query=negative_query,
            split="development",
            is_negative=True,
            query_type="corpus_negative",
            difficulty="hard",
            court=None,
            jurisdiction="CZ",
            legal_area="asylum_dublin",
            source_document_id=None,
            expected_document_ids=[],
            primary_expected_block_id=None,
            expected_block_ids=[],
            accepted_alternative_block_ids=[],
            hard_negative_block_ids=[],
            evidence_excerpt=None,
            grounding_note=None,
            negative_rationale=(
                "Legally plausible Dublin III / Hungary transfer question. Offline token-overlap "
                "scan over all development canonical blocks found only unrelated Czech "
                "constitutional, civil, commercial, or tax-crime passages. Manual inspection of "
                "the top candidates confirmed none states transfer conditions, systemic asylum "
                "deficiencies, or Hungary reception standards. Therefore the pilot corpus cannot "
                "answer the query."
            ),
            inspected_negative_candidates=inspected,
        )
    )

    report = validate_retrieval_golden_dataset(
        items,
        blocks_by_id=corpus.blocks_by_id,
        dataset_path=str(args.output),
    )
    if not report.ok:
        raise SystemExit(
            "validation failed:\n"
            + "\n".join(f"- {issue.code}: {issue.message}" for issue in report.issues)
        )

    write_jsonl(args.output, items)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "ok",
        "output": str(args.output),
        "item_count": report.item_count,
        "positive_count": report.positive_count,
        "negative_count": report.negative_count,
        "development_documents": [ref.document_id for ref in corpus.documents],
        "negative_query_id": "nalus-rg-pilot-030",
        "negative_top_candidates": [c.model_dump() for c in inspected],
        "by_document": _count_by(items, "source_document_id"),
        "by_court": _count_by(items, "court"),
        "by_query_type": _count_by(items, "query_type"),
        "by_difficulty": _count_by(items, "difficulty"),
    }
    args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _count_by(items: list[RetrievalGoldenItem], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = getattr(item, field)
        key = str(value) if value is not None else "null"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


if __name__ == "__main__":
    raise SystemExit(main())
