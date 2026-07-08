#!/usr/bin/env python3
"""Build nalus_client_longform_eval_v1.json from pilot markers + client questions."""

from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
PILOT = json.loads((BASE / "nalus_eval.json").read_text(encoding="utf-8"))
OUT = BASE / "nalus_client_longform_eval_v1.json"

PILOT_BY_ID = {c["id"]: c for c in PILOT["cases"]}

CASES = [
    {
        "id": "client-longform-01",
        "pilot_id": "nsoud-positive-01",
        "question": (
            "Jsem vlastník bytu a druhá osoba v něm zůstala bydlet i po tom, co podle mě už neměla normální nájemní vztah. "
            "Tvrdí, že měla ještě nějaké právo v bytě bydlet, protože nebyla vyřešená bytová náhrada. "
            "Já ale řeším, jestli mi má platit částku odpovídající normálnímu nájemnému, nebo jen nějakou nižší úhradu. "
            "Soud nižší instance to posoudil tak, že dokud nebyla zajištěna bytová náhrada, nemůže jít o plné bezdůvodné obohacení. "
            "Potřebuji najít judikaturu, která řeší, kdy vzniká nárok vlastníka bytu na peněžní náhradu za další užívání bytu po skončení původního vztahu."
        ),
        "test_type": "client_longform",
    },
    {
        "id": "client-longform-02",
        "pilot_id": "nsoud-positive-02",
        "question": (
            "V trestní věci byl člověk odsouzen, ale tvrdí, že soudy vzaly důkazy úplně jednostranně. "
            "Podle něj z provedených důkazů vůbec nevyplývá skutkový závěr, na kterém stojí odsouzení. "
            "Některé důkazy podle něj nebyly použitelné nebo nebyly vůbec provedeny, i když mohly být důležité. "
            "Potřebuji najít rozhodnutí Nejvyššího soudu, kde se vysvětluje, kdy může dovolací soud zasáhnout kvůli extrémnímu nesouladu mezi skutkovými závěry a důkazy."
        ),
        "test_type": "client_longform",
    },
    {
        "id": "client-longform-03",
        "pilot_id": "nsoud-positive-03",
        "question": (
            "Obviněný nechce v dovolání jen znovu opakovat, že se skutek nestal jinak, ale tvrdí, že i kdyby soud vzal skutkový stav tak, jak je popsán, právně to bylo posouzeno špatně. "
            "Řeší se, jestli jednání vůbec naplňuje trestný čin nebo jestli byl uložen druh trestu, který není přiměřený právnímu posouzení. "
            "Potřebuji judikaturu, která vysvětluje hranici mezi námitkami proti skutku a námitkami proti právní kvalifikaci v dovolacím řízení."
        ),
        "test_type": "client_longform",
    },
    {
        "id": "client-longform-04",
        "pilot_id": "nsoud-positive-04",
        "question": (
            "Klient se odvolal proti rozsudku, ale odvolací soud jeho odvolání zamítl po věcném přezkoumání. "
            "Teď chce podat dovolání a tvrdí, že chyby vznikly už před soudem prvního stupně. "
            "Potřebuji najít rozhodnutí, která vysvětlují, jak se v dovolání argumentuje v situaci, kdy odvolací soud odvolání zamítl, ale dovolatel tvrdí, že už předchozí řízení mělo právní vadu. "
            "Zajímá mě hlavně vazba mezi rozhodnutím odvolacího soudu a dovolacími důvody vztahujícími se k předchozímu řízení."
        ),
        "test_type": "client_longform",
    },
    {
        "id": "client-longform-05",
        "pilot_id": "nsoud-positive-05",
        "question": (
            "V exekuční věci není jasné, který okresní soud má být místně příslušný. "
            "Povinný nemá dohledatelný pobyt v České republice, případně není jasné, kde se skutečně zdržuje nebo kde má majetek. "
            "Nižší soud nechce věc řešit, protože neví, podle čeho určit místní příslušnost. "
            "Potřebuji najít judikaturu Nejvyššího soudu k tomu, co se dělá, když české soudy mají věc projednat, ale místní příslušnost nejde běžně určit."
        ),
        "test_type": "client_longform",
    },
    {
        "id": "client-longform-06",
        "pilot_id": "nsoud-positive-06",
        "question": (
            "Klient byl v konfliktu, kde druhá osoba bouchala do auta, bránila mu odjet a vyzývala ho ke rvačce. "
            "Klient pak z auta vystoupil a došlo k fyzickému střetu. Soudy to posoudily jako běžné vzájemné napadání, ale klient tvrdí, že se jen bránil trvajícímu útoku. "
            "Potřebuji najít rozhodnutí Nejvyššího soudu, které rozlišuje, kdy ještě jde o vzájemnou potyčku a kdy může jít o obranu proti útoku."
        ),
        "test_type": "client_longform",
    },
    {
        "id": "client-longform-07",
        "pilot_id": "nsoud-positive-07",
        "question": (
            "Klient prohrál dovolací řízení a Nejvyšší soud dovolání odmítl. "
            "Teď řeší, jestli mu může být uloženo zaplatit druhé straně náklady dovolacího řízení a jak soud takový výrok typicky formuluje. "
            "Nejde mi o samotnou hmotněprávní otázku sporu, ale o procesní výsledek dovolacího řízení a náklady po odmítnutí dovolání. "
            "Potřebuji najít podobná rozhodnutí, kde Nejvyšší soud rozhodoval o nákladech po odmítnutí dovolání."
        ),
        "test_type": "client_longform",
    },
    {
        "id": "client-longform-08",
        "pilot_id": "nsoud-positive-08",
        "question": (
            "Klient koupil rodinný dům a po koupi zjistil vady, o kterých tvrdí, že při podpisu smlouvy nebyly zjevné. "
            "Mluví o vlhkosti, plísních, špatném zdivu, vadných rozvodech nebo dalších problémech domu. "
            "Chce po prodávajícím peníze zpět jako snížení ceny, protože dům podle něj neměl vlastnosti, které při koupi očekával. "
            "Potřebuji najít judikaturu Nejvyššího soudu k odpovědnosti prodávajícího za vady nemovitosti a ke slevě z kupní ceny."
        ),
        "test_type": "client_longform",
    },
]


def main() -> None:
    cases_out = []
    for spec in CASES:
        pilot = PILOT_BY_ID[spec["pilot_id"]]
        cases_out.append(
            {
                "id": spec["id"],
                "question": spec["question"],
                "expected_answer_type": pilot["expected_answer_type"],
                "test_type": pilot["test_type"],
                "source_scope": pilot["source_scope"],
                "required_evidence": pilot["required_evidence"],
                "minimum_coverage": pilot["minimum_coverage"],
                "allow_partial": pilot["allow_partial"],
                "expected_citation_count_min": pilot["expected_citation_count_min"],
                "difficulty": "hard",
                "language": "cs",
                "expected_long_context": True,
                "minimum_context_chars": 200,
                "pilot_case_id": spec["pilot_id"],
            }
        )

    payload = {
        "dataset_id": "nalus-client-longform-v1",
        "name": "NALUS NSOud Client Long-Form Retrieval Eval v1",
        "description": (
            "Long-form client-style legal problem descriptions for semantic retrieval evaluation. "
            "Questions describe facts and conflicts in natural language without ECLI or benchmark markers."
        ),
        "project_name": "NALUS Scraper",
        "metadata": {
            "court": "Nejvyšší soud ČR",
            "batch": "2025_01_03",
            "chunking_strategy": "document_section_aware",
            "eval_style": "client_longform",
            "pilot_dataset_id": "nalus-nsoud-pilot-v1",
            "question_style": "client_narrative",
        },
        "cases": cases_out,
        "source_documents": PILOT["source_documents"],
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(cases_out)} cases)")


if __name__ == "__main__":
    main()
