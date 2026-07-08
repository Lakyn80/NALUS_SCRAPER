from __future__ import annotations

from app.rag.clarification.models import (
    AmbiguityType,
    LegalDomain,
    ProcedureStage,
    RuleAssessment,
)
from app.rag.clarification.text_utils import contains_any, count_matches, simplify_text

CRIMINAL_DOMAIN_TERMS = (
    "trestni",
    "trestni rad",
    "trestniho radu",
    "tr r",
    "obvineny",
    "obzalovany",
    "odsouzeny",
    "obhajce",
    "trestny cin",
    "tdo",
)
CIVIL_DOMAIN_TERMS = (
    "o s r",
    "obcansky soudni rad",
    "civilni",
    "zalobce",
    "zalovany",
    "navrhovatel",
    "odporce",
    "cdo",
    "navrh",
)
EXECUTION_DOMAIN_TERMS = (
    "exekuc",
    "exekutor",
    "povinny",
    "opravneny",
    "nd",
)
FAMILY_DOMAIN_TERMS = (
    "rodinny",
    "pecovatelska",
    "vychova",
    "rozvod",
    "manzel",
    "vyzivne",
    "alimenty",
)
ADMIN_DOMAIN_TERMS = (
    "spravni",
    "spravni soud",
    "spravni zaloba",
    "us",
)

PROCEDURE_STAGE_TERMS = {
    "first_instance": ("soud prvniho stupne", "prvostupnov", "nalozac", "okresni soud", "krajsky soud"),
    "appeal": ("odvolani", "odvolaci soud", "odvolac"),
    "dovolani": ("dovolani", "dovolac", "nsoud"),
    "execution": ("exekuc", "vykon rozhodnuti", "exekutor"),
}

ROLE_CRIMINAL_TERMS = ("obvineny", "obzalovany", "odsouzeny", "poskozeny")
ROLE_CIVIL_TERMS = ("zalobce", "zalovany", "navrhovatel", "odporce", "ucastnik")
ROLE_EXECUTION_TERMS = ("povinny", "opravneny", "dluznik", "veritel")

REMEDY_TERMS = (
    "dovolani",
    "odvolani",
    "zaloba",
    "navrh",
    "exekuc",
    "odskodneni",
    "naklady rizeni",
    "zruseni rozhodnuti",
    "navrat veci",
)

JURISDICTION_TERMS = (
    "nejvyssi soud",
    "ustavni soud",
    "nsoud",
    "usoud",
    "cdo",
    "tdo",
    "nd",
)

GENERIC_PROCEDURE_TERMS = (
    "odvolani",
    "dovolani",
    "soud prvniho stupne",
    "odvolaci soud",
    "rozsudek",
    "rozhodnuti",
)


def _detect_legal_domain(query: str) -> tuple[LegalDomain, float]:
    family_specific_hits = count_matches(query, ("vyzivne", "alimenty", "pecovatelska", "vychova"))
    if family_specific_hits >= 1:
        return "family", min(0.95, 0.68 + family_specific_hits * 0.1)

    scores = {
        "criminal": count_matches(query, CRIMINAL_DOMAIN_TERMS),
        "civil": count_matches(query, CIVIL_DOMAIN_TERMS),
        "execution": count_matches(query, EXECUTION_DOMAIN_TERMS),
        "family": count_matches(query, FAMILY_DOMAIN_TERMS),
        "administrative": count_matches(query, ADMIN_DOMAIN_TERMS),
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_domain, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    if top_score == 0:
        return "unknown", 0.2
    if top_score == second_score and top_score > 0:
        return "unknown", 0.35
    confidence = min(0.95, 0.55 + top_score * 0.15)
    return top_domain, confidence  # type: ignore[return-value]


def _detect_procedure_stage(query: str) -> tuple[ProcedureStage, float]:
    if contains_any(
        query,
        (
            "podat dovolani",
            "chce podat dovolani",
            "teď chce podat dovolani",
            "ted chce podat dovolani",
            "dovolani k nejvyssimu soudu",
            "dovolatel",
        ),
    ):
        return "dovolani", 0.9

    scores = {
        stage: count_matches(query, terms)
        for stage, terms in PROCEDURE_STAGE_TERMS.items()
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_stage, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    if top_score == 0:
        return "unknown", 0.25
    if top_score == second_score and top_score > 0:
        return "unknown", 0.4
    confidence = min(0.95, 0.5 + top_score * 0.2)
    return top_stage, confidence  # type: ignore[return-value]


def _detect_role_confidence(query: str) -> float:
    role_hits = (
        count_matches(query, ROLE_CRIMINAL_TERMS)
        + count_matches(query, ROLE_CIVIL_TERMS)
        + count_matches(query, ROLE_EXECUTION_TERMS)
    )
    if role_hits >= 1:
        return min(0.9, 0.6 + role_hits * 0.1)
    if contains_any(query, ("klient", "muj klient", "moje klientka")):
        return 0.45
    return 0.25


def _detect_remedy_confidence(query: str) -> float:
    remedy_hits = count_matches(query, REMEDY_TERMS)
    if remedy_hits >= 2:
        return min(0.9, 0.55 + remedy_hits * 0.12)
    if remedy_hits == 1:
        return 0.55
    return 0.2


def _detect_jurisdiction_confidence(query: str) -> float:
    hits = count_matches(query, JURISDICTION_TERMS)
    if hits >= 1:
        return min(0.9, 0.6 + hits * 0.1)
    return 0.3


def _build_query_signature(
    *,
    query: str,
    ambiguity_types: list[AmbiguityType],
    detected_procedure_stage: ProcedureStage,
) -> str:
    if "legal_domain_ambiguous" in ambiguity_types and (
        detected_procedure_stage == "dovolani"
        or contains_any(query, ("dovolani", "odvolani", "odvolaci soud"))
    ):
        return "appeal_dovolani_previous_proceeding_ambiguous"
    if "legal_domain_ambiguous" in ambiguity_types:
        return "legal_domain_under_specified"
    if "procedure_stage_ambiguous" in ambiguity_types:
        return "procedure_stage_under_specified"
    if "retrieval_domain_mismatch" in ambiguity_types:
        return "retrieval_domain_mismatch"
    return "general_legal_ambiguity"


def assess_query_rules(query: str) -> RuleAssessment:
    detected_legal_domain, domain_confidence = _detect_legal_domain(query)
    detected_procedure_stage, procedure_confidence = _detect_procedure_stage(query)
    role_confidence = _detect_role_confidence(query)
    remedy_confidence = _detect_remedy_confidence(query)
    jurisdiction_confidence = _detect_jurisdiction_confidence(query)

    ambiguity_types: list[AmbiguityType] = []
    missing_slots: list[str] = []
    reasons: list[str] = []

    has_generic_procedure = contains_any(query, GENERIC_PROCEDURE_TERMS)
    has_domain_anchor = domain_confidence >= 0.6 and detected_legal_domain != "unknown"

    if has_generic_procedure and not has_domain_anchor:
        ambiguity_types.append("legal_domain_ambiguous")
        missing_slots.append("legal_domain")
        reasons.append(
            "Dotaz popisuje procesní stádium, ale neříká jasně, zda jde o trestní, civilní nebo jinou větev práva."
        )

    if count_matches(query, ("navrh", "odvolani")) >= 2:
        if "remedy_ambiguous" not in ambiguity_types:
            ambiguity_types.append("remedy_ambiguous")
            missing_slots.append("remedy_type")
            reasons.append("Není jasné, zda má jít o návrh, odvolání, nebo jiný procesní krok.")

    if has_generic_procedure and procedure_confidence < 0.55:
        ambiguity_types.append("procedure_stage_ambiguous")
        missing_slots.append("procedure_stage")
        reasons.append("Není zřejmé, v jakém procesním stadiu se věc nachází.")

    if role_confidence < 0.5 and contains_any(query, ("klient", "osoba", "strana")):
        ambiguity_types.append("role_ambiguous")
        missing_slots.append("party_role")
        reasons.append("Není jasná procesní role klienta.")

    if remedy_confidence < 0.5 and contains_any(query, ("potrebuji najit", "judikatur", "rozhodnuti")):
        ambiguity_types.append("remedy_ambiguous")
        missing_slots.append("remedy_type")
        reasons.append("Není jasné, jaký procesní cíl nebo opravný prostředek má být podložen judikaturou.")

    if jurisdiction_confidence < 0.45 and has_generic_procedure and detected_legal_domain == "unknown":
        ambiguity_types.append("jurisdiction_or_court_ambiguous")
        missing_slots.append("court_branch")
        reasons.append("Není jasné, která větev judikatury Nejvyššího soudu je relevantní.")

    query_signature = _build_query_signature(
        query=query,
        ambiguity_types=ambiguity_types,
        detected_procedure_stage=detected_procedure_stage,
    )
    reason_cs = " ".join(reasons) if reasons else "Dotaz obsahuje dostatečné právní ukotvení pro bezpečné vyhledávání."

    return RuleAssessment(
        domain_confidence=domain_confidence,
        procedure_confidence=procedure_confidence,
        role_confidence=role_confidence,
        remedy_confidence=remedy_confidence,
        jurisdiction_confidence=jurisdiction_confidence,
        detected_legal_domain=detected_legal_domain,
        detected_procedure_stage=detected_procedure_stage,
        ambiguity_types=tuple(ambiguity_types),
        missing_slots=tuple(missing_slots),
        query_signature=query_signature,
        reason_cs=reason_cs,
    )


def should_ask_clarification(assessment: RuleAssessment) -> bool:
    if "legal_domain_ambiguous" in assessment.ambiguity_types:
        return True
    if "remedy_ambiguous" in assessment.ambiguity_types:
        return True
    blocking_types = {
        "procedure_stage_ambiguous",
        "remedy_ambiguous",
        "jurisdiction_or_court_ambiguous",
    }
    blocking_hits = [item for item in assessment.ambiguity_types if item in blocking_types]
    if len(blocking_hits) >= 2:
        return True
    if (
        "procedure_stage_ambiguous" in assessment.ambiguity_types
        and assessment.procedure_confidence < 0.45
    ):
        return True
    return False


def normalized_query_signature(query: str) -> str:
    return simplify_text(query)
