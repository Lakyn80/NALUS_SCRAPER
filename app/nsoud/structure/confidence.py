from __future__ import annotations


def calculate_structure_confidence(structure: dict) -> dict[str, float | str | bool]:
    marker_flags = structure.get("marker_flags", {})
    score = 0.0
    if marker_flags.get("document_type"):
        score += 0.15
    if marker_flags.get("nejvyssi_soud_rozhodl"):
        score += 0.15
    if marker_flags.get("takto"):
        score += 0.15
    if marker_flags.get("oduvodneni"):
        score += 0.20
    if marker_flags.get("numbered_paragraphs"):
        score += 0.15
    if marker_flags.get("pouceni"):
        score += 0.10
    if marker_flags.get("closing"):
        score += 0.10

    confidence = round(score, 2)
    if confidence >= 0.85:
        status = "strong"
    elif confidence >= 0.65:
        status = "medium"
    else:
        status = "weak"

    return {
        "structure_confidence": confidence,
        "structure_status": status,
        "needs_review": confidence < 0.65,
    }
