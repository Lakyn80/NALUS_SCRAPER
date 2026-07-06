#!/usr/bin/env python3
"""Export per-question retrieval results for the benchmark winner (bge_m3 hybrid)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from rag_eval.adapters.factory import build_backend
from rag_eval.config import load_benchmark_config

BASE = Path(__file__).resolve().parent
CONFIG_PATH = BASE / "rerun" / "bge_m3_hybrid.yaml"
DATASET_PATH = BASE / "nalus_eval.json"
RANKING_PATH = BASE / "out_rerun_bge_m3" / "ranking.json"
OUT_JSON = BASE / "out_combined" / "winner_bge_m3_qa.json"
OUT_MD = BASE / "out_combined" / "winner_bge_m3_qa.md"


def _marker_hit(text: str, marker: str, aliases: list[str]) -> bool:
    haystack = text.casefold()
    needles = [marker, *aliases]
    return any(needle and needle.casefold() in haystack for needle in needles)


def _preview(text: str, limit: int = 600) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def main() -> None:
    config = load_benchmark_config(CONFIG_PATH)
    ranking = json.loads(RANKING_PATH.read_text(encoding="utf-8"))
    winner = ranking["winner"]
    collection_name = winner["collection_name"]
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    backend = build_backend(config)
    backend.embed_source(source_id=config.source_id, model_code="bge_m3")

    cases_out: list[dict] = []
    for case in dataset["cases"]:
        question = case["question"]
        markers = case.get("required_evidence") or []
        response = backend.retrieve(
            profile_id=config.profile_id,
            source_id=config.source_id,
            query=question,
            model_code="bge_m3",
            collection_name=collection_name,
            top_k=config.top_k,
            retrieval_mode="dense_plus_bm25",
        )

        hits = []
        for rank, result in enumerate(response.results, start=1):
            meta = result.payload_metadata or {}
            document_id = meta.get("document_id") or meta.get("source_document_id")
            hit_markers = []
            for item in markers:
                marker = item.get("marker", "")
                aliases = item.get("aliases") or []
                if _marker_hit(result.text, marker, aliases):
                    hit_markers.append(marker)
            hits.append(
                {
                    "rank": rank,
                    "score": result.score,
                    "chunk_id": result.chunk_id,
                    "document_id": document_id,
                    "matched_markers": hit_markers,
                    "text_preview": _preview(result.text),
                    "text": result.text,
                }
            )

        top = hits[0] if hits else None
        cases_out.append(
            {
                "case_id": case["id"],
                "question": question,
                "expected_markers": [m.get("marker") for m in markers],
                "top_answer_preview": top["text_preview"] if top else None,
                "top_answer": top["text"] if top else None,
                "top_document_id": top.get("document_id") if top else None,
                "top_matched_markers": top.get("matched_markers") if top else [],
                "retrieval_hits": hits,
            }
        )

    payload = {
        "exported_at": datetime.now(UTC).isoformat(),
        "winner_config_id": winner["config_id"],
        "model_code": winner["model_code"],
        "collection_name": collection_name,
        "retrieval_mode": "dense_plus_bm25",
        "run_id": ranking.get("run_id"),
        "metrics": winner.get("metrics"),
        "case_count": len(cases_out),
        "cases": cases_out,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# NALUS RAG Eval — bge_m3 hybrid Q&A export",
        "",
        f"- Winner: `{winner['config_id']}`",
        f"- Collection: `{collection_name}`",
        f"- Run: `{ranking.get('run_id')}`",
        f"- Cases: `{len(cases_out)}`",
        "",
        "Poznámka: „Odpověď“ = text nejlepšího retrieved chunku (top-1), ne LLM generace.",
        "",
    ]
    for index, item in enumerate(cases_out, start=1):
        lines.extend(
            [
                f"## {index}. {item['case_id']}",
                "",
                f"**Otázka:** {item['question']}",
                "",
                f"**Odpověď (top-1 chunk):** {item['top_answer_preview'] or '(žádný výsledek)'}",
                "",
            ]
        )
        if item.get("top_document_id"):
            lines.append(f"- document_id: `{item['top_document_id']}`")
        if item.get("top_matched_markers"):
            lines.append(f"- matched markers: `{', '.join(item['top_matched_markers'])}`")
        lines.append("")

    OUT_MD.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
