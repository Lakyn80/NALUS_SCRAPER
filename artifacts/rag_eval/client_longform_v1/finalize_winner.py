#!/usr/bin/env python3
"""Export winner QA and run legal quality evaluation for client long-form benchmark."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from rag_eval.adapters.factory import build_backend
from rag_eval.config import load_benchmark_config

BASE = Path(__file__).resolve().parent
RAG_EVAL = BASE.parent
sys.path.insert(0, str(RAG_EVAL))

from legal_quality_eval import evaluate_winner_export, render_legal_quality_report

DATASET_PATH = RAG_EVAL / "nalus_client_longform_eval_v1.json"
RANKING_PATH = BASE / "ranking.json"
OUT_QA_JSON = BASE / "winner_qa.json"
OUT_QA_MD = BASE / "winner_qa.md"
OUT_LEGAL_JSON = BASE / "winner_legal_eval.json"
OUT_LEGAL_MD = BASE / "legal_quality_report.md"

CONFIG_BY_CONFIG_ID = {
    "multilingual_e5_small__dense": BASE / "configs" / "dense_e5_small.yaml",
    "bm25__bm25": BASE / "configs" / "bm25_only.yaml",
    "multilingual_e5_small__dense_plus_bm25": BASE / "configs" / "hybrid_e5_small.yaml",
    "multilingual_e5_base__dense_plus_bm25": BASE / "configs" / "hybrid_e5_base.yaml",
    "multilingual_e5_large__dense_plus_bm25": BASE / "configs" / "hybrid_e5_large.yaml",
    "paraphrase_multilingual_mpnet_base_v2__dense_plus_bm25": BASE / "configs" / "hybrid_mpnet.yaml",
    "bge_m3__dense_plus_bm25": BASE / "configs" / "hybrid_bge_m3.yaml",
}


def _marker_hit(text: str, marker: str, aliases: list[str]) -> bool:
    haystack = text.casefold()
    needles = [marker, *aliases]
    return any(needle and needle.casefold() in haystack for needle in needles)


def _preview(text: str, limit: int = 600) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def export_winner_qa() -> Path:
    merged = json.loads(RANKING_PATH.read_text(encoding="utf-8"))
    winner = merged["winner"]
    config_id = winner["config_id"]
    config_path = CONFIG_BY_CONFIG_ID[config_id]
    config = load_benchmark_config(config_path)
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    collection_name = winner["collection_name"]
    model_code = winner["model_code"]
    retrieval_mode = winner["retrieval_mode"]

    backend = build_backend(config)
    if retrieval_mode in {"dense", "dense_plus_bm25"}:
        backend.embed_source(source_id=config.source_id, model_code=model_code)

    cases_out: list[dict] = []
    for case in dataset["cases"]:
        question = case["question"]
        markers = case.get("required_evidence") or []
        response = backend.retrieve(
            profile_id=config.profile_id,
            source_id=config.source_id,
            query=question,
            model_code=model_code,
            collection_name=collection_name,
            top_k=config.top_k,
            retrieval_mode=retrieval_mode,
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
        "eval_style": "client_longform",
        "winner_config_id": config_id,
        "model_code": model_code,
        "collection_name": collection_name,
        "retrieval_mode": retrieval_mode,
        "metrics": {
            "hit_rate": winner.get("hit_rate"),
            "recall_at_k": winner.get("recall_at_k"),
            "mrr": winner.get("mrr"),
            "evidence_marker_coverage": winner.get("evidence_marker_coverage"),
            "missing_expected_marker_count": winner.get("missing_expected_marker_count"),
            "false_positive_count": winner.get("false_positive_count"),
        },
        "case_count": len(cases_out),
        "cases": cases_out,
    }
    OUT_QA_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# NALUS Client Long-Form — Winner Q&A Export",
        "",
        f"- Winner: `{config_id}`",
        f"- Collection: `{collection_name}`",
        f"- Cases: `{len(cases_out)}`",
        "",
    ]
    for index, item in enumerate(cases_out, start=1):
        lines.extend(
            [
                f"## {index}. {item['case_id']}",
                "",
                f"**Otázka:** {item['question'][:500]}{'...' if len(item['question']) > 500 else ''}",
                "",
                f"**Odpověď (top-1):** {item['top_answer_preview'] or '(žádný výsledek)'}",
                "",
            ]
        )
        if item.get("top_document_id"):
            lines.append(f"- document_id: `{item['top_document_id']}`")
        lines.append("")

    OUT_QA_MD.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {OUT_QA_JSON}")
    return OUT_QA_JSON


def run_legal_eval(qa_path: Path) -> None:
    payload = evaluate_winner_export(winner_qa_path=qa_path, dataset_path=DATASET_PATH)
    payload["eval_style"] = "client_longform"
    OUT_LEGAL_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = render_legal_quality_report(payload)
    report = report.replace(
        "# NALUS RAG Eval — Legal Quality Report",
        "# NALUS Client Long-Form — Legal Quality Report",
    )
    report = report.replace(
        "pilotních otázek",
        "client-style long-form otázek",
    )
    OUT_LEGAL_MD.write_text(report, encoding="utf-8")
    print(f"Wrote {OUT_LEGAL_JSON}")
    print(f"Wrote {OUT_LEGAL_MD}")


def main() -> None:
    qa_path = export_winner_qa()
    run_legal_eval(qa_path)


if __name__ == "__main__":
    main()
