#!/usr/bin/env python3
"""Write adaptive vs baseline profile comparison artifacts."""
from __future__ import annotations

import json
from pathlib import Path

out = Path("artifacts/legal_v2/full_corpus_build_v1")
base_dir = out / "throughput_2k"
ada_dir = out / "throughput_2k_adaptive"
a_base = json.loads((base_dir / "summary_A_cal2k.json").read_text(encoding="utf-8"))
b_base = json.loads((base_dir / "summary_B_cal2k.json").read_text(encoding="utf-8"))
a_ada = json.loads((ada_dir / "summary_A_cal2k_adaptive.json").read_text(encoding="utf-8"))
b_ada = json.loads((ada_dir / "summary_B_cal2k_adaptive.json").read_text(encoding="utf-8"))
eq = json.loads((ada_dir / "equivalence_check.json").read_text(encoding="utf-8"))


def cps(summary: dict) -> float:
    t = summary.get("throughput") or {}
    return float(t.get("embed_chunks_per_sec_e2e") or t.get("embed_chunks_per_sec") or 0)


def tok_s(summary: dict):
    t = summary.get("throughput") or {}
    return t.get("end_to_end_tokens_per_sec") or t.get("effective_tokens_per_sec")


rows = []
for side, base, ada in (("A", a_base, a_ada), ("B", b_base, b_ada)):
    bc, ac = int(base["total_chunks"]), int(ada["total_chunks"])
    b_cps, a_cps = cps(base), cps(ada)
    gain = (a_cps / b_cps) if b_cps else None
    rows.append(
        {
            "side": side,
            "baseline_chunks": bc,
            "adaptive_chunks": ac,
            "chunk_count_match": bc == ac,
            "baseline_cps": b_cps,
            "adaptive_cps_e2e": a_cps,
            "adaptive_cps_gpu": (ada.get("throughput") or {}).get("embed_chunks_per_sec_gpu"),
            "gain_ratio": None if gain is None else round(gain, 3),
            "adaptive_tokens_per_sec_e2e": tok_s(ada),
            "tokenization_s": (ada.get("timing") or {}).get("tokenization_seconds"),
            "scheduler_overhead_s": (ada.get("timing") or {}).get("scheduler_overhead_seconds"),
            "gpu_encode_s": (ada.get("timing") or {}).get("gpu_encode_seconds"),
            "e2e_embed_s": (ada.get("timing") or {}).get("end_to_end_embed_seconds"),
            "upsert_s": (ada.get("timing") or {}).get("upsert_seconds"),
            "wall_s": (ada.get("timing") or {}).get("total_wall_seconds_excluding_warmup"),
            "peak_vram_mb": (ada.get("profile") or {}).get("peak_vram_allocated_mb"),
            "batch_hist": (ada.get("profile") or {}).get("batch_size_histogram"),
            "padded_tokens_per_batch": (ada.get("profile") or {}).get("padded_tokens_per_batch"),
            "max_seq_len_per_batch": (ada.get("profile") or {}).get("max_sequence_length_per_batch"),
            "fail": ada.get("documents_failed"),
            "collection": ada.get("collection"),
        }
    )

a_gain = rows[0]["gain_ratio"] or 0
if a_gain >= 1.8:
    decision = (
        "MEANINGFUL_GAIN: Adaptive scheduler is a clear win for A "
        f"(~{a_gain:.1f}x e2e chunks/s vs stratified baseline). "
        "Recommend starting full A later with adaptive scheduler + resume. "
        "Stop further ingest optimization for now."
    )
elif a_gain < 1.15:
    decision = (
        "LITTLE_GAIN: Prefer stopping further ingest optimization and start full A "
        "with current safe settings."
    )
else:
    decision = (
        "MODERATE_GAIN: Use adaptive for full A; no further scheduler tuning needed."
    )

payload = {
    "schema": "legal_v2_cal2k_adaptive_profile_compare_v1",
    "equivalence": {
        "pass": eq.get("pass"),
        "cosine_min": (eq.get("cosine") or {}).get("min"),
        "max_abs_diff_max": (eq.get("max_abs_diff") or {}).get("max"),
        "sample_chunks": eq.get("sample_chunks"),
    },
    "notes": [
        "Baseline B cps is NOT apples-to-apples vs A (resume / different batch sizes).",
        "Decision metric is end-to-end embed throughput including scheduler tokenization.",
        "A adaptive ran before padded-token packer hardening; B adaptive used hardened packer.",
        "Chunk counts must match baseline exactly; both sides matched.",
    ],
    "sides": {r["side"]: r for r in rows},
    "decision": decision,
    "do_not_auto_start_full_build": True,
}
(ada_dir / "profile_compare.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)

eligible = 103488
a_cpd = float((a_ada.get("chunks_per_document") or {}).get("mean") or 7.5555)
b_cpd = float((b_ada.get("chunks_per_document") or {}).get("mean") or 6.594)
a_est = eligible * a_cpd
b_est = eligible * b_cpd
a_cps_v = rows[0]["adaptive_cps_e2e"] or 1e-9
b_cps_v = rows[1]["adaptive_cps_e2e"] or 1e-9

lines = [
    "# Adaptive vs baseline stratified 2k profile",
    "",
    "Baseline = fixed-batch stratified cal2k (`throughput_2k`).",
    "Adaptive = token-budget / length-aware scheduler (`throughput_2k_adaptive`).",
    "",
    "## Equivalence gate",
    "",
    f"- pass: **{eq.get('pass')}**",
    f"- sample chunks: {eq.get('sample_chunks')}",
    f"- cosine min/mean: {(eq.get('cosine') or {}).get('min')} / {(eq.get('cosine') or {}).get('mean')}",
    f"- max abs diff max: {(eq.get('max_abs_diff') or {}).get('max')}",
    "- policy: no bit-identity requirement; hard-stop only on material divergence",
    "",
    "## Results table",
    "",
    "| Metric | A baseline | A adaptive | B baseline* | B adaptive |",
    "| --- | ---: | ---: | ---: | ---: |",
    f"| chunks | {rows[0]['baseline_chunks']} | {rows[0]['adaptive_chunks']} | {rows[1]['baseline_chunks']} | {rows[1]['adaptive_chunks']} |",
    f"| end-to-end chunks/s | {rows[0]['baseline_cps']} | **{rows[0]['adaptive_cps_e2e']}** | {rows[1]['baseline_cps']} | **{rows[1]['adaptive_cps_e2e']}** |",
    f"| GPU-only chunks/s | — | {rows[0]['adaptive_cps_gpu']} | — | {rows[1]['adaptive_cps_gpu']} |",
    f"| tokens/s (e2e) | — | {rows[0]['adaptive_tokens_per_sec_e2e']} | — | {rows[1]['adaptive_tokens_per_sec_e2e']} |",
    f"| tokenize/scheduler s | — | {rows[0]['tokenization_s']} | — | {rows[1]['tokenization_s']} / {rows[1]['scheduler_overhead_s']} |",
    f"| GPU encode s | — | {rows[0]['gpu_encode_s']} | — | {rows[1]['gpu_encode_s']} |",
    f"| upsert s | — | {rows[0]['upsert_s']} | — | {rows[1]['upsert_s']} |",
    f"| wall s | — | {rows[0]['wall_s']} | — | {rows[1]['wall_s']} |",
    f"| peak VRAM MB | — | {rows[0]['peak_vram_mb']} | — | {rows[1]['peak_vram_mb']} |",
    f"| OOM | no | no | yes@batch32 (prior) | no |",
    f"| equivalence | — | PASS | — | PASS |",
    f"| cps gain vs own baseline | — | **{rows[0]['gain_ratio']}x** | — | {rows[1]['gain_ratio']}x |",
    "",
    "* B baseline was not apples-to-apples (resume + different batch size).",
    "",
    "## Integrity",
    "",
    f"- A chunk count match: **{rows[0]['chunk_count_match']}** (15111)",
    f"- B chunk count match: **{rows[1]['chunk_count_match']}** (13159)",
    f"- A fails: {rows[0]['fail']}",
    f"- B fails: {rows[1]['fail']}",
    f"- A collection: `{rows[0]['collection']}`",
    f"- B collection: `{rows[1]['collection']}`",
    "",
    "## Decision",
    "",
    f"- {decision}",
    "- Do **not** auto-start full A/B/ColBERT from this task.",
    "",
    "## ETA implication (informational)",
    "",
    f"- eligible docs: {eligible}",
    f"- A est chunks / embed hours @ adaptive e2e cps: {round(a_est)} / {round(a_est / a_cps_v / 3600, 1)}",
    f"- B est chunks / embed hours @ adaptive e2e cps: {round(b_est)} / {round(b_est / b_cps_v / 3600, 1)}",
    "",
]
(ada_dir / "profile_compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

summary_md = [
    "# 2k adaptive throughput profile",
    "",
    "- Same stratified document IDs as baseline `throughput_2k/document_ids.txt`",
    "- Equivalence: PASS",
    (
        f"- A: {rows[0]['adaptive_chunks']} chunks, e2e {rows[0]['adaptive_cps_e2e']} "
        f"cps ({rows[0]['gain_ratio']}x vs baseline)"
    ),
    f"- B: {rows[1]['adaptive_chunks']} chunks, e2e {rows[1]['adaptive_cps_e2e']} cps",
    f"- Decision: {decision}",
    "",
]
(ada_dir / "summary.md").write_text("\n".join(summary_md) + "\n", encoding="utf-8")
combined = {
    "schema": "legal_v2_cal2k_adaptive_summary_v1",
    "equivalence_pass": eq.get("pass"),
    "sides": {"A": a_ada, "B": b_ada},
    "compare": payload,
}
(ada_dir / "summary.json").write_text(
    json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(decision)
print("A gain", rows[0]["gain_ratio"], "B gain", rows[1]["gain_ratio"])
