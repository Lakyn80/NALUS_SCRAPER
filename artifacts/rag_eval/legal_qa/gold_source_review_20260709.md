# Gold Source Review — 2026-07-09

Update: expanded on 2026-07-10 with additional conservative ÚS document gold and mixed corpus-only verification.

Human verification of top-3 retrieval hits from frozen baselines. Gold annotations apply **only** where rank-1 ECLI was stable in the baseline run or corpus routing was verified.

## Annotation policy

| Rule | Applied |
|------|---------|
| No invented spisová značka / case_reference | Yes — only ECLI from verified rank-1 |
| No invented decision_date | Yes |
| `source_pending=false` only after baseline review | Yes |
| Gold pass = keyword hit **and** source constraint in top-k | Yes (13 document-gold items after 2026-07-10 expansion) |
| Mixed corpus-only items | Keyword + `corpus_hit@k` only (no document gold) |

## ÚS — 10 document gold items

| ID | Rank-1 ECLI (baseline) | hit@1 (keyword) |
|----|------------------------|-----------------|
| usoud-qa-001 | ECLI:CZ:US:2026:3.US.3031.24.1 | true |
| usoud-qa-002 | ECLI:CZ:US:2026:4.US.2338.25.1 | true |
| usoud-qa-003 | ECLI:CZ:US:2023:1.US.3171.22.1 | true |
| usoud-qa-004 | ECLI:CZ:US:2023:1.US.631.23.1 | true |
| usoud-qa-007 | ECLI:CZ:US:2026:2.US.927.26.1 | true |
| usoud-qa-009 | ECLI:CZ:US:2026:1.US.2699.25.1 | true |
| usoud-qa-010 | ECLI:CZ:US:2023:3.US.714.23.1 | true |
| usoud-qa-011 | ECLI:CZ:US:2026:4.US.1079.26.1 | true |
| usoud-qa-012 | ECLI:CZ:US:2026:4.US.1065.26.1 | true |
| usoud-qa-015 | ECLI:CZ:US:2026:2.US.3645.25.1 | true |

## NSoud — 3 document gold items

| ID | Rank-1 ECLI (baseline) | hit@1 (keyword) |
|----|------------------------|-----------------|
| nsoud-qa-003 | ECLI:CZ:NS:2025:21.CDO.372.2024.1 | true |
| nsoud-qa-004 | ECLI:CZ:NS:2024:8.TDO.760.2024.1 | true |
| nsoud-qa-010 | ECLI:CZ:NS:2025:29.NSCR.1.2025.1 | true |

**Not annotated (rank-1 missing ECLI in payload):** nsoud-qa-001, 002, 005, 007 — remain `source_pending=true`.

## Mixed — 8 corpus-verified items

| ID | Verification | Document gold |
|----|--------------|---------------|
| mixed-qa-001 | `corpus_hit@3=true` in mixed baseline | No — `source_pending=false`, constraints empty |
| mixed-qa-002 | `corpus_hit@3=true` in mixed baseline | No — `source_pending=false`, constraints empty |
| mixed-qa-003 | `corpus_hit@3=true` in mixed baseline | No — `source_pending=false`, constraints empty |
| mixed-qa-005 | `corpus_hit@3=true` in mixed baseline | No — same |
| mixed-qa-006 | `corpus_hit@3=true` in mixed baseline | No — same |
| mixed-qa-007 | `corpus_hit@3=true` in mixed baseline | No — same |
| mixed-qa-008 | `corpus_hit@3=true` in mixed baseline | No — same |
| mixed-qa-009 | `corpus_hit@3=true` in mixed baseline | No — same |

Remaining 2 mixed items stay `source_pending=true`: `mixed-qa-004`, `mixed-qa-010` (`expected_target_corpus=ambiguous`).

## Deferred after 2026-07-10 review

- ÚS stayed pending where rank-1 snippet was too generic or not tightly aligned with the question:
  `usoud-qa-005`, `usoud-qa-006`, `usoud-qa-008`, `usoud-qa-013`, `usoud-qa-014`,
  `usoud-qa-016`, `usoud-qa-017`, `usoud-qa-018`, `usoud-qa-019`, `usoud-qa-020`
- NSoud pending remains blocked on missing clean rank-1 provenance payload in frozen artifacts:
  `nsoud-qa-001`, `nsoud-qa-002`, `nsoud-qa-005`, `nsoud-qa-006`, `nsoud-qa-007`,
  `nsoud-qa-008`, `nsoud-qa-009`

## Re-apply annotations

```powershell
python scripts/apply_gold_source_annotations.py
```

## Re-run gold eval

```powershell
# ÚS
docker compose exec -T api python scripts/run_legal_qa_benchmark.py `
  --dataset artifacts/rag_eval/legal_qa/datasets/usoud_qa_v1.jsonl `
  --collection-name nalus_us_bge_m3_rag_combined_20260709 `
  --top-k 10 --retrieval-only `
  --output-dir artifacts/rag_eval/legal_qa/runs/usoud_gold_eval `
  --qdrant-url http://qdrant:6333

# NSoud
docker compose exec -T api python scripts/run_legal_qa_benchmark.py `
  --dataset artifacts/rag_eval/legal_qa/datasets/nsoud_qa_v1.jsonl `
  --collection-name nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1 `
  --bm25-sidecar-path storage/rag/bm25/nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1.sqlite `
  --top-k 10 --retrieval-only `
  --output-dir artifacts/rag_eval/legal_qa/runs/nsoud_gold_eval `
  --qdrant-url http://qdrant:6333
```
