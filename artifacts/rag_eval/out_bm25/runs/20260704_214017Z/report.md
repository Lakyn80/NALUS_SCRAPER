# RAG Embedding Benchmark Report

- Run ID: `20260704_214017Z`
- Dataset: `nalus-nsoud-pilot-v1`
- Preflight issues: `0`

## Winner

- Model: `bm25`
- Config: `bm25__bm25`
- Collection: `nalus_rag_eval__bm25__rag_eval__nalus_nsoud_pilot_v1__63119240e1`

### Metrics

- hit_rate: `1.0`
- evidence_marker_coverage: `1.0`
- recall_at_k: `1.0`
- mrr: `0.71875`
- forbidden_marker_rate: `0.0`

## Ranking

1. `bm25` (hit_rate=1.0, coverage=1.0, forbidden_rate=0.0)
2. `multilingual_e5_small` (hit_rate=1.0, coverage=1.0, forbidden_rate=0.0)

## Failed Models

- `bge_m3` [FAILED]: Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply when loading files with safetensors.
See the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434
- `multilingual_e5_base` [FAILED]: timed out
- `multilingual_e5_large` [FAILED]: timed out
- `paraphrase_multilingual_mpnet_base_v2` [FAILED]: timed out
