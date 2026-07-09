# Mixed retrieval benchmark summary

- Generated: 2026-07-09T17:24:34Z
- Dataset: `/app/artifacts/rag_eval/legal_qa/datasets/mixed_qa_v1.jsonl`
- Mode: two-pass mixed retrieval
- ÚS collection: `nalus_us_bge_m3_rag_combined_20260709`
- NSoud collection: `nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1`
- ÚS BM25 sidecar: `/app/storage/rag/bm25/nalus_us_bge_m3_rag_combined_20260709.sqlite`
- NSoud BM25 sidecar: `/app/storage/rag/bm25/nalus_client_lf__bge_m3__rag_eval__nalus_client_longform_v1__63119240e1.sqlite`
- Questions: 10
- Corpus-scored questions: 8
- Ambiguous questions: 2
- Source pending: 8
- Top-k: 10
- Retrieval only: True
- Redis cache: False

## Metrics

- corpus_hit@1: 0.000
- corpus_hit@3: 1.000
- corpus_hit@5: 1.000
- retrieval_hit@1: 1.000
- retrieval_hit@3: 1.000
- retrieval_hit@5: 1.000
- retrieval_hit@10: 1.000
- mean keyword coverage: 1.000
- pass rate: 1.000
- usoud_win_rate@1: 0.000
- nsoud_win_rate@1: 1.000


