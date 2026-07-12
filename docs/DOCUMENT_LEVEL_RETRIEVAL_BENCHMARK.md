# Document-level retrieval benchmark

Status: offline evaluation framework.

## Scope

This benchmark evaluates retrieval quality only. It does not answer legal questions, generate legal advice, summarize decisions, call an LLM, modify Qdrant, change embeddings, change BM25, change RRF, or modify retrieval ranking.

The benchmark compares:

- chunk-level retrieval output;
- document-level retrieval output produced from a bounded candidate chunk pool.

The existing legal QA benchmark remains unchanged.

## Dataset format

Input is JSONL. Each line is one object:

```json
{
  "id": "doc-qa-001",
  "corpus": "usoud",
  "question": "Question text",
  "relevant_document_ids": [
    "ECLI:CZ:US:2026:1.TEST.1",
    "ECLI:CZ:US:2026:2.TEST.2"
  ],
  "legal_topic": "optional topic",
  "difficulty": "optional difficulty"
}
```

`relevant_document_ids` supports arbitrary counts, including zero, one, or many relevant documents. Duplicate identifiers are normalized and deduplicated deterministically.

## Metrics

Aggregate metrics include:

- chunk recall at 10, 20, 50, and 100;
- document recall at 10, 20, 50, and 100;
- precision at 10, 20, 50, and 100;
- unique document coverage;
- candidate pool coverage;
- duplicate rate;
- zero result rate;
- average retrieved documents;
- average candidate chunks;
- average retrieval latency;
- document aggregation latency.

Candidate pool coverage measures whether relevant documents appeared in the candidate chunks before document aggregation and threshold filtering.

Final document recall measures relevant documents after grouping, scoring, threshold filtering, and returned-document limits.

## Failure categories

Per-question failure categories are deterministic:

- `relevant_document_never_retrieved`
- `relevant_document_removed_by_aggregation`
- `relevant_document_removed_by_threshold`
- `relevant_document_removed_by_returned_document_limit`
- `duplicate_handling_issue`
- `metadata_issue`
- `unknown`

## Reports

The writer produces:

- `metrics.json`
- `summary.json`
- `per_question.jsonl`
- `per_question.csv`
- `summary.md`

Per-question reports use question ids and metrics. Raw query text is not written to `per_question.jsonl`.

Generated benchmark reports are local artifacts and should not be committed unless a task explicitly requires that exact report.

## Runner

Use:

```powershell
python scripts/run_document_retrieval_benchmark.py `
  --dataset path/to/document_benchmark.jsonl `
  --retrieval-only `
  --collection-name nalus_us_bge_m3_rag_combined_20260709 `
  --candidate-pool-size 200 `
  --max-returned-documents 100 `
  --max-supporting-chunks 3 `
  --document-threshold 0.0
```

Redis cache is intentionally rejected for benchmark runs.

## Observability

The existing `app.observability.eval_metrics_exporter` can expose benchmark summaries from:

`artifacts/rag_eval/legal_qa/document_retrieval_benchmark`

Prometheus labels remain bounded:

- `run_name`
- `corpus`

Do not expose raw query text, document ids, ECLI values, or other sensitive/high-cardinality values as Prometheus labels.

## Extension points

Future changes can add:

- new document scoring strategies;
- document-level reranking benchmarks;
- corpus-specific benchmark datasets;
- candidate-pool tuning reports;
- comparison dashboards using the existing Prometheus/Grafana stack.
