# Legal Retrieval v2

Status: implemented as an isolated, disabled-by-default pipeline. It is not production traffic.

## Runtime Boundary

- New endpoint: `POST /api/rag/search-v2`.
- Feature flag: `NALUS_LEGAL_V2_SEARCH_ENABLED=0`.
- New Qdrant collection: `nalus_legal_paragraph_chunks_v2`.
- New BM25 sidecar: `storage/rag/bm25/nalus_legal_paragraph_bm25_v2.sqlite`.
- Existing `/api/rag/retrieve`, `/api/rag/query`, `/api/rag/retrieve-documents`, `/api/rag/retrieve-verified`, frontend behavior, cache behavior, and production retrieval profile are unchanged.

## Pipeline

```text
Original query
-> DeepSeek QuerySpec v2 interpreter
-> deterministic hard-constraint preservation validation
-> BGE-M3 dense retrieval from v2 collection
-> BM25 lexical retrieval from v2 sidecar
-> RRF fusion
-> document aggregation
-> paragraph-aware evidence selection
-> DeepSeek semantic verifier
-> deterministic terminal gate
-> verified documents with paragraph evidence
```

Unit tests use deterministic fake providers and do not call DeepSeek.

## Parser Audit

Run the parse-only audit before any embedding or index build:

```powershell
python scripts/legal_v2/audit_corpus.py --output-dir artifacts/legal_v2/parse_audit
```

The audit is read-only. It does not call DeepSeek, create embeddings, write Qdrant, or write BM25. Documents with material invariant failures are excluded from indexing.

## Parser Quality Gate

Create the human-review artifact:

```powershell
python scripts/legal_v2/parser_quality_gate.py --output-dir artifacts/legal_v2/parser_quality_gate
```

The review manifest supports `approved`, `rejected`, and `needs_review`.
The generator selects a bounded representative sample instead of blindly taking
the first N documents. The generated artifact includes explicit review fields
for beginning/end parsing, headings, numbered paragraphs, legal reasoning,
boilerplate, reconstruction, child chunks, parent windows, and cross-document
mixing. Items remain `needs_review` unless the review manifest explicitly marks
them otherwise.

Create a source inventory before a full parse audit:

```powershell
python scripts/legal_v2/source_inventory.py
```

The inventory reports discovered document counts, source files, date coverage,
missing identifiers/text, duplicate source-document identifiers, unreadable
files, and unsupported formats.

## Index Builder

Build only after parser audit and quality review pass:

```powershell
python scripts/legal_v2/build_index.py --overwrite-bm25 --recreate-v2-collection
```

The builder writes only `nalus_legal_paragraph_chunks_v2` and the v2 BM25 sidecar. It validates dense/BM25 chunk identity consistency and writes `legal_v2_build_manifest.json`.

## Live Smoke

Run only when the v2 index exists and DeepSeek credentials are configured:

```powershell
python scripts/legal_v2/live_smoke.py --query "únos dítěte matkou z Česka do Ruska"
```

The smoke checks the v2 collection, v2 BM25 sidecar, DeepSeek QuerySpec interpretation, hybrid retrieval, evidence selection, DeepSeek final verification, and deterministic gate. Secrets are not printed.

DeepSeek configuration is read from the runtime environment. Docker Compose loads `.env`
and then applies service `environment` entries; `LLM_MODEL_DEEPSEEK` is now passed
through as `${LLM_MODEL_DEEPSEEK:-deepseek-v4-flash}` so `.env` is not masked by a
hard-coded compose value.

For a bounded provider diagnostic before the full Legal v2 smoke:

```powershell
python scripts/legal_v2/deepseek_smoke.py --mode direct
python scripts/legal_v2/deepseek_smoke.py --mode provider
```

These diagnostics print only safe configuration fields, request shape summaries,
status codes, provider error codes/messages, and short output previews. They do not
print the API key or full prompts.

Legal v2 QuerySpec and verifier calls use `NALUS_LEGAL_V2_LLM_MAX_TOKENS` when
set. If it is unset and generic `LLM_MAX_TOKENS` is lower than 2400, Legal v2 uses
2400 for those structured calls because DeepSeek v4 responses can otherwise spend
the 800-token budget on `reasoning_content` and truncate the final JSON.

## Rollback

Set:

```env
NALUS_LEGAL_V2_SEARCH_ENABLED=0
```

No production index, sidecar, endpoint, cache, or frontend rollback is required because v2 is isolated.

## Known Limitations

- Parser readiness still depends on manual review of the QA artifact.
- Full v2 index build requires local `qdrant_client`, Qdrant, and the offline BGE-M3 model.
- Live semantic control requires configured DeepSeek credentials.
- The v2 endpoint returns no thematic fallback when hard constraints are not proven.
