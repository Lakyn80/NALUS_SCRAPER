# NALUS RAG eval — split runs

Dense baseline (partial, preserved 2026-07-05):
- Config: `nalus.rag_eval.yaml` (dense-only, default modes)
- Output: `out_dense_baseline/`
- Run ID: `20260704_150825Z`
- Note: only `multilingual_e5_small` completed eval; others timed out in that run.

BM25 + hybrid run:
- Config: `nalus_bm25_eval.yaml`
- Modes: `bm25`, `dense_plus_bm25` (6 candidates: 1 BM25 + 5 hybrid)
- Output: `out_bm25/`
- Log: `benchmark_bm25_run.log`

After BM25 run completes, merge comparison:
```powershell
python artifacts/rag_eval/merge_rankings.py
# or inside container (scripts/ is not mounted):
docker compose exec api python /app/artifacts/rag_eval/merge_rankings.py
```
Output: `out_combined/combined_ranking.json` + `combined_report.md`

Check if benchmark is running (pgrep is not in the slim image):
```powershell
docker compose exec api ps aux
```

Package: `rag-embedding-benchmark[sql-qdrant-bm25]` **v0.2.0** from GitHub (`@v0.2.0`).

Incomplete hybrid reruns (4 candidates):
```powershell
docker compose exec -d api sh -c "nohup sh /app/artifacts/rag_eval/run_all_reruns.sh > /app/artifacts/rag_eval/rerun_logs/master.log 2>&1 &"
```
Monitor: `artifacts/rag_eval/rerun_logs/*.log`
