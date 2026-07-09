# Merge mvp_5y + mvp_recent_3h into one RAG candidate collection and build BM25 sidecar.
# Production aliases/collections are never modified.

$ErrorActionPreference = "Stop"
Set-Location "C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper"

$TargetCollection = "nalus_us_bge_m3_rag_combined_20260709"
$OutputDir = "artifacts/nalus_update/usoud_bge_m3_rag_combined_20260709"
$Bm25Sqlite = "storage/rag/bm25/nalus_us_bge_m3_rag_combined_20260709.sqlite"

docker compose cp scripts/merge_bge_m3_candidate_collections.py api:/app/scripts/merge_bge_m3_candidate_collections.py
docker compose cp scripts/build_bm25_sidecar_from_qdrant.py api:/app/scripts/build_bm25_sidecar_from_qdrant.py

function Invoke-Merge {
    param([string[]]$ExtraArgs, [string]$Label)
    Write-Host ""
    Write-Host "=== $Label ===" -ForegroundColor Cyan
    docker compose exec -e PYTHONUNBUFFERED=1 api python scripts/merge_bge_m3_candidate_collections.py `
        --source-collections nalus_us_bge_m3_mvp_5y_20260708 nalus_us_bge_m3_mvp_recent_3h_20260709 `
        --target-collection $TargetCollection `
        --output-dir $OutputDir `
        @ExtraArgs
    if ($LASTEXITCODE -ne 0) { throw "Merge failed ($Label)" }
}

# 1) Merge dry-run
Invoke-Merge -ExtraArgs @("--dry-run") -Label "Merge dry-run"

# 2) Merge execute (first run only)
Invoke-Merge -ExtraArgs @("--execute", "--recreate-target-collection") -Label "Merge execute"

# 3) BM25 sidecar export
Write-Host ""
Write-Host "=== BM25 sidecar export ===" -ForegroundColor Cyan
docker compose exec -e PYTHONUNBUFFERED=1 api python scripts/build_bm25_sidecar_from_qdrant.py `
    --collection-name $TargetCollection `
    --sqlite-path $Bm25Sqlite `
    --overwrite
if ($LASTEXITCODE -ne 0) { throw "BM25 export failed" }

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "Target collection: $TargetCollection"
Write-Host "BM25 sidecar: $Bm25Sqlite"
Write-Host ""
Write-Host "API wiring (add to .env, then restart api):" -ForegroundColor Yellow
Write-Host "QDRANT_COLLECTION_NAME=$TargetCollection"
Write-Host "BM25_SIDECAR_PATH=/app/$Bm25Sqlite"
Write-Host "BM25_INDEX_ID=nalus_bge_m3_dense_bm25_rrf_v1"
Write-Host ""
Write-Host "Smoke after restart:"
Write-Host "curl http://localhost:8029/health"
Write-Host "curl -X POST http://localhost:8029/api/rag/query -H `"Content-Type: application/json`" -d `"{\`"query\`":\`"právo na spravedlivý proces\`",\`"top_k\`":5}`""
