# Case-similarity golden v1 baseline (Docker). Local Python often lacks qdrant_client.
$ErrorActionPreference = "Stop"
$Repo = "C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper-parser-fix"
$Model = "/root/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
$Collection = "nalus_legal_paragraph_chunks_v2_pilot_600"
$Bm25Id = "nalus_legal_paragraph_bm25_v2_pilot_600"
$Bm25Path = "/app/storage/rag/bm25/nalus_legal_paragraph_bm25_v2_pilot_600.sqlite"

Set-Location $Repo

# Prefer the Stage 1 local API container; fall back to the historical scraper API name.
$VolumeSource = @(
  "nalus-scraper-parser-fix-api-1",
  "nalus-scraper-api-1",
  "nalus-scraper-api-1-prev"
) | Where-Object {
  docker inspect $_ 2>$null | Out-Null
  $LASTEXITCODE -eq 0
} | Select-Object -First 1

if (-not $VolumeSource) {
  throw "No API container found for --volumes-from (expected Stage 1 or nalus-scraper-api-1)."
}

# Explicit Qdrant DNS avoids resolving to an empty worktree qdrant service.
$QdrantUrl = "http://nalus-scraper-qdrant-1:6333"

docker run --rm `
  --network nalus-scraper_default `
  --volumes-from $VolumeSource `
  -v "${Repo}:/work" `
  -w /work `
  -e "EMBEDDING_MODEL_NAME=$Model" `
  -e EMBEDDING_LOCAL_FILES_ONLY=1 `
  -e "NALUS_LEGAL_V2_QDRANT_COLLECTION=$Collection" `
  -e "NALUS_LEGAL_V2_BM25_INDEX_ID=$Bm25Id" `
  -e "NALUS_LEGAL_V2_BM25_SIDECAR_PATH=$Bm25Path" `
  nalus-scraper-parser-fix-api `
  python scripts/legal_v2/evaluate_case_similarity_golden_v1.py `
  --qdrant-url $QdrantUrl `
  --bm25-sidecar-path $Bm25Path `
  --bm25-index-id $Bm25Id
