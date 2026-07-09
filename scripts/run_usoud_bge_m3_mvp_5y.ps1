# Stage 3 MVP ingest: US/NALUS BGE-M3, rolling last 5 years only.
# Production collections are never touched. Progress is streamed live.
# After interruption, rerun the RESUME block (no --recreate-full-collection).

$ErrorActionPreference = "Stop"
Set-Location "C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper"

$Collection = "nalus_us_bge_m3_mvp_5y_20260708"
$OutputDir = "artifacts/nalus_update/usoud_bge_m3_mvp_5y_20260708"
$LogDir = "$OutputDir\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = "$LogDir\mvp_5y_$Stamp.log"

$CommonArgs = @(
    "scripts/build_usoud_bge_m3_candidate.py",
    "--mode", "full",
    "--limit", "0",
    "--collection-name", $Collection,
    "--source-manifest", "batches/manifest.json",
    "--output-dir", $OutputDir,
    "--no-alias-update",
    "--years-back", "5",
    "--ingest-slice", "mvp_5y",
    "--embedding-batch-size", "16",
    "--full-record-batch-size", "50"
)

function Invoke-Builder {
    param(
        [string[]]$ExtraArgs,
        [string]$Label
    )
    Write-Host ""
    Write-Host "=== $Label ===" -ForegroundColor Cyan
    Write-Host "Log: $LogFile"
    $cmd = @("compose", "exec", "-e", "PYTHONUNBUFFERED=1", "api", "python") + $CommonArgs + $ExtraArgs
    docker @cmd 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Builder failed ($Label) with exit code $LASTEXITCODE"
    }
}

# 1) Dry-run planning (no Qdrant writes)
Invoke-Builder -ExtraArgs @("--dry-run") -Label "MVP 5y dry-run"

# 2) First execute (creates collection + checkpoint). Comment out after first successful start
#    if you only want resume on the next run.
Invoke-Builder -ExtraArgs @("--execute", "--recreate-full-collection") -Label "MVP 5y execute (fresh)"

# 3) Resume after interruption (uncomment and comment block 2 when continuing)
# Invoke-Builder -ExtraArgs @("--execute") -Label "MVP 5y execute (resume)"

# 4) Later: append older decisions in another slice (example, adjust date bounds first)
# Invoke-Builder -ExtraArgs @(
#     "--execute",
#     "--append-full-slice",
#     "--ingest-slice", "pre_mvp_5y",
#     "--decision-date-to", "2021-07-08"
# ) -Label "MVP append older slice"

Write-Host ""
Write-Host "Done. Checkpoint:" "$OutputDir\execute_checkpoint.json" -ForegroundColor Green
Write-Host "Report: artifacts/nalus_update/usoud_bge_m3_stage3_full_report.md"
