# MVP ~3h ingest: 600 newest US/NALUS decisions up to 2026-07-09 (BGE-M3, non-production).
# Stop any running mvp_5y execute first (Ctrl+C) — this uses a separate collection.
# scripts/ is not mounted in Docker; always cp the builder before running.

$ErrorActionPreference = "Stop"
Set-Location "C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper"

docker compose cp scripts/build_usoud_bge_m3_candidate.py api:/app/scripts/build_usoud_bge_m3_candidate.py

$Collection = "nalus_us_bge_m3_mvp_recent_3h_20260709"
$OutputDir = "artifacts/nalus_update/usoud_bge_m3_mvp_recent_3h_20260709"
$RecordLimit = "600"

$CommonArgs = @(
    "scripts/build_usoud_bge_m3_candidate.py",
    "--mode", "full",
    "--limit", $RecordLimit,
    "--collection-name", $Collection,
    "--source-manifest", "batches/manifest.json",
    "--output-dir", $OutputDir,
    "--no-alias-update",
    "--ingest-slice", "mvp_recent_3h",
    "--decision-date-to", "2026-07-09",
    "--newest-first",
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
    $cmd = @("compose", "exec", "-e", "PYTHONUNBUFFERED=1", "api", "python") + $CommonArgs + $ExtraArgs
    docker @cmd
    if ($LASTEXITCODE -ne 0) {
        throw "Builder failed ($Label) with exit code $LASTEXITCODE"
    }
}

# 1) Dry-run (plan chunk count for newest 600 decisions)
Invoke-Builder -ExtraArgs @("--dry-run") -Label "MVP recent 3h dry-run"

# 2) Fresh execute (first run only)
Invoke-Builder -ExtraArgs @("--execute", "--recreate-full-collection") -Label "MVP recent 3h execute"

# 3) Resume after interruption (uncomment block 2 off, use this instead)
# Invoke-Builder -ExtraArgs @("--execute") -Label "MVP recent 3h resume"

Write-Host ""
Write-Host "Done. Checkpoint:" "$OutputDir\execute_checkpoint.json" -ForegroundColor Green
