# Court Staging Daily Updater — Windows Task Scheduler
# Registers a daily job that ONLY writes to artifacts/court_staging.
# Never merges to batches/, never indexes Qdrant.

param(
    [string]$RepoRoot = "C:\Users\lukas\Desktop\PYTHON_PROJECTS_DESKTOP\PYTHON_PROJECTS\nalus-scraper",
    [string]$TaskName = "NALUS-CourtStaging-Daily",
    [string]$Time = "03:30",
    [switch]$Register,
    [switch]$Unregister,
    [switch]$RunOnce
)

$ErrorActionPreference = "Stop"
$Python = (Get-Command python -ErrorAction SilentlyContinue)?.Source
if (-not $Python) { $Python = "python" }

$Updater = Join-Path $RepoRoot "scripts\court_staging_updater.py"
$LogDir = Join-Path $RepoRoot "artifacts\court_staging\updater"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Invoke-Updater {
    Set-Location $RepoRoot
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $log = Join-Path $LogDir "scheduler_$stamp.log"
    & $Python $Updater --courts us,ns,nss --mode incremental --overlap-days 7 `
        2>&1 | Tee-Object -FilePath $log
    return $LASTEXITCODE
}

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Unregistered $TaskName"
    exit 0
}

if ($RunOnce) {
    exit (Invoke-Updater)
}

if ($Register) {
    $action = New-ScheduledTaskAction -Execute $Python -Argument "`"$Updater`" --courts us,ns,nss --mode incremental --overlap-days 7" -WorkingDirectory $RepoRoot
    $trigger = New-ScheduledTaskTrigger -Daily -At $Time
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
    Write-Host "Registered daily task '$TaskName' at $Time → staging only"
    Write-Host "Log dir: $LogDir"
    exit 0
}

Write-Host @"
Usage:
  .\scripts\court_staging\register_daily_task.ps1 -Register
  .\scripts\court_staging\register_daily_task.ps1 -RunOnce
  .\scripts\court_staging\register_daily_task.ps1 -Unregister

Task runs: python scripts/court_staging_updater.py --courts us,ns,nss --mode incremental
Output:    artifacts/court_staging/ (NO batches merge, NO indexing)
"@
