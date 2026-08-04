param(
    [switch]$List,
    [switch]$Status,
    [string]$DocumentId,
    [ValidateSet("Summary", "Raw", "Blocks", "Lines", "Boundaries")]
    [string]$View = "Lines",
    [string]$LineId,
    [string]$BoundaryId,
    [switch]$Accept,
    [ValidateSet("metadata", "heading", "numbered_paragraph_start", "numbered_paragraph_continuation", "prose_start", "prose_continuation", "citation_continuation", "list_or_table", "signature", "instruction", "layout_noise", "unresolved")]
    [string]$OverrideClass,
    [ValidateSet("split", "merge", "preserve_parser", "unresolved")]
    [string]$BoundaryDecision,
    [switch]$Unresolved,
    [string]$Comment = "",
    [switch]$OpenGrid,
    [string]$ReviewDir = "artifacts/legal_v2/visual_parser_review"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

function Ensure-Snapshot {
    if (-not (Test-Path (Join-Path $ReviewDir "review_manifest.json"))) {
        python scripts/legal_v2/build_visual_parser_review.py --review-dir $ReviewDir | Out-Host
    }
}

function Write-Decision {
    param([hashtable]$Payload)
    $tmpDir = Join-Path $ReviewDir ".tmp"
    New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
    $tmp = Join-Path $tmpDir ("decision-" + [guid]::NewGuid().ToString("N") + ".json")
    $Payload | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 -Path $tmp
    try {
        python scripts/legal_v2/validate_manual_parser_review.py --review-dir $ReviewDir --record-decision-json $tmp | Out-Host
    } finally {
        Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
    }
}

Ensure-Snapshot

if ($List) {
    python -m scripts.legal_v2.parser_review.presentation list --review-dir $ReviewDir
    return
}

if ($Status) {
    python -m scripts.legal_v2.parser_review.presentation status --review-dir $ReviewDir
    return
}

if ($LineId) {
    $status = "overridden"
    if ($Accept) { $status = "accepted" }
    if ($Unresolved) { $status = "unresolved"; $OverrideClass = "unresolved" }
    if (-not $OverrideClass) { $OverrideClass = "unresolved" }
    Write-Decision @{
        item_type = "line"
        item_id = $LineId
        document_id = $DocumentId
        decision_status = $status
        manual_class = $OverrideClass
        reviewer_comment = $Comment
        interface = "powershell"
    }
    return
}

if ($BoundaryId) {
    $status = "overridden"
    if ($Accept) { $status = "accepted"; $BoundaryDecision = "preserve_parser" }
    if ($Unresolved) { $status = "unresolved"; $BoundaryDecision = "unresolved" }
    if (-not $BoundaryDecision) { $BoundaryDecision = "preserve_parser" }
    Write-Decision @{
        item_type = "boundary"
        item_id = $BoundaryId
        document_id = $DocumentId
        decision_status = $status
        manual_boundary_decision = $BoundaryDecision
        reviewer_comment = $Comment
        interface = "powershell"
    }
    return
}

if ($OpenGrid) {
    if (-not (Get-Command Out-GridView -ErrorAction SilentlyContinue)) {
        throw "Out-GridView is not available in this PowerShell session."
    }
    $lines = Get-Content (Join-Path $ReviewDir "review_lines.jsonl") | ForEach-Object { $_ | ConvertFrom-Json }
    if ($DocumentId) { $lines = $lines | Where-Object { $_.document_id -eq $DocumentId -or $_.source_document_id -eq $DocumentId } }
    $lines | Out-GridView -Title "NALUS parser review lines"
    return
}

if ($DocumentId) {
    python -m scripts.legal_v2.parser_review.presentation view --review-dir $ReviewDir --document-id $DocumentId --view $View.ToLowerInvariant()
    return
}

python -m scripts.legal_v2.parser_review.presentation status --review-dir $ReviewDir
