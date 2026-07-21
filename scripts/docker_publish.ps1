[CmdletBinding()]
param(
    [string]$DockerHubNamespace = $env:DOCKERHUB_NAMESPACE,
    [string]$Tag = $(if ($env:IMAGE_TAG) { $env:IMAGE_TAG } else { "latest" }),
    [string]$Registry = $(if ($env:DOCKER_REGISTRY) { $env:DOCKER_REGISTRY } else { "docker.io" }),
    [switch]$SkipExporter,
    [switch]$AlsoTagLatest,
    [switch]$UseExistingImages,
    [string]$LocalApiImage = "nalus-scraper-api:latest",
    [string]$LocalExporterImage = "nalus-scraper-nalus-eval-metrics-exporter:latest"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Docker {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host ">> docker $($Arguments -join ' ')"
    & docker @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Docker command failed: docker $($Arguments -join ' ')"
    }
}

function Build-And-PushImage {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Repository,
        [Parameter(Mandatory = $true)]
        [string]$DockerfilePath,
        [switch]$PublishLatestAlias
    )

    $primaryImage = "${Registry}/${DockerHubNamespace}/${Repository}:${Tag}"
    $latestImage = "${Registry}/${DockerHubNamespace}/${Repository}:latest"

    Invoke-Docker -Arguments @("build", "-f", $DockerfilePath, "-t", $primaryImage, ".")

    if ($PublishLatestAlias -and $Tag -ne "latest") {
        Invoke-Docker -Arguments @("tag", $primaryImage, $latestImage)
    }

    Invoke-Docker -Arguments @("push", $primaryImage)

    if ($PublishLatestAlias -and $Tag -ne "latest") {
        Invoke-Docker -Arguments @("push", $latestImage)
    }

    return $primaryImage
}

function Tag-And-PushExistingImage {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LocalImage,
        [Parameter(Mandatory = $true)]
        [string]$Repository,
        [switch]$PublishLatestAlias
    )

    $primaryImage = "${Registry}/${DockerHubNamespace}/${Repository}:${Tag}"
    $latestImage = "${Registry}/${DockerHubNamespace}/${Repository}:latest"

    Invoke-Docker -Arguments @("image", "inspect", $LocalImage)
    Invoke-Docker -Arguments @("tag", $LocalImage, $primaryImage)

    if ($PublishLatestAlias -and $Tag -ne "latest") {
        Invoke-Docker -Arguments @("tag", $LocalImage, $latestImage)
    }

    Invoke-Docker -Arguments @("push", $primaryImage)

    if ($PublishLatestAlias -and $Tag -ne "latest") {
        Invoke-Docker -Arguments @("push", $latestImage)
    }

    return $primaryImage
}

if ([string]::IsNullOrWhiteSpace($DockerHubNamespace)) {
    throw "Provide -DockerHubNamespace or set DOCKERHUB_NAMESPACE."
}

Get-Command docker -ErrorAction Stop | Out-Null

$repoRoot = Split-Path -Path $PSScriptRoot -Parent
Push-Location $repoRoot

try {
    Write-Host "Publishing Docker images from $repoRoot"
    if ($UseExistingImages) {
        $apiImage = Tag-And-PushExistingImage -LocalImage $LocalApiImage -Repository "nalus-scraper-api" -PublishLatestAlias:$AlsoTagLatest
    }
    else {
        $apiImage = Build-And-PushImage -Repository "nalus-scraper-api" -DockerfilePath "Dockerfile" -PublishLatestAlias:$AlsoTagLatest
    }

    $exporterImage = $null
    if (-not $SkipExporter) {
        if ($UseExistingImages) {
            $exporterImage = Tag-And-PushExistingImage -LocalImage $LocalExporterImage -Repository "nalus-scraper-eval-exporter" -PublishLatestAlias:$AlsoTagLatest
        }
        else {
            $exporterImage = Build-And-PushImage -Repository "nalus-scraper-eval-exporter" -DockerfilePath "Dockerfile.eval-exporter" -PublishLatestAlias:$AlsoTagLatest
        }
    }

    Write-Host ""
    Write-Host "Published images:"
    Write-Host "  $apiImage"
    if ($exporterImage) {
        Write-Host "  $exporterImage"
    }
}
finally {
    Pop-Location
}
