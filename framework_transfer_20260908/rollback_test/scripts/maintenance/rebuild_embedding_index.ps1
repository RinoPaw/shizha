param(
    [int]$BatchSize = 64,
    [int]$Workers = 6,
    [double]$RequestTimeout = 5,
    [int]$MaxRounds = 20,
    [double]$RetryDelay = 2,
    [int]$Limit = 0,
    [switch]$NoResume
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $ProjectRoot

$PythonCandidates = @(
    (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
    "python"
)

$Python = $null
foreach ($Candidate in $PythonCandidates) {
    if ($Candidate -eq "python") {
        $Python = $Candidate
        break
    }
    if (Test-Path $Candidate) {
        $Python = $Candidate
        break
    }
}

$Arguments = @(
    ".\scripts\maintenance\rebuild_embedding_index.py",
    "--batch-size", $BatchSize,
    "--workers", $Workers,
    "--request-timeout", $RequestTimeout,
    "--retry-delay", $RetryDelay,
    "--max-rounds", $MaxRounds
)

if ($Limit -gt 0) {
    $Arguments += @("--limit", $Limit)
}

if ($NoResume) {
    $Arguments += "--no-resume"
}

Write-Host "Building embedding index..."
Write-Host "Python: $Python"
Write-Host "BatchSize=$BatchSize Workers=$Workers RequestTimeout=${RequestTimeout}s MaxRounds=$MaxRounds"
Write-Host ""

& $Python @Arguments
exit $LASTEXITCODE
