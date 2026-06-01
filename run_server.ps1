param(
    [int]$Port = 8001,
    [string]$ModelPath = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$DefaultModelPath = Join-Path $ProjectRoot "models\poker_policy.joblib"
$Python = $null

if (!$ModelPath) {
    $ModelPath = $DefaultModelPath
}

$PythonCandidates = @(
    (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
    (Join-Path $ProjectRoot "env\Scripts\python.exe")
)

$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($PythonCommand) {
    $PythonCandidates += $PythonCommand.Source
}

foreach ($Candidate in $PythonCandidates) {
    if ($Candidate -and (Test-Path $Candidate)) {
        & $Candidate --version *> $null
        if ($LASTEXITCODE -eq 0) {
            $Python = $Candidate
            break
        }
    }
}

if (!$Python) {
    Write-Error "Working Python was not found. Run .\install.ps1 first or install Python 3.11+."
}

if (!(Test-Path $ModelPath)) {
    Write-Error "Model not found: $ModelPath"
}

$env:POKER_POLICY_PATH = $ModelPath
Set-Location $ProjectRoot

Write-Host ""
Write-Host "Poker Decision Agent is starting..." -ForegroundColor Green
Write-Host "Application: http://127.0.0.1:$Port/predict"
Write-Host "API docs: http://127.0.0.1:$Port/docs"
Write-Host "Health: http://127.0.0.1:$Port/health"
Write-Host "Model: $ModelPath"
Write-Host ""
Write-Host "Press Ctrl+C to stop the server."
Write-Host ""

& $Python -m uvicorn poker_agent.service:app --host 127.0.0.1 --port $Port
