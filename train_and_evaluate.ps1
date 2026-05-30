param(
    [string]$Dataset = "C:\Users\user\Desktop\AllFile\dataset",
    [string]$ModelOut = "",
    [int]$MaxExamples = 0,
    [int]$Epochs = 12,
    [double]$LearningRate = 0.015,
    [ValidateSet("none", "sqrt_balanced", "balanced")]
    [string]$ClassWeighting = "none"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (!$ModelOut) {
    $ModelOut = Join-Path $ProjectRoot "models\poker_policy.json"
}

$PythonCandidates = @(
    (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
    (Join-Path $ProjectRoot "env\Scripts\python.exe")
)

$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($PythonCommand) {
    $PythonCandidates += $PythonCommand.Source
}

$Python = $null
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

if (!(Test-Path $Dataset)) {
    Write-Error "Dataset folder not found: $Dataset"
}

Set-Location $ProjectRoot

Write-Host "Training policy model..." -ForegroundColor Green
& $Python scripts\train_policy.py `
    --dataset $Dataset `
    --model-out $ModelOut `
    --epochs $Epochs `
    --learning-rate $LearningRate `
    --class-weighting $ClassWeighting `
    --max-examples $MaxExamples

Write-Host ""
Write-Host "Evaluating saved model..." -ForegroundColor Green
& $Python scripts\evaluate_policy.py `
    --dataset $Dataset `
    --model $ModelOut `
    --max-examples $MaxExamples
