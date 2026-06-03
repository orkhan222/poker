param(
    [string]$Dataset = "C:\Users\user\Desktop\AllFile\dataset",
    [string]$ModelOut = "",
    [int]$MaxExamples = 0,
    [ValidateSet("hist_gradient_boosting", "xgboost", "lightgbm", "catboost", "extra_trees", "random_forest", "mlp", "softmax")]
    [string]$Policy = "hist_gradient_boosting",
    [int]$Epochs = 12,
    [int]$MaxIter = 90,
    [double]$LearningRate = 0.05,
    [ValidateSet("none", "sqrt_balanced", "balanced")]
    [string]$ClassWeighting = "sqrt_balanced",
    [double]$MaxClassWeight = 6.0,
    [ValidateSet("drop", "flag", "keep")]
    [string]$MissingHoleCards = "drop",
    [ValidateSet("stratified_hand_group", "random_action")]
    [string]$SplitStrategy = "stratified_hand_group",
    [switch]$SinglePolicy
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (!$ModelOut) {
    if ($SinglePolicy) {
        $ModelOut = Join-Path $ProjectRoot "models\poker_policy.joblib"
    } else {
        $ModelOut = Join-Path $ProjectRoot "models\poker_policy_bundle.joblib"
    }
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

if ($SinglePolicy) {
    Write-Host "Training single policy model..." -ForegroundColor Green
    & $Python scripts\train_policy.py `
        --dataset $Dataset `
        --model-out $ModelOut `
        --policy $Policy `
        --epochs $Epochs `
        --max-iter $MaxIter `
        --learning-rate $LearningRate `
        --class-weighting $ClassWeighting `
        --max-class-weight $MaxClassWeight `
        --missing-hole-cards $MissingHoleCards `
        --split-strategy $SplitStrategy `
        --max-examples $MaxExamples
} else {
    Write-Host "Training routed policy bundle..." -ForegroundColor Green
    & $Python scripts\train_policy_bundle.py `
        --dataset $Dataset `
        --model-out $ModelOut `
        --observed-policy $Policy `
        --context-policy $Policy `
        --max-iter $MaxIter `
        --learning-rate $LearningRate `
        --class-weighting $ClassWeighting `
        --max-class-weight $MaxClassWeight `
        --max-examples $MaxExamples
}

Write-Host ""
Write-Host "Evaluating saved model..." -ForegroundColor Green
& $Python scripts\evaluate_policy.py `
    --dataset $Dataset `
    --model $ModelOut `
    --missing-hole-cards $MissingHoleCards `
    --max-examples $MaxExamples
