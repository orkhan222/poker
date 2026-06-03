param(
    [string]$Dataset = "C:\Users\user\Desktop\AllFile\dataset",
    [string]$ModelOut = "",
    [string]$ReportsDir = "",
    [switch]$SkipTrain,
    [switch]$TrainBundle,
    [switch]$AllowGateFailure
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (!$ModelOut) {
    if ($TrainBundle) {
        $ModelOut = Join-Path $ProjectRoot "models\poker_policy_bundle.joblib"
    } else {
        $ModelOut = Join-Path $ProjectRoot "models\poker_policy.joblib"
    }
}
if (!$ReportsDir) {
    $ReportsDir = Join-Path $ProjectRoot "reports"
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
        & $Candidate -c "import sys; print(sys.executable)" *> $null
        if ($LASTEXITCODE -eq 0) {
            $Python = $Candidate
            break
        }
    }
}
if (!$Python) {
    Write-Error "Working Python was not found. Run .\install.ps1 first."
}
if (!(Test-Path $Dataset)) {
    Write-Error "Dataset folder not found: $Dataset"
}

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null

$AuditReport = Join-Path $ReportsDir "dataset_audit.json"
$RepositoryAuditReport = Join-Path $ReportsDir "repository_audit.json"
$GateReport = Join-Path $ReportsDir "production_gate.json"
$RepoHygieneReport = Join-Path $ReportsDir "repo_hygiene.json"
$EventBenchmarkReport = Join-Path $ReportsDir "llm_event_benchmark.json"
$EventMethodologyReport = Join-Path $ReportsDir "llm_event_methodology.md"
$GoldEvalReport = Join-Path $ReportsDir "llm_event_gold_eval.json"
$GoldPredictionsReport = Join-Path $ReportsDir "llm_event_gold_predictions.jsonl"
$GoldMarkdownReport = Join-Path $ReportsDir "llm_event_gold_report.md"

Write-Host "1/8 Auditing dataset..." -ForegroundColor Green
& $Python scripts\audit_dataset.py `
    --dataset $Dataset `
    --out $AuditReport `
    --missing-hole-cards flag `
    --max-feature-examples 50000

if (!$SkipTrain) {
    if ($TrainBundle) {
        Write-Host "2/8 Training routed policy bundle..." -ForegroundColor Green
        & $Python scripts\train_policy_bundle.py `
            --dataset $Dataset `
            --model-out $ModelOut `
            --max-examples 50000 `
            --max-iter 60 `
            --learning-rate 0.05 `
            --max-leaf-nodes 31 `
            --l2-regularization 0.02 `
            --class-weighting sqrt_balanced `
            --max-class-weight 6
    } else {
        Write-Host "2/8 Training leakage-aware single policy..." -ForegroundColor Green
        & $Python scripts\train_policy.py `
            --dataset $Dataset `
            --model-out $ModelOut `
            --policy hist_gradient_boosting `
            --max-examples 0 `
            --max-iter 90 `
            --learning-rate 0.05 `
            --max-leaf-nodes 31 `
            --l2-regularization 0.02 `
            --class-weighting sqrt_balanced `
            --max-class-weight 6 `
            --missing-hole-cards drop `
            --split-strategy stratified_hand_group
    }
} else {
    Write-Host "2/8 Skipping training; using existing model." -ForegroundColor Yellow
}

Write-Host "3/8 Running production gate..." -ForegroundColor Green
& $Python scripts\production_gate.py `
    --model $ModelOut `
    --audit-report $AuditReport `
    --out $GateReport
$GateExit = $LASTEXITCODE
if ($GateExit -ne 0 -and !$AllowGateFailure) {
    Write-Error "Production gate failed. Use -AllowGateFailure only when preparing a research/prototype delivery."
}

Write-Host "4/8 Running event extraction benchmark..." -ForegroundColor Green
& $Python scripts\llm_event_benchmark.py `
    --input (Join-Path $ProjectRoot "dataset\logs") `
    --out $EventBenchmarkReport `
    --methodology-out $EventMethodologyReport `
    --prompt (Join-Path $ProjectRoot "configs\prompts\event_extraction_prompt.txt") `
    --provider local_rules `
    --max-files 2 `
    --max-records 1000 `
    --min-confidence 0.2

Write-Host "5/8 Running gold event extraction evaluation..." -ForegroundColor Green
& $Python scripts\llm_event_gold_eval.py `
    --gold (Join-Path $ProjectRoot "evaluation\event_extraction_gold.jsonl") `
    --out $GoldEvalReport `
    --predictions-out $GoldPredictionsReport `
    --report-out $GoldMarkdownReport `
    --minimal-prompt (Join-Path $ProjectRoot "configs\prompts\event_extraction_minimal.txt") `
    --permissive-prompt (Join-Path $ProjectRoot "configs\prompts\event_extraction_permissive.txt") `
    --strict-prompt (Join-Path $ProjectRoot "configs\prompts\event_extraction_strict.txt")

Write-Host "6/8 Auditing repository..." -ForegroundColor Green
& $Python scripts\audit_repository.py `
    --root $ProjectRoot `
    --out $RepositoryAuditReport

Write-Host "7/8 Checking repository hygiene..." -ForegroundColor Green
& $Python scripts\check_repo_hygiene.py `
    --root $ProjectRoot `
    --json-out $RepoHygieneReport
if ($LASTEXITCODE -ne 0) {
    Write-Error "Repository hygiene check failed. Remove local tool metadata or delivery-only comments before rebuilding the ZIP."
}

Write-Host "8/8 Rebuilding delivery ZIP..." -ForegroundColor Green
$ZipPath = Join-Path $ProjectRoot "release\poker-decision-agent.zip"
$Items = Get-ChildItem -Force | Where-Object {
    $_.Name -notin @(".git", ".qodo", ".venv", "env", "dataset", "sample_out", "smoke_dataset", "__pycache__", "release", "research_runs")
}
Compress-Archive -Path $Items.FullName -DestinationPath $ZipPath -Force

Write-Host ""
Write-Host "Delivery workflow complete." -ForegroundColor Green
Write-Host "Model: $ModelOut"
Write-Host "Audit: $AuditReport"
Write-Host "Repository audit: $RepositoryAuditReport"
Write-Host "Gate: $GateReport"
Write-Host "Repo hygiene: $RepoHygieneReport"
Write-Host "Event benchmark: $EventBenchmarkReport"
Write-Host "Event methodology: $EventMethodologyReport"
Write-Host "Gold event eval: $GoldEvalReport"
Write-Host "Gold event report: $GoldMarkdownReport"
Write-Host "ZIP: $ZipPath"
