param(
    [string]$Dataset = "C:\Users\user\Desktop\AllFile\dataset",
    [string]$ModelOut = "",
    [string]$ReportsDir = "",
    [string]$PythonExecutable = "",
    [switch]$SkipTrain,
    [switch]$TrainBundle,
    [switch]$AllowGateFailure,
    [switch]$RunTransformerEval,
    [switch]$RunDecisionContextEval
)

$ErrorActionPreference = "Stop"
$env:PYTHONDONTWRITEBYTECODE = "1"

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

function Test-PythonRuntime {
    param([string]$Candidate)
    if (!$Candidate -or !(Test-Path $Candidate)) {
        return $false
    }
    try {
        $PreviousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $Candidate -c "import sys, joblib, pandas, sklearn; print(sys.executable)" *> $null
        $ExitCode = $LASTEXITCODE
        $ErrorActionPreference = $PreviousErrorActionPreference
        return ($ExitCode -eq 0)
    } catch {
        $ErrorActionPreference = $PreviousErrorActionPreference
        return $false
    }
}

$PythonCandidates = @()
if ($PythonExecutable) {
    $PythonCandidates += $PythonExecutable
}
if ($env:POKER_AGENT_PYTHON) {
    $PythonCandidates += $env:POKER_AGENT_PYTHON
}
$PythonCandidates += @(
    (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
    (Join-Path $ProjectRoot "env\Scripts\python.exe"),
    "C:\Users\user\AppData\Local\poker-qwen-venv\Scripts\python.exe"
)
$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($PythonCommand) {
    $PythonCandidates += $PythonCommand.Source
}

$Python = $null
foreach ($Candidate in $PythonCandidates) {
    if (Test-PythonRuntime -Candidate $Candidate) {
        $Python = $Candidate
        break
    }
}
if (!$Python) {
    Write-Error "Working Python with required ML dependencies was not found. Run .\install.ps1 first or pass -PythonExecutable."
}
if (!(Test-Path $Dataset)) {
    Write-Error "Dataset folder not found: $Dataset"
}

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null


function Remove-DeliveryArtifacts {
    param([string]$Root)
    foreach ($Relative in @(".qodo", "__pycache__", "poker_agent\__pycache__", "scripts\__pycache__")) {
        $Target = Join-Path $Root $Relative
        if (Test-Path -LiteralPath $Target) {
            Remove-Item -LiteralPath $Target -Recurse -Force
        }
    }
}

function Remove-FailedHydraRuns {
    param([string]$Root)
    $HydraRoot = Join-Path $Root "reports\hydra"
    if (!(Test-Path -LiteralPath $HydraRoot)) {
        return
    }
    $ResolvedRoot = (Resolve-Path -LiteralPath $Root).Path
    foreach ($RunFile in Get-ChildItem -LiteralPath $HydraRoot -Recurse -File -Filter "run.json" -ErrorAction SilentlyContinue) {
        try {
            $Run = Get-Content -LiteralPath $RunFile.FullName -Raw | ConvertFrom-Json
        } catch {
            continue
        }
        if ($Run.status -eq "failed") {
            $RunDirectory = $RunFile.Directory.FullName
            if (!$RunDirectory.StartsWith($ResolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "Refusing outside-root removal: $RunDirectory"
            }
            Remove-Item -LiteralPath $RunDirectory -Recurse -Force
        }
    }
}

$AuditReport = Join-Path $ReportsDir "dataset_audit.json"
$RepositoryAuditReport = Join-Path $ReportsDir "repository_audit.json"
$GateReport = Join-Path $ReportsDir "production_gate.json"
$RepoHygieneReport = Join-Path $ReportsDir "repo_hygiene.json"
$EventBenchmarkReport = Join-Path $ReportsDir "llm_event_benchmark.json"
$EventMethodologyReport = Join-Path $ReportsDir "llm_event_methodology.md"
$GoldEvalReport = Join-Path $ReportsDir "llm_event_gold_eval.json"
$GoldPredictionsReport = Join-Path $ReportsDir "llm_event_gold_predictions.jsonl"
$GoldMarkdownReport = Join-Path $ReportsDir "llm_event_gold_report.md"
$DecisionContextReport = Join-Path $ReportsDir "llm_decision_context.json"
$DecisionContextMarkdownReport = Join-Path $ReportsDir "llm_decision_context.md"
$DecisionContextSmokeReport = Join-Path $ReportsDir "llm_decision_context_smoke.json"
$DecisionContextSmokePredictions = Join-Path $ReportsDir "llm_decision_context_smoke_predictions.jsonl"
$DecisionContextSmokeMarkdown = Join-Path $ReportsDir "llm_decision_context_smoke.md"
$DecisionHoldoutReport = Join-Path $ReportsDir "decision_context_holdout.json"
$DecisionHoldoutData = Join-Path $ProjectRoot "evaluation\decision_context_human_holdout.jsonl"
$DecisionQwenReport = Join-Path $ReportsDir "llm_decision_context_qwen25.json"
$DecisionGateReport = Join-Path $ReportsDir "llm_decision_gate.json"
$DecisionGateMarkdown = Join-Path $ReportsDir "llm_decision_gate.md"
$CandidateRankerReport = Join-Path $ReportsDir "llm_decision_candidate_ranker_qwen25.json"
$CandidateGateReport = Join-Path $ReportsDir "llm_decision_candidate_gate.json"
$CandidateGateMarkdown = Join-Path $ReportsDir "llm_decision_candidate_gate.md"
$ArchitectureComparisonReport = Join-Path $ReportsDir "llm_architecture_comparison.json"
$ArchitectureComparisonMarkdown = Join-Path $ReportsDir "llm_architecture_comparison.md"
$LlmRoleBoundaryReport = Join-Path $ReportsDir "llm_role_boundary.json"
$LlmRoleBoundaryMarkdownReport = Join-Path $ReportsDir "llm_role_boundary.md"
$QloraNextStageReport = Join-Path $ReportsDir "qlora_next_stage.json"
$QloraNextStageMarkdownReport = Join-Path $ReportsDir "qlora_next_stage.md"
$TransformerEvalReport = Join-Path $ReportsDir "llm_transformer_gold_eval.json"
$TransformerMarkdownReport = Join-Path $ReportsDir "llm_transformer_gold_report.md"
$ScopeContractReport = Join-Path $ReportsDir "scope_contract.json"
$ScopeContractMarkdownReport = Join-Path $ReportsDir "scope_contract.md"
$ProjectCompletionReport = Join-Path $ReportsDir "project_completion.json"
$ProjectCompletionMarkdownReport = Join-Path $ReportsDir "project_completion.md"
$FinalDeliveryAcceptanceReport = Join-Path $ReportsDir "final_delivery_acceptance.json"
$FinalDeliveryAcceptanceMarkdownReport = Join-Path $ReportsDir "final_delivery_acceptance.md"
$FinalStrategyQualityStatusReport = Join-Path $ReportsDir "final_strategy_quality_status.json"
$FinalStrategyQualityStatusMarkdownReport = Join-Path $ReportsDir "final_strategy_quality_status.md"
$ProductionRuntimeMonitoringReport = Join-Path $ReportsDir "production_runtime_monitoring.json"
# $ProductionRuntimeMonitoringMarkdownReport = Join-Path $ReportsDir "production_runtime_monitoring.md"
$ModelRiskRegisterReport = Join-Path $ReportsDir "model_risk_register.json"
$ModelRiskRegisterMarkdownReport = Join-Path $ReportsDir "model_risk_register.md"
$ProductionApprovalReport = Join-Path $ReportsDir "production_approval.json"
$ProductionApprovalMarkdownReport = Join-Path $ReportsDir "production_approval.md"
$StrategyStackMaturityReport = Join-Path $ReportsDir "strategy_stack_maturity.json"
$StrategyStackMaturityMarkdownReport = Join-Path $ReportsDir "strategy_stack_maturity.md"
$BehavioralRevalidationReport = Join-Path $ReportsDir "behavioral_revalidation.json"
$BehavioralRevalidationMarkdownReport = Join-Path $ReportsDir "behavioral_revalidation.md"
$BehavioralRevalidationProofReport = Join-Path $ReportsDir "behavioral_revalidation_proof.json"
$BehavioralRevalidationProofMarkdownReport = Join-Path $ReportsDir "behavioral_revalidation_proof.md"
$HumanLikenessEvidenceReport = Join-Path $ReportsDir "human_likeness_evidence.json"
$HumanLikenessEvidenceMarkdownReport = Join-Path $ReportsDir "human_likeness_evidence.md"
$HumanLikenessClaimGateReport = Join-Path $ReportsDir "human_likeness_claim_gate.json"
$HumanLikenessClaimGateMarkdownReport = Join-Path $ReportsDir "human_likeness_claim_gate.md"
$HoleCardDataQualityReport = Join-Path $ReportsDir "hole_card_data_quality.json"
$HoleCardDataQualityMarkdownReport = Join-Path $ReportsDir "hole_card_data_quality.md"
$BetTimingCalibrationReport = Join-Path $ReportsDir "bet_timing_calibration.json"
$BetTimingCalibrationMarkdownReport = Join-Path $ReportsDir "bet_timing_calibration.md"
$RawModelStatusReport = Join-Path $ReportsDir "raw_model_status.json"
$RawModelStatusMarkdownReport = Join-Path $ReportsDir "raw_model_status.md"
$ChallengerStrategyQualityReport = Join-Path $ReportsDir "challenger_strategy_quality.json"
$ChallengerStrategyQualityMarkdownReport = Join-Path $ReportsDir "challenger_strategy_quality.md"
$ClientHandoffReport = Join-Path $ReportsDir "client_handoff.json"
$ClientHandoffMarkdownReport = Join-Path $ReportsDir "client_handoff.md"
$TrainingClusterReport = Join-Path $ReportsDir "training_cluster_requirements.json"
$TrainingClusterMarkdownReport = Join-Path $ReportsDir "training_cluster_requirements.md"
$TodayTrainingReport = Join-Path $ReportsDir "today_acceptance_training.json"
$TodayTrainingMarkdownReport = Join-Path $ReportsDir "today_acceptance_training.md"
$TodayTrainingGateReport = Join-Path $ReportsDir "today_acceptance_production_gate.json"
$ClientGpuTrainingResponseReport = Join-Path $ReportsDir "client_gpu_training_response.json"
$ClientGpuTrainingResponseMarkdown = Join-Path $ReportsDir "client_gpu_training_response.md"
$MultiAgentTrainingStatusReport = Join-Path $ReportsDir "multi_agent_training_status.json"
$MultiAgentTrainingStatusMarkdown = Join-Path $ReportsDir "multi_agent_training_status.md"
$TodayTrainingModelOut = Join-Path $ProjectRoot "models\poker_policy_bundle.joblib"

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

Write-Host "5a/8 Building LLM decision context contract..." -ForegroundColor Green
& $Python scripts\build_llm_decision_context.py `
    --out $DecisionContextReport `
    --report-out $DecisionContextMarkdownReport

Write-Host "5b/8 Running decision-context contract ablation..." -ForegroundColor Green
& $Python scripts\llm_decision_context_eval.py `
    --data (Join-Path $ProjectRoot "evaluation\decision_context_smoke.jsonl") `
    --provider rule_baseline `
    --model-id unused `
    --device cpu `
    --torch-dtype auto `
    --max-new-tokens 192 `
    --max-examples 0 `
    --seed 42 `
    --dataset-kind smoke `
    --context-modes "minimal_zero_shot,rules_grounded,full_in_context" `
    --out $DecisionContextSmokeReport `
    --predictions-out $DecisionContextSmokePredictions `
    --report-out $DecisionContextSmokeMarkdown

Write-Host "5c/8 Building grouped human decision holdout..." -ForegroundColor Green
& $Python scripts\build_decision_context_holdout.py `
    --data-dir $Dataset `
    --out $DecisionHoldoutData `
    --report-out $DecisionHoldoutReport `
    --hands 800 `
    --examples-per-action 4 `
    --seed 42

if ($RunDecisionContextEval) {
    Write-Host "5d/8 Running Qwen decision-context evaluation..." -ForegroundColor Green
    & $Python scripts\run_hydra_experiment.py `
        experiments=llm_decision_context_qwen25 `
        "python_executable=$Python"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Qwen decision-context evaluation failed."
    }
    Write-Host "5d-2/8 Running Qwen candidate-ranking evaluation..." -ForegroundColor Green
    & $Python scripts\run_hydra_experiment.py `
        experiments=llm_decision_candidate_ranker_qwen25 `
        "python_executable=$Python"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Qwen candidate-ranking evaluation failed."
    }
}

if (Test-Path $DecisionQwenReport) {
    Write-Host "5e/8 Building LLM decision-model gate..." -ForegroundColor Green
    & $Python scripts\build_llm_decision_gate.py `
        --benchmark $DecisionQwenReport `
        --holdout-report $DecisionHoldoutReport `
        --out $DecisionGateReport `
        --report-out $DecisionGateMarkdown `
        --min-examples 20 `
        --min-macro-f1 0.40 `
        --min-schema-valid-rate 0.95 `
        --min-legal-action-rate 0.99 `
        --max-average-latency-ms 5000
}

if (Test-Path $CandidateRankerReport) {
    Write-Host "5e-2/8 Building candidate-ranker gate..." -ForegroundColor Green
    & $Python scripts\build_llm_decision_gate.py `
        --benchmark $CandidateRankerReport `
        --holdout-report $DecisionHoldoutReport `
        --out $CandidateGateReport `
        --report-out $CandidateGateMarkdown `
        --min-examples 20 `
        --min-macro-f1 0.40 `
        --min-schema-valid-rate 0.95 `
        --min-legal-action-rate 0.99 `
        --max-average-latency-ms 5000
}

if ((Test-Path $DecisionQwenReport) -and (Test-Path $CandidateRankerReport) -and (Test-Path $DecisionGateReport) -and (Test-Path $CandidateGateReport)) {
    Write-Host "5e-3/8 Building LLM architecture comparison..." -ForegroundColor Green
    & $Python scripts\build_llm_architecture_comparison.py `
        --generation $DecisionQwenReport `
        --candidate-ranker $CandidateRankerReport `
        --generation-gate $DecisionGateReport `
        --candidate-gate $CandidateGateReport `
        --out $ArchitectureComparisonReport `
        --report-out $ArchitectureComparisonMarkdown
}

Write-Host "5e-4/8 Building LLM role boundary contract..." -ForegroundColor Green
& $Python scripts\build_llm_role_boundary.py `
    --project-root $ProjectRoot `
    --out $LlmRoleBoundaryReport `
    --markdown-out $LlmRoleBoundaryMarkdownReport

Write-Host "5e-5/8 Building QLoRA next-stage boundary contract..." -ForegroundColor Green
& $Python scripts\build_qlora_next_stage.py `
    --project-root $ProjectRoot `
    --out $QloraNextStageReport `
    --markdown-out $QloraNextStageMarkdownReport

if ($RunTransformerEval) {
    Write-Host "5f/8 Running local instruction-model evaluation..." -ForegroundColor Green
    & $Python scripts\run_hydra_experiment.py `
        experiments=llm_transformer_gold_eval `
        "python_executable=$Python"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Local instruction-model evaluation failed."
    }
} elseif (!(Test-Path $TransformerEvalReport) -or !(Test-Path $TransformerMarkdownReport)) {
    Write-Warning "Instruction-model reports are missing. Skipping optional transformer eval; re-run with -RunTransformerEval when GPU/runtime is available."
}

Write-Host "6/8 Auditing repository..." -ForegroundColor Green
& $Python scripts\audit_repository.py `
    --root $ProjectRoot `
    --out $RepositoryAuditReport

Remove-DeliveryArtifacts -Root $ProjectRoot
Write-Host "7/8 Checking repository hygiene..." -ForegroundColor Green
& $Python scripts\check_repo_hygiene.py `
    --root $ProjectRoot `
    --json-out $RepoHygieneReport
if ($LASTEXITCODE -ne 0) {
    Write-Error "Repository hygiene check failed. Remove local tool metadata or delivery-only comments before rebuilding the ZIP."
}

Remove-DeliveryArtifacts -Root $ProjectRoot
Remove-FailedHydraRuns -Root $ProjectRoot
Write-Host "7b/8 Building model risk register..." -ForegroundColor Green
& $Python scripts\build_model_risk_register.py `
    --project-root $ProjectRoot `
    --out $ModelRiskRegisterReport `
    --markdown-out $ModelRiskRegisterMarkdownReport

Write-Host "7c/8 Building production approval contract..." -ForegroundColor Green
& $Python scripts\build_production_approval.py `
    --project-root $ProjectRoot `
    --out $ProductionApprovalReport `
    --markdown-out $ProductionApprovalMarkdownReport



Write-Host "7c-0/8 Building behavioral revalidation contract..." -ForegroundColor Green
& $Python scripts\build_behavioral_revalidation.py `
    --project-root $ProjectRoot `
    --out $BehavioralRevalidationReport `
    --markdown-out $BehavioralRevalidationMarkdownReport


Write-Host "7c-0b/8 Building behavioral revalidation proof..." -ForegroundColor Green
& $Python scripts\build_behavioral_revalidation_proof.py `
    --project-root $ProjectRoot `
    --out $BehavioralRevalidationProofReport `
    --markdown-out $BehavioralRevalidationProofMarkdownReport

Write-Host "7c-1/8 Building strategy stack maturity contract..." -ForegroundColor Green
& $Python scripts\build_strategy_stack_maturity.py `
    --project-root $ProjectRoot `
    --out $StrategyStackMaturityReport `
    --markdown-out $StrategyStackMaturityMarkdownReport

Write-Host "7c-2/8 Building raw model status contract..." -ForegroundColor Green
& $Python scripts\build_raw_model_status.py `
    --project-root $ProjectRoot `
    --out $RawModelStatusReport `
    --markdown-out $RawModelStatusMarkdownReport
Write-Host "7c-2b/8 Building challenger strategy-quality boundary..." -ForegroundColor Green
& $Python scripts\build_challenger_strategy_quality.py `
    --project-root $ProjectRoot `
    --out $ChallengerStrategyQualityReport `
    --markdown-out $ChallengerStrategyQualityMarkdownReport

Write-Host "7d/8 Building client handoff statement..." -ForegroundColor Green
& $Python scripts\build_client_handoff.py `
    --project-root $ProjectRoot `
    --out $ClientHandoffReport `
    --markdown-out $ClientHandoffMarkdownReport

Write-Host "7e/8 Building training cluster requirements..." -ForegroundColor Green
& $Python scripts\build_training_cluster_requirements.py `
    --project-root $ProjectRoot `
    --out $TrainingClusterReport `
    --markdown-out $TrainingClusterMarkdownReport

Write-Host "7f/8 Building today acceptance training report..." -ForegroundColor Green
& $Python scripts\run_today_acceptance_training.py `
    --project-root $ProjectRoot `
    --dataset $Dataset `
    --model-out $TodayTrainingModelOut `
    --report-out $TodayTrainingReport `
    --markdown-out $TodayTrainingMarkdownReport `
    --gate-out $TodayTrainingGateReport `
    --max-examples 1000 `
    --skip-training

Write-Host "7f-0/8 Building bet-sizing and timing calibration contract..." -ForegroundColor Green
& $Python scripts\build_bet_timing_calibration.py `
    --project-root $ProjectRoot `
    --out $BetTimingCalibrationReport `
    --markdown-out $BetTimingCalibrationMarkdownReport

Write-Host "7f-0b/8 Building human-likeness evidence contract..." -ForegroundColor Green
& $Python scripts\build_human_likeness_evidence.py `
    --project-root $ProjectRoot `
    --out $HumanLikenessEvidenceReport `
    --markdown-out $HumanLikenessEvidenceMarkdownReport

Write-Host "7f-0c/8 Building human-likeness final-claim gate..." -ForegroundColor Green
& $Python scripts\build_human_likeness_claim_gate.py `
    --project-root $ProjectRoot `
    --out $HumanLikenessClaimGateReport `
    --markdown-out $HumanLikenessClaimGateMarkdownReport

Write-Host "7f-1/8 Building hole-card data-quality contract..." -ForegroundColor Green
& $Python scripts\build_hole_card_data_quality.py `
    --project-root $ProjectRoot `
    --out $HoleCardDataQualityReport `
    --markdown-out $HoleCardDataQualityMarkdownReport

Write-Host "7f-2/8 Building client GPU training response..." -ForegroundColor Green
& $Python scripts\build_client_gpu_training_response.py `
    --project-root $ProjectRoot `
    --out $ClientGpuTrainingResponseReport `
    --markdown-out $ClientGpuTrainingResponseMarkdown

Write-Host "7f-3/8 Building multi-agent training status boundary..." -ForegroundColor Green
& $Python scripts\build_multi_agent_training_status.py `
    --project-root $ProjectRoot `
    --out $MultiAgentTrainingStatusReport `
    --markdown-out $MultiAgentTrainingStatusMarkdown
Write-Host "7g/8 Building scope contract..." -ForegroundColor Green
& $Python scripts\build_scope_contract.py `
    --project-root $ProjectRoot `
    --out $ScopeContractReport `
    --markdown-out $ScopeContractMarkdownReport

Write-Host "7h/8 Building project completion contract..." -ForegroundColor Green
& $Python scripts\build_project_completion.py `
    --project-root $ProjectRoot `
    --out $ProjectCompletionReport `
    --markdown-out $ProjectCompletionMarkdownReport

Write-Host "7h-1/8 Building final delivery acceptance contract..." -ForegroundColor Green
& $Python scripts\build_final_delivery_acceptance.py `
    --project-root $ProjectRoot `
    --out $FinalDeliveryAcceptanceReport `
    --markdown-out $FinalDeliveryAcceptanceMarkdownReport

Write-Host "7h-2/8 Building final strategy quality status contract..." -ForegroundColor Green
& $Python scripts\build_final_strategy_quality_status.py `
    --project-root $ProjectRoot `
    --out $FinalStrategyQualityStatusReport `
    --markdown-out $FinalStrategyQualityStatusMarkdownReport

Write-Host "8/8 Rebuilding delivery ZIP..." -ForegroundColor Green

$GeneratedDirs = Get-ChildItem -Force -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | Where-Object {
    $_.FullName -notlike "*\.venv\*" -and $_.FullName -notlike "*\env\*"
}
foreach ($Dir in $GeneratedDirs) {
    if ($Dir.FullName.StartsWith($ProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $Dir.FullName -Recurse -Force
    }
}
$GeneratedFiles = Get-ChildItem -Force -Recurse -File -ErrorAction SilentlyContinue | Where-Object {
    ($_.Extension -in @(".pyc", ".pyo", ".pyd") -or $_.Name -eq "requirements-research.txt") -and
    $_.FullName -notlike "*\.venv\*" -and $_.FullName -notlike "*\env\*"
}
foreach ($File in $GeneratedFiles) {
    if ($File.FullName.StartsWith($ProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $File.FullName -Force
    }
}
$ZipPath = Join-Path $ProjectRoot "release\poker-decision-agent.zip"
$Items = Get-ChildItem -Force | Where-Object {
    $_.Name -notin @(".git", ".qodo", ".venv", "env", "data", "dataset", "sample_out", "smoke_dataset", "__pycache__", "release", "research_runs") -and
    $_.Name -notlike ".venv.corrupt-*"
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
Write-Host "Decision context: $DecisionContextReport"
Write-Host "Decision context smoke: $DecisionContextSmokeReport"
Write-Host "Decision Qwen benchmark: $DecisionQwenReport"
Write-Host "Decision model gate: $DecisionGateReport"
Write-Host "Candidate ranker: $CandidateRankerReport"
Write-Host "Architecture comparison: $ArchitectureComparisonReport"
Write-Host "LLM role boundary: $LlmRoleBoundaryReport"
Write-Host "Instruction-model eval: $TransformerEvalReport"
Write-Host "Instruction-model report: $TransformerMarkdownReport"
Write-Host "Scope contract: $ScopeContractReport"
Write-Host "Project completion: $ProjectCompletionReport"
Write-Host "Final delivery acceptance: $FinalDeliveryAcceptanceReport"
Write-Host "Final strategy quality status: $FinalStrategyQualityStatusReport"
Write-Host "Production runtime monitoring: $ProductionRuntimeMonitoringReport"
Write-Host "QLoRA next-stage boundary: $QloraNextStageReport"
Write-Host "Model risk register: $ModelRiskRegisterReport"
Write-Host "Production approval: $ProductionApprovalReport"
Write-Host "Behavioral revalidation: $BehavioralRevalidationReport"
Write-Host "Behavioral revalidation proof: $BehavioralRevalidationProofReport"
Write-Host "Human-likeness evidence: $HumanLikenessEvidenceReport"
Write-Host "Human-likeness claim gate: $HumanLikenessClaimGateReport"
Write-Host "Bet/timing calibration: $BetTimingCalibrationReport"
Write-Host "Hole-card data quality: $HoleCardDataQualityReport"
Write-Host "Strategy stack maturity: $StrategyStackMaturityReport"
Write-Host "Raw model status: $RawModelStatusReport"
Write-Host "Challenger strategy quality: $ChallengerStrategyQualityReport"
Write-Host "Client handoff: $ClientHandoffReport"
Write-Host "Training cluster requirements: $TrainingClusterReport"
Write-Host "Today acceptance training: $TodayTrainingReport"
Write-Host "Multi-agent training status: $MultiAgentTrainingStatusReport"
Write-Host "ZIP: $ZipPath"

