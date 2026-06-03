# Poker Decision Agent

FastAPI service and reproducible ML research workspace for poker action
prediction from OCR/event-log data.

## Current Delivery Status

```text
repository_audit=PASS
repo_hygiene=PASS
delivery_verification=PASS
model_production_gate=FAIL
```

The repository is packaged and reproducible. The trained poker policy is not
approved for production decision-policy deployment because the dataset audit
still contains blocker findings and validation metrics remain below deployment
thresholds.

## Project Structure

```text
.
|-- poker_agent/
|-- scripts/
|-- configs/
|   |-- dataset/
|   |-- model/
|   |-- training/
|   |-- evaluation/
|   |-- inference/
|   |-- logging/
|   |-- prompts/
|   `-- experiments/
|-- evaluation/
|-- reports/
|-- models/
|-- release/
|-- install.ps1
|-- run_server.ps1
|-- complete_delivery.ps1
|-- verify_delivery.ps1
`-- README.md
```

## Install

```powershell
cd "C:\Users\user\Desktop\Secop\files-mentioned-by-the-user-poker-2"
.\install.ps1
```

## Run API

```powershell
.\run_server.ps1
```

Open:

```text
http://127.0.0.1:8001/predict
http://127.0.0.1:8001/docs
http://127.0.0.1:8001/health.json
```

## Hydra Experiments

All experiments are launched through one entrypoint:

```powershell
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=<name> python_executable=.venv/Scripts/python.exe
```

Configured experiments:

```text
repo_audit
audit_dataset
train_single_hgb
evaluate_policy
research_compare_tabular
production_gate
train_routed_bundle_smoke
llm_event_extraction_smoke
llm_event_benchmark
llm_event_gold_eval
verify_delivery
```

Example commands:

```powershell
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=repo_audit python_executable=.venv/Scripts/python.exe
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=llm_event_benchmark python_executable=.venv/Scripts/python.exe
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=llm_event_gold_eval python_executable=.venv/Scripts/python.exe
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=verify_delivery python_executable=.venv/Scripts/python.exe
```

Override example:

```powershell
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=train_single_hgb training.max_examples=5000 model.max_iter=40
```

Each Hydra run writes:

```text
reports\hydra\<experiment-name>\<timestamp>\
|-- resolved_config.yaml
|-- command.txt
|-- stdout.txt
|-- stderr.txt
`-- run.json
```

## LLM/Text Event Results

Objective: convert OCR/dealer records into structured poker events for cleaner
betting-history reconstruction.

Weak-label benchmark on 1000 records:

```text
value_only_baseline: event_accuracy=0.4150, macro_f1=0.3284
local_rules:         event_accuracy=1.0000, macro_f1=1.0000
```

Gold-label evaluation on 24 manually specified examples:

```text
minimal_action_only:      event_accuracy=0.6667, macro_f1=0.4091
permissive_prompt_rules:  event_accuracy=0.8333, macro_f1=0.8545
strict_schema_rules:      event_accuracy=1.0000, macro_f1=1.0000
```

Known extraction limitation:

```text
strict_schema_rules card_exact_match=0.8000
```

The remaining card error is caused by permissive card-token parsing. The next
milestone is strict card validation against rank/suit grammar before using
extracted cards as supervised labels.

## Key Reports

```text
reports\repository_audit.json
reports\repo_hygiene.json
reports\dataset_audit.json
reports\production_gate.json
reports\llm_event_benchmark.json
reports\llm_event_gold_eval.json
reports\llm_event_gold_report.md
reports\delivery_verification.json
```

## Full Delivery Build

```powershell
.\complete_delivery.ps1 -SkipTrain -AllowGateFailure
```

Use `-SkipTrain` when rebuilding the package around the existing model. Remove
it when retraining is required.

## Verify Delivery

```powershell
.\verify_delivery.ps1
```

Expected result:

```text
"status": "PASS"
```

## Latest Model Metrics

```text
policy=hist_gradient_boosting
split=stratified_hand_group_holdout
valid_accuracy=0.6798
valid_balanced_accuracy=0.4415
valid_macro_f1=0.4135
valid_weighted_f1=0.6636
valid_majority_baseline_accuracy=0.7029
valid_lift_vs_majority=-0.0231
```

The model should be treated as a corrected research/API deliverable, not as an
approved profitable strategy engine.

## Remaining Risks

- Hole-card coverage is too low for strong card-strength modeling.
- Target distribution is imbalanced and fold-dominant.
- Current model does not beat the majority-class accuracy baseline on the
  strict holdout split.
- Gold event extraction fixture is small and must be expanded with reviewed
  production logs.
