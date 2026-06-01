# Poker Decision Agent

Poker Decision Agent is a FastAPI service for real-time poker action prediction.
The package includes the API, browser form, bundled trained model, local run
scripts, Docker files, evaluation tooling, and a research experiment runner for
model comparison.

## Project Layout

```text
.
|-- poker_agent/
|   |-- agents.py
|   |-- evaluator.py
|   |-- features.py
|   |-- model.py
|   |-- schemas.py
|   |-- sequence_models.py
|   `-- service.py
|-- scripts/
|   |-- audit_dataset.py
|   |-- evaluate_policy.py
|   |-- research_experiment.py
|   `-- train_policy.py
|-- models/
|   |-- poker_policy.joblib
|   `-- poker_policy.json
|-- install.ps1
|-- run_server.ps1
|-- train_and_evaluate.ps1
|-- verify_docker.ps1
|-- requirements.txt
|-- requirements-research.txt
|-- USAGE.md
|-- MODEL_REVIEW_RESPONSE.md
|-- RESEARCH_MODEL_DESIGN_AZ.md
|-- SENIOR_ML_REVIEW.md
`-- DELIVERY_NOTE_AZ.md
```

## Current Model Pipeline

The current pipeline is no longer the tiny sample-log baseline. Training uses
the real CSV dataset with `hands.csv`, `players.csv`, `actions.csv`, and
`stack_events.csv`.

Implemented research fixes:

- Missing hole cards are dropped by default, with `flag` and `keep` modes
  available for ablation experiments.
- `to_call`, `min_raise`, and action-time pot are reconstructed from ordered
  actions plus negative stack events.
- Board cards are street-visible only: preflop sees 0 board cards, flop sees 3,
  turn sees 4, river sees 5. This avoids future-card leakage.
- Final-hand pot is not used as an action-time feature.
- Sparse raw seat labels are compressed into stable position groups.
- Rare `all_in` labels are merged into `raise` by default because the OCR data
  does not contain enough reliable all-in samples.
- Temporal betting context features are added: action count, aggression ratio,
  call/check/fold counts, players-acted ratio, hero commitment, table pressure,
  last aggressor group, and facing-bet indicators.
- API requests can include `betting_history` / `action_history`.
- The evaluator reports accuracy, balanced accuracy, cross entropy, Brier loss,
  ECE@10, macro F1, weighted F1, precision/recall/F1 per class, predicted
  class counts, and confusion matrix.
- `train_policy.py` now defaults to stratified hand-group holdout instead of
  random action split.
- `scripts/audit_dataset.py` produces dataset quality findings before model
  training.
- The model layer supports `hist_gradient_boosting`, `extra_trees`,
  `random_forest`, `mlp`, and optional `xgboost`, `lightgbm`, `catboost`.
- `poker_agent/sequence_models.py` contains a Transformer/focal-loss scaffold
  for future full-sequence experiments.

## Latest Honest Evaluation

Bundled model:

```text
models\poker_policy.joblib
```

Training command used for the delivered model:

```powershell
python scripts\train_policy.py `
  --dataset "C:\Users\user\Desktop\AllFile\dataset" `
  --model-out ".\models\poker_policy.joblib" `
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
```

Validation result with stratified hand-group holdout:

```text
examples=150152
train_examples=127613
valid_examples=22539
valid_accuracy=0.6798
valid_cross_entropy=0.8077
valid_balanced_accuracy=0.4415
valid_macro_f1=0.4135
valid_weighted_f1=0.6636
valid_brier_loss=0.4432
valid_ece_10=0.0762
valid_majority_baseline_accuracy=0.7029
valid_lift_vs_majority=-0.0231
```

An earlier experiment produced a much higher score, but it was rejected because
one stack-event feature leaked the current action amount. The delivered metrics
above are the honest leakage-safer numbers. Under group holdout, accuracy is
still below the majority-class baseline, so this should not be approved as a
production decision model yet.

## Dataset Audit

Run the audit before any model claim:

```powershell
python scripts\audit_dataset.py `
  --dataset "C:\Users\user\Desktop\AllFile\dataset" `
  --out ".\reports\dataset_audit.json" `
  --max-feature-examples 50000 `
  --missing-hole-cards flag
```

## Research Model Comparison

Run a hand-level holdout comparison:

```powershell
python scripts\research_experiment.py `
  --dataset "C:\Users\user\Desktop\AllFile\dataset" `
  --out-dir ".\research_runs\full" `
  --policies hist_gradient_boosting,extra_trees,random_forest,mlp `
  --class-weighting sqrt_balanced `
  --max-class-weight 6 `
  --missing-hole-cards drop `
  --save-models
```

Optional research backends:

```powershell
pip install -r requirements-research.txt
```

Then include:

```text
xgboost,lightgbm,catboost
```

The comparison report is written to:

```text
research_runs\full\model_comparison.json
```

## Local Run

```powershell
cd "C:\Users\user\Desktop\Secop\files-mentioned-by-the-user-poker-2"
.\install.ps1
.\run_server.ps1
```

Open:

```text
http://127.0.0.1:8001/predict
```

API docs:

```text
http://127.0.0.1:8001/docs
```

Health / model status:

```text
http://127.0.0.1:8001/health.json
```

Stop the server with `Ctrl+C`.

## API Request Example

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8001/predict" `
  -ContentType "application/json" `
  -Body '{"position":"BTN","street":"preflop","hole_cards":["Ah","Kd"],"board_cards":[],"pot":2.5,"to_call":1.0,"stack":100.0,"min_raise":2.0,"player_count":6,"betting_history":[]}'
```

## Docker

```powershell
docker build -t poker-decision-agent:latest .
docker run --rm -p 8001:8001 poker-decision-agent:latest
```

Docker Desktop must be running. If Docker returns daemon errors, restart Docker
Desktop and verify with `docker info`.

## Delivery Position

This package is a corrected supervised imitation model and API. It is stronger
than the previous 0.34 baseline in pipeline quality, but the current
stratified hand-group validation result is still below the majority-class
accuracy baseline. It must not be described as a deployable poker strategy,
GTO solver, or guaranteed profitable agent. Next research work should focus on
cleaner OCR/card extraction, source-level holdout evaluation, calibrated
boosting models, and full betting-sequence models.
