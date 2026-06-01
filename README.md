# Poker Decision Agent

Poker Decision Agent is a FastAPI service that predicts poker actions from a
structured game state. The project includes the training pipeline, evaluation
tools, bundled model file, local installer, Docker files, and delivery notes.

## Project Layout

```text
.
|-- build_poker_dataset_optimized.py
|-- poker_agent/
|   |-- agents.py
|   |-- evaluator.py
|   |-- features.py
|   |-- model.py
|   |-- schemas.py
|   `-- service.py
|-- scripts/
|   |-- train_policy.py
|   `-- evaluate_policy.py
|-- models/
|   |-- poker_policy.joblib
|   `-- poker_policy.json
|-- install.ps1
|-- run_server.ps1
|-- train_and_evaluate.ps1
|-- verify_docker.ps1
|-- Dockerfile
|-- docker-compose.yml
|-- USAGE.md
`-- DELIVERY_NOTE_AZ.md
```

## Current Model Pipeline

The training pipeline no longer uses the tiny sample log as the main dataset.
It trains from the CSV dataset containing `hands.csv`, `players.csv`,
`actions.csv`, and `stack_events.csv`.

Implemented fixes:

- Rows with missing hole cards are filtered by default, so the card strength
  signal is not mostly zero.
- `pot`, `to_call`, and `min_raise` are reconstructed at action time from the
  hand/action sequence and negative stack events.
- Raw seat labels such as `Player1_Bottom` are compressed into stable position
  groups instead of hundreds of sparse one-hot labels.
- Rare `all_in` labels are merged into `raise` by default because the OCR logs
  contain too few all-in examples for a reliable standalone class.
- The trainer supports `none`, `sqrt_balanced`, and `balanced` loss weighting.
  The bundled model uses the validation-selected setting.
- The delivered model is now a non-linear `ExtraTreesClassifier` ensemble saved
  as a joblib artifact. The older JSON softmax model is kept only as a fallback.
- `scikit-learn`, `numpy`, `scipy`, and `joblib` are pinned in
  `requirements.txt` because joblib model artifacts are version-sensitive.
- Evaluation reports majority baseline, lift versus baseline, cross entropy,
  macro F1, predicted class counts, and per-class metrics.

## Latest Evaluation

Bundled model:

```text
models\poker_policy.joblib
```

Training command used for the delivered model:

```powershell
python scripts\train_policy.py `
  --dataset "C:\Users\user\Desktop\AllFile\dataset" `
  --model-out ".\models\poker_policy.joblib" `
  --policy extra_trees `
  --max-examples 0 `
  --n-estimators 120 `
  --class-weighting sqrt_balanced `
  --max-class-weight 6
```

Full filtered dataset evaluation:

```text
examples=150152
accuracy=0.6501
cross_entropy=0.8773
macro_f1=0.4665
majority_baseline_accuracy=0.5948
lift_vs_majority=0.0553
```

Important note: these metrics are supervised imitation metrics from OCR/event
logs. The ensemble is above the majority-class baseline and is stronger than
the previous softmax baseline, but it is still not a profitability or
GTO-strength claim.

## Local Run

```powershell
cd "C:\Users\user\Desktop\Secop\files-mentioned-by-the-user-poker-2"
.\install.ps1
.\run_server.ps1
```

Open:

```text
http://127.0.0.1:8000/predict
```

API docs:

```text
http://127.0.0.1:8000/docs
```

Health / model status:

```text
http://127.0.0.1:8000/health.json
```

Stop the server with `Ctrl+C`.

## Train And Evaluate

Default:

```powershell
.\train_and_evaluate.ps1
```

Custom dataset:

```powershell
.\train_and_evaluate.ps1 -Dataset "C:\path\to\dataset"
```

Class-weighted experiment:

```powershell
.\train_and_evaluate.ps1 -ClassWeighting sqrt_balanced
```

## API Request Example

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/predict" `
  -ContentType "application/json" `
  -Body '{"position":"BTN","street":"preflop","hole_cards":["Ah","Kd"],"board_cards":[],"pot":2.5,"to_call":1.0,"stack":100.0,"min_raise":2.0,"player_count":6}'
```

## Docker

```powershell
docker build -t poker-decision-agent:latest .
docker run --rm -p 8000:8000 poker-decision-agent:latest
```

Docker Desktop must be running. On this workstation Docker Desktop previously
returned a daemon `500 Internal Server Error`; the Docker files are included,
but Docker itself must be healthy to run the verification script.
