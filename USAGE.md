# Poker Decision Agent Usage

## Start Locally

```powershell
cd "C:\Users\user\Desktop\Secop\files-mentioned-by-the-user-poker-2"
.\install.ps1
.\run_server.ps1
```

Open:

```text
http://127.0.0.1:8001/predict
```

Stop with `Ctrl+C`.

## Health / Model Status

```text
http://127.0.0.1:8001/health.json
```

Example response:

```json
{
  "status": "ok",
  "model": "C:\\Users\\user\\Desktop\\Secop\\files-mentioned-by-the-user-poker-2\\models\\poker_policy.joblib",
  "model_status": "loaded"
}
```

Field meanings:

- `status`: API service status.
- `model`: policy model loaded by the service.
- `model_status`: `loaded` means the bundled model file was found and loaded.

## API Docs

```text
http://127.0.0.1:8001/docs
```

Visible endpoint groups:

- `Prediction`: poker action prediction.
- `POST /predict`: accepts game state JSON and returns action probabilities.
- `System`: service and model status.
- `GET /health.json`: confirms that the API and model are available.

## Prediction Request

```json
{
  "position": "BTN",
  "street": "preflop",
  "hole_cards": ["Ah", "Kd"],
  "board_cards": [],
  "pot": 2.5,
  "to_call": 1.0,
  "stack": 100.0,
  "min_raise": 2.0,
  "player_count": 6,
  "betting_history": [
    {"position": "UTG", "action": "fold"},
    {"position": "CO", "action": "raise", "amount": 2.5}
  ]
}
```

`betting_history` is optional, but recommended when available. It lets the
feature pipeline add temporal betting context such as aggression count, last
aggressor, and players-acted ratio.

## Retrain Delivered Model

```powershell
.\train_and_evaluate.ps1
```

Default dataset path:

```text
C:\Users\user\Desktop\AllFile\dataset
```

The retraining script writes:

```text
models\poker_policy.joblib
```

## Research Model Comparison

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

For XGBoost, LightGBM, CatBoost, and Transformer research dependencies:

```powershell
pip install -r requirements-research.txt
```

## Latest Model Metrics

```text
examples=150152
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

These are supervised imitation metrics from OCR/event logs. Present the package
as a poker action prediction API prototype, not as an approved production model
or guaranteed profitable poker strategy.

## Docker

```powershell
docker build -t poker-decision-agent:latest .
docker run --rm -p 8001:8001 poker-decision-agent:latest
```

Docker verification:

```powershell
.\verify_docker.ps1
```

Docker Desktop must be running and healthy before Docker commands will work.
