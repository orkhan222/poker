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
  "model": "C:\\Users\\user\\Desktop\\Secop\\files-mentioned-by-the-user-poker-2\\models\\poker_policy.json",
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
  "board_cards": [2c, 3d, QS],
  "pot": 2.5,
  "to_call": 1.0,
  "stack": 100.0,
  "min_raise": 2.0,
  "player_count": 6
}
```

## Retrain

```powershell
.\train_and_evaluate.ps1
```

The default dataset path is:

```text
C:\Users\user\Desktop\AllFile\dataset
```

The retraining script writes:

```text
models\poker_policy.json
```

## Latest Model Metrics

```text
examples=150152
accuracy=0.6174
cross_entropy=0.9580
macro_f1=0.3453
majority_baseline_accuracy=0.5973
lift_vs_majority=0.0201
```

The model is an OCR-log imitation model. It should be presented as a decision
prediction API, not as a guaranteed profitable poker strategy.

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
