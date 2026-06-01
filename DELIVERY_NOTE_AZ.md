# Poker Decision Agent - Tehvil Qeydi

Bu paket musteriyye API ve ML supervised imitation baseline kimi tehvil vermek
ucun hazirlanib. Layihede FastAPI servis, browser sehifesi, bundled model,
installer, run script, Docker fayllari, training/evaluation scriptleri,
research experiment runner ve istifade senedleri var.

## Daxil Olanlar

- FastAPI backend
- Browser application sehifesi: `GET /predict`
- Prediction API: `POST /predict`
- Health endpoint: `GET /health.json`
- Bundled model: `models\poker_policy.joblib`
- Local installer: `install.ps1`
- Local run script: `run_server.ps1`
- Retrain/evaluate script: `train_and_evaluate.ps1`
- Research comparison script: `scripts\research_experiment.py`
- Optional research dependencies: `requirements-research.txt`
- Dockerfile ve `docker-compose.yml`
- Docker yoxlama scripti: `verify_docker.ps1`
- Istifade senedi: `USAGE.md`
- Texniki README: `README.md`
- Model review cavabi: `MODEL_REVIEW_RESPONSE.md`
- Research dizayn qeydi: `RESEARCH_MODEL_DESIGN_AZ.md`

## Edilen Esas Duzelisler

Evvelki review-da qeyd olunan data/feature problemleri kod seviyyesinde
duzeldilib:

- Model repo daxilindeki kicik sample fayldan yox, esas CSV datasetden train
  olunur.
- Hole cards olmayan row-lar default olaraq trainingden cixarilir.
- `to_call`, `min_raise` ve action-vaxti pot deyerleri `actions.csv` +
  `stack_events.csv` ardicilligindan hesablanir.
- Board cards yalniz cari street ucun gorunen kartlarla mehdudlasdirilir:
  preflop 0, flop 3, turn 4, river 5. Bu future-card leakage riskini azaldir.
- Final hand pot action-vaxti feature kimi istifade olunmur.
- `Player1_Bottom` kimi coxsayli seat label-lari yerine stabil position
  qruplari istifade olunur.
- Cox nadir `all_in` sinfi default olaraq `raise` ile birlesdirilir.
- Temporal betting features elave olunub: action count, aggression ratio,
  call/check/fold counts, players acted ratio, hero commitment, table pressure,
  facing bet flag ve last aggressor group.
- Class imbalance ucun trainerde `none`, `sqrt_balanced`, `balanced` rejimleri
  var.
- Model layer artiq `hist_gradient_boosting`, `extra_trees`, `random_forest`,
  `mlp`, optional `xgboost`, `lightgbm`, `catboost` modellerini destekleyir.
- Transformer/focal-loss research scaffold `poker_agent\sequence_models.py`
  faylina elave olunub.
- Evaluation majority baseline, lift, cross entropy, macro F1, weighted F1,
  per-class precision/recall/F1 ve confusion matrix cixarir.

## Son Model Neticesi

Bundled model:

```text
models\poker_policy.joblib
```

Filtered real dataset validation:

```text
examples=150152
train_examples=127629
valid_examples=22523
valid_accuracy=0.6544
valid_cross_entropy=0.8011
valid_macro_f1=0.5138
valid_weighted_f1=0.6503
valid_majority_baseline_accuracy=0.5948
valid_lift_vs_majority=0.0596
```

Qeyd: Daha yuksek bir experiment neticesi redd edildi, cunki current-action
stack amount feature label leakage yaradirirdi. Yuxaridaki reqemler
leakage-safe honest validation neticesidir.

Bu, OCR/event log-larindan supervised imitation metric-dir. Netice
majority/fold baseline-dan yuxaridir, amma bunu profitable poker strategy ve ya
GTO seviyye iddiasi kimi teqdim etmek olmaz.

## Local Run

PowerShell-de project folderine kecin:

```powershell
cd "C:\Users\user\Desktop\Secop\files-mentioned-by-the-user-poker-2"
```

Ilk defe:

```powershell
.\install.ps1
```

Serveri baslatmaq:

```powershell
.\run_server.ps1
```

Application:

```text
http://127.0.0.1:8000/predict
```

API docs:

```text
http://127.0.0.1:8000/docs
```

Health:

```text
http://127.0.0.1:8000/health.json
```

Serveri dayandirmaq ucun terminalda `Ctrl+C` basin.

## Research Compare

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

Optional backend-ler ucun:

```powershell
pip install -r requirements-research.txt
```

## Docker Run

Docker Desktop islek olmalidir.

```powershell
docker build -t poker-decision-agent:latest .
docker run --rm -p 8001:8001 poker-decision-agent:latest
```

Docker yoxlama:

```powershell
.\verify_docker.ps1
```

Bu komputerde Docker Desktop daemon evvel `500 Internal Server Error`
qaytarirdi. Bu Docker fayllarinin yox, Docker Desktop servisinin problemidir.

## Musteriye Qisa Izah

```text
Poker Decision Agent real-time game state JSON qebul eden FastAPI servisidir.
Model OCR/event log-larindan qurulan real CSV dataset uzerinde train edilib.
Training pipeline-da missing-card row-lar filtreden kecir, action-vaxti
to_call/min_raise/pot feature-lari stack event ardicilligindan hesablanir,
future-card leakage qarsisi alinir ve evaluation majority baseline ile birlikde
report olunur.
```
