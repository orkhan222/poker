# Poker Decision Agent - Tehvil Qeydi

Bu paket musteriyye API baseline kimi tehvil vermek ucun hazirlanib. Layihede
local installer, run script, bundled model, Docker fayllari, training/evaluation
scriptleri ve istifade senedi var.

## Daxil Olanlar

- FastAPI backend
- Browser application sehifesi: `GET /predict`
- Prediction API: `POST /predict`
- Health endpoint: `GET /health.json`
- Bundled model: `models\poker_policy.joblib`
- Local installer: `install.ps1`
- Local run script: `run_server.ps1`
- Retrain/evaluate script: `train_and_evaluate.ps1`
- Dockerfile ve `docker-compose.yml`
- Docker yoxlama scripti: `verify_docker.ps1`
- Istifade senedi: `USAGE.md`
- Texniki README: `README.md`
- Review cavabi: `MODEL_REVIEW_RESPONSE.md`

## Edilen Esas Duzelisler

Evvelki review-da qeyd olunan data/feature problemleri kod seviyyesinde
duzeldilib:

- Model repo daxilindeki kicik sample fayldan yox, esas CSV datasetden train
  olunur.
- Hole cards olmayan row-lar default olaraq trainingden cixarilir.
- `to_call`, `min_raise` ve action-vaxti pot deyerleri `actions.csv` +
  `stack_events.csv` ardicilligindan hesablanir.
- `Player1_Bottom` kimi coxsayli seat label-lari yerine stabil position
  qruplari istifade olunur.
- Cox nadir `all_in` sinfi default olaraq `raise` ile birlesdirilir.
- Class imbalance ucun trainerde `none`, `sqrt_balanced`, `balanced` rejimleri
  var. Bundled model validation neticesine gore secilen parametrle train
  edilib.
- Bundled model artiq simple softmax deyil, `ExtraTreesClassifier` non-linear
  ensemble modelidir. Kohnə JSON softmax fayli yalniz fallback ucun saxlanilib.
- Evaluation majority baseline, lift, cross entropy, macro F1 ve per-class
  metrikleri cixarir.

## Son Model Neticesi

Bundled model:

```text
models\poker_policy.joblib
```

Filtered real dataset:

```text
examples=150152
accuracy=0.6501
cross_entropy=0.8773
macro_f1=0.4665
majority_baseline_accuracy=0.5948
lift_vs_majority=0.0553
```

Qeyd: Bu, OCR/event log-larindan supervised imitation metric-dir. Netice
majority/fold baseline-dan yuxaridir, amma bunu profitable poker strategy veya
GTO seviyye iddiasi kimi teqdim etmek olmaz. Model baseline softmax-dan daha
guclu professional ensemble seviyyesine qaldirilib, amma tam avtonom poker
agent kimi teqdim olunmamali ve novbeti data/model merheleleri davam
etdirilmelidir.

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
http://127.0.0.1:8001/predict
```

API docs:

```text
http://127.0.0.1:8001/docs
```

Health:

```text
http://127.0.0.1:8001/health.json
```

Serveri dayandirmaq ucun terminalda `Ctrl+C` basin.

## Retrain

```powershell
.\train_and_evaluate.ps1
```

Default dataset yolu:

```text
C:\Users\user\Desktop\AllFile\dataset
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
Docker Desktop saglam isleyende yuxaridaki command-lar istifade olunur.

## Musteriye Qisa Izah

Layiheni bu formada teqdim etmek olar:

```text
Poker Decision Agent real-time game state JSON qebul eden FastAPI servisidir.
Model OCR/event log-larindan qurulan real CSV dataset uzerinde train edilib.
Training pipeline-da missing-card row-lar filtreden kecir, action-vaxti
to_call/min_raise/pot feature-lari stack event ardicilligindan hesablanir ve
evaluation majority baseline ile birlikde report olunur.
```

## Musteri Review-una Cavab

Musterinin model review-u ucun hazir cavab fayli:

```text
MODEL_REVIEW_RESPONSE.md
```

Bu faylda evvelki problemler qebul edilir, edilen duzelisler izah olunur ve
qalan limitler aciq yazilir.
