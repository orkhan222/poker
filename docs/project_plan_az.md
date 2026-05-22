# Poker ML Agent Layihə Planı

## Məqsəd

Bu layihənin məqsədi poker oyun vəziyyətindən optimal qərar çıxaran agent
hazırlamaqdır. Agent tarixi insan oyunlarından öyrənir, daha sonra evaluation
metrikləri ilə ölçülür və FastAPI mikroservisi kimi istifadə olunur.

## Faza 1: Dataset Hazırlığı

Giriş məlumatı HoldemHub/OCR event log fayllarıdır. Fayllar `json` və ya
`jsonl` formatında ola bilər.

Çıxış cədvəlləri:

- `hands.csv`: hər poker hand üçün ümumi məlumat.
- `players.csv`: hər hand daxilində oyunçu məlumatı.
- `actions.csv`: oyunçuların fold/call/raise kimi qərarları.
- `stack_events.csv`: stack dəyişiklikləri.

Əsas skript:

```powershell
python build_poker_dataset_optimized.py --input "C:\path\logs" --out-dir "C:\path\dataset"
```

## Faza 2: Supervised Policy Model

Model `actions.csv` faylındakı insan qərarlarını label kimi istifadə edir.
Input xüsusiyyətləri:

- street: preflop/flop/turn/river
- hole cards
- board cards
- pot
- stack
- position
- sadə hand strength proxy

Train:

```powershell
python scripts\train_policy.py --dataset "C:\path\dataset" --model-out "C:\path\poker_policy.json"
```

## Faza 3: Evaluation

Əsas metriklər:

- action accuracy
- cross entropy loss

Evaluation:

```powershell
python scripts\evaluate_policy.py --dataset "C:\path\dataset" --model "C:\path\poker_policy.json"
```

## Faza 4: Deployment

Servis FastAPI ilə işləyir.

```powershell
$env:POKER_POLICY_PATH="C:\path\poker_policy.json"
uvicorn poker_agent.service:app --host 127.0.0.1 --port 8000
```

Endpoint:

```text
POST /predict
```

Nümunə input:

```json
{
  "position": "BTN",
  "street": "preflop",
  "hole_cards": ["Ah", "Kd"],
  "board_cards": [],
  "pot": 2.5,
  "to_call": 1.0,
  "stack": 100.0,
  "player_count": 6
}
```

Nümunə output:

```json
{
  "action": "call",
  "probabilities": {
    "fold": 0.12,
    "call": 0.51,
    "raise": 0.22,
    "check": 0.10,
    "bet": 0.05
  }
}
```

## Növbəti İnkişaf Addımları

Bu versiya layihənin işlək baseline formasıdır. Sonrakı mərhələlərdə:

- daha zəngin feature engineering,
- PyTorch policy network,
- self-play environment,
- PPO fine-tuning,
- real LLM API adapteri,
- latency/cost benchmark

əlavə edilə bilər.

