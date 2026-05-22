# Poker Agent ML Project

This project turns HoldemHub-style poker OCR/event logs into model-ready CSV
tables, trains a supervised poker action policy, evaluates it, and exposes it
through a FastAPI `/predict` endpoint.

## Project Layout

```text
.
├── build_poker_dataset_optimized.py   # JSON/JSONL -> CSV dataset builder
├── poker_agent/
│   ├── agents.py                      # Rule, ML, and LLM-style agent classes
│   ├── evaluator.py                   # Accuracy / cross entropy evaluation
│   ├── features.py                    # Structured game state -> feature vector
│   ├── model.py                       # Lightweight softmax policy model
│   ├── schemas.py                     # Request/response dataclasses
│   └── service.py                     # FastAPI microservice
├── scripts/
│   ├── train_policy.py                # Train supervised baseline
│   └── evaluate_policy.py             # Evaluate saved model
├── sample_poker_log.jsonl             # Tiny smoke-test input
└── requirements.txt
```

## 1. Build Dataset

```powershell
python build_poker_dataset_optimized.py `
  --input "C:\Users\user\Desktop\AllFile\logs" `
  --out-dir "C:\Users\user\Desktop\AllFile\dataset"
```

It writes:

- `hands.csv`
- `players.csv`
- `actions.csv`
- `stack_events.csv`

## 2. Train Supervised Policy

```powershell
python scripts\train_policy.py `
  --dataset "C:\Users\user\Desktop\AllFile\dataset" `
  --model-out "C:\Users\user\Desktop\AllFile\poker_policy.json"
```

## 3. Evaluate

```powershell
python scripts\evaluate_policy.py `
  --dataset "C:\Users\user\Desktop\AllFile\dataset" `
  --model "C:\Users\user\Desktop\AllFile\poker_policy.json"
```

## 4. Serve

```powershell
uvicorn poker_agent.service:app --host 127.0.0.1 --port 8000
```

Example request:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/predict" `
  -ContentType "application/json" `
  -Body '{"position":"BTN","street":"preflop","hole_cards":["Ah","Kd"],"board_cards":[],"pot":2.5,"to_call":1.0,"stack":100.0}'
```

Set `POKER_POLICY_PATH` to load a trained model:

```powershell
$env:POKER_POLICY_PATH="C:\Users\user\Desktop\AllFile\poker_policy.json"
uvicorn poker_agent.service:app --host 127.0.0.1 --port 8000
```

