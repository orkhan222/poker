# Model Review Response

The review feedback was valid. The previous setup was not acceptable as a
production-ready model evaluation because the main issue was the data and
feature pipeline, not only the model architecture.

## What Was Fixed

- Training now uses the representative CSV dataset instead of the tiny
  `sample_poker_log.jsonl` smoke-test file.
- Rows where OCR did not capture two hole cards are filtered by default, so the
  card-strength signal is no longer mostly zero.
- `to_call`, `min_raise`, and action-time pot values are reconstructed from
  `actions.csv` and `stack_events.csv`.
- Position encoding was reduced from sparse raw seat labels to stable position
  groups.
- Extremely rare `all_in` labels are merged into `raise` by default.
- Class-weighted training modes were added for imbalance experiments:
  `sqrt_balanced` and `balanced`.
- Evaluation now reports majority baseline, lift versus baseline, cross entropy,
  macro F1, predicted class counts, and per-class precision/recall/F1.

## Delivered Model

The bundled model is:

```text
models\poker_policy.json
```

It was trained with:

```powershell
python scripts\train_policy.py `
  --dataset "C:\Users\user\Desktop\AllFile\dataset" `
  --model-out ".\models\poker_policy.json" `
  --max-examples 0 `
  --epochs 12 `
  --learning-rate 0.015 `
  --class-weighting none
```

The validation comparison showed that class-weighted variants improved minority
pressure but reduced overall validation accuracy on this noisy OCR dataset, so
the delivered model uses the validation-selected setting. The class-weighted
modes remain available for further experiments.

## Current Metrics

```text
examples=150152
accuracy=0.6174
cross_entropy=0.9580
macro_f1=0.3453
majority_baseline_accuracy=0.5973
lift_vs_majority=0.0201
```

## Important Limitation

This is a supervised imitation model trained from OCR/event logs. The current
metrics show that the fixed pipeline performs above the majority-class baseline,
but the model should not be presented as a profitable poker strategy, GTO bot,
or complete autonomous poker agent.

Recommended next stage: improve OCR/card extraction coverage and add a cleaner
hand-history source before changing the model architecture further.
