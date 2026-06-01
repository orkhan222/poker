# Response To Model Review

Thank you for the detailed review. I agree with the main criticism: the old
0.34 result was not acceptable as a model-quality claim. The core issue was the
data and feature pipeline first, and the simple model architecture second.

The project has now been reworked into a stronger supervised imitation pipeline
with leakage controls, temporal betting features, class-imbalance handling, and
a model-comparison path.

## What Was Fixed

1. Representative dataset

   Training now uses the real CSV dataset:

   ```text
   hands.csv
   players.csv
   actions.csv
   stack_events.csv
   ```

   The tiny `sample_poker_log.jsonl` file is only a smoke-test sample.

2. Missing hole cards

   Rows where OCR did not capture two hole cards are dropped by default. The
   loader also supports `flag` and `keep` modes for ablation experiments.

3. Pot odds and betting amounts

   `to_call`, `min_raise`, and action-time pot are reconstructed from the
   ordered action stream and stack-event deltas. They are no longer left as zero
   inside `load_training_examples`.

4. Future-card and final-pot leakage

   Board cards are limited to the current street: preflop 0, flop 3, turn 4,
   river 5. The model no longer sees final-board information at earlier
   decisions. Final-hand pot is also not used as an action-time feature.

5. Position encoding

   Sparse labels such as `Player1_Bottom` are compressed into stable position
   groups. This reduces noisy one-hot feature explosion.

6. Rare all-in labels

   `all_in` is merged into `raise` by default because the OCR labels contain too
   few reliable all-in examples. The original class can still be kept for
   experiments.

7. Temporal betting features

   The feature pipeline now includes action count, aggressive count,
   call/check/fold count, aggression ratio, players-acted ratio, hero
   commitment, table pressure, facing-bet flags, call price, raise pressure, and
   last-aggressor group. API users can also pass `betting_history`.

8. Class imbalance

   The trainer supports `none`, `sqrt_balanced`, and `balanced` sample
   weighting. This directly addresses fold dominance without pretending that
   accuracy alone is sufficient.

9. Model families

   The code now supports:

   ```text
   hist_gradient_boosting
   extra_trees
   random_forest
   mlp
   xgboost      optional
   lightgbm     optional
   catboost     optional
   softmax      fallback baseline
   ```

   A Transformer/focal-loss scaffold is included in
   `poker_agent/sequence_models.py` for future full betting-sequence research.

10. Evaluation

   Evaluation now reports accuracy, balanced accuracy, cross entropy, Brier
   loss, ECE@10, macro F1, weighted F1, precision/recall/F1 per class,
   majority baseline, lift versus baseline, predicted class counts, and
   confusion matrix.

## Current Delivered Model

Bundled model:

```text
models\poker_policy.joblib
```

Training command:

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

Validation result:

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

An earlier score was higher, but it was rejected because a current-action stack
amount feature leaked label information. The metrics above are the honest
leakage-safer result. It is not sufficient for production approval because the
group-holdout accuracy is still below the majority-class baseline and minority
action recall remains weak.

## Research Experiment Runner

The new comparison script runs a hand-level holdout:

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

The report is saved as:

```text
research_runs\full\model_comparison.json
```

The dataset audit script should be run before model claims:

```powershell
python scripts\audit_dataset.py `
  --dataset "C:\Users\user\Desktop\AllFile\dataset" `
  --out ".\reports\dataset_audit.json" `
  --missing-hole-cards flag
```

## Honest Status

This is now a corrected supervised research scaffold with a working API,
feature extraction fixes, class-weighted training, temporal betting features,
dataset audit tooling, and model-comparison tooling.

It should still not be marketed as a finished profitable poker bot or GTO
engine. The current result is a technical rejection for production deployment;
the next research priorities are cleaner card/OCR coverage, source-level
holdout splits, calibrated boosting experiments, richer betting histories, and
sequence models trained on full hand trajectories.
