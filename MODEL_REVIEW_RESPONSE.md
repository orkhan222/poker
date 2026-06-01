# Response To Model Review

Thank you for the detailed review. I agree with the main point: the previous
0.34 accuracy result should not have been presented as an acceptable model
result. The root problem was first the data and feature pipeline, and after
those fixes the remaining issue was that the model itself was still too simple.

I have reworked the project so the training and evaluation flow now addresses
both points: pipeline correctness and a stronger non-linear supervised model.

## Changes Made

1. Representative dataset

   The trainer now uses the real CSV dataset containing:

   ```text
   hands.csv
   players.csv
   actions.csv
   stack_events.csv
   ```

   The small `sample_poker_log.jsonl` file is kept only as a smoke-test sample,
   not as the main training source.

2. Missing hole cards

   Training rows where OCR did not capture two hole cards are filtered by
   default. This prevents `strength_proxy` from being zero for most training
   samples.

3. Pot-odds features

   `to_call`, `min_raise`, and action-time pot are reconstructed from the
   action sequence and negative stack events. These values are no longer left at
   zero inside `load_training_examples`.

4. Position encoding

   Sparse raw position labels such as `Player1_Bottom` are compressed into
   stable position groups. This avoids hundreds of low-value one-hot features.

5. Rare all-in labels

   `all_in` appears too rarely in the OCR labels to train as a reliable
   standalone class. It is merged into `raise` by default, while the original
   class can still be kept with a training flag for experiments.

6. Class imbalance handling

   The trainer supports `none`, `sqrt_balanced`, and `balanced` weighting modes.
   The delivered model uses `sqrt_balanced` sample weighting.

7. Model architecture

   The delivered artifact is no longer the simple in-repo softmax baseline. It
   is a non-linear `ExtraTreesClassifier` ensemble saved as a joblib model. The
   softmax JSON model is kept only as a compatibility fallback.

   The sklearn/numpy/joblib versions are pinned in `requirements.txt` because
   persisted sklearn models are version-sensitive.

8. Evaluation reporting

   Evaluation reports:

   ```text
   accuracy
   cross_entropy
   macro_f1
   majority_baseline_accuracy
   lift_vs_majority
   class_counts
   predicted_class_counts
   per_class precision / recall / f1
   ```

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
  --policy extra_trees `
  --max-examples 0 `
  --n-estimators 120 `
  --class-weighting sqrt_balanced `
  --max-class-weight 6
```

Current validation result on the filtered real dataset:

```text
examples=150152
train_examples=127629
valid_examples=22523
accuracy=0.6501
cross_entropy=0.8773
macro_f1=0.4665
majority_baseline_accuracy=0.5948
lift_vs_majority=0.0553
```

## Honest Status

This is now a corrected supervised imitation model with a working API, fixed
training pipeline, class-weighted training, and a non-linear ensemble model. It
is above the majority-class baseline and stronger than the previous softmax
baseline.

I would still not present it as a finished profitable poker strategy, GTO
engine, or fully production-grade autonomous poker agent. It is a stronger
professional supervised baseline for the current OCR/event-log dataset.

The right next step is to continue with the remaining stages:

1. Improve OCR/card extraction coverage.
2. Add a cleaner hand-history source if available.
3. Add a source-file/session-level holdout split to reduce leakage risk.
4. Run stronger model experiments only after data quality improves.
5. Track macro F1 and per-class recall, not only accuracy.

In short: the previous criticism was valid. The current package fixes the
pipeline-level issues and upgrades the model from a simple softmax baseline to a
weighted non-linear ensemble. It should still be delivered with the limitations
above clearly stated.
