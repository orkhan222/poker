# Senior ML Review And Redesign Decision

## Decision

Rejected for production deployment and rejected for publication-quality model
claims in the current state.

Approved only as a corrected research scaffold and API prototype. The pipeline
is materially better than the original 0.34 baseline, but the current
stratified hand-group validation result is still below the majority-class
accuracy baseline:

```text
valid_accuracy=0.6798
valid_balanced_accuracy=0.4415
valid_macro_f1=0.4135
valid_weighted_f1=0.6636
valid_majority_baseline_accuracy=0.7029
valid_lift_vs_majority=-0.0231
```

The practical interpretation is simple: the model has learned useful minority
signals for some classes, but it is not yet a reliable action policy. Accuracy
alone is misleading because the validation split is fold-heavy.

## 1. Validation Split Was Architecturally Wrong

Current/suboptimal behavior:

The old trainer used random action-level split. That can put actions from the
same hand into both train and validation, sharing cards, positions, stack
trajectory, OCR artifacts, and betting context.

Expected impact:

High. This can inflate metrics and hide generalization failure.

Improved implementation:

- Added `poker_agent/validation.py`.
- `train_policy.py` now defaults to `--split-strategy stratified_hand_group`.
- `research_experiment.py` uses the same group-holdout splitter.

Risk/limitation:

Group holdout is stricter and metrics will look worse. That is the point. A
future production report should move from hand-level holdout to source-file or
session-level holdout if source/session ids are available.

## 2. Dataset Quality Was Not Gated

Current/suboptimal behavior:

The project trained models without a formal data audit. That is unacceptable for
OCR-derived poker logs where missing cards, broken stack events, and class
imbalance dominate model behavior.

Expected impact:

High. Most model improvements will be fake if the data pipeline is not audited.

Improved implementation:

- Added `scripts/audit_dataset.py`.
- It reports action distribution, hole-card coverage, stack-event health,
  feature zero/missing rates, and blocker-level findings.

Run:

```powershell
python scripts\audit_dataset.py `
  --dataset "C:\Users\user\Desktop\AllFile\dataset" `
  --out ".\reports\dataset_audit.json" `
  --missing-hole-cards flag
```

Risk/limitation:

The audit detects symptoms, not root causes. If OCR is wrong, the correct fix is
upstream OCR/card extraction, not another classifier.

## 3. Hole-Card Missingness Remains A Blocker

Current/suboptimal behavior:

Dropping missing hole-card rows improves card-strength signal integrity, but it
biases the training distribution toward hands where OCR succeeded. Keeping them
without special handling makes `strength_proxy` mostly zero.

Expected impact:

Very high. Hole cards are among the strongest features in poker. The smoke audit
showed `strength_proxy` zero for most flag-mode examples.

Improved implementation:

- Loader supports `drop`, `flag`, and `keep`.
- Default training uses `drop`.
- Audit uses `flag` to quantify the missing-card problem.

Recommended next implementation:

- Train two models: visible-hole-card model and no-hole-card/context-only model.
- Route inference based on `hole_card_observed_ratio`.
- Track metrics separately for observed-card and missing-card slices.

Risk/limitation:

The no-card model will mostly learn population tendencies and betting context,
not true hand strength. It can be useful operationally but should not be sold as
strong poker reasoning.

## 4. Feature Reconstruction Is Still Approximate

Current/suboptimal behavior:

`to_call`, `min_raise`, and pot are reconstructed from action order and negative
stack deltas. This is better than leaving them zero, but it is not equivalent to
a canonical hand-history parser.

Expected impact:

High for call/fold/raise decisions. Pot odds and stack pressure are primary
decision variables.

Improved implementation:

- Action-time pot and commitments are reconstructed in `features.py`.
- Final pot is not used as a feature.
- Future board cards are excluded by street.

Risk/limitation:

Stack-event matching by frame window can still be wrong when OCR/action timing
is noisy. The production-grade fix is to ingest canonical hand histories or add
an event reconciliation layer with unit-tested invariants.

## 5. Evaluation Was Too Narrow

Current/suboptimal behavior:

Accuracy and macro F1 were not enough. In imbalanced classification, accuracy
can improve while minority recall collapses. Calibration was also unmeasured.

Expected impact:

Medium to high. Evaluation quality determines whether model selection is real or
cosmetic.

Improved implementation:

- Added balanced accuracy.
- Added Brier loss.
- Added ECE@10 calibration error.
- Kept macro/weighted F1, per-class precision/recall/F1, confusion matrix, and
  majority baseline.

Risk/limitation:

These are still imitation metrics. They do not measure EV, exploitability, or
profitability.

## 6. Model Family Alone Will Not Fix This

Current/suboptimal behavior:

The project previously treated model architecture as the main lever. That is
wrong. With missing cards, noisy labels, and weak splits, XGBoost or a neural net
will mostly overfit better.

Expected impact:

Medium. Stronger models matter after data/split/feature correctness.

Improved implementation:

- `model.py` supports HistGradientBoosting, ExtraTrees, RandomForest, MLP, and
  optional XGBoost/LightGBM/CatBoost.
- `research_experiment.py` compares them under the same group split.

Recommended priority:

1. LightGBM/XGBoost after data audit passes.
2. CatBoost if richer categorical features are retained.
3. MLP only as an ablation.
4. Transformer only after full ordered betting sequences are preserved.

Risk/limitation:

Optional backends add dependency and reproducibility cost. Do not add them to
the production runtime unless they clearly outperform sklearn baselines under
group/session holdout.

## 7. Betting History Is Aggregated, Not Truly Sequential

Current/suboptimal behavior:

The current temporal features summarize action counts and last aggressor. This
is useful, but it loses exact order, bet sizing trajectory, and player-specific
action patterns.

Expected impact:

Medium now, high if the dataset contains reliable full action histories.

Improved implementation:

- Added temporal aggregate features.
- Added `poker_agent/sequence_models.py` with Transformer and focal-loss
  scaffold.

Risk/limitation:

The Transformer scaffold should not be trained on flattened rows. It needs full
hand trajectories with causal masking and strict sequence-time feature cuts.

## 8. Artifact Governance Was Weak

Current/suboptimal behavior:

Model artifacts did not store enough training metadata. That makes review and
reproducibility weak.

Expected impact:

Medium. It does not directly improve accuracy but prevents unreviewable model
claims.

Improved implementation:

- `SoftmaxPolicy` and `SklearnPolicy` now save `metadata`.
- `train_policy.py` stores dataset path, split info, class weighting, and
  train/validation metrics inside the model artifact.

Risk/limitation:

This is not full MLflow/DVC lineage. For production, dataset version hashes and
source-file manifests should be added.

## 9. Production Approval Criteria

This project should not be approved until all are true:

- Validation split is source/session-level or at least hand-group holdout.
- Accuracy beats majority baseline on the same split.
- Macro F1 improves materially, not just weighted F1.
- Rare action recall is acceptable by product requirement.
- Observed-card and missing-card slices are reported separately.
- Calibration is measured and acceptable.
- Dataset audit has no blocker findings.
- API model artifact includes metrics and training metadata.

## 10. Highest-Impact Next Work

1. Fix OCR/card extraction coverage.
   Expected impact: very high.

2. Replace reconstructed events with canonical hand-history parsing if possible.
   Expected impact: very high.

3. Add source/session holdout and slice metrics.
   Expected impact: high.

4. Run LightGBM/XGBoost/CatBoost after audit gates pass.
   Expected impact: medium to high.

5. Train a context-only missing-card model and route by card availability.
   Expected impact: medium.

6. Move from imitation metrics to EV/backtesting metrics.
   Expected impact: required for any poker-strategy claim.
