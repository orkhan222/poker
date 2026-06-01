# Research-Level Poker Model Redesign

Bu sened musterinin ML research review-una cavab olaraq hazirlanib. Meqsed
layiheni beginner tutorial kimi yox, publication-quality supervised poker action
prediction pipeline kimi strukturlaşdırmaqdir.

## 1. Existing Feature Pipeline Analysis

Evvelki pipeline-in esas problemi modelden once data idi:

- Tiny sample fayl ile train edilirdi.
- Fold sinfi dominant idi, rare action-lar itirdi.
- Hole cards cox vaxt missing oldugu ucun card strength signal sifira dusurdu.
- `to_call`, `min_raise`, pot ve stack context duzgun propagate olunmurdu.
- Raw seat label-lar feature space-i sisirdirdi.
- Random split eyni hand-in action-larini train ve validation-a sala bilerdi.

Cari kodda bu hisseler duzeldilib:

- `poker_agent/features.py`: leakage-safe feature extraction.
- `poker_agent/model.py`: multiple model families.
- `poker_agent/evaluator.py`: macro/weighted F1, precision/recall, confusion matrix.
- `scripts/research_experiment.py`: hand-level holdout model comparison.

## 2. Weaknesses And Leakage Risks

Poker action predictionde esas leakage riskleri:

- Preflop/flop decision-a river board kartlarini vermek.
- Action vaxtinda bilinmeyen final pot deyerini feature kimi istifade etmek.
- Current action amount-u feature kimi vermek ve label-i dolayi yolla acmaq.
- Eyni hand daxilindeki action-lari random split ile hem train, hem validation-a salmaq.
- OCR post-processing-den sonradan yaranan target-dependent feature istifade etmek.

Kodda gorulen qarsilama:

- `visible_board_cards()` yalniz cari street-de gorunen board kartlarini qaytarir.
- Training loader final pot yerine action-time running pot istifade edir.
- Current-action stack amount feature saxlanilmir.
- `research_experiment.py` hand_id group holdout istifade edir.

## 3. Research-Grade Feature Set

Feature set dord qrupa bolunur:

1. Card texture features

   - hole high/low rank
   - suited, paired, connected
   - preflop bucket score
   - made hand score
   - flush/straight draw pressure
   - board wetness
   - top pair / overpair indicators

2. Pot and stack features

   - pot odds
   - call_to_stack
   - raise_to_stack
   - SPR
   - hero commitment ratio
   - table commitment pressure
   - call price ratio

3. Position and table context

   - normalized position group
   - player count
   - players acted ratio
   - last aggressor group
   - last aggressor is hero

4. Temporal betting history

   - street action count
   - street aggressive count
   - call/check/fold counts
   - street aggression ratio
   - facing bet or raise flag
   - optional request-level `betting_history`

## 4. Missing Hole Cards

Uc strategiya desteklenir:

- `drop`: OCR two-card signal yoxdursa row trainingden cixarilir. Bu cari default-dur.
- `flag`: row saxlanilir, amma `hole_cards_missing` ve observed-ratio feature-lari elave olunur.
- `keep`: raw missing state saxlanilir, ablation ucun.

Research ucun en saglam yol:

1. Main model: `drop`.
2. Robustness test: `flag`.
3. Production OCR improvement: card detector coverage-ni artirmaq.
4. Opponent unknown cards ucun hand-range distribution feature-lari elave etmek.

## 5. Class Imbalance

Cari kod `none`, `sqrt_balanced`, `balanced` sample weighting destekleyir.

Recommended experiments:

- Weighted cross entropy / sample_weight for tree models.
- Focal loss for neural models (`poker_agent/sequence_models.py`).
- Rare-class oversampling only inside train split.
- Macro F1 model selection.
- Per-class recall guardrails, especially call/bet/check.

Accuracy tek basina yeterli deyil, cunki fold-heavy datasetde majority baseline
yuksek gorune biler.

## 6. Model Comparison

| Model | Strength | Weakness | Recommendation |
|---|---|---|---|
| XGBoost | Strong tabular nonlinear interactions | Extra dependency, tuning needed | High-priority experiment |
| LightGBM | Fast on large tabular data | Can overfit noisy categories | High-priority experiment |
| CatBoost | Robust categorical handling | Slower, extra dependency | Good if raw categorical features grow |
| Random Forest | Stable baseline | Often weaker calibration | Secondary baseline |
| ExtraTrees | Strong variance reduction | Less calibrated | Secondary baseline |
| MLP | Handles dense interactions | Needs scaling, more data | Useful ablation |
| Transformer | Best for ordered betting histories | Requires full sequences | Research-stage next step |

Most likely near-term winner: LightGBM/XGBoost with leakage-safe tabular
features and `sqrt_balanced` weighting. They usually outperform simple SGD and
plain forests on mixed numeric sparse poker-state features because they capture
nonlinear thresholds like pot odds x stack depth x position x aggression.

## 7. Evaluation Framework

Required metrics:

- Accuracy
- Cross entropy
- Macro F1
- Weighted F1
- Per-class precision
- Per-class recall
- Per-class F1
- Confusion matrix
- Majority baseline
- Lift versus majority baseline

Research split:

- Prefer hand_id/source/session holdout.
- Do not random-split individual actions from the same hand when reporting
  final validation metrics.
- Keep a separate test set untouched until model selection is finished.

## 8. Production Code Entry Points

Train delivered model:

```powershell
python scripts\train_policy.py `
  --dataset "C:\Users\user\Desktop\AllFile\dataset" `
  --model-out ".\models\poker_policy.joblib" `
  --policy hist_gradient_boosting `
  --max-iter 90 `
  --learning-rate 0.05 `
  --max-leaf-nodes 31 `
  --l2-regularization 0.02 `
  --class-weighting sqrt_balanced `
  --max-class-weight 6 `
  --missing-hole-cards drop
```

Run research comparison:

```powershell
python scripts\research_experiment.py `
  --dataset "C:\Users\user\Desktop\AllFile\dataset" `
  --out-dir ".\research_runs\full" `
  --policies hist_gradient_boosting,extra_trees,random_forest,mlp,xgboost,lightgbm,catboost `
  --class-weighting sqrt_balanced `
  --max-class-weight 6 `
  --missing-hole-cards drop `
  --save-models
```

Install optional research libraries:

```powershell
pip install -r requirements-research.txt
```

Evaluate a saved policy:

```powershell
python scripts\evaluate_policy.py `
  --dataset "C:\Users\user\Desktop\AllFile\dataset" `
  --model ".\models\poker_policy.joblib" `
  --missing-hole-cards drop
```

## 9. Current Honest Result

```text
valid_accuracy=0.6544
valid_macro_f1=0.5138
valid_weighted_f1=0.6503
majority_baseline_accuracy=0.5948
lift_vs_majority=0.0596
```

Bu artiq 0.34 baseline-dan xeyli gucludur ve majority baseline-dan yuxaridir.
Amma bunu final poker intelligence kimi yox, duzeldilmis professional supervised
baseline kimi teqdim etmek lazimdir.

## 10. Next Research Step

En boyuk potensial:

1. OCR/card extraction coverage-ni artirmaq.
2. Hand-history source varsa, OCR-dan daha temiz label/data almaq.
3. Full betting sequence saxlanilsa Transformer experiment etmek.
4. XGBoost/LightGBM/CatBoost grid search-i hand-level holdout ile aparmaq.
5. Model selection metric kimi macro F1 ve rare-class recall istifade etmek.
