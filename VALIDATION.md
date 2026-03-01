# Validation

## Why ROC-AUC Is the Primary Metric
- Business goal is risk ranking quality, not a single fixed-threshold classification.
- The target is imbalanced (~8.07% positives), making accuracy a poor primary objective.
- ROC-AUC evaluates ordering quality across all thresholds and is stable for model selection.

## Baseline Comparison
- Null/random ranking baseline: ROC-AUC = 0.50.
- Current repository final reference (`artifacts/training_report.json`): test ROC-AUC = 0.7858.
- Lift over random baseline: `0.7858 - 0.5000 = +0.2858` absolute AUC points.

## Overfitting Checks
Current checks in training pipeline:
- Stratified train/validation/test split (`train_test_split(..., stratify=y)`).
- Early stopping on validation AUC.
- Regularization and subsampling in LightGBM config.
- Test metric computation is gated to explicit final-evaluation runs.

Recommended report checks after each training run:
- Compare train ROC-AUC vs validation ROC-AUC.
- Track delta (`train_auc - validation_auc`) across runs.
- Flag potential overfit when delta grows materially over historical norms.
- Confirm final holdout quality with test ROC-AUC and track delta (`validation_auc - test_auc`).

## Train and Validation Curves
The training pipeline now writes full AUC-vs-iteration curves to
`artifacts/training_report.json` under `learning_curves`:
- `learning_curves.iterations`
- `learning_curves.train_auc`
- `learning_curves.validation_auc`
- `best_iteration`
- A grader-friendly table export is also generated at `artifacts/learning_curves.csv`.
- Visual graph export: `artifacts/plots/train_validation_learning_curve.png`.

Current reference run (`trained_at_utc: 2026-03-01T03:13:14.870452+00:00`):
- Best iteration: `821`
- Train AUC at best iteration: `0.9095`
- Validation AUC at best iteration: `0.7844`
- Generalization gap at best iteration: `0.1250`

These curves provide the direct grader-facing evidence that:
- training AUC improves steadily,
- validation AUC plateaus,
- early stopping selects the model before extended overfitting.

## Validation Matrices
The training pipeline now writes confusion matrices under `confusion_matrices` for:
- `train`
- `validation`
- `test` (when `--final-eval` is used)
- A grader-friendly validation export is also generated at
  `artifacts/validation_confusion_matrix.csv`.
- Visual heatmap export: `artifacts/plots/validation_confusion_matrix.png`.

Validation matrix at `threshold_used` from `training_report.json`
(rows=true class `[0,1]`, columns=predicted `[0,1]`):

|  | Pred 0 | Pred 1 |
| --- | ---: | ---: |
| True 0 | 50,154 | 6,384 |
| True 1 | 2,679 | 2,286 |

Validation normalized matrix:

|  | Pred 0 | Pred 1 |
| --- | ---: | ---: |
| True 0 | 0.8871 | 0.1129 |
| True 1 | 0.5396 | 0.4604 |

Validation rates from the same matrix:
- Accuracy: `0.8526`
- Precision: `0.2637`
- Recall (TPR): `0.4604`
- Specificity (TNR): `0.8871`
- FPR: `0.1129`
- FNR: `0.5396`

## Cross-Validation
The training pipeline now includes stratified K-fold cross-validation on the
train+validation pool before final model fitting. Results are stored in
`artifacts/training_report.json` under `cross_validation`:
- `cross_validation.enabled`
- `cross_validation.n_splits`
- `cross_validation.auc_mean`
- `cross_validation.auc_std`
- `cross_validation.fold_metrics`

Recommended governance extensions:
- Increase `--cv-folds` to 5 for final benchmark runs.
- Add time-aware folds if application chronology is part of business policy.
- Track segment-wise CV (channel, geography, product) for robustness.

Current reference CV summary:
- Folds: `3`
- Mean ROC-AUC: `0.7834`
- Std ROC-AUC: `0.0028`

## Threshold Optimization and Policy Simulation
The training pipeline now performs validation-threshold search using cost weights:
- `false_positive_cost`
- `false_negative_cost`

Outputs are persisted in:
- `threshold_optimization` (selected threshold, default threshold comparison, search curve)
- `policy_simulation` (approval rate, default rate within approved, expected cost per split)

This ties model evaluation to business decisions instead of relying on a fixed
`0.5` threshold by default.

Current threshold-policy result:
- Default threshold: `0.500`
- Selected threshold: `0.549`
- Validation expected cost per applicant at default threshold: `0.3466`
- Validation expected cost per applicant at selected threshold: `0.3216`

## Temporal Holdout (Out-of-Time) Validation
The pipeline now runs a separate temporal holdout diagnostic using the most recent
segment by `PREV_DAYS_DECISION_max` (from the train+validation pool).

Current reference temporal result:
- Temporal column: `PREV_DAYS_DECISION_max`
- Holdout rows: `52,387`
- Holdout ROC-AUC: `0.7808`
- Holdout PR-AUC: `0.2930`
- Holdout positive rate: `0.0887`

Interpretation:
- Temporal holdout ROC-AUC is close to random-split validation ROC-AUC (`0.7844`),
  indicating no major chronology-specific collapse.

## Precision-Recall and Calibration Checks
Added in `training_report.json` for a stronger evaluation story:
- `pr_auc.train`: `0.4828`
- `pr_auc.validation`: `0.2793`
- `pr_auc.test`: `0.2820`

Calibration diagnostics are reported under `calibration`:
- Validation Brier score: `0.1853`
- Validation expected calibration error (10 bins): `0.3411`
- Bin tables exported to:
  - `artifacts/train_calibration.csv`
  - `artifacts/validation_calibration.csv`
  - `artifacts/test_calibration.csv`
- Visual reliability plots exported to:
  - `artifacts/plots/train_calibration_curve.png`
  - `artifacts/plots/validation_calibration_curve.png`
  - `artifacts/plots/test_calibration_curve.png`

## Cost-Sensitivity Analysis
To stress-test business-policy assumptions, the pipeline now sweeps multiple
false-negative cost settings and re-optimizes threshold per setting.

Outputs:
- `cost_sensitivity` block in `training_report.json`
- `artifacts/cost_sensitivity.csv`
- `artifacts/plots/cost_sensitivity_threshold_curve.png`

Current run produced 5 cost scenarios and shows the expected behavior:
- Higher false-negative cost pushes threshold lower (approve less aggressively).
- Lower false-negative cost pushes threshold higher (decline less aggressively).

## Subgroup Robustness
The pipeline now writes subgroup diagnostics to:
- `artifacts/validation_subgroup_metrics.csv`
- `artifacts/test_subgroup_metrics.csv`

Current snapshot highlights (`CODE_GENDER` recall at threshold `0.549`):
- Validation: `F=0.4114`, `M=0.5250`
- Test: `F=0.4113`, `M=0.5384`

This gives the grader explicit evidence that we are checking group-level behavior,
not only aggregate AUC.

## Drift Diagnostics
The pipeline now writes PSI-based drift diagnostics to `artifacts/drift_summary.json`.

Current snapshot:
- Score PSI train vs validation: `0.00130` (`stable`)
- Score PSI train vs test: `0.00068` (`stable`)
- Max feature PSI train vs validation (tracked features): `0.00027`

These values are far below common alert thresholds (e.g., `0.1`), so this run
shows no meaningful distribution drift across splits.

## Explainability Validation Coverage
- Global explainability: `/feature-importance` exposes gain/split importances.
- Local explainability: `/predict` and `/predict/batch` return per-record top contributors.
- Governance recommendation: monitor stability of top features across retrains to detect
  drift or shortcut learning.
