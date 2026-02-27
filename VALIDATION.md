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

## Cross-Validation Discussion
Cross-validation is not yet implemented in code; current evaluation uses one stratified holdout.
For enterprise rigor, consider:
- 5-fold stratified CV with mean/std ROC-AUC.
- Time-aware or application-window splits if data chronology affects leakage risk.
- Segment-wise validation (channel, geography, product) for robustness.

## Explainability Validation Coverage
- Global explainability: `/feature-importance` exposes gain/split importances.
- Local explainability: `/predict` and `/predict/batch` return per-record top contributors.
- Governance recommendation: monitor stability of top features across retrains to detect
  drift or shortcut learning.
