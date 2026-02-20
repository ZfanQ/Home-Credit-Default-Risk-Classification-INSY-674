# Hypothesis

## Predictable Outcome
Applicant-level features from `application_train` and aggregated historical credit behavior
(`bureau`, `previous_application`, `POS_CASH`, `installments`, `credit_card`) contain enough
signal to rank future default risk meaningfully above random ranking.

## Statistical Hypotheses
- Null hypothesis (`H0`): model ranking is not better than chance on unseen data
  (`ROC-AUC <= 0.5`).
- Alternative hypothesis (`H1`): model ranking is better than chance
  (`ROC-AUC > 0.5`).

## Type I and Type II Errors in Credit Context
- Type I (false positive): classify a reliable borrower as high risk.
  Impact: lost revenue, reduced customer acquisition, possible fairness concerns.
- Type II (false negative): classify a high-risk borrower as low risk.
  Impact: elevated defaults and write-offs.

For lending economics, Type II error is usually costlier in direct loss terms, but Type I
error can degrade growth and customer experience; threshold policy balances both.

## Statistical Reasoning
- ROC-AUC is threshold-independent and robust to class imbalance when measuring rank quality.
- The dataset has strong imbalance (~8.07% positive class), making accuracy misleading.
- Stratified split in training preserves class ratio for stable validation estimates.

## Threshold Decision Logic
- Default threshold in service: `0.50` (`HC_PREDICTION_THRESHOLD`, configurable).
- Decision rule:
  - `p(default) >= threshold` -> `default_risk`
  - `p(default) < threshold` -> `repay_normal`
- Threshold should be set using business loss ratio:
  - If false negatives are much more expensive, use a lower threshold.
  - If false positives are too costly for growth, use a higher threshold.
- Recommended governance: evaluate thresholds on expected value curves, not accuracy alone.
