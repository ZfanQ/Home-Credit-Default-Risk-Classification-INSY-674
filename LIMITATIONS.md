# Limitations

## Assumptions
- Training data is representative of future applicants and economic conditions.
- Applicant identifier joins across source tables are complete and correct.
- Missingness patterns in production remain similar to historical training data.

## Data Leakage Risks
- Aggregated historical tables may contain fields that are only available after application
  decision time unless strictly time-aligned.
- Without explicit event-time cutoffs in feature engineering, leakage can inflate validation.

## Data Drift Risks
- Macroeconomic changes and policy shifts can alter applicant risk profile.
- Product mix and channel changes can shift feature distributions.
- Categorical value drift may increase unknown-category rate at inference.

## External Validity
- Model performance measured on Home Credit historical data may not transfer directly to
  different geographies, lenders, or regulatory regimes.
- Business outcomes depend on downstream policy decisions, not score quality alone.

## Aggregation Bias
- Applicant-level aggregates can obscure recency and trajectory effects.
- Summary statistics (mean/sum/std) may overweight customers with long history depth.
- One-to-many table compression can lose sequence context that may matter for risk.

## Reproducibility Limitations
- Current evaluation is single holdout split (no full CV uncertainty interval).
- Reproducibility depends on external package and platform behavior despite fixed seeds.
- Artifact bundles store the trained estimator but not full raw data snapshot hashes by
  default.
