# Explainability

## Global Explainability
- Source: LightGBM gain/split feature importance generated during training.
- Exposure: `GET /feature-importance?limit=N`.
- Use case: identify dominant risk drivers at portfolio level and track changes across model
  versions.

## Local Explainability
- Source: LightGBM contribution values (`pred_contrib=True`) at inference time.
- Exposure: `POST /predict` and `POST /predict/batch` return `top_contributors`.
- Output fields:
  - `feature`
  - `contribution`
  - `raw_value`
  - `base_value` (model bias term)

## Governance Guidance
- Compare top global features across retrains for stability.
- Review contributor patterns for false positives/false negatives from adjudicated samples.
- Pair model explanations with policy rules to ensure actionability and fairness review.
