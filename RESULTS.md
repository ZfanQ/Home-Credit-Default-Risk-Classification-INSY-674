# Results

## Current Performance Snapshot
- Reference validation ROC-AUC: `0.7881` (`artifacts/training_report.json`).
- Baseline random ROC-AUC: `0.5000`.
- Absolute lift over random: `+0.2881`.

## Operational Interpretation
- The model provides meaningful risk ranking separation between likely defaulters and likely
  non-defaulters.
- Ranking quality supports downstream policy choices:
  - Auto-approve low-risk band.
  - Manual review medium-risk band.
  - Decline or stricter terms for high-risk band.

## Decision-Threshold Implications
- Threshold is a policy lever, not the training objective.
- Lower threshold generally increases risk capture (fewer false negatives) at the cost of
  more false positives.
- Higher threshold generally preserves approvals but can increase default losses.

## Financial Signal (From Business Context Assumptions)
Using the assumptions documented in `BUSINESS_CONTEXT.md`, a moderate reduction in approved
defaults driven by improved ROC-AUC corresponds to an estimated savings range of
`$0.52M` to `$1.17M` per year, pending policy simulation and A/B validation.
