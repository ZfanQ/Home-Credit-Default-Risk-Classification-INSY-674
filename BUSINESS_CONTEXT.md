# Business Context

## Problem Framing
Home Credit extends credit to applicants with limited or non-traditional credit history.
The core business problem is to estimate probability of payment difficulty (`TARGET=1`)
for each applicant so underwriting and pricing decisions can be risk-adjusted rather than
rule-based.

## Quantified Objective
- Primary ML objective: maximize ROC-AUC on held-out validation data.
- Operational objective: improve ranking quality of risky applicants while keeping approval
  throughput stable.
- Current reference validation ROC-AUC in this repository: `0.7881`
  (`artifacts/training_report.json`).

## Business Value Explanation
- Better ranking reduces expected credit losses by improving who is approved, declined, or
  escalated for manual review.
- A probability score can also drive risk-based pricing and credit line assignment.
- Explainable outputs (`/feature-importance`, per-prediction contributors) support model
  governance and analyst review workflows.

## Financial Impact Estimation (Assumption-Based)
Assumptions for a planning scenario:
- 100,000 applications/year.
- 60% approval rate.
- Average funded amount: $8,000.
- Baseline default rate in approved population: 6%.
- Loss given default (LGD): 45%.

Estimated baseline annual loss:
- Approved loans: `100,000 * 60% = 60,000`
- Defaulted loans: `60,000 * 6% = 3,600`
- Expected loss: `3,600 * 8,000 * 45% = $12.96M`

If ROC-AUC improves by +0.02 to +0.04 and policy uses top-risk deciles more effectively,
an assumed 4% to 9% reduction in defaulted approved loans yields:
- Conservative savings: `$12.96M * 4% = $0.52M/year`
- Moderate savings: `$12.96M * 6.5% = $0.84M/year`
- Upside savings: `$12.96M * 9% = $1.17M/year`

These estimates are directional and should be validated with champion/challenger policy
simulation on historical decision logs.
