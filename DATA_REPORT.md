# Data Report

## Data Sources Used
All raw CSV files are loaded from the configured data directory
(`homecreditdefaultriskdata` by default):

| File | Rows | Columns | Role |
| --- | ---: | ---: | --- |
| `application_train.csv` | 307,511 | 122 | Main training table with `TARGET` |
| `application_test.csv` | 48,744 | 121 | Holdout scoring table (no target) |
| `bureau.csv` | 1,716,428 | 17 | Credit bureau history |
| `bureau_balance.csv` | 27,299,925 | 3 | Monthly bureau status history |
| `previous_application.csv` | 1,670,214 | 37 | Prior application records |
| `POS_CASH_balance.csv` | 10,001,358 | 8 | POS/cash loan monthly status |
| `installments_payments.csv` | 13,605,401 | 8 | Installment payment history |
| `credit_card_balance.csv` | 3,840,312 | 23 | Credit card monthly balances |

## Aggregation Logic
- Primary key: `SK_ID_CURR` (applicant).
- `application_train` is enriched with ratio features:
  - `CREDIT_INCOME_RATIO`
  - `ANNUITY_INCOME_RATIO`
  - `CREDIT_ANNUITY_RATIO`
  - `DAYS_EMPLOYED_PCT`
- Auxiliary tables are aggregated by `SK_ID_CURR` using:
  - `mean`, `max`, `min`, `sum`, `std`, and row count per source.
- Temporal leakage guard is applied before aggregation:
  - only non-future records are kept using relative-time columns such as
    `DAYS_CREDIT`, `DAYS_DECISION`, `MONTHS_BALANCE`,
    `DAYS_ENTRY_PAYMENT`, and `DAYS_INSTALMENT` (rows with values `<= 0` or missing).
- Aggregates are left-joined back to the base applicant table.

## Missing Value Strategy
- Sentinel cleaning: `DAYS_EMPLOYED=365243` is replaced with `NaN`.
- Safe ratio generation avoids division by zero (`inf` converted to `NaN`).
- Numeric casting uses coercion (`errors="coerce"`) at transform time.
- Categorical features are ordinal-encoded with:
  - `__MISSING__` token for missing values.
  - Unknown/missing inference categories mapped to `-1`.
- Data directory/file validation is explicit before training; missing files raise a clear
  `FileNotFoundError` listing required filenames.

## Class Imbalance Statistics
From `application_train.csv`:
- Positive class (`TARGET=1`): 24,825
- Negative class (`TARGET=0`): 282,686
- Positive rate: 8.07%
- Negative rate: 91.93%

The trainer computes `scale_pos_weight = negatives / positives` on the training split and
passes it to LightGBM.

## Profiling Summary
- `application_train` has 41 columns with >50% missingness and 50 columns with >20%.
- Highest-missing columns include:
  - `COMMONAREA_MEDI`, `COMMONAREA_AVG`, `COMMONAREA_MODE` (~69.87% missing)
  - `NONLIVINGAPARTMENTS_*` (~69.43% missing)
  - `LIVINGAPARTMENTS_*` (~68.35% missing)
- These patterns justify model/encoder pathways that can absorb sparse fields without
  dropping large segments of applicants.
