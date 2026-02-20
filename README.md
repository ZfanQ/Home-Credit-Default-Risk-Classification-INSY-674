# Home Credit Default Risk Modeling Service

This project provides a production-ready FastAPI service for **Home Credit Default Risk** binary classification.

## What This Project Does
- Predicts the probability of loan payment difficulty for each applicant.
- Uses target definition:
  - `TARGET = 1`: applicant has payment difficulties (default risk).
  - `TARGET = 0`: applicant repays normally.
- Optimizes **ROC-AUC** (not accuracy).
- Handles class imbalance with LightGBM `scale_pos_weight`.
- Returns interpretable outputs:
  - Global feature importance.
  - Per-prediction top feature contributors.

## Project Structure
- `src/homecredit_service/features.py`: raw data loading and feature aggregation to `SK_ID_CURR`.
- `src/homecredit_service/modeling.py`: preprocessing, training, metrics, serialization, inference transforms.
- `src/homecredit_service/service.py`: prediction logic + contributor extraction.
- `src/homecredit_service/main.py`: FastAPI app and endpoints.
- `src/homecredit_service/train.py`: training CLI entrypoint.
- `tests/`: API/modeling tests.
- `homecreditdefaultriskdata/`: required raw CSV inputs.
- `artifacts/`: trained model outputs.

## Presentation Framework Docs
- `BUSINESS_CONTEXT.md`: business framing, quantified objective, and financial impact estimate.
- `HYPOTHESIS.md`: statistical hypotheses, error types, and threshold logic.
- `DATA_REPORT.md`: data sources, aggregation, missingness, and class imbalance profile.
- `RESULTS.md`: model performance summary and decision-policy interpretation.
- `VALIDATION.md`: metric justification, baseline comparison, overfitting checks, CV discussion.
- `EXPLAINABILITY.md`: global and local model explainability coverage.
- `LIMITATIONS.md`: assumptions, threats to validity, and reproducibility limits.

## Prerequisites
- Python `3.11` (project pins `>=3.11,<3.13`).
- `uv` installed.
- macOS only: `brew install libomp` (required by LightGBM).

## Data Requirement
The following files must exist under `homecreditdefaultriskdata/`:
- `application_train.csv`
- `application_test.csv`
- `bureau.csv`
- `bureau_balance.csv`
- `previous_application.csv`
- `POS_CASH_balance.csv`
- `installments_payments.csv`
- `credit_card_balance.csv`

## Setup Commands
### Option A (recommended): `uv` + Make
1. Install all dependencies (runtime + dev):
```bash
make install
```
What it does: creates/updates `.venv` using `uv sync --all-groups`.

2. Rebuild lockfile:
```bash
make lock
```
What it does: updates `uv.lock`.

### Option B: pip requirements
1. Create/activate a virtual environment.
2. Install requirements:
```bash
pip install -r requirements.txt
```
What it does: installs runtime + developer tooling listed in `requirements.txt`.

## Quality Commands
1. Lint code:
```bash
make lint
```
What it does: runs `ruff check .`.

2. Auto-format code:
```bash
make format
```
What it does: runs `ruff format .`.

3. Type-check:
```bash
make typecheck
```
What it does: runs `ty check src tests`.

4. Run tests:
```bash
make test
```
What it does: runs `pytest`.

## Training Commands
1. Full training on all rows:
```bash
make train
```
What it does:
- Loads and aggregates all Home Credit raw tables.
- Trains LightGBM with AUC metric and class weighting.
- Writes:
  - `artifacts/model_bundle.joblib`
  - `artifacts/training_report.json`

2. Quick smoke training with sampled rows:
```bash
uv run train-homecredit --data-dir homecreditdefaultriskdata --artifact-dir artifacts --sample-size 5000
```
What it does: same training flow on a subset for faster iteration.

Optional metadata argument:
```bash
uv run train-homecredit --threshold 0.5
```
What it does: stores decision-threshold metadata in `training_report.json` (does not change
training objective).

## Run API
1. Start service:
```bash
make run
```
What it does: launches Uvicorn on `http://0.0.0.0:8000` using `artifacts/model_bundle.joblib`.

2. Start service with explicit artifact path:
```bash
HC_ARTIFACT_PATH=artifacts/model_bundle.joblib uv run uvicorn homecredit_service.main:app --host 0.0.0.0 --port 8000
```
What it does: overrides model artifact path from environment.

## API Commands (Examples)
1. Health check:
```bash
curl -s http://127.0.0.1:8000/health
```

2. Model metadata:
```bash
curl -s http://127.0.0.1:8000/metadata
```

3. Top feature importance:
```bash
curl -s "http://127.0.0.1:8000/feature-importance?limit=10"
```

4. Single prediction:
```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "record": {
      "AMT_INCOME_TOTAL": 180000,
      "AMT_CREDIT": 600000,
      "AMT_ANNUITY": 28000,
      "DAYS_BIRTH": -13000,
      "CODE_GENDER": "F"
    },
    "top_n": 5
  }'
```

5. Batch prediction:
```bash
curl -s -X POST http://127.0.0.1:8000/predict/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "records": [
      {"AMT_INCOME_TOTAL": 180000, "AMT_CREDIT": 600000, "CODE_GENDER": "F"},
      {"AMT_INCOME_TOTAL": 120000, "AMT_CREDIT": 350000, "CODE_GENDER": "M"}
    ],
    "top_n": 3
  }'
```

## Docker Commands
1. Build image:
```bash
docker build -t homecredit-risk-service .
```

2. Run container:
```bash
docker run --rm -p 8000:8000 \
  -e HC_ARTIFACT_PATH=/app/artifacts/model_bundle.joblib \
  -v $(pwd)/artifacts:/app/artifacts \
  homecredit-risk-service
```

## Expected Outputs
- `artifacts/model_bundle.joblib`: trained model + preprocessing bundle.
- `artifacts/training_report.json`: metrics and metadata including class distribution,
  split counts, hyperparameters, random seed, threshold, and dependency snapshot.

## Notes
- Full training is compute and memory intensive.
- Use sample training command during development.
- Primary model selection metric is ROC-AUC.
