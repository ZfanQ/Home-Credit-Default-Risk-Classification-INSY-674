from __future__ import annotations

import numpy as np
import pandas as pd

from homecredit_service.modeling import prepare_inference_frame, train_lightgbm_model


def test_training_bundle_and_inference_transform() -> None:
    rng = np.random.default_rng(42)
    row_count = 300

    train_df = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(row_count),
            "TARGET": rng.binomial(1, 0.08, size=row_count),
            "AMT_INCOME_TOTAL": rng.normal(180000, 20000, size=row_count),
            "AMT_CREDIT": rng.normal(600000, 50000, size=row_count),
            "AMT_ANNUITY": rng.normal(25000, 3000, size=row_count),
            "DAYS_BIRTH": -rng.integers(9000, 25000, size=row_count),
            "CODE_GENDER": rng.choice(["M", "F"], size=row_count),
        }
    )

    bundle = train_lightgbm_model(train_df, random_state=42, valid_size=0.2)

    assert 0.0 <= bundle["metrics"]["validation_auc"] <= 1.0
    assert bundle["scale_pos_weight"] > 1.0
    assert len(bundle["feature_columns"]) > 0

    records = [{"AMT_INCOME_TOTAL": 200000, "AMT_CREDIT": 650000, "CODE_GENDER": "F"}]
    _, transformed = prepare_inference_frame(records, bundle)

    assert transformed.shape[0] == 1
    assert transformed.shape[1] == len(bundle["feature_columns"])
