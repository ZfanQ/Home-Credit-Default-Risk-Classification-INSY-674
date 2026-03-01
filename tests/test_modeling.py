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

    bundle = train_lightgbm_model(train_df, random_state=42, valid_size=0.2, evaluate_test=True)

    assert 0.0 <= bundle["metrics"]["validation_auc"] <= 1.0
    assert 0.0 <= bundle["metrics"]["test_auc"] <= 1.0
    assert 0.0 <= bundle["metrics"]["train_pr_auc"] <= 1.0
    assert 0.0 <= bundle["metrics"]["validation_pr_auc"] <= 1.0
    assert 0.0 <= bundle["metrics"]["test_pr_auc"] <= 1.0
    assert bundle["scale_pos_weight"] > 1.0
    assert len(bundle["feature_columns"]) > 0
    assert bundle["best_iteration"] >= 1
    assert len(bundle["learning_curves"]["iterations"]) > 0
    assert len(bundle["learning_curves"]["train_auc"]) == len(
        bundle["learning_curves"]["iterations"]
    )
    assert len(bundle["learning_curves"]["validation_auc"]) == len(
        bundle["learning_curves"]["iterations"]
    )
    assert "validation" in bundle["confusion_matrices"]
    assert "test" in bundle["confusion_matrices"]
    assert bundle["confusion_matrices"]["validation"]["matrix"][0][0] >= 0
    assert 0.0 <= bundle["threshold_used"] <= 1.0
    assert "threshold_optimization" in bundle
    assert "policy_simulation" in bundle
    assert "validation" in bundle["policy_simulation"]["splits"]
    assert "calibration" in bundle
    assert "validation" in bundle["calibration"]
    assert "brier_score" in bundle["calibration"]["validation"]
    assert "cost_sensitivity" in bundle
    assert len(bundle["cost_sensitivity"]["rows"]) >= 3
    assert "temporal_validation" in bundle
    assert "enabled" in bundle["temporal_validation"]
    assert "subgroup_performance" in bundle
    assert "validation" in bundle["subgroup_performance"]
    assert "drift_summary" in bundle
    assert "score_distribution" in bundle["drift_summary"]
    assert "cross_validation" in bundle
    assert "enabled" in bundle["cross_validation"]

    records = [{"AMT_INCOME_TOTAL": 200000, "AMT_CREDIT": 650000, "CODE_GENDER": "F"}]
    _, transformed = prepare_inference_frame(records, bundle)

    assert transformed.shape[0] == 1
    assert transformed.shape[1] == len(bundle["feature_columns"])


def test_training_without_final_eval_does_not_compute_test_auc() -> None:
    rng = np.random.default_rng(123)
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(220),
            "TARGET": rng.binomial(1, 0.08, size=220),
            "AMT_INCOME_TOTAL": rng.normal(180000, 20000, size=220),
            "AMT_CREDIT": rng.normal(600000, 50000, size=220),
            "AMT_ANNUITY": rng.normal(25000, 3000, size=220),
            "DAYS_BIRTH": -rng.integers(9000, 25000, size=220),
            "CODE_GENDER": rng.choice(["M", "F"], size=220),
        }
    )

    bundle = train_lightgbm_model(frame, random_state=123, valid_size=0.2, evaluate_test=False)

    assert 0.0 <= bundle["metrics"]["validation_auc"] <= 1.0
    assert "test_auc" not in bundle["metrics"]
    assert 0.0 <= bundle["metrics"]["train_pr_auc"] <= 1.0
    assert 0.0 <= bundle["metrics"]["validation_pr_auc"] <= 1.0
    assert "test_pr_auc" not in bundle["metrics"]
    assert bundle["best_iteration"] >= 1
    assert "validation" in bundle["confusion_matrices"]
    assert "test" not in bundle["confusion_matrices"]
    assert "threshold_optimization" in bundle
    assert bundle["threshold_optimization"]["enabled"] is True
    assert "temporal_validation" in bundle
    assert "subgroup_performance" in bundle
    assert "drift_summary" in bundle
    assert "cross_validation" in bundle
    assert bundle["test_evaluated"] is False


def test_training_with_threshold_optimization_disabled_uses_default_threshold() -> None:
    rng = np.random.default_rng(321)
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(260),
            "TARGET": rng.binomial(1, 0.08, size=260),
            "AMT_INCOME_TOTAL": rng.normal(180000, 20000, size=260),
            "AMT_CREDIT": rng.normal(600000, 50000, size=260),
            "AMT_ANNUITY": rng.normal(25000, 3000, size=260),
            "DAYS_BIRTH": -rng.integers(9000, 25000, size=260),
            "CODE_GENDER": rng.choice(["M", "F"], size=260),
        }
    )

    bundle = train_lightgbm_model(
        frame,
        random_state=321,
        valid_size=0.2,
        evaluate_test=False,
        prediction_threshold=0.42,
        optimize_threshold=False,
    )

    assert bundle["threshold_default"] == 0.42
    assert bundle["threshold_used"] == 0.42
    assert bundle["threshold_optimization"]["enabled"] is False


def test_temporal_validation_enables_with_temporal_feature() -> None:
    rng = np.random.default_rng(777)
    row_count = 320
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(row_count),
            "TARGET": rng.binomial(1, 0.1, size=row_count),
            "AMT_INCOME_TOTAL": rng.normal(180000, 20000, size=row_count),
            "AMT_CREDIT": rng.normal(600000, 50000, size=row_count),
            "AMT_ANNUITY": rng.normal(25000, 3000, size=row_count),
            "DAYS_BIRTH": -rng.integers(9000, 25000, size=row_count),
            "CODE_GENDER": rng.choice(["M", "F"], size=row_count),
            "PREV_DAYS_DECISION_max": -rng.integers(1, 3650, size=row_count),
        }
    )
    frame["TARGET"] = (
        (frame["PREV_DAYS_DECISION_max"] > -700).astype(int)
        | rng.binomial(1, 0.05, size=row_count).astype(int)
    )

    bundle = train_lightgbm_model(
        frame,
        random_state=777,
        valid_size=0.2,
        evaluate_test=False,
        cv_folds=0,
        temporal_holdout_fraction=0.2,
        temporal_max_estimators=120,
        subgroup_min_size=20,
    )

    assert bundle["temporal_validation"]["enabled"] is True
    assert bundle["temporal_validation"]["holdout_rows"] > 0
    assert "holdout_auc" in bundle["temporal_validation"]
