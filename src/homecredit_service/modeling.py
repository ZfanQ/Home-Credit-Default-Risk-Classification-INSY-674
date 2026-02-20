from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from homecredit_service.features import ID_COLUMN, TARGET_COLUMN

MISSING_TOKEN = "__MISSING__"


def split_feature_types(feature_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_columns = (
        feature_df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    )
    numeric_columns = [column for column in feature_df.columns if column not in categorical_columns]
    return categorical_columns, numeric_columns


def fit_encoder(train_df: pd.DataFrame, categorical_columns: list[str]) -> OrdinalEncoder | None:
    if not categorical_columns:
        return None

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
        dtype=np.float32,
    )

    categorical_train = train_df[categorical_columns].astype("string").fillna(MISSING_TOKEN)
    encoder.fit(categorical_train)
    return encoder


def transform_features(
    feature_df: pd.DataFrame,
    categorical_columns: list[str],
    numeric_columns: list[str],
    encoder: OrdinalEncoder | None,
) -> pd.DataFrame:
    numeric_frame = feature_df.reindex(columns=numeric_columns).apply(
        pd.to_numeric, errors="coerce"
    )

    if not categorical_columns:
        return numeric_frame.astype("float32")

    if encoder is None:
        msg = "Encoder is required when categorical columns are present."
        raise ValueError(msg)

    categorical_frame = (
        feature_df.reindex(columns=categorical_columns).astype("string").fillna(MISSING_TOKEN)
    )
    encoded = encoder.transform(categorical_frame)
    encoded_frame = pd.DataFrame(encoded, columns=categorical_columns, index=feature_df.index)

    combined = pd.concat([numeric_frame, encoded_frame], axis=1)
    return combined.astype("float32")


def build_feature_importance(
    model: LGBMClassifier, feature_names: list[str]
) -> list[dict[str, float | str]]:
    booster = model.booster_
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_gain": gain,
            "importance_split": split,
        }
    ).sort_values("importance_gain", ascending=False)

    records = importance_df.to_dict(orient="records")
    normalized: list[dict[str, float | str]] = []
    for row in records:
        normalized_row: dict[str, float | str] = {}
        for key, value in row.items():
            if isinstance(value, (np.floating, np.integer)):
                normalized_row[key] = float(value)
            else:
                normalized_row[key] = value
        normalized.append(normalized_row)
    return normalized


def train_lightgbm_model(
    train_df: pd.DataFrame,
    random_state: int = 42,
    valid_size: float = 0.2,
) -> dict[str, Any]:
    y = train_df[TARGET_COLUMN].astype("int32")
    X = train_df.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore")

    categorical_columns, numeric_columns = split_feature_types(X)
    X_train_raw, X_valid_raw, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=valid_size,
        random_state=random_state,
        stratify=y,
    )

    encoder = fit_encoder(X_train_raw, categorical_columns)
    X_train = transform_features(X_train_raw, categorical_columns, numeric_columns, encoder)
    X_valid = transform_features(X_valid_raw, categorical_columns, numeric_columns, encoder)

    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    scale_pos_weight = negatives / positives if positives > 0 else 1.0

    model = LGBMClassifier(
        objective="binary",
        n_estimators=2500,
        learning_rate=0.03,
        num_leaves=64,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        metric="auc",
        verbosity=-1,
        scale_pos_weight=scale_pos_weight,
        importance_type="gain",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[early_stopping(120, verbose=False), log_evaluation(0)],
    )

    valid_proba = model.predict_proba(X_valid)[:, 1]
    valid_auc = float(roc_auc_score(y_valid, valid_proba))
    feature_names = X_train.columns.tolist()
    feature_importance = build_feature_importance(model, feature_names)

    return {
        "model": model,
        "encoder": encoder,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "feature_columns": feature_names,
        "metrics": {"validation_auc": valid_auc},
        "scale_pos_weight": scale_pos_weight,
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "train_rows": len(X_train),
        "valid_rows": len(X_valid),
        "feature_importance": feature_importance,
    }


def save_bundle(bundle: dict[str, Any], artifact_path: Path) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path)


def load_bundle(artifact_path: Path) -> dict[str, Any]:
    return joblib.load(artifact_path)


def prepare_inference_frame(
    records: list[dict[str, Any]], bundle: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_frame = pd.DataFrame(records)
    base_columns = bundle["numeric_columns"] + bundle["categorical_columns"]
    ordered = raw_frame.reindex(columns=base_columns)
    transformed = transform_features(
        ordered,
        categorical_columns=bundle["categorical_columns"],
        numeric_columns=bundle["numeric_columns"],
        encoder=bundle["encoder"],
    )
    return ordered, transformed
