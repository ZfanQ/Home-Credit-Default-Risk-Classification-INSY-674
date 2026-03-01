from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder

from homecredit_service.features import ID_COLUMN, TARGET_COLUMN

MISSING_TOKEN = "__MISSING__"
FOCAL_GAMMA = 2.0
FOCAL_EPS = 1e-12
TEMPORAL_COLUMN_CANDIDATES = (
    "PREV_DAYS_DECISION_max",
    "BUREAU_DAYS_CREDIT_max",
    "INST_DAYS_ENTRY_PAYMENT_max",
    "INST_DAYS_INSTALMENT_max",
    "CC_MONTHS_BALANCE_max",
    "POS_MONTHS_BALANCE_max",
    "BUREAU_BAL_MONTHS_BALANCE_max",
)
SUBGROUP_COLUMN_CANDIDATES = (
    "CODE_GENDER",
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
)
DRIFT_FEATURE_CANDIDATES = (
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
)


@dataclass(frozen=True)
class FocalLossObjective:
    alpha: float
    gamma: float = FOCAL_GAMMA

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y_true = y_true.astype(np.float32)
        probability = 1.0 / (1.0 + np.exp(-y_pred))
        probability = np.clip(probability, FOCAL_EPS, 1.0 - FOCAL_EPS)

        positive_grad = self.alpha * (1.0 - probability) ** self.gamma * (
            self.gamma * probability * np.log(probability) + probability - 1.0
        )
        negative_grad = (1.0 - self.alpha) * probability**self.gamma * (
            probability - self.gamma * (1.0 - probability) * np.log(1.0 - probability)
        )
        grad = np.where(y_true == 1, positive_grad, negative_grad)

        positive_hess = self.alpha * probability * (1.0 - probability) * (
            (1.0 - probability) ** (self.gamma - 1.0)
            * (
                -self.gamma
                * (self.gamma * probability * np.log(probability) + probability - 1.0)
                + (1.0 - probability) * (self.gamma * np.log(probability) + self.gamma + 1.0)
            )
        )
        negative_hess = (1.0 - self.alpha) * probability * (1.0 - probability) * (
            probability ** (self.gamma - 1.0)
            * (
                self.gamma
                * (
                    probability
                    - self.gamma * (1.0 - probability) * np.log(1.0 - probability)
                )
                + probability * (1.0 + self.gamma * np.log(1.0 - probability) + self.gamma)
            )
        )
        hess = np.where(y_true == 1, positive_hess, negative_hess)
        hess = np.maximum(hess, FOCAL_EPS)
        return grad, hess


def make_focal_loss_objective(alpha: float, gamma: float = FOCAL_GAMMA):
    return FocalLossObjective(alpha=alpha, gamma=gamma)


def split_feature_types(feature_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_columns = feature_df.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
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


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def build_confusion_matrix_summary(
    y_true: pd.Series, positive_probability: np.ndarray, threshold: float
) -> dict[str, Any]:
    y_pred = (positive_probability >= threshold).astype("int32")
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix.astype("float64"),
        row_totals,
        out=np.zeros_like(matrix, dtype="float64"),
        where=row_totals != 0,
    )

    return {
        "threshold": float(threshold),
        "matrix": matrix.astype("int64").tolist(),
        "normalized_matrix": normalized.tolist(),
        "counts": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "rates": {
            "accuracy": _safe_divide(tp + tn, tp + tn + fp + fn),
            "precision": _safe_divide(tp, tp + fp),
            "recall": _safe_divide(tp, tp + fn),
            "specificity": _safe_divide(tn, tn + fp),
            "fpr": _safe_divide(fp, fp + tn),
            "fnr": _safe_divide(fn, fn + tp),
        },
    }


def build_auc_learning_curves(model: LGBMClassifier) -> dict[str, Any]:
    evals_result = getattr(model, "evals_result_", {}) or {}

    train_curve = [
        float(value) for value in evals_result.get("train", {}).get("auc", [])
    ]
    validation_curve = [
        float(value) for value in evals_result.get("validation", {}).get("auc", [])
    ]

    curve_len = min(len(train_curve), len(validation_curve))
    if curve_len == 0:
        return {
            "iterations": [],
            "train_auc": [],
            "validation_auc": [],
            "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        }

    best_iteration = int(getattr(model, "best_iteration_", 0) or curve_len)
    return {
        "iterations": list(range(1, curve_len + 1)),
        "train_auc": train_curve[:curve_len],
        "validation_auc": validation_curve[:curve_len],
        "best_iteration": best_iteration,
    }


def build_calibration_summary(
    y_true: pd.Series, positive_probability: np.ndarray, n_bins: int = 10
) -> dict[str, Any]:
    grid_bins = max(2, int(n_bins))
    probabilities = np.clip(positive_probability.astype("float64"), 0.0, 1.0)
    y_values = y_true.to_numpy(dtype="int32")
    brier = float(brier_score_loss(y_values, probabilities))

    bin_edges = np.linspace(0.0, 1.0, grid_bins + 1)
    bin_ids = np.digitize(probabilities, bin_edges[1:-1], right=False)
    total_count = len(probabilities)

    bin_rows: list[dict[str, float | int]] = []
    weighted_gap = 0.0
    for bin_idx in range(grid_bins):
        mask = bin_ids == bin_idx
        count = int(np.sum(mask))
        lower = float(bin_edges[bin_idx])
        upper = float(bin_edges[bin_idx + 1])
        if count == 0:
            row = {
                "bin_index": int(bin_idx),
                "bin_lower": lower,
                "bin_upper": upper,
                "count": 0,
                "mean_predicted_probability": 0.0,
                "observed_default_rate": 0.0,
                "absolute_gap": 0.0,
            }
            bin_rows.append(row)
            continue

        mean_pred = float(np.mean(probabilities[mask]))
        observed = float(np.mean(y_values[mask]))
        abs_gap = abs(mean_pred - observed)
        weighted_gap += abs_gap * (count / total_count)
        row = {
            "bin_index": int(bin_idx),
            "bin_lower": lower,
            "bin_upper": upper,
            "count": count,
            "mean_predicted_probability": mean_pred,
            "observed_default_rate": observed,
            "absolute_gap": abs_gap,
        }
        bin_rows.append(row)

    return {
        "brier_score": brier,
        "expected_calibration_error": float(weighted_gap),
        "n_bins": int(grid_bins),
        "bins": bin_rows,
    }


def build_policy_simulation_summary(
    counts: dict[str, int],
    false_positive_cost: float,
    false_negative_cost: float,
) -> dict[str, float | int]:
    tn = int(counts.get("tn", 0))
    fp = int(counts.get("fp", 0))
    fn = int(counts.get("fn", 0))
    tp = int(counts.get("tp", 0))
    total = tn + fp + fn + tp

    approvals = tn + fn
    declines = tp + fp
    expected_cost = float(fp * false_positive_cost + fn * false_negative_cost)

    return {
        "approvals": approvals,
        "declines": declines,
        "approval_rate": _safe_divide(approvals, total),
        "decline_rate": _safe_divide(declines, total),
        "approved_default_count": fn,
        "approved_good_count": tn,
        "default_rate_within_approved": _safe_divide(fn, approvals),
        "good_rate_within_approved": _safe_divide(tn, approvals),
        "false_decline_count": fp,
        "captured_default_count": tp,
        "expected_cost": expected_cost,
        "expected_cost_per_applicant": _safe_divide(expected_cost, total),
        "expected_cost_per_1000": _safe_divide(expected_cost * 1000.0, total),
    }


def build_threshold_policy_curve(
    y_true: pd.Series,
    positive_probability: np.ndarray,
    false_positive_cost: float,
    false_negative_cost: float,
    grid_size: int = 101,
) -> dict[str, Any]:
    grid = max(3, int(grid_size))
    thresholds = np.linspace(0.01, 0.99, grid)

    curve: list[dict[str, float]] = []
    best_idx = 0
    best_cost = float("inf")
    best_recall = -1.0
    best_distance_to_midpoint = float("inf")

    for idx, threshold in enumerate(thresholds):
        confusion = build_confusion_matrix_summary(y_true, positive_probability, float(threshold))
        policy = build_policy_simulation_summary(
            counts=confusion["counts"],
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        )
        row = {
            "threshold": float(threshold),
            "expected_cost_per_applicant": float(policy["expected_cost_per_applicant"]),
            "approval_rate": float(policy["approval_rate"]),
            "default_rate_within_approved": float(policy["default_rate_within_approved"]),
            "recall": float(confusion["rates"]["recall"]),
            "specificity": float(confusion["rates"]["specificity"]),
        }
        curve.append(row)

        current_cost = row["expected_cost_per_applicant"]
        current_recall = row["recall"]
        current_distance = abs(row["threshold"] - 0.5)
        if (
            current_cost < best_cost
            or (
                np.isclose(current_cost, best_cost)
                and (
                    current_recall > best_recall
                    or (
                        np.isclose(current_recall, best_recall)
                        and current_distance < best_distance_to_midpoint
                    )
                )
            )
        ):
            best_idx = idx
            best_cost = current_cost
            best_recall = current_recall
            best_distance_to_midpoint = current_distance

    return {
        "grid_size": grid,
        "curve": curve,
        "best_threshold": float(curve[best_idx]["threshold"]),
        "best_expected_cost_per_applicant": float(curve[best_idx]["expected_cost_per_applicant"]),
    }


def build_cost_sensitivity_summary(
    y_true: pd.Series,
    positive_probability: np.ndarray,
    false_positive_cost: float,
    false_negative_cost: float,
    grid_size: int = 101,
) -> dict[str, Any]:
    fp_cost = float(false_positive_cost)
    base_fn = float(false_negative_cost)
    fn_cost_options = sorted(
        {
            max(0.5, round(base_fn * 0.5, 4)),
            base_fn,
            round(base_fn * 1.5, 4),
            round(base_fn * 2.0, 4),
            round(base_fn * 3.0, 4),
        }
    )

    rows: list[dict[str, float]] = []
    for fn_cost in fn_cost_options:
        threshold_curve = build_threshold_policy_curve(
            y_true=y_true,
            positive_probability=positive_probability,
            false_positive_cost=fp_cost,
            false_negative_cost=fn_cost,
            grid_size=grid_size,
        )
        chosen_threshold = float(threshold_curve["best_threshold"])
        confusion = build_confusion_matrix_summary(y_true, positive_probability, chosen_threshold)
        policy = build_policy_simulation_summary(
            counts=confusion["counts"],
            false_positive_cost=fp_cost,
            false_negative_cost=fn_cost,
        )
        rows.append(
            {
                "false_positive_cost": fp_cost,
                "false_negative_cost": float(fn_cost),
                "selected_threshold": chosen_threshold,
                "expected_cost_per_applicant": float(policy["expected_cost_per_applicant"]),
                "recall": float(confusion["rates"]["recall"]),
                "specificity": float(confusion["rates"]["specificity"]),
                "approval_rate": float(policy["approval_rate"]),
                "default_rate_within_approved": float(policy["default_rate_within_approved"]),
            }
        )

    return {
        "base_cost_weights": {
            "false_positive_cost": fp_cost,
            "false_negative_cost": base_fn,
        },
        "rows": rows,
    }


def _compute_psi(reference: np.ndarray, comparison: np.ndarray, bins: int = 10) -> float | None:
    ref = reference[np.isfinite(reference)]
    cmp = comparison[np.isfinite(comparison)]
    if len(ref) < 10 or len(cmp) < 10:
        return None

    quantiles = np.linspace(0.0, 1.0, max(2, int(bins)) + 1)
    raw_edges = np.quantile(ref, quantiles)
    unique_edges = np.unique(raw_edges)
    if len(unique_edges) < 2:
        return None

    histogram_edges = np.concatenate(([-np.inf], unique_edges[1:-1], [np.inf]))
    ref_counts, _ = np.histogram(ref, bins=histogram_edges)
    cmp_counts, _ = np.histogram(cmp, bins=histogram_edges)

    ref_total = max(1, int(ref_counts.sum()))
    cmp_total = max(1, int(cmp_counts.sum()))
    ref_ratio = np.clip(ref_counts.astype("float64") / ref_total, 1e-6, 1.0)
    cmp_ratio = np.clip(cmp_counts.astype("float64") / cmp_total, 1e-6, 1.0)
    psi = float(np.sum((cmp_ratio - ref_ratio) * np.log(cmp_ratio / ref_ratio)))
    return psi


def _psi_severity(psi_value: float | None) -> str:
    if psi_value is None:
        return "not_available"
    if psi_value < 0.1:
        return "stable"
    if psi_value < 0.25:
        return "minor_shift"
    return "major_shift"


def _pick_temporal_column(feature_df: pd.DataFrame) -> str | None:
    for column in TEMPORAL_COLUMN_CANDIDATES:
        if column not in feature_df.columns:
            continue
        valid_count = int(pd.to_numeric(feature_df[column], errors="coerce").notna().sum())
        if valid_count >= 100:
            return column
    return None


def _temporal_range(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric.to_numpy(dtype="float64"))]
    if len(finite) == 0:
        return {"min": 0.0, "max": 0.0}
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def _group_gap(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(max(values) - min(values))


def build_subgroup_performance_summary(
    split_name: str,
    y_true: pd.Series,
    positive_probability: np.ndarray,
    raw_features: pd.DataFrame,
    threshold: float,
    false_positive_cost: float,
    false_negative_cost: float,
    min_group_size: int = 500,
) -> dict[str, Any]:
    y_values = y_true.to_numpy(dtype="int32")
    columns_payload: dict[str, Any] = {}
    effective_min_size = max(10, int(min_group_size))

    for column in SUBGROUP_COLUMN_CANDIDATES:
        if column not in raw_features.columns:
            continue
        groups = (
            raw_features[column]
            .astype("string")
            .fillna(MISSING_TOKEN)
            .replace("<NA>", MISSING_TOKEN)
        )
        top_groups = groups.value_counts().head(12).index.tolist()

        rows: list[dict[str, Any]] = []
        auc_values: list[float] = []
        recall_values: list[float] = []
        approval_values: list[float] = []
        positive_rate_values: list[float] = []
        for group_label in top_groups:
            mask = (groups == group_label).to_numpy(dtype=bool)
            group_size = int(mask.sum())
            if group_size < effective_min_size:
                continue
            y_group = y_values[mask]
            p_group = positive_probability[mask]
            y_group_series = pd.Series(y_group)

            class_count = int(np.unique(y_group).size)
            group_auc = (
                float(roc_auc_score(y_group, p_group))
                if class_count >= 2
                else None
            )
            group_pr_auc = (
                float(average_precision_score(y_group, p_group))
                if class_count >= 2
                else None
            )
            confusion = build_confusion_matrix_summary(y_group_series, p_group, threshold)
            policy = build_policy_simulation_summary(
                counts=confusion["counts"],
                false_positive_cost=false_positive_cost,
                false_negative_cost=false_negative_cost,
            )
            row = {
                "group": str(group_label),
                "count": group_size,
                "positive_rate": float(np.mean(y_group)),
                "auc": group_auc,
                "pr_auc": group_pr_auc,
                "recall": float(confusion["rates"]["recall"]),
                "specificity": float(confusion["rates"]["specificity"]),
                "approval_rate": float(policy["approval_rate"]),
                "default_rate_within_approved": float(policy["default_rate_within_approved"]),
            }
            rows.append(row)

            if group_auc is not None:
                auc_values.append(group_auc)
            recall_values.append(float(row["recall"]))
            approval_values.append(float(row["approval_rate"]))
            positive_rate_values.append(float(row["positive_rate"]))

        if not rows:
            continue

        columns_payload[column] = {
            "groups": rows,
            "max_auc_gap": _group_gap(auc_values),
            "max_recall_gap": _group_gap(recall_values),
            "max_approval_rate_gap": _group_gap(approval_values),
            "max_positive_rate_gap": _group_gap(positive_rate_values),
        }

    if not columns_payload:
        return {
            "enabled": False,
            "split": split_name,
            "reason": "No configured subgroup columns available with enough rows.",
            "min_group_size": int(effective_min_size),
        }
    return {
        "enabled": True,
        "split": split_name,
        "threshold": float(threshold),
        "min_group_size": int(effective_min_size),
        "columns": columns_payload,
    }


def build_drift_summary(
    X_train_raw: pd.DataFrame,
    X_valid_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame | None,
    train_probability: np.ndarray,
    validation_probability: np.ndarray,
    test_probability: np.ndarray | None,
) -> dict[str, Any]:
    score_train_vs_validation_psi = _compute_psi(train_probability, validation_probability, bins=10)
    score_train_vs_test_psi = (
        _compute_psi(train_probability, test_probability, bins=10)
        if test_probability is not None
        else None
    )

    feature_rows: list[dict[str, Any]] = []
    for feature_name in DRIFT_FEATURE_CANDIDATES:
        if feature_name not in X_train_raw.columns or feature_name not in X_valid_raw.columns:
            continue
        train_feature = pd.to_numeric(X_train_raw[feature_name], errors="coerce").to_numpy(
            dtype="float64"
        )
        valid_feature = pd.to_numeric(X_valid_raw[feature_name], errors="coerce").to_numpy(
            dtype="float64"
        )
        train_vs_validation_psi = _compute_psi(train_feature, valid_feature, bins=10)
        train_vs_test_psi: float | None = None
        if (
            test_probability is not None
            and X_test_raw is not None
            and feature_name in X_test_raw.columns
        ):
            test_feature = pd.to_numeric(X_test_raw[feature_name], errors="coerce").to_numpy(
                dtype="float64"
            )
            train_vs_test_psi = _compute_psi(train_feature, test_feature, bins=10)
        feature_rows.append(
            {
                "feature": feature_name,
                "train_vs_validation_psi": train_vs_validation_psi,
                "train_vs_test_psi": train_vs_test_psi,
                "train_vs_validation_severity": _psi_severity(train_vs_validation_psi),
                "train_vs_test_severity": _psi_severity(train_vs_test_psi),
            }
        )

    valid_feature_psis = [
        float(row["train_vs_validation_psi"])
        for row in feature_rows
        if isinstance(row.get("train_vs_validation_psi"), float)
    ]
    test_feature_psis = [
        float(row["train_vs_test_psi"])
        for row in feature_rows
        if isinstance(row.get("train_vs_test_psi"), float)
    ]
    return {
        "score_distribution": {
            "train_vs_validation_psi": score_train_vs_validation_psi,
            "train_vs_validation_severity": _psi_severity(score_train_vs_validation_psi),
            "train_vs_test_psi": score_train_vs_test_psi,
            "train_vs_test_severity": _psi_severity(score_train_vs_test_psi),
        },
        "feature_distribution": {
            "rows": feature_rows,
            "max_train_vs_validation_psi": max(valid_feature_psis) if valid_feature_psis else None,
            "max_train_vs_test_psi": max(test_feature_psis) if test_feature_psis else None,
        },
    }


def run_temporal_holdout_validation(
    X_raw: pd.DataFrame,
    y: pd.Series,
    categorical_columns: list[str],
    numeric_columns: list[str],
    random_state: int,
    hyperparameters: dict[str, Any],
    holdout_fraction: float,
    max_estimators: int,
    false_positive_cost: float,
    false_negative_cost: float,
) -> dict[str, Any]:
    temporal_column = _pick_temporal_column(X_raw)
    if temporal_column is None:
        return {
            "enabled": False,
            "reason": "No suitable temporal reference column found.",
        }

    temporal_values = pd.to_numeric(X_raw[temporal_column], errors="coerce")
    valid_mask = temporal_values.notna()
    if int(valid_mask.sum()) < 100:
        return {
            "enabled": False,
            "reason": "Too few rows with temporal values for a reliable holdout.",
            "temporal_column": temporal_column,
        }

    X_time = X_raw.loc[valid_mask].copy()
    y_time = y.loc[valid_mask].astype("int32")
    time_values = temporal_values.loc[valid_mask]
    sorted_idx = time_values.sort_values(ascending=False).index
    holdout_rows = max(100, round(len(sorted_idx) * holdout_fraction))
    holdout_rows = min(holdout_rows, len(sorted_idx) - 100)
    if holdout_rows <= 0:
        return {
            "enabled": False,
            "reason": "Invalid temporal holdout size after constraints.",
            "temporal_column": temporal_column,
        }

    holdout_idx = sorted_idx[:holdout_rows]
    train_idx = sorted_idx[holdout_rows:]

    X_time_train_raw = X_time.loc[train_idx]
    X_time_holdout_raw = X_time.loc[holdout_idx]
    y_time_train = y_time.loc[train_idx]
    y_time_holdout = y_time.loc[holdout_idx]
    if y_time_train.nunique() < 2 or y_time_holdout.nunique() < 2:
        return {
            "enabled": False,
            "reason": "Temporal split produced single-class train or holdout segment.",
            "temporal_column": temporal_column,
        }

    temporal_encoder = fit_encoder(X_time_train_raw, categorical_columns)
    X_time_train = transform_features(
        X_time_train_raw,
        categorical_columns,
        numeric_columns,
        temporal_encoder,
    )
    X_time_holdout = transform_features(
        X_time_holdout_raw,
        categorical_columns,
        numeric_columns,
        temporal_encoder,
    )

    time_positives = float(y_time_train.sum())
    time_negatives = float(len(y_time_train) - time_positives)
    time_scale_pos_weight = time_negatives / time_positives if time_positives > 0 else 1.0
    n_estimators_cap = min(int(max_estimators), int(hyperparameters.get("n_estimators", 600)))
    temporal_model = LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators_cap,
        learning_rate=float(hyperparameters.get("learning_rate", 0.025)),
        num_leaves=int(hyperparameters.get("num_leaves", 48)),
        min_child_samples=int(hyperparameters.get("min_child_samples", 70)),
        subsample=float(hyperparameters.get("subsample", 0.9)),
        colsample_bytree=float(hyperparameters.get("colsample_bytree", 0.8)),
        reg_alpha=float(hyperparameters.get("reg_alpha", 0.05)),
        reg_lambda=float(hyperparameters.get("reg_lambda", 0.8)),
        max_depth=int(hyperparameters.get("max_depth", 12)),
        random_state=random_state,
        n_jobs=-1,
        metric="auc",
        verbosity=-1,
        importance_type="gain",
        scale_pos_weight=time_scale_pos_weight,
    )
    callbacks = [early_stopping(60, verbose=False), log_evaluation(0)]
    temporal_model.fit(
        X_time_train,
        y_time_train,
        eval_set=[(X_time_holdout, y_time_holdout)],
        eval_metric="auc",
        callbacks=callbacks,
    )

    holdout_proba = temporal_model.predict_proba(X_time_holdout)[:, 1]
    holdout_auc = float(roc_auc_score(y_time_holdout, holdout_proba))
    holdout_pr_auc = float(average_precision_score(y_time_holdout, holdout_proba))
    holdout_confusion = build_confusion_matrix_summary(y_time_holdout, holdout_proba, threshold=0.5)
    holdout_policy = build_policy_simulation_summary(
        counts=holdout_confusion["counts"],
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
    )
    return {
        "enabled": True,
        "temporal_column": temporal_column,
        "holdout_fraction": float(holdout_fraction),
        "train_rows": len(X_time_train),
        "holdout_rows": len(X_time_holdout),
        "train_positive_rate": _safe_divide(
            float(y_time_train.sum()),
            float(len(y_time_train)),
        ),
        "holdout_positive_rate": _safe_divide(
            float(y_time_holdout.sum()),
            float(len(y_time_holdout)),
        ),
        "holdout_auc": holdout_auc,
        "holdout_pr_auc": holdout_pr_auc,
        "holdout_threshold": 0.5,
        "holdout_confusion_rates": holdout_confusion["rates"],
        "holdout_policy": holdout_policy,
        "best_iteration": int(getattr(temporal_model, "best_iteration_", 0) or 0),
        "time_range_train": _temporal_range(time_values.loc[train_idx]),
        "time_range_holdout": _temporal_range(time_values.loc[holdout_idx]),
    }


def run_cross_validation_auc(
    X_raw: pd.DataFrame,
    y: pd.Series,
    categorical_columns: list[str],
    numeric_columns: list[str],
    random_state: int,
    hyperparameters: dict[str, Any],
    n_splits: int,
    max_estimators: int,
) -> dict[str, Any]:
    if n_splits < 2:
        return {
            "enabled": False,
            "reason": "cv_folds < 2; skipped.",
        }

    positive_count = int(y.sum())
    negative_count = int(len(y) - positive_count)
    minority_count = min(positive_count, negative_count)
    if minority_count < 2:
        return {
            "enabled": False,
            "reason": "Not enough minority-class rows for stratified CV.",
        }

    effective_splits = min(int(n_splits), minority_count)
    if effective_splits < 2:
        return {
            "enabled": False,
            "reason": "effective cv folds < 2; skipped.",
        }

    n_estimators_cap = min(int(max_estimators), int(hyperparameters.get("n_estimators", 600)))
    splitter = StratifiedKFold(
        n_splits=effective_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_metrics: list[dict[str, float | int]] = []
    callbacks = [early_stopping(60, verbose=False), log_evaluation(0)]

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(X_raw, y), start=1):
        X_fold_train_raw = X_raw.iloc[train_idx]
        X_fold_valid_raw = X_raw.iloc[valid_idx]
        y_fold_train = y.iloc[train_idx]
        y_fold_valid = y.iloc[valid_idx]

        fold_encoder = fit_encoder(X_fold_train_raw, categorical_columns)
        X_fold_train = transform_features(
            X_fold_train_raw, categorical_columns, numeric_columns, fold_encoder
        )
        X_fold_valid = transform_features(
            X_fold_valid_raw, categorical_columns, numeric_columns, fold_encoder
        )

        fold_positives = float(y_fold_train.sum())
        fold_negatives = float(len(y_fold_train) - fold_positives)
        fold_scale_pos_weight = fold_negatives / fold_positives if fold_positives > 0 else 1.0

        cv_model = LGBMClassifier(
            objective="binary",
            n_estimators=n_estimators_cap,
            learning_rate=float(hyperparameters.get("learning_rate", 0.025)),
            num_leaves=int(hyperparameters.get("num_leaves", 48)),
            min_child_samples=int(hyperparameters.get("min_child_samples", 70)),
            subsample=float(hyperparameters.get("subsample", 0.9)),
            colsample_bytree=float(hyperparameters.get("colsample_bytree", 0.8)),
            reg_alpha=float(hyperparameters.get("reg_alpha", 0.05)),
            reg_lambda=float(hyperparameters.get("reg_lambda", 0.8)),
            max_depth=int(hyperparameters.get("max_depth", 12)),
            random_state=random_state,
            n_jobs=-1,
            metric="auc",
            verbosity=-1,
            importance_type="gain",
            scale_pos_weight=fold_scale_pos_weight,
        )
        cv_model.fit(
            X_fold_train,
            y_fold_train,
            eval_set=[(X_fold_valid, y_fold_valid)],
            eval_metric="auc",
            callbacks=callbacks,
        )

        fold_proba = cv_model.predict_proba(X_fold_valid)[:, 1]
        fold_auc = float(roc_auc_score(y_fold_valid, fold_proba))
        fold_metrics.append(
            {
                "fold": int(fold_idx),
                "auc": fold_auc,
                "best_iteration": int(getattr(cv_model, "best_iteration_", 0) or 0),
            }
        )

    fold_auc_values = [float(row["auc"]) for row in fold_metrics]
    return {
        "enabled": True,
        "n_splits": int(effective_splits),
        "model_strategy": "binary_scale_pos_weight",
        "n_estimators_cap": int(n_estimators_cap),
        "auc_mean": float(np.mean(fold_auc_values)),
        "auc_std": float(np.std(fold_auc_values)),
        "fold_metrics": fold_metrics,
    }


def train_lightgbm_model(
    train_df: pd.DataFrame,
    random_state: int = 42,
    valid_size: float = 0.2,
    test_size: float = 0.1,
    prediction_threshold: float = 0.5,
    evaluate_test: bool = False,
    optimize_threshold: bool = True,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 5.0,
    threshold_grid_size: int = 101,
    cv_folds: int = 3,
    cv_max_estimators: int = 600,
    temporal_holdout_fraction: float = 0.2,
    temporal_max_estimators: int = 600,
    subgroup_min_size: int = 500,
) -> dict[str, Any]:
    if not 0.0 < valid_size < 1.0:
        msg = "valid_size must be between 0 and 1."
        raise ValueError(msg)
    if not 0.0 < test_size < 1.0:
        msg = "test_size must be between 0 and 1."
        raise ValueError(msg)
    if valid_size + test_size >= 1.0:
        msg = "valid_size + test_size must be less than 1."
        raise ValueError(msg)
    if false_positive_cost <= 0.0:
        msg = "false_positive_cost must be > 0."
        raise ValueError(msg)
    if false_negative_cost <= 0.0:
        msg = "false_negative_cost must be > 0."
        raise ValueError(msg)
    if threshold_grid_size < 3:
        msg = "threshold_grid_size must be >= 3."
        raise ValueError(msg)
    if cv_folds < 0:
        msg = "cv_folds must be >= 0."
        raise ValueError(msg)
    if cv_max_estimators < 10:
        msg = "cv_max_estimators must be >= 10."
        raise ValueError(msg)
    if not 0.05 <= temporal_holdout_fraction <= 0.4:
        msg = "temporal_holdout_fraction must be between 0.05 and 0.40."
        raise ValueError(msg)
    if temporal_max_estimators < 10:
        msg = "temporal_max_estimators must be >= 10."
        raise ValueError(msg)
    if subgroup_min_size < 10:
        msg = "subgroup_min_size must be >= 10."
        raise ValueError(msg)

    y = train_df[TARGET_COLUMN].astype("int32")
    X = train_df.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore")

    dataset_rows = len(train_df)
    dataset_columns = int(train_df.shape[1])
    positive_count = int(y.sum())
    negative_count = int(len(y) - positive_count)
    positive_rate = float(positive_count / len(y)) if len(y) > 0 else 0.0
    negative_rate = float(1.0 - positive_rate)

    categorical_columns, numeric_columns = split_feature_types(X)

    X_train_valid_raw, X_test_raw, y_train_valid, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    valid_share_of_train_valid = valid_size / (1.0 - test_size)
    X_train_raw, X_valid_raw, y_train, y_valid = train_test_split(
        X_train_valid_raw,
        y_train_valid,
        test_size=valid_share_of_train_valid,
        random_state=random_state,
        stratify=y_train_valid,
    )

    encoder = fit_encoder(X_train_raw, categorical_columns)
    X_train = transform_features(X_train_raw, categorical_columns, numeric_columns, encoder)
    X_valid = transform_features(X_valid_raw, categorical_columns, numeric_columns, encoder)
    X_test = transform_features(X_test_raw, categorical_columns, numeric_columns, encoder)

    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    scale_pos_weight = negatives / positives if positives > 0 else 1.0
    focal_alpha = negatives / (positives + negatives) if (positives + negatives) > 0 else 0.5
    focal_gamma = FOCAL_GAMMA

    hyperparameters: dict[str, Any] = {
        "objective": "binary",
        "n_estimators": 3000,
        "learning_rate": 0.025,
        "num_leaves": 48,
        "min_child_samples": 70,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 0.8,
        "max_depth": 12,
        "random_state": random_state,
        "n_jobs": -1,
        "metric": "auc",
        "verbosity": -1,
        "importance_type": "gain",
    }

    cross_validation = run_cross_validation_auc(
        X_raw=X_train_valid_raw,
        y=y_train_valid,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        random_state=random_state,
        hyperparameters=hyperparameters,
        n_splits=cv_folds,
        max_estimators=cv_max_estimators,
    )
    temporal_validation = run_temporal_holdout_validation(
        X_raw=X_train_valid_raw,
        y=y_train_valid,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        random_state=random_state,
        hyperparameters=hyperparameters,
        holdout_fraction=temporal_holdout_fraction,
        max_estimators=temporal_max_estimators,
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
    )

    callbacks = [early_stopping(120, verbose=False), log_evaluation(0)]

    def fit_candidate(candidate_params: dict[str, Any]) -> dict[str, Any]:
        model = LGBMClassifier(**candidate_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid), (X_train, y_train)],
            eval_names=["validation", "train"],
            eval_metric="auc",
            callbacks=callbacks,
        )
        use_raw_score = callable(candidate_params.get("objective"))
        if use_raw_score:
            train_raw = model.predict(X_train, raw_score=True)
            valid_raw = model.predict(X_valid, raw_score=True)
            train_proba = 1.0 / (1.0 + np.exp(-train_raw))
            valid_proba = 1.0 / (1.0 + np.exp(-valid_raw))
        else:
            train_proba = model.predict_proba(X_train)[:, 1]
            valid_proba = model.predict_proba(X_valid)[:, 1]
        train_auc = float(roc_auc_score(y_train, train_proba))
        valid_auc = float(roc_auc_score(y_valid, valid_proba))
        learning_curves = build_auc_learning_curves(model)
        return {
            "model": model,
            "metrics": {
                "train_auc": train_auc,
                "validation_auc": valid_auc,
            },
            "hyperparameters": candidate_params,
            "learning_curves": learning_curves,
            "probabilities": {
                "train": train_proba,
                "validation": valid_proba,
            },
        }

    weighted_params = dict(hyperparameters)
    weighted_params["scale_pos_weight"] = scale_pos_weight

    focal_params = dict(hyperparameters)
    focal_params["objective"] = make_focal_loss_objective(alpha=focal_alpha, gamma=focal_gamma)

    weighted_candidate = fit_candidate(weighted_params)
    focal_candidate = fit_candidate(focal_params)

    use_focal = (
        focal_candidate["metrics"]["validation_auc"]
        > weighted_candidate["metrics"]["validation_auc"]
    )

    chosen = focal_candidate if use_focal else weighted_candidate
    model = chosen["model"]
    metrics = dict(chosen["metrics"])
    train_proba = chosen["probabilities"]["train"]
    valid_proba = chosen["probabilities"]["validation"]
    metrics["train_pr_auc"] = float(average_precision_score(y_train, train_proba))
    metrics["validation_pr_auc"] = float(average_precision_score(y_valid, valid_proba))
    test_proba: np.ndarray | None = None
    if evaluate_test:
        use_raw_score = callable(chosen["hyperparameters"].get("objective"))
        if use_raw_score:
            test_raw = model.predict(X_test, raw_score=True)
            test_proba = 1.0 / (1.0 + np.exp(-test_raw))
        else:
            test_proba = model.predict_proba(X_test)[:, 1]
        metrics["test_auc"] = float(roc_auc_score(y_test, test_proba))
        metrics["test_pr_auc"] = float(average_precision_score(y_test, test_proba))

    threshold_curve = build_threshold_policy_curve(
        y_true=y_valid,
        positive_probability=valid_proba,
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
        grid_size=threshold_grid_size,
    )
    selected_threshold = (
        float(threshold_curve["best_threshold"])
        if optimize_threshold
        else float(prediction_threshold)
    )

    confusion_matrices = {
        "train": build_confusion_matrix_summary(y_train, train_proba, selected_threshold),
        "validation": build_confusion_matrix_summary(y_valid, valid_proba, selected_threshold),
    }
    if evaluate_test and test_proba is not None:
        confusion_matrices["test"] = build_confusion_matrix_summary(
            y_test, test_proba, selected_threshold
        )

    policy_simulation = {
        "cost_weights": {
            "false_positive_cost": float(false_positive_cost),
            "false_negative_cost": float(false_negative_cost),
        },
        "threshold_used": float(selected_threshold),
        "splits": {
            split_name: build_policy_simulation_summary(
                counts=payload["counts"],
                false_positive_cost=false_positive_cost,
                false_negative_cost=false_negative_cost,
            )
            for split_name, payload in confusion_matrices.items()
        },
    }
    validation_default_threshold_confusion = build_confusion_matrix_summary(
        y_valid, valid_proba, float(prediction_threshold)
    )
    threshold_optimization = {
        "enabled": bool(optimize_threshold),
        "default_threshold": float(prediction_threshold),
        "selected_threshold": float(selected_threshold),
        "cost_weights": {
            "false_positive_cost": float(false_positive_cost),
            "false_negative_cost": float(false_negative_cost),
        },
        "validation_policy_at_selected_threshold": build_policy_simulation_summary(
            counts=confusion_matrices["validation"]["counts"],
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        ),
        "validation_policy_at_default_threshold": build_policy_simulation_summary(
            counts=validation_default_threshold_confusion["counts"],
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        ),
        "search_curve": threshold_curve,
    }
    calibration = {
        "train": build_calibration_summary(y_train, train_proba, n_bins=10),
        "validation": build_calibration_summary(y_valid, valid_proba, n_bins=10),
    }
    if evaluate_test and test_proba is not None:
        calibration["test"] = build_calibration_summary(y_test, test_proba, n_bins=10)
    cost_sensitivity = build_cost_sensitivity_summary(
        y_true=y_valid,
        positive_probability=valid_proba,
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
        grid_size=threshold_grid_size,
    )
    subgroup_performance = {
        "validation": build_subgroup_performance_summary(
            split_name="validation",
            y_true=y_valid,
            positive_probability=valid_proba,
            raw_features=X_valid_raw,
            threshold=selected_threshold,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
            min_group_size=subgroup_min_size,
        )
    }
    if evaluate_test and test_proba is not None:
        subgroup_performance["test"] = build_subgroup_performance_summary(
            split_name="test",
            y_true=y_test,
            positive_probability=test_proba,
            raw_features=X_test_raw,
            threshold=selected_threshold,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
            min_group_size=subgroup_min_size,
        )
    drift_summary = build_drift_summary(
        X_train_raw=X_train_raw,
        X_valid_raw=X_valid_raw,
        X_test_raw=X_test_raw if evaluate_test else None,
        train_probability=train_proba,
        validation_probability=valid_proba,
        test_probability=test_proba if evaluate_test else None,
    )

    feature_names = X_train.columns.tolist()
    feature_importance = build_feature_importance(model, feature_names)

    train_positive_count = int(y_train.sum())
    train_negative_count = int(len(y_train) - train_positive_count)
    valid_positive_count = int(y_valid.sum())
    valid_negative_count = int(len(y_valid) - valid_positive_count)
    test_positive_count = int(y_test.sum())
    test_negative_count = int(len(y_test) - test_positive_count)
    serializable_hyperparameters = dict(chosen["hyperparameters"])
    if callable(serializable_hyperparameters.get("objective")):
        serializable_hyperparameters["objective"] = "custom_focal_loss"

    return {
        "model": model,
        "encoder": encoder,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "feature_columns": feature_names,
        "metrics": metrics,
        "scale_pos_weight": scale_pos_weight,
        "imbalance_strategy": "focal_loss" if use_focal else "scale_pos_weight",
        "focal_params": {
            "alpha": float(focal_alpha),
            "gamma": float(focal_gamma),
        },
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "train_rows": len(X_train),
        "valid_rows": len(X_valid),
        "test_rows": len(X_test),
        "feature_importance": feature_importance,
        "dataset_summary": {
            "rows": dataset_rows,
            "columns": dataset_columns,
        },
        "class_distribution": {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_rate": positive_rate,
            "negative_rate": negative_rate,
        },
        "split_counts": {
            "train_rows": len(X_train),
            "valid_rows": len(X_valid),
            "test_rows": len(X_test),
            "train_positive_count": train_positive_count,
            "train_negative_count": train_negative_count,
            "valid_positive_count": valid_positive_count,
            "valid_negative_count": valid_negative_count,
            "test_positive_count": test_positive_count,
            "test_negative_count": test_negative_count,
        },
        "hyperparameters": serializable_hyperparameters,
        "candidate_metrics": {
            "scale_pos_weight": weighted_candidate["metrics"],
            "focal_loss": focal_candidate["metrics"],
        },
        "random_seed": random_state,
        "threshold_used": float(selected_threshold),
        "threshold_default": float(prediction_threshold),
        "threshold_optimization": threshold_optimization,
        "policy_simulation": policy_simulation,
        "cost_sensitivity": cost_sensitivity,
        "calibration": calibration,
        "subgroup_performance": subgroup_performance,
        "drift_summary": drift_summary,
        "learning_curves": chosen["learning_curves"],
        "best_iteration": chosen["learning_curves"]["best_iteration"],
        "confusion_matrices": confusion_matrices,
        "cross_validation": cross_validation,
        "temporal_validation": temporal_validation,
        "test_evaluated": bool(evaluate_test),
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
