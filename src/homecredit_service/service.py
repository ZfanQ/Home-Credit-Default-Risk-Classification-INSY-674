from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from homecredit_service.modeling import load_bundle, prepare_inference_frame


class PredictionService:
    def __init__(self, artifact_path: Path, threshold: float = 0.5) -> None:
        self.artifact_path = artifact_path
        self.threshold = threshold
        self.bundle = load_bundle(artifact_path)

    def metadata(self) -> dict[str, Any]:
        metrics = self.bundle.get("metrics", {})
        raw_test_auc = metrics.get("test_auc")
        return {
            "trained_at_utc": self.bundle.get("trained_at_utc", ""),
            "test_auc": float(raw_test_auc) if raw_test_auc is not None else None,
            "test_evaluated": bool(self.bundle.get("test_evaluated", False)),
            "scale_pos_weight": float(self.bundle.get("scale_pos_weight", 1.0)),
            "train_rows": int(self.bundle.get("train_rows", 0)),
            "valid_rows": int(self.bundle.get("valid_rows", 0)),
            "test_rows": int(self.bundle.get("test_rows", 0)),
            "feature_count": len(self.bundle.get("feature_columns", [])),
            "categorical_feature_count": len(self.bundle.get("categorical_columns", [])),
        }

    def feature_importance(self, limit: int = 20) -> list[dict[str, Any]]:
        importance = self.bundle.get("feature_importance", [])
        return importance[:limit]

    def predict(self, records: list[dict[str, Any]], top_n: int = 5) -> list[dict[str, Any]]:
        raw_input, transformed = prepare_inference_frame(records, self.bundle)
        model = self.bundle["model"]

        proba_or_raw = model.predict_proba(transformed)
        if isinstance(proba_or_raw, np.ndarray) and proba_or_raw.ndim == 2:
            probabilities = proba_or_raw[:, 1]
        else:
            raw_scores = model.predict(transformed, raw_score=True)
            probabilities = 1.0 / (1.0 + np.exp(-raw_scores))
        contributions = model.booster_.predict(transformed, pred_contrib=True)
        feature_names = transformed.columns.tolist()

        predictions: list[dict[str, Any]] = []
        for row_index, probability in enumerate(probabilities):
            contributor_values = contributions[row_index, :-1]
            base_value = float(contributions[row_index, -1])

            top_indices = np.argsort(np.abs(contributor_values))[::-1][:top_n]
            top_contributors = []
            for feature_idx in top_indices:
                feature = feature_names[int(feature_idx)]
                raw_value = raw_input.iloc[row_index].get(feature)
                if isinstance(raw_value, (float, np.floating)) and np.isnan(raw_value):
                    normalized_raw_value = None
                elif isinstance(raw_value, np.generic):
                    normalized_raw_value = raw_value.item()
                else:
                    normalized_raw_value = raw_value
                top_contributors.append(
                    {
                        "feature": feature,
                        "contribution": float(contributor_values[int(feature_idx)]),
                        "raw_value": normalized_raw_value,
                    }
                )

            decision = "default_risk" if probability >= self.threshold else "repay_normal"
            predictions.append(
                {
                    "default_probability": float(probability),
                    "non_default_probability": float(1.0 - probability),
                    "decision": decision,
                    "base_value": base_value,
                    "top_contributors": top_contributors,
                }
            )

        return predictions
