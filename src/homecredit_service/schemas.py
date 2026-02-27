from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class FeatureContribution(BaseModel):
    feature: str
    contribution: float
    raw_value: Any


class PredictionResult(BaseModel):
    default_probability: float
    non_default_probability: float
    decision: str
    base_value: float
    top_contributors: list[FeatureContribution]


class SinglePredictRequest(BaseModel):
    record: dict[str, Any] = Field(default_factory=dict)
    top_n: int = Field(default=5, ge=1, le=20)


class BatchPredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(default_factory=list, min_length=1)
    top_n: int = Field(default=5, ge=1, le=20)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictionResult]


class FeatureImportanceResponse(BaseModel):
    feature: str
    importance_gain: float
    importance_split: float


class MetadataResponse(BaseModel):
    trained_at_utc: str
    test_auc: float | None
    test_evaluated: bool
    scale_pos_weight: float
    train_rows: int
    valid_rows: int
    test_rows: int
    feature_count: int
    categorical_feature_count: int
