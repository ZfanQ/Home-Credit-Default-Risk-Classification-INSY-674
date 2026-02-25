from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from homecredit_service.main import create_app
from homecredit_service.modeling import save_bundle, train_lightgbm_model


def build_bundle(tmp_path: Path) -> Path:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(250),
            "TARGET": rng.binomial(1, 0.08, size=250),
            "AMT_INCOME_TOTAL": rng.normal(170000, 30000, size=250),
            "AMT_CREDIT": rng.normal(550000, 45000, size=250),
            "AMT_ANNUITY": rng.normal(26000, 3200, size=250),
            "DAYS_BIRTH": -rng.integers(9000, 25000, size=250),
            "CODE_GENDER": rng.choice(["M", "F"], size=250),
        }
    )
    bundle = train_lightgbm_model(frame, random_state=7, valid_size=0.2)
    artifact = tmp_path / "model_bundle.joblib"
    save_bundle(bundle, artifact_path=artifact)
    return artifact


def test_api_predict_and_metadata(tmp_path: Path) -> None:
    artifact_path = build_bundle(tmp_path)
    app = create_app(artifact_path=artifact_path)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True

        cors = client.get("/health", headers={"Origin": "null"})
        assert cors.status_code == 200
        assert cors.headers.get("access-control-allow-origin") == "*"

        metadata = client.get("/metadata")
        assert metadata.status_code == 200
        assert 0.0 <= metadata.json()["validation_auc"] <= 1.0

        response = client.post(
            "/predict",
            json={
                "record": {
                    "AMT_INCOME_TOTAL": 180000,
                    "AMT_CREDIT": 600000,
                    "AMT_ANNUITY": 26000,
                    "DAYS_BIRTH": -13000,
                    "CODE_GENDER": "F",
                },
                "top_n": 3,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["default_probability"] <= 1.0
        assert len(body["top_contributors"]) == 3
