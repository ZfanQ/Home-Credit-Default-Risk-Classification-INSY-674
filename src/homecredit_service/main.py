from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from homecredit_service.config import get_settings
from homecredit_service.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    MetadataResponse,
    PredictionResult,
    SinglePredictRequest,
)
from homecredit_service.service import PredictionService

logger = logging.getLogger(__name__)


def create_app(artifact_path: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.prediction_service = None
        settings = get_settings()
        path = artifact_path or settings.artifact_path
        if path.exists():
            app.state.prediction_service = PredictionService(
                artifact_path=path,
                threshold=settings.prediction_threshold,
            )
            logger.info("Model artifact loaded from %s", path)
        else:
            logger.warning("Model artifact not found at %s", path)
        yield

    app = FastAPI(title="Home Credit Default Risk API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_prediction_service() -> PredictionService:
        service = app.state.prediction_service
        if service is None:
            raise HTTPException(status_code=503, detail="Model artifact is not loaded.")
        return service

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        loaded = app.state.prediction_service is not None
        return HealthResponse(status="ok", model_loaded=loaded)

    @app.get("/metadata", response_model=MetadataResponse)
    def metadata() -> MetadataResponse:
        service = get_prediction_service()
        return MetadataResponse(**service.metadata())

    @app.get("/feature-importance")
    def feature_importance(
        limit: int = Query(default=20, ge=1, le=100),
    ) -> list[dict[str, float | str]]:
        service = get_prediction_service()
        return service.feature_importance(limit=limit)

    @app.post("/predict", response_model=PredictionResult)
    def predict(payload: SinglePredictRequest) -> PredictionResult:
        service = get_prediction_service()
        predictions = service.predict([payload.record], top_n=payload.top_n)
        return PredictionResult(**predictions[0])

    @app.post("/predict/batch", response_model=BatchPredictResponse)
    def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
        service = get_prediction_service()
        predictions = service.predict(payload.records, top_n=payload.top_n)
        parsed = [PredictionResult(**entry) for entry in predictions]
        return BatchPredictResponse(predictions=parsed)

    return app


app = create_app()
