from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    artifact_path: Path = Path("artifacts/model_bundle.joblib")
    prediction_threshold: float = 0.5

    model_config = SettingsConfigDict(env_prefix="HC_", env_file=".env", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
