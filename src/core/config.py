"""Application configuration via environment variables."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


ROOT_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"


class Settings(BaseSettings):
    app_name: str = "ml-inference-api"
    app_version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    # model
    default_model: Literal["xgboost", "lightgbm"] = "lightgbm"
    models_dir: Path = MODELS_DIR

    # Redis (optional caching)
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    enable_cache: bool = False

    # rate limiting
    rate_limit_requests: int = 60
    rate_limit_window: str = "minute"

    # API key auth (comma-separated keys; empty = auth disabled)
    api_keys: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def valid_api_keys(self) -> set[str]:
        if not self.api_keys:
            return set()
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


settings = Settings()
