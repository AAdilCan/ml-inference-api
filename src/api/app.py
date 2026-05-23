"""FastAPI application factory."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import health, predict
from src.core.config import settings
from src.core.logging import configure_logging
from src.model.loader import ModelRegistry


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    ModelRegistry.get()  # warm-up models at startup
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(predict.router, prefix="/api/v1")
    return app


app = create_app()
