"""FastAPI application factory."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.api.limiter import limiter
from src.api.middleware.auth import ApiKeyMiddleware
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

    # rate limiter state
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # middleware — added last-in, first-executed
    app.add_middleware(ApiKeyMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    # routes
    app.include_router(health.router)
    app.include_router(predict.router, prefix="/api/v1")

    return app


app = create_app()
