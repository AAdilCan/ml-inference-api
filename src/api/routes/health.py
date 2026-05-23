"""Health check endpoint."""
from __future__ import annotations

from fastapi import APIRouter

from src.core.config import settings
from src.model.loader import ModelRegistry
from src.schemas.predict import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    registry = ModelRegistry.get()
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        models_loaded=registry.loaded_names(),
    )
