"""Prediction endpoints — /predict and /batch-predict."""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.limiter import limiter
from src.cache.redis_cache import RedisCache
from src.core.config import settings
from src.model.loader import ModelRegistry
from src.schemas.predict import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)

log = logging.getLogger(__name__)
router = APIRouter()

_cache: RedisCache | None = None


def _get_cache() -> RedisCache | None:
    global _cache
    if _cache is None and settings.enable_cache:
        _cache = RedisCache(settings.redis_url, settings.cache_ttl_seconds)
    return _cache


@router.post("/predict", response_model=PredictResponse, tags=["inference"])
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}")
async def predict(
    request: Request,
    body: PredictRequest,
    cache: RedisCache | None = Depends(_get_cache),
) -> PredictResponse:
    registry = ModelRegistry.get()
    if body.model not in registry.loaded_names():
        raise HTTPException(status_code=400, detail=f"Model '{body.model}' not available")

    features = body.features.to_feature_dict()
    request_id = request.headers.get("X-Request-ID")

    if cache is not None:
        cached = cache.get(features, body.model)
        if cached:
            log.info("cache hit", extra={"model": body.model, "request_id": request_id})
            return PredictResponse(**cached)

    pred, prob = registry.predict(features, body.model)
    response = PredictResponse.build(pred, prob, body.model, request_id)

    if cache is not None:
        cache.set(features, body.model, response.model_dump())

    log.info(
        "prediction completed",
        extra={"model": body.model, "prediction": pred, "probability": round(prob, 4)},
    )
    return response


@router.post("/batch-predict", response_model=BatchPredictResponse, tags=["inference"])
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}")
async def batch_predict(
    request: Request,
    body: BatchPredictRequest,
) -> BatchPredictResponse:
    registry = ModelRegistry.get()
    if body.model not in registry.loaded_names():
        raise HTTPException(status_code=400, detail=f"Model '{body.model}' not available")

    records = [r.to_feature_dict() for r in body.records]
    results = registry.predict_batch(records, body.model)
    predictions = [PredictResponse.build(p, prob, body.model) for p, prob in results]

    log.info("batch prediction completed", extra={"model": body.model, "count": len(predictions)})
    return BatchPredictResponse.build(predictions)
