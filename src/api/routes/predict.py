"""Prediction endpoints — /predict and /batch-predict."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from src.model.loader import ModelRegistry
from src.schemas.predict import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(request: Request, body: PredictRequest) -> PredictResponse:
    registry = ModelRegistry.get()
    if body.model not in registry.loaded_names():
        raise HTTPException(status_code=400, detail=f"Model '{body.model}' not available")

    features = body.features.to_feature_dict()
    request_id = request.headers.get("X-Request-ID")
    pred, prob = registry.predict(features, body.model)

    log.info(
        "prediction completed",
        extra={"model": body.model, "prediction": pred, "probability": round(prob, 4)},
    )
    return PredictResponse.build(pred, prob, body.model, request_id)


@router.post("/batch-predict", response_model=BatchPredictResponse, tags=["inference"])
async def batch_predict(request: Request, body: BatchPredictRequest) -> BatchPredictResponse:
    registry = ModelRegistry.get()
    if body.model not in registry.loaded_names():
        raise HTTPException(status_code=400, detail=f"Model '{body.model}' not available")

    records = [r.to_feature_dict() for r in body.records]
    results = registry.predict_batch(records, body.model)
    predictions = [PredictResponse.build(p, prob, body.model) for p, prob in results]

    log.info("batch prediction completed", extra={"model": body.model, "count": len(predictions)})
    return BatchPredictResponse.build(predictions)
