"""Model registry — loads XGBoost/LightGBM artifacts once at startup."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.core.config import settings
from src.features.preprocessor import FeatureProcessor

log = logging.getLogger(__name__)


class ModelRegistry:
    """Singleton that holds loaded models and the feature processor."""

    _instance: "ModelRegistry | None" = None

    def __init__(self, models_dir: Path) -> None:
        self.processor = FeatureProcessor(models_dir)
        self._models: dict[str, Any] = {}
        self._load_models(models_dir)

    @classmethod
    def get(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = cls(settings.models_dir)
        return cls._instance

    def _load_models(self, models_dir: Path) -> None:
        for name, filename in [
            ("xgboost", "xgboost_model.pkl"),
            ("lightgbm", "lightgbm_model.pkl"),
        ]:
            path = models_dir / filename
            if path.exists():
                self._models[name] = joblib.load(path)
                log.info("loaded model %s from %s", name, path)
            else:
                log.warning("model artifact not found: %s", path)

    def loaded_names(self) -> list[str]:
        return list(self._models.keys())

    def predict(self, features: dict[str, Any], model_name: str) -> tuple[int, float]:
        """Return (predicted_class, positive_class_probability)."""
        model = self._models[model_name]
        arr = self.processor.transform(features).reshape(1, -1)
        prob = float(model.predict_proba(arr)[0, 1])
        return int(prob >= 0.5), prob

    def predict_batch(
        self, records: list[dict[str, Any]], model_name: str
    ) -> list[tuple[int, float]]:
        model = self._models[model_name]
        arr = self.processor.transform_batch(records)
        probas = model.predict_proba(arr)[:, 1]
        return [(int(p >= 0.5), float(p)) for p in probas]
