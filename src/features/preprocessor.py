"""Feature preprocessing pipeline for inference."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger(__name__)


class FeatureProcessor:
    """Transform raw input dicts into model-ready numpy arrays.

    Mirrors the training-time pipeline: impute → label-encode → reorder.
    Unseen categories fall back to the 'nan' class (or index 0 if absent).
    Missing numerics fall back to 0.0, consistent with training imputation.
    """

    def __init__(self, models_dir: Path) -> None:
        schema_path = models_dir / "feature_schema.json"
        encoders_path = models_dir / "encoders.pkl"

        with schema_path.open() as f:
            schema = json.load(f)

        self.numeric_cols: list[str] = schema["numeric_cols"]
        self.categorical_cols: list[str] = schema["categorical_cols"]
        self.feature_order: list[str] = schema["feature_order"]
        self.encoders: dict[str, LabelEncoder] = joblib.load(encoders_path)

        # cache the index of 'nan' class per categorical column for fast fallback
        self._nan_index: dict[str, int] = {}
        for col, enc in self.encoders.items():
            classes = list(enc.classes_)
            self._nan_index[col] = classes.index("nan") if "nan" in classes else 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _impute_numeric(self, value: Any) -> float:
        try:
            v = float(value)
            return 0.0 if np.isnan(v) else v
        except (TypeError, ValueError):
            return 0.0

    def _encode_categorical(self, value: Any, col: str) -> int:
        raw = str(value).strip() if value is not None else "nan"
        enc = self.encoders[col]
        if raw in enc.classes_:
            return int(enc.transform([raw])[0])
        return self._nan_index[col]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, record: dict[str, Any]) -> np.ndarray:
        """Transform a single input dict to a 1-D float32 feature array."""
        row: dict[str, float] = {}

        for col in self.numeric_cols:
            row[col] = self._impute_numeric(record.get(col))

        for col in self.categorical_cols:
            row[col] = self._encode_categorical(record.get(col), col)

        return np.array([row[col] for col in self.feature_order], dtype=np.float32)

    def transform_batch(self, records: list[dict[str, Any]]) -> np.ndarray:
        """Transform a list of input dicts to a 2-D feature matrix."""
        return np.stack([self.transform(r) for r in records])
