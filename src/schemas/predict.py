"""Pydantic v2 schemas for prediction endpoints."""
from __future__ import annotations

import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class AdultFeatures(BaseModel):
    """Input features matching the UCI Adult dataset schema.

    Hyphenated column names are exposed as aliases so callers can send
    the original dataset field names (e.g. ``"education-num": 13``).
    """

    age: float = Field(..., ge=0, le=120)
    fnlwgt: float = Field(..., ge=0)
    education_num: float = Field(..., ge=1, le=16, alias="education-num")
    capital_gain: float = Field(0.0, ge=0, alias="capital-gain")
    capital_loss: float = Field(0.0, ge=0, alias="capital-loss")
    hours_per_week: float = Field(..., ge=1, le=99, alias="hours-per-week")

    workclass: Optional[str] = None
    education: Optional[str] = None
    marital_status: Optional[str] = Field(None, alias="marital-status")
    occupation: Optional[str] = None
    relationship: Optional[str] = None
    race: Optional[str] = None
    sex: Optional[str] = None
    native_country: Optional[str] = Field(None, alias="native-country")

    model_config = {"populate_by_name": True}

    def to_feature_dict(self) -> dict[str, Any]:
        """Return a dict keyed by original hyphenated column names for the preprocessor."""
        return {
            "age": self.age,
            "fnlwgt": self.fnlwgt,
            "education-num": self.education_num,
            "capital-gain": self.capital_gain,
            "capital-loss": self.capital_loss,
            "hours-per-week": self.hours_per_week,
            "workclass": self.workclass,
            "education": self.education,
            "marital-status": self.marital_status,
            "occupation": self.occupation,
            "relationship": self.relationship,
            "race": self.race,
            "sex": self.sex,
            "native-country": self.native_country,
        }


class PredictRequest(BaseModel):
    features: AdultFeatures
    model: Literal["xgboost", "lightgbm"] = "lightgbm"


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prediction: int = Field(..., description="0 = <=50K, 1 = >50K")
    probability: float = Field(..., ge=0.0, le=1.0)
    label: str = Field(..., description="Human-readable income class")
    model_used: str

    @classmethod
    def build(
        cls,
        prediction: int,
        probability: float,
        model_used: str,
        request_id: Optional[str] = None,
    ) -> "PredictResponse":
        kwargs: dict[str, Any] = {
            "prediction": prediction,
            "probability": round(float(probability), 4),
            "label": ">50K" if prediction == 1 else "<=50K",
            "model_used": model_used,
        }
        if request_id is not None:
            kwargs["request_id"] = request_id
        return cls(**kwargs)


class BatchPredictRequest(BaseModel):
    records: list[AdultFeatures] = Field(..., min_length=1, max_length=500)
    model: Literal["xgboost", "lightgbm"] = "lightgbm"


class BatchPredictResponse(BaseModel):
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    count: int
    predictions: list[PredictResponse]

    @classmethod
    def build(
        cls,
        predictions: list[PredictResponse],
    ) -> "BatchPredictResponse":
        return cls(count=len(predictions), predictions=predictions)


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: list[str]
