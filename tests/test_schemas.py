"""Unit tests for Pydantic schemas — no HTTP, no models needed."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas.predict import (
    AdultFeatures,
    BatchPredictRequest,
    PredictResponse,
)


_VALID: dict = {
    "age": 39,
    "fnlwgt": 77516,
    "education-num": 13,
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
}


# ---------------------------------------------------------------------------
# AdultFeatures — field aliases
# ---------------------------------------------------------------------------


def test_features_accept_hyphenated_aliases():
    feat = AdultFeatures(**_VALID)
    assert feat.education_num == 13
    assert feat.capital_gain == 2174
    assert feat.hours_per_week == 40


def test_features_accept_snake_case():
    snake = {k.replace("-", "_"): v for k, v in _VALID.items()}
    feat = AdultFeatures(**snake)
    assert feat.age == 39


def test_feature_dict_uses_hyphenated_keys():
    d = AdultFeatures(**_VALID).to_feature_dict()
    assert "education-num" in d
    assert "capital-gain" in d
    assert "hours-per-week" in d
    assert "education_num" not in d


# ---------------------------------------------------------------------------
# AdultFeatures — field validation
# ---------------------------------------------------------------------------


def test_age_below_zero_rejected():
    with pytest.raises(ValidationError):
        AdultFeatures(**{**_VALID, "age": -1})


def test_age_above_120_rejected():
    with pytest.raises(ValidationError):
        AdultFeatures(**{**_VALID, "age": 121})


def test_hours_per_week_zero_rejected():
    with pytest.raises(ValidationError):
        AdultFeatures(**{**_VALID, "hours-per-week": 0})


def test_missing_required_age_rejected():
    incomplete = {k: v for k, v in _VALID.items() if k != "age"}
    with pytest.raises(ValidationError):
        AdultFeatures(**incomplete)


def test_missing_required_fnlwgt_rejected():
    incomplete = {k: v for k, v in _VALID.items() if k != "fnlwgt"}
    with pytest.raises(ValidationError):
        AdultFeatures(**incomplete)


# ---------------------------------------------------------------------------
# PredictResponse.build
# ---------------------------------------------------------------------------


def test_response_build_high_income():
    r = PredictResponse.build(1, 0.82, "lightgbm")
    assert r.prediction == 1
    assert r.label == ">50K"
    assert r.probability == 0.82
    assert r.model_used == "lightgbm"
    assert r.request_id  # auto-generated UUID


def test_response_build_low_income():
    r = PredictResponse.build(0, 0.23, "xgboost")
    assert r.prediction == 0
    assert r.label == "<=50K"


def test_response_build_custom_request_id():
    r = PredictResponse.build(1, 0.9, "lightgbm", request_id="req-abc")
    assert r.request_id == "req-abc"


# ---------------------------------------------------------------------------
# BatchPredictRequest — length constraints
# ---------------------------------------------------------------------------


def test_batch_empty_records_rejected():
    with pytest.raises(ValidationError):
        BatchPredictRequest(records=[], model="lightgbm")
