"""Shared test fixtures."""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from src.core.config import settings
from src.model.loader import ModelRegistry


# A real Adult-dataset record with known structure
SAMPLE_FEATURES: dict = {
    "age": 39,
    "fnlwgt": 77516,
    "education-num": 13,
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "workclass": "State-gov",
    "education": "Bachelors",
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "native-country": "United-States",
}


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset the ModelRegistry singleton before each test."""
    ModelRegistry._instance = None
    yield
    ModelRegistry._instance = None


@pytest.fixture
def client(monkeypatch) -> TestClient:
    """HTTP test client — cache disabled, API key auth disabled."""
    monkeypatch.setattr(settings, "enable_cache", False)
    monkeypatch.setattr(settings, "api_keys", "")
    from src.api.app import create_app
    app = create_app()
    with TestClient(app) as tc:
        yield tc


@pytest.fixture
def auth_client(monkeypatch) -> TestClient:
    """HTTP test client — cache disabled, API key auth enabled."""
    monkeypatch.setattr(settings, "enable_cache", False)
    monkeypatch.setattr(settings, "api_keys", "test-secret-key")
    from src.api.app import create_app
    app = create_app()
    with TestClient(app) as tc:
        yield tc


@pytest.fixture
def sample_features() -> dict:
    return dict(SAMPLE_FEATURES)


@pytest.fixture
def predict_payload(sample_features) -> dict:
    return {"features": sample_features, "model": "lightgbm"}


@pytest.fixture
def xgboost_payload(sample_features) -> dict:
    return {"features": sample_features, "model": "xgboost"}
