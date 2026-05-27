"""Tests for GET /health."""
from __future__ import annotations


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_status_ok(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"


def test_health_contains_required_fields(client):
    data = client.get("/health").json()
    assert "status" in data
    assert "version" in data
    assert "models_loaded" in data


def test_health_both_models_loaded(client):
    models = client.get("/health").json()["models_loaded"]
    assert "xgboost" in models
    assert "lightgbm" in models


def test_health_version_semver_format(client):
    version = client.get("/health").json()["version"]
    parts = version.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)
