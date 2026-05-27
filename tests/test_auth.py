"""Tests for API key authentication middleware."""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Auth disabled (default)
# ---------------------------------------------------------------------------


def test_predict_no_key_required_when_auth_disabled(client, predict_payload):
    resp = client.post("/api/v1/predict", json=predict_payload)
    assert resp.status_code == 200


def test_health_always_accessible_without_key(client):
    resp = client.get("/health")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Auth enabled
# ---------------------------------------------------------------------------


def test_missing_key_returns_401(auth_client, predict_payload):
    resp = auth_client.post("/api/v1/predict", json=predict_payload)
    assert resp.status_code == 401


def test_wrong_key_returns_401(auth_client, predict_payload):
    resp = auth_client.post(
        "/api/v1/predict",
        json=predict_payload,
        headers={"X-API-Key": "not-the-right-key"},
    )
    assert resp.status_code == 401


def test_correct_key_in_header_passes(auth_client, predict_payload):
    resp = auth_client.post(
        "/api/v1/predict",
        json=predict_payload,
        headers={"X-API-Key": "test-secret-key"},
    )
    assert resp.status_code == 200


def test_correct_key_via_query_param_passes(auth_client, predict_payload):
    resp = auth_client.post(
        "/api/v1/predict?api_key=test-secret-key",
        json=predict_payload,
    )
    assert resp.status_code == 200


def test_health_exempt_from_auth(auth_client):
    resp = auth_client.get("/health")
    assert resp.status_code == 200


def test_metrics_exempt_from_auth(auth_client):
    resp = auth_client.get("/metrics")
    assert resp.status_code == 200
