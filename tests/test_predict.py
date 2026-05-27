"""Integration tests for POST /api/v1/predict."""
from __future__ import annotations


def test_predict_lightgbm_200(client, predict_payload):
    resp = client.post("/api/v1/predict", json=predict_payload)
    assert resp.status_code == 200


def test_predict_xgboost_200(client, xgboost_payload):
    resp = client.post("/api/v1/predict", json=xgboost_payload)
    assert resp.status_code == 200


def test_predict_response_has_required_fields(client, predict_payload):
    data = client.post("/api/v1/predict", json=predict_payload).json()
    for field in ("prediction", "probability", "label", "model_used", "request_id"):
        assert field in data, f"missing field: {field}"


def test_predict_prediction_is_binary(client, predict_payload):
    data = client.post("/api/v1/predict", json=predict_payload).json()
    assert data["prediction"] in (0, 1)


def test_predict_probability_in_unit_interval(client, predict_payload):
    data = client.post("/api/v1/predict", json=predict_payload).json()
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_label_consistent_with_prediction(client, predict_payload):
    data = client.post("/api/v1/predict", json=predict_payload).json()
    expected = ">50K" if data["prediction"] == 1 else "<=50K"
    assert data["label"] == expected


def test_predict_model_used_matches_request(client, predict_payload):
    data = client.post("/api/v1/predict", json=predict_payload).json()
    assert data["model_used"] == predict_payload["model"]


def test_predict_xgboost_model_used(client, xgboost_payload):
    data = client.post("/api/v1/predict", json=xgboost_payload).json()
    assert data["model_used"] == "xgboost"


def test_predict_invalid_model_name_422(client, sample_features):
    payload = {"features": sample_features, "model": "random-forest"}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


def test_predict_missing_required_field_422(client):
    # fnlwgt and hours-per-week are required — omitting them fails schema
    payload = {"features": {"age": 39}, "model": "lightgbm"}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


def test_predict_age_negative_422(client, sample_features):
    sample_features["age"] = -1
    resp = client.post("/api/v1/predict", json={"features": sample_features, "model": "lightgbm"})
    assert resp.status_code == 422


def test_predict_with_zero_capital_gain(client, sample_features):
    sample_features["capital-gain"] = 0
    resp = client.post("/api/v1/predict", json={"features": sample_features, "model": "lightgbm"})
    assert resp.status_code == 200


def test_predict_with_null_optional_fields(client, sample_features):
    # All categorical fields are Optional — null values are legal
    for field in ("workclass", "education", "occupation"):
        sample_features[field] = None
    resp = client.post("/api/v1/predict", json={"features": sample_features, "model": "lightgbm"})
    assert resp.status_code == 200


def test_predict_request_id_propagated(client, predict_payload):
    resp = client.post(
        "/api/v1/predict",
        json=predict_payload,
        headers={"X-Request-ID": "trace-abc-123"},
    )
    assert resp.status_code == 200
    assert resp.json()["request_id"] == "trace-abc-123"


def test_predict_auto_generates_request_id(client, predict_payload):
    data = client.post("/api/v1/predict", json=predict_payload).json()
    assert data["request_id"]  # non-empty UUID
    assert len(data["request_id"]) == 36  # UUID4 format
