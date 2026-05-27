"""Integration tests for POST /api/v1/batch-predict."""
from __future__ import annotations


def _payload(records, model="lightgbm"):
    return {"records": records, "model": model}


def test_batch_single_record_200(client, sample_features):
    resp = client.post("/api/v1/batch-predict", json=_payload([sample_features]))
    assert resp.status_code == 200


def test_batch_multiple_records_200(client, sample_features):
    records = [dict(sample_features) for _ in range(5)]
    resp = client.post("/api/v1/batch-predict", json=_payload(records))
    assert resp.status_code == 200


def test_batch_count_matches_input(client, sample_features):
    records = [dict(sample_features) for _ in range(7)]
    data = client.post("/api/v1/batch-predict", json=_payload(records)).json()
    assert data["count"] == 7
    assert len(data["predictions"]) == 7


def test_batch_response_has_batch_id(client, sample_features):
    data = client.post("/api/v1/batch-predict", json=_payload([sample_features])).json()
    assert "batch_id" in data
    assert len(data["batch_id"]) == 36  # UUID4


def test_batch_each_prediction_structure(client, sample_features):
    data = client.post("/api/v1/batch-predict", json=_payload([sample_features])).json()
    pred = data["predictions"][0]
    for field in ("prediction", "probability", "label", "model_used"):
        assert field in pred, f"missing field: {field}"


def test_batch_all_probabilities_valid(client, sample_features):
    records = [dict(sample_features) for _ in range(4)]
    data = client.post("/api/v1/batch-predict", json=_payload(records, "xgboost")).json()
    for pred in data["predictions"]:
        assert 0.0 <= pred["probability"] <= 1.0
        assert pred["prediction"] in (0, 1)


def test_batch_all_labels_consistent(client, sample_features):
    records = [dict(sample_features) for _ in range(3)]
    data = client.post("/api/v1/batch-predict", json=_payload(records)).json()
    for pred in data["predictions"]:
        expected = ">50K" if pred["prediction"] == 1 else "<=50K"
        assert pred["label"] == expected


def test_batch_empty_records_422(client):
    resp = client.post("/api/v1/batch-predict", json=_payload([]))
    assert resp.status_code == 422


def test_batch_invalid_model_422(client, sample_features):
    resp = client.post("/api/v1/batch-predict", json=_payload([sample_features], "svm"))
    assert resp.status_code == 422


def test_batch_xgboost_model(client, sample_features):
    records = [dict(sample_features) for _ in range(2)]
    data = client.post("/api/v1/batch-predict", json=_payload(records, "xgboost")).json()
    assert all(p["model_used"] == "xgboost" for p in data["predictions"])
