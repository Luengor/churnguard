import math

import pytest
from fastapi.testclient import TestClient

from serve import CATEGORIES, app, meta, raw

client = TestClient(app)


@pytest.fixture
def customer() -> dict:
    row = raw.iloc[0].to_dict()
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict(customer):
    response = client.post("/predict", json=customer)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn"], bool)


def test_predict_rejects_invalid_category(customer):
    column = next(iter(CATEGORIES))
    customer[column] = "not-a-real-category"
    assert client.post("/predict", json=customer).status_code == 422


def test_explain(customer):
    response = client.post("/explain", json=customer)
    assert response.status_code == 200
    body = response.json()
    n_features = len(meta.feature_columns)
    assert len(body["shap_values"]) == n_features
    assert len(body["data"]) == n_features
    assert len(body["feature_names"]) == n_features
    assert isinstance(body["base_value"], float)


def test_explain_reconciles_with_predict(customer):
    explanation = client.post("/explain", json=customer).json()
    probability = client.post("/predict", json=customer).json()["churn_probability"]
    margin = explanation["base_value"] + sum(explanation["shap_values"])
    assert 1 / (1 + math.exp(-margin)) == pytest.approx(probability, abs=1e-5)


def test_metrics_exposes_request_counter():
    client.get("/health")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text
