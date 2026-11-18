import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.schemas import CustomerFeatures  # noqa: E402

VALID = CustomerFeatures.model_config["json_schema_extra"]["example"]


@pytest.fixture()
def client(model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    from src.api import app

    with TestClient(app) as c:  # el "with" ejecuta el lifespan (carga el modelo)
        yield c


def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_valid_returns_probability(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn_prediction"], bool)
    assert body["threshold"] == pytest.approx(0.4)


def test_predict_invalid_returns_422(client):
    bad = {**VALID, "Contract": "Three year"}
    assert client.post("/predict", json=bad).status_code == 422
    incomplete = {k: v for k, v in VALID.items() if k != "tenure"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_api_matches_pipeline_probabilities(client, model_path):
    import joblib
    import pandas as pd

    pipeline = joblib.load(model_path)["pipeline"]
    expected = float(pipeline.predict_proba(pd.DataFrame([VALID]))[0, 1])
    got = client.post("/predict", json=VALID).json()["churn_probability"]
    assert got == pytest.approx(expected, abs=1e-6)
