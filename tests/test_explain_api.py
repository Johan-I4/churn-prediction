import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.schemas import CustomerFeatures  # noqa: E402

VALID = CustomerFeatures.model_config["json_schema_extra"]["example"]


@pytest.fixture()
def client(model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    for name in ("ANTHROPIC_API_KEY", "LLM_MODEL", "LLM_BACKEND"):
        monkeypatch.delenv(name, raising=False)
    from src.api import app

    with TestClient(app) as c:
        yield c


def test_explain_returns_prediction_factors_and_text(client):
    r = client.post("/explain", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["threshold"] == pytest.approx(0.4)
    assert body["backend"] == "template" and body["fallback_reason"] is None
    assert body["explanation"]
    assert all(f["effect"] == "aumenta" for f in body["factors_up"])
    assert all(f["effect"] == "reduce" for f in body["factors_down"])
    assert 0 < len(body["context_sources"]) <= 3


def test_explain_probability_matches_predict(client):
    explained = client.post("/explain", json=VALID).json()
    predicted = client.post("/predict", json=VALID).json()
    assert explained["churn_probability"] == pytest.approx(predicted["churn_probability"], abs=1e-6)
    assert explained["churn_prediction"] == predicted["churn_prediction"]


def test_explain_invalid_input_returns_422(client):
    assert client.post("/explain", json={**VALID, "Contract": "Three year"}).status_code == 422


def test_failing_llm_backend_falls_back_to_template(client):
    class Broken:
        name = "anthropic"

        def explain(self, case):
            raise ConnectionError("sin red")

    client.app.state.explainer = Broken()
    body = client.post("/explain", json=VALID).json()
    assert body["backend"] == "template"
    assert body["fallback_reason"] == "anthropic falló: ConnectionError"
    assert body["explanation"]


def test_health_and_predict_still_work_without_knowledge_base(model_path, monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("KNOWLEDGE_DIR", str(tmp_path / "no_existe"))
    from src.api import app

    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        assert c.post("/predict", json=VALID).status_code == 200
        body = c.post("/explain", json=VALID).json()
        assert body["context_sources"] == []
