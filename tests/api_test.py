from fastapi.testclient import TestClient

from mlops.api import model_loader
from mlops.api.main import app


class DummyModel:
    def predict(self, X):
        # Regresa siempre la misma clase para simplificar el test
        return ["happy"]


def test_predict_endpoint(monkeypatch):
    # Parchea get_model para no depender de MLflow en el test
    monkeypatch.setattr(model_loader, "get_model", lambda: DummyModel())

    client = TestClient(app)
    payload = {"features": {"feature1": 0.5, "feature2": 1.2}}

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] == "happy"
    assert "model_uri" in body


# def test_health_endpoint(monkeypatch):
#     monkeypatch.setattr(model_loader, "get_model", lambda: DummyModel())

#     client = TestClient(app)
#     resp = client.get("/health")
#     assert resp.status_code == 200
#     body = resp.json()
#     assert body["status"] in ("ok", "degraded")
#     assert "model_uri" in body
