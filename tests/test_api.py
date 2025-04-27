from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_predict_text():
    response = client.post("/predict_text", json={"text": "Сегодня был отличный футбольный матч"})
    assert response.status_code == 200
    assert "simple_nn" in response.json()
    assert "distilbert" in response.json()


def test_rate_model():
    response = client.post("/rate_model", json={"model_name": "simple_nn", "rating": 5})
    assert response.status_code == 200

