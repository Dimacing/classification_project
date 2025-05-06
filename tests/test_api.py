from fastapi.testclient import TestClient
from api.main import app
import pytest
import os

client = TestClient(app)

all_models_loaded = (
    "simple_nn" in client.app.state.models and
    "random_forest" in client.app.state.models and
    "logistic_regression" in client.app.state.models
)
expected_models = {"simple_nn", "random_forest", "logistic_regression"}

@pytest.mark.skipif(not all_models_loaded, reason="One or more models did not load, skipping API tests")
def test_predict_text_success():
    response = client.post("/predict_text", json={"text": "Это текст про спорт и немного про юмор"})
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == expected_models
    for model_name in expected_models:
        assert isinstance(data[model_name], dict)
        assert "спорт" in data[model_name]
        assert isinstance(data[model_name]["спорт"], (float, int))

def test_predict_text_empty():
    response = client.post("/predict_text", json={"text": " "})
    assert response.status_code == 400
    assert "detail" in response.json()

def test_predict_text_no_json():
    response = client.post("/predict_text", data="не json")
    assert response.status_code == 422

dummy_file_content = "Текст из файла для теста API."
dummy_file_path = "test_upload.txt"
with open(dummy_file_path, "w", encoding="utf-8") as f:
    f.write(dummy_file_content)

@pytest.mark.skipif(not all_models_loaded, reason="Models did not load, skipping API tests")
def test_predict_file_success():
    with open(dummy_file_path, "rb") as f:
        response = client.post("/predict_file", files={"file": ("test_upload.txt", f, "text/plain")})
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == expected_models
    assert isinstance(data["simple_nn"], dict)
    assert isinstance(data["random_forest"], dict)
    assert isinstance(data["logistic_regression"], dict)

try:
    os.remove(dummy_file_path)
except OSError:
    pass

@pytest.fixture(params=list(expected_models))
def model_to_rate(request):
    model_name = request.param
    if model_name in client.app.state.models:
        return model_name
    else:
        pytest.skip(f"Model {model_name} not loaded, skipping rating test for it.")

def test_rate_model_success(model_to_rate):
    rating_value = 4
    response = client.post("/rate_model", json={"model_name": model_to_rate, "rating": rating_value})
    assert response.status_code == 200
    assert f"Рейтинг {rating_value} сохранен" in response.json()["message"]
    assert model_to_rate in response.json()["message"]

def test_rate_model_invalid_name():
    response = client.post("/rate_model", json={"model_name": "non_existent_model", "rating": 3})
    assert response.status_code == 404

@pytest.mark.skipif(not all_models_loaded, reason="Models did not load, skipping invalid rating tests")
@pytest.mark.parametrize("invalid_rating", [0, 6, -1, 5.5])
def test_rate_model_invalid_rating(invalid_rating):
    model_name = "simple_nn" if "simple_nn" in client.app.state.models else list(client.app.state.models.keys())[0]
    response = client.post("/rate_model", json={"model_name": model_name, "rating": invalid_rating})
    assert response.status_code in [400, 422]

@pytest.mark.skipif(not all_models_loaded, reason="Models did not load, skipping get ratings test")
def test_get_ratings():
    for model_name in client.app.state.models.keys():
        client.post("/rate_model", json={"model_name": model_name, "rating": 3})

    response = client.get("/model_ratings")
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == expected_models
    for model_name in expected_models:
        assert isinstance(data[model_name], float)

def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "Классификация текста по темам" in response.text
    if all_models_loaded:
        for model_name in expected_models:
            assert model_name in response.text
