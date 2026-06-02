from fastapi.testclient import TestClient
from main import app
import base64
import numpy as np
from PIL import Image
import io

client = TestClient(app)


def make_test_image(digit_like: str = "blank") -> str:
    """Create a test image — blank white canvas encoded as base64."""
    img = Image.new('RGB', (280, 280), color='white')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_blank_canvas():
    # Blank canvas should return a prediction (model handles all-zero input)
    image_data = make_test_image()
    response = client.post("/predict", json={"image_data": image_data})
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "digit" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert len(data["probabilities"]) == 10
        assert 0 <= data["digit"] <= 9
        assert 0.0 <= data["confidence"] <= 1.0


def test_predict_invalid_image():
    response = client.post("/predict",
        json={"image_data": "not-valid-base64!!!"})
    assert response.status_code == 400


def test_model_info():
    response = client.get("/api/model-info")
    assert response.status_code == 200


def test_training_history():
    response = client.get("/api/training-history")
    assert response.status_code == 200
