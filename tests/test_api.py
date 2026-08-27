"""Smoke, unit and integration tests for the YOLO inference API."""

import base64
import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
os.environ.setdefault("MODEL_NAME", "yolov8n.pt")

from app.main import _decode_image, app

client = TestClient(app)
ASSETS = Path(__file__).parent / "assets"


class TestSmoke:
    def test_health_status_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_payload_structure(self):
        data = client.get("/health").json()
        assert {"status", "model_loaded", "model_name"} <= data.keys()

    def test_metrics_endpoint_accessible(self):
        assert client.get("/metrics").status_code == 200


class TestDecodeImage:
    @staticmethod
    def _make_b64_image(width=32, height=32, fmt="JPEG"):
        image = Image.new("RGB", (width, height), color=(128, 64, 192))
        buffer = io.BytesIO()
        image.save(buffer, format=fmt)
        return base64.b64encode(buffer.getvalue()).decode()

    def test_returns_numpy_array(self):
        assert isinstance(_decode_image(self._make_b64_image()), np.ndarray)

    def test_correct_shape(self):
        assert _decode_image(self._make_b64_image(64, 48)).shape == (48, 64, 3)

    def test_png_format(self):
        assert _decode_image(self._make_b64_image(fmt="PNG")).shape[2] == 3

    def test_invalid_base64_raises(self):
        with pytest.raises(Exception):
            _decode_image("dado_invalido_nao_e_base64")


class TestPredictEndpoint:
    @pytest.fixture
    def zidane_b64(self):
        return base64.b64encode((ASSETS / "zidane.jpg").read_bytes()).decode()

    def test_predict_returns_200(self, zidane_b64):
        response = client.post(
            "/predict", json={"image_base64": zidane_b64, "confidence": 0.3}
        )
        assert response.status_code == 200

    def test_predict_detects_at_least_one_object(self, zidane_b64):
        data = client.post(
            "/predict", json={"image_base64": zidane_b64, "confidence": 0.3}
        ).json()
        assert len(data["detections"]) >= 1

    def test_predict_response_schema(self, zidane_b64):
        data = client.post(
            "/predict", json={"image_base64": zidane_b64, "confidence": 0.3}
        ).json()
        assert {
            "detections",
            "inference_ms",
            "model_used",
            "image_width",
            "image_height",
        } <= data.keys()
        assert data["inference_ms"] > 0

    def test_predict_detection_fields(self, zidane_b64):
        data = client.post(
            "/predict", json={"image_base64": zidane_b64, "confidence": 0.3}
        ).json()
        for detection in data["detections"]:
            assert isinstance(detection["label"], str)
            assert 0.0 <= detection["confidence"] <= 1.0
            assert len(detection["bbox"]) == 4

    def test_predict_missing_input_returns_422(self):
        response = client.post("/predict", json={"confidence": 0.3})
        assert response.status_code == 422


class TestBatchEndpoint:
    @pytest.fixture
    def two_images_b64(self):
        encoded = base64.b64encode((ASSETS / "zidane.jpg").read_bytes()).decode()
        return [encoded, encoded]

    def test_batch_returns_correct_count(self, two_images_b64):
        data = client.post(
            "/predict/batch",
            json={"images_base64": two_images_b64, "confidence": 0.3},
        ).json()
        assert len(data["results"]) == 2

    def test_batch_total_ms_is_positive(self, two_images_b64):
        data = client.post(
            "/predict/batch",
            json={"images_base64": two_images_b64, "confidence": 0.3},
        ).json()
        assert data["total_inference_ms"] > 0
