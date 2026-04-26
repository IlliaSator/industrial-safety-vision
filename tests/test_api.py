import numpy as np
from fastapi.testclient import TestClient

from industrial_safety_vision.api.main import create_app
from industrial_safety_vision.api.routes import InferenceService
from industrial_safety_vision.core import BoundingBox, Detection
from industrial_safety_vision.utils.image_io import write_image


class FakeService:
    alerts: list[dict[str, object]] = []

    def __init__(self) -> None:
        self.processed_images = 0

    def model_info(self) -> dict[str, object]:
        return {
            "backend": "fake",
            "model_path": "fake.pt",
            "confidence": 0.35,
            "iou": 0.45,
            "device": "cpu",
            "classes": {0: "person"},
        }

    def predict_image_bytes(self, content: bytes) -> tuple[list[Detection], float]:
        self.processed_images += 1
        return [Detection(0, "person", 0.9, BoundingBox(1, 2, 10, 20))], 1.5

    def metrics(self) -> dict[str, object]:
        return {
            "processed_images": self.processed_images,
            "processed_videos": 0,
            "average_inference_latency_ms": 1.5 if self.processed_images else 0.0,
            "average_fps": 0.0,
            "total_alerts_generated": 0,
        }


def build_client() -> TestClient:
    app = create_app()
    app.state.service = FakeService()
    return TestClient(app)


def test_health_endpoint() -> None:
    response = build_client().get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_endpoint_with_mocked_service() -> None:
    response = build_client().get("/model/info")

    assert response.status_code == 200
    assert response.json()["backend"] == "fake"


def test_predict_image_endpoint_with_mocked_service() -> None:
    response = build_client().post(
        "/predict/image",
        files={"file": ("image.jpg", b"fake-image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["detections"][0]["class_name"] == "person"
    assert body["latency_ms"] == 1.5


def test_metrics_endpoint_with_mocked_service() -> None:
    client = build_client()
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.json()["processed_images"] == 0


def test_api_smoke_with_real_mock_service(tmp_path) -> None:
    image_path = tmp_path / "sample.jpg"
    write_image(image_path, np.zeros((120, 160, 3), dtype=np.uint8))
    app = create_app(service=InferenceService(mock_detector=True))
    client = TestClient(app)

    info = client.get("/model/info")
    prediction = client.post(
        "/predict/image",
        files={"file": ("sample.jpg", image_path.read_bytes(), "image/jpeg")},
    )
    metrics = client.get("/metrics")

    assert info.status_code == 200
    assert info.json()["backend"] == "mock_safety_detector"
    assert prediction.status_code == 200
    assert prediction.json()["detections"][0]["class_name"] == "person"
    assert metrics.json()["processed_images"] == 1
    assert metrics.json()["mode"] == "mock"
