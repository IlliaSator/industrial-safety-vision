"""API routes and lightweight in-memory inference service."""

from __future__ import annotations

import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, File, Request, UploadFile

from industrial_safety_vision.api.schemas import (
    AlertResponse,
    HealthResponse,
    ImagePredictionResponse,
    MetricsResponse,
    ModelInfoResponse,
    VideoPredictionResponse,
)
from industrial_safety_vision.config.settings import settings
from industrial_safety_vision.core import Detection
from industrial_safety_vision.inference.detector import YOLODetector
from industrial_safety_vision.inference.video_inference import run_video_inference

router = APIRouter()


class InferenceService:
    """Simple in-memory service state for demos and tests."""

    def __init__(self) -> None:
        self._detector: YOLODetector | None = None
        self.processed_images = 0
        self.processed_videos = 0
        self.latencies_ms: list[float] = []
        self.video_fps_values: list[float] = []
        self.alerts: list[dict[str, object]] = []

    @property
    def detector(self) -> YOLODetector:
        if self._detector is None:
            self._detector = YOLODetector(
                settings.model_path,
                confidence=settings.inference_confidence,
                iou=settings.inference_iou,
                device=settings.device,
            )
        return self._detector

    def model_info(self) -> dict[str, Any]:
        return self.detector.model_info()

    def predict_image_bytes(self, content: bytes) -> tuple[list[Detection], float]:
        frame = _decode_image_bytes(content)
        started = time.perf_counter()
        detections = self.detector.predict_frame(frame)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.processed_images += 1
        self.latencies_ms.append(latency_ms)
        return detections, latency_ms

    def predict_video_file(self, input_path: Path) -> dict[str, Any]:
        summary = run_video_inference(input_path, "data/outputs/api_annotated_video.mp4")
        self.processed_videos += 1
        self.video_fps_values.append(summary.fps)
        self.alerts.extend(_read_alerts(summary.alerts_path))
        return summary.to_dict()

    def metrics(self) -> dict[str, Any]:
        return {
            "processed_images": self.processed_images,
            "processed_videos": self.processed_videos,
            "average_inference_latency_ms": _average(self.latencies_ms),
            "average_fps": _average(self.video_fps_values),
            "total_alerts_generated": len(self.alerts),
        }


def get_service(request: Request) -> InferenceService:
    return request.app.state.service


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info(service: InferenceService = Depends(get_service)) -> dict[str, Any]:
    return service.model_info()


@router.post("/predict/image", response_model=ImagePredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    service: InferenceService = Depends(get_service),
) -> dict[str, Any]:
    detections, latency_ms = service.predict_image_bytes(await file.read())
    return {
        "detections": [detection.to_dict() for detection in detections],
        "latency_ms": latency_ms,
    }


@router.post("/predict/video", response_model=VideoPredictionResponse)
async def predict_video(
    file: UploadFile = File(...),
    service: InferenceService = Depends(get_service),
) -> dict[str, Any]:
    suffix = Path(file.filename or "video.mp4").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        input_path = Path(tmp.name)

    summary = service.predict_video_file(input_path)
    alert_summary = Counter(alert.get("alert_type", "unknown") for alert in service.alerts)
    return {
        "output_video_path": summary.get("output_path"),
        "fps": summary.get("fps", 0.0),
        "processed_frames": summary.get("processed_frames", 0),
        "number_of_alerts": summary.get("total_alerts", 0),
        "alert_summary": dict(alert_summary),
    }


@router.get("/alerts", response_model=AlertResponse)
def alerts(service: InferenceService = Depends(get_service)) -> dict[str, Any]:
    return {"alerts": service.alerts[-100:]}


@router.get("/metrics", response_model=MetricsResponse)
def metrics(service: InferenceService = Depends(get_service)) -> dict[str, Any]:
    return service.metrics()


def _decode_image_bytes(content: bytes) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment-specific
        msg = "OpenCV is required to decode uploaded images."
        raise RuntimeError(msg) from exc
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file is not a decodable image.")
    return image


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _read_alerts(path: str | None) -> list[dict[str, object]]:
    if not path or not Path(path).exists():
        return []
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))
