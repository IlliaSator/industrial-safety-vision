"""Pydantic schemas for the inference API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBoxResponse(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBoxResponse


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "industrial-safety-vision"


class ModelInfoResponse(BaseModel):
    backend: str
    model_path: str
    confidence: float
    iou: float
    device: str
    classes: dict[int, str] = Field(default_factory=dict)


class ImagePredictionResponse(BaseModel):
    detections: list[DetectionResponse]
    latency_ms: float


class VideoPredictionResponse(BaseModel):
    output_video_path: str | None
    fps: float
    processed_frames: int
    number_of_alerts: int
    alert_summary: dict[str, int]


class AlertResponse(BaseModel):
    alerts: list[dict[str, object]]


class MetricsResponse(BaseModel):
    processed_images: int
    processed_videos: int
    average_inference_latency_ms: float
    average_fps: float
    total_alerts_generated: int
