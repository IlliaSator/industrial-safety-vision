"""YOLO detector abstraction returning project domain objects."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from industrial_safety_vision.core import BoundingBox, Detection


class ModelLoadError(RuntimeError):
    """Raised when a detector backend cannot be loaded."""


class YOLODetector:
    """Thin adapter around Ultralytics YOLO.

    The adapter deliberately returns only :class:`Detection` objects so the rest
    of the system can be tested without importing or depending on Ultralytics
    result internals.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence: float = 0.35,
        iou: float = 0.45,
        device: str = "auto",
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.confidence = confidence
        self.iou = iou
        self.device = "cpu" if device == "auto" else device
        self.model = self._load_model(model_factory)
        self.class_names = self._extract_names(self.model)

    def predict_image(self, image_path: str | Path) -> list[Detection]:
        """Run inference for an image path."""

        path = Path(image_path)
        if not path.exists():
            msg = f"Image does not exist: {path}"
            raise FileNotFoundError(msg)
        return self._predict(path)

    def predict_frame(self, frame: np.ndarray) -> list[Detection]:
        """Run inference for an already-decoded frame."""

        if frame.size == 0:
            raise ValueError("Cannot run inference on an empty frame.")
        return self._predict(frame)

    def model_info(self) -> dict[str, object]:
        """Return lightweight model metadata suitable for API responses."""

        return {
            "backend": "ultralytics_yolo",
            "model_path": self.model_path,
            "confidence": self.confidence,
            "iou": self.iou,
            "device": self.device,
            "classes": self.class_names,
        }

    def _load_model(self, model_factory: Callable[[str], Any] | None) -> Any:
        if model_factory is not None:
            return model_factory(self.model_path)
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            msg = "Ultralytics is not installed. Install project dependencies before loading YOLO."
            raise ModelLoadError(msg) from exc
        try:
            return YOLO(self.model_path)
        except Exception as exc:  # pragma: no cover - backend-specific errors
            msg = (
                f"Failed to load YOLO model '{self.model_path}'. Provide a valid checkpoint "
                "or model name such as yolov8n.pt for demo mode."
            )
            raise ModelLoadError(msg) from exc

    def _predict(self, source: str | Path | np.ndarray) -> list[Detection]:
        results = self.model.predict(
            source=source,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        return self._convert_result(results[0])

    def _convert_result(self, result: Any) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = _to_numpy(getattr(boxes, "xyxy", []))
        classes = _to_numpy(getattr(boxes, "cls", []))
        confidences = _to_numpy(getattr(boxes, "conf", []))

        detections: list[Detection] = []
        for coords, class_id_raw, confidence_raw in zip(xyxy, classes, confidences, strict=False):
            class_id = int(class_id_raw)
            x1, y1, x2, y2 = [float(value) for value in coords]
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self.class_names.get(class_id, str(class_id)),
                    confidence=float(confidence_raw),
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )
        return detections

    @staticmethod
    def _extract_names(model: Any) -> dict[int, str]:
        names = getattr(model, "names", {})
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, list):
            return dict(enumerate(str(value) for value in names))
        return {}


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)
