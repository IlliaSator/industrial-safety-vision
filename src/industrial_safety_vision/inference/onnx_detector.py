"""ONNX Runtime detector backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from industrial_safety_vision.core import Detection


class ONNXDetector:
    """Minimal ONNX Runtime adapter for exported YOLO models.

    The class currently focuses on optimized model execution and benchmarking.
    Full YOLO post-processing can be added once the exported model variant and
    class mapping are fixed for a target deployment.
    """

    def __init__(self, model_path: str | Path, *, input_size: int = 640) -> None:
        self.model_path = Path(model_path)
        self.input_size = input_size
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for ONNX inference.") from exc

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict_frame(self, frame: np.ndarray) -> list[Detection]:
        """Execute ONNX model and return detections when post-processing is implemented."""

        _ = self.run_raw(frame)
        return []

    def run_raw(self, frame: np.ndarray) -> list[Any]:
        blob = self._preprocess(frame)
        return self.session.run(None, {self.input_name: blob})

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for ONNX preprocessing.") from exc
        resized = cv2.resize(frame, (self.input_size, self.input_size))
        blob = resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(blob, axis=0)
