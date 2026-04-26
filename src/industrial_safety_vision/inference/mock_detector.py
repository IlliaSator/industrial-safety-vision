"""Deterministic detector used for smoke tests and synthetic demos."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from industrial_safety_vision.core import BoundingBox, Detection


@dataclass
class MockSafetyDetector:
    """Return deterministic safety-like detections without model weights."""

    include_helmet: bool = True
    include_vest: bool = True
    include_vehicle: bool = True
    moving_person: bool = False

    def __post_init__(self) -> None:
        self._frame_index = 0

    def predict_frame(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        person_width = width * 0.18
        person_height = height * 0.55
        x1 = width * 0.18
        if self.moving_person:
            x1 += min(self._frame_index, 40) * width * 0.01
        x1 = min(x1, width - person_width - 4)
        y1 = height * 0.28
        person = BoundingBox(x1, y1, x1 + person_width, y1 + person_height)
        detections = [Detection(0, "person", 0.99, person)]

        if self.include_helmet:
            detections.append(
                Detection(
                    1,
                    "helmet",
                    0.94,
                    BoundingBox(
                        person.x1 + person.width * 0.25,
                        person.y1,
                        person.x2 - person.width * 0.25,
                        person.y1 + person.height * 0.14,
                    ),
                )
            )
        if self.include_vest:
            detections.append(
                Detection(
                    2,
                    "safety_vest",
                    0.92,
                    BoundingBox(
                        person.x1 + person.width * 0.15,
                        person.y1 + person.height * 0.28,
                        person.x2 - person.width * 0.15,
                        person.y1 + person.height * 0.68,
                    ),
                )
            )
        if self.include_vehicle:
            detections.append(
                Detection(
                    3,
                    "forklift",
                    0.88,
                    BoundingBox(width * 0.62, height * 0.42, width * 0.9, height * 0.74),
                )
            )

        self._frame_index += 1
        return detections

    def model_info(self) -> dict[str, object]:
        return {
            "backend": "mock_safety_detector",
            "model_path": "mock://deterministic-safety-demo",
            "confidence": 1.0,
            "iou": 1.0,
            "device": "cpu",
            "classes": {0: "person", 1: "helmet", 2: "safety_vest", 3: "forklift"},
        }
