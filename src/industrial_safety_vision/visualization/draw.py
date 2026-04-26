"""Drawing helpers for detections, tracks, zones, and alerts."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import cv2
import numpy as np

from industrial_safety_vision.core import Detection
from industrial_safety_vision.safety.alert_types import Alert
from industrial_safety_vision.tracking.track_types import Track

Color = tuple[int, int, int]

CLASS_COLORS: dict[str, Color] = {
    "person": (60, 180, 75),
    "helmet": (0, 215, 255),
    "safety_vest": (255, 190, 0),
    "forklift": (0, 100, 255),
    "vehicle": (0, 100, 255),
}
ALERT_COLOR: Color = (0, 0, 255)


def draw_detections(frame: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    annotated = frame.copy()
    for detection in detections:
        color = CLASS_COLORS.get(detection.class_name, (220, 220, 220))
        _draw_box(
            annotated,
            detection.bbox.to_xyxy(),
            f"{detection.class_name} {detection.confidence:.2f}",
            color,
        )
    return annotated


def draw_tracks(frame: np.ndarray, tracks: Iterable[Track]) -> np.ndarray:
    annotated = frame.copy()
    for track in tracks:
        color = CLASS_COLORS.get(track.class_name, (120, 220, 120))
        _draw_box(
            annotated,
            track.bbox.to_xyxy(),
            f"#{track.track_id} {track.class_name} {track.confidence:.2f}",
            color,
        )
    return annotated


def draw_danger_zones(
    frame: np.ndarray,
    zones: Sequence[tuple[str, Sequence[tuple[float, float]]]],
) -> np.ndarray:
    annotated = frame.copy()
    for name, polygon in zones:
        points = np.array(polygon, dtype=np.int32)
        cv2.polylines(annotated, [points], isClosed=True, color=(30, 30, 255), thickness=2)
        if len(points) > 0:
            cv2.putText(
                annotated,
                name,
                tuple(points[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (30, 30, 255),
                2,
                cv2.LINE_AA,
            )
    return annotated


def draw_alerts(frame: np.ndarray, alerts: Iterable[Alert]) -> np.ndarray:
    annotated = frame.copy()
    for alert in alerts:
        _draw_box(annotated, alert.bbox.to_xyxy(), alert.alert_type.value, ALERT_COLOR, thickness=3)
    return annotated


def _draw_box(
    frame: np.ndarray,
    xyxy: tuple[float, float, float, float],
    label: str,
    color: Color,
    *,
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    label_y = max(18, y1 - 8)
    cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
