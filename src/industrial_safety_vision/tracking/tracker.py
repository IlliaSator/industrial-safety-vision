"""Simple tracker implementations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

from industrial_safety_vision.core import BoundingBox, Detection
from industrial_safety_vision.tracking.track_types import Track
from industrial_safety_vision.utils.geometry import bbox_iou


class Tracker(Protocol):
    def update(self, detections: list[Detection]) -> list[Track]:
        """Update tracker state from detections."""


@dataclass
class _TrackState:
    track_id: int
    class_name: str
    bbox: BoundingBox
    confidence: float
    age: int = 1
    missed_frames: int = 0

    def to_track(self) -> Track:
        return Track(
            track_id=self.track_id,
            class_name=self.class_name,
            bbox=self.bbox,
            confidence=self.confidence,
            age=self.age,
            missed_frames=self.missed_frames,
        )


class SimpleIoUTracker:
    """IoU-based tracker for stable IDs on simple real-time pipelines."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        max_missed_frames: int = 10,
        track_class_names: set[str] | None = None,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.track_class_names = track_class_names or {"person"}
        self._next_track_id = 1
        self._tracks: dict[int, _TrackState] = {}

    def update(self, detections: list[Detection]) -> list[Track]:
        eligible = [d for d in detections if d.class_name in self.track_class_names]
        unmatched_track_ids = set(self._tracks)
        unmatched_detection_indices = set(range(len(eligible)))
        matches: list[tuple[int, int]] = []

        candidate_pairs: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(eligible):
                iou = bbox_iou(track.bbox, detection.bbox)
                if iou >= self.iou_threshold:
                    candidate_pairs.append((iou, track_id, detection_index))

        for _, track_id, detection_index in sorted(candidate_pairs, reverse=True):
            if (
                track_id not in unmatched_track_ids
                or detection_index not in unmatched_detection_indices
            ):
                continue
            matches.append((track_id, detection_index))
            unmatched_track_ids.remove(track_id)
            unmatched_detection_indices.remove(detection_index)

        for track_id, detection_index in matches:
            detection = eligible[detection_index]
            current = self._tracks[track_id]
            self._tracks[track_id] = _TrackState(
                track_id=track_id,
                class_name=detection.class_name,
                bbox=detection.bbox,
                confidence=detection.confidence,
                age=current.age + 1,
                missed_frames=0,
            )

        for track_id in list(unmatched_track_ids):
            current = self._tracks[track_id]
            missed = current.missed_frames + 1
            if missed > self.max_missed_frames:
                del self._tracks[track_id]
                continue
            self._tracks[track_id] = _TrackState(
                track_id=track_id,
                class_name=current.class_name,
                bbox=current.bbox,
                confidence=current.confidence,
                age=current.age + 1,
                missed_frames=missed,
            )

        for detection_index in sorted(unmatched_detection_indices):
            detection = eligible[detection_index]
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = _TrackState(
                track_id=track_id,
                class_name=detection.class_name,
                bbox=detection.bbox,
                confidence=detection.confidence,
            )

        return [
            track.to_track()
            for track in sorted(self._tracks.values(), key=lambda item: item.track_id)
        ]


def load_tracker_from_config(path: str | Path) -> SimpleIoUTracker:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    config = payload.get("tracker", {})
    return SimpleIoUTracker(
        iou_threshold=float(config.get("iou_threshold", 0.3)),
        max_missed_frames=int(config.get("max_missed_frames", 10)),
        track_class_names=set(config.get("class_names", ["person"])),
    )
