"""PPE association helpers."""

from __future__ import annotations

from collections.abc import Sequence

from industrial_safety_vision.core import Detection
from industrial_safety_vision.tracking.track_types import Track
from industrial_safety_vision.utils.geometry import bbox_overlap_ratio


def has_associated_ppe(
    track: Track,
    detections: Sequence[Detection],
    *,
    ppe_class_names: set[str],
    min_overlap_ratio: float,
) -> bool:
    """Return whether any PPE detection sufficiently overlaps a tracked person."""

    for detection in detections:
        if detection.class_name not in ppe_class_names:
            continue
        if bbox_overlap_ratio(track.bbox, detection.bbox) >= min_overlap_ratio:
            return True
    return False
