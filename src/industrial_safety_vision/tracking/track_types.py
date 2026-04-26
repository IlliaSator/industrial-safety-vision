"""Track domain types."""

from __future__ import annotations

from dataclasses import dataclass

from industrial_safety_vision.core import BoundingBox


@dataclass(frozen=True)
class Track:
    """Tracked object state returned by tracker implementations."""

    track_id: int
    class_name: str
    bbox: BoundingBox
    confidence: float
    age: int = 1
    missed_frames: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
            "age": self.age,
            "missed_frames": self.missed_frames,
        }
