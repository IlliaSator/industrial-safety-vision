"""Safety alert domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from industrial_safety_vision.core import BoundingBox


class AlertType(StrEnum):
    MISSING_HELMET = "missing_helmet"
    MISSING_VEST = "missing_vest"
    DANGER_ZONE_VIOLATION = "danger_zone_violation"
    VEHICLE_PROXIMITY = "vehicle_proximity"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Alert:
    """Safety alert emitted after rule smoothing and cooldown checks."""

    frame_index: int
    track_id: int
    alert_type: AlertType
    severity: Severity
    message: str
    bbox: BoundingBox
    confidence: float
    metadata: dict[str, object] = field(default_factory=dict)
    alert_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, object]:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "frame_index": self.frame_index,
            "track_id": self.track_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
            "metadata": self.metadata,
        }
