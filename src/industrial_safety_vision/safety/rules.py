"""Configurable safety-rule engine with smoothing and cooldown."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from industrial_safety_vision.core import Detection
from industrial_safety_vision.safety.alert_types import Alert, AlertType, Severity
from industrial_safety_vision.safety.danger_zone import DangerZone, track_in_zone
from industrial_safety_vision.safety.ppe_compliance import has_associated_ppe
from industrial_safety_vision.tracking.track_types import Track
from industrial_safety_vision.utils.geometry import bbox_center_distance


@dataclass(frozen=True)
class TemporalRuleConfig:
    enabled: bool = True
    consecutive_frames: int = 3
    cooldown_frames: int = 50


@dataclass(frozen=True)
class PPERuleConfig(TemporalRuleConfig):
    min_overlap_ratio: float = 0.05


@dataclass(frozen=True)
class DangerZoneRuleConfig(TemporalRuleConfig):
    position: str = "bottom_center"
    zones: list[DangerZone] = field(default_factory=list)


@dataclass(frozen=True)
class VehicleProximityRuleConfig(TemporalRuleConfig):
    distance_threshold_px: float = 120.0


@dataclass(frozen=True)
class SafetyRulesConfig:
    missing_helmet: PPERuleConfig = field(default_factory=lambda: PPERuleConfig())
    missing_vest: PPERuleConfig = field(default_factory=lambda: PPERuleConfig(min_overlap_ratio=0.08))
    danger_zone: DangerZoneRuleConfig = field(default_factory=DangerZoneRuleConfig)
    vehicle_proximity: VehicleProximityRuleConfig = field(default_factory=VehicleProximityRuleConfig)


class SafetyRulesEngine:
    """Evaluate safety rules over tracked people and detector outputs."""

    def __init__(self, config: SafetyRulesConfig | None = None) -> None:
        self.config = config or SafetyRulesConfig()
        self._violation_counts: dict[tuple[int, AlertType, str], int] = {}
        self._last_alert_frame: dict[tuple[int, AlertType, str], int] = {}

    def evaluate(
        self,
        *,
        tracks: list[Track],
        detections: list[Detection],
        frame_index: int,
    ) -> list[Alert]:
        alerts: list[Alert] = []
        person_tracks = [track for track in tracks if track.class_name == "person"]

        for track in person_tracks:
            alerts.extend(self._evaluate_missing_helmet(track, detections, frame_index))
            alerts.extend(self._evaluate_missing_vest(track, detections, frame_index))
            alerts.extend(self._evaluate_danger_zone(track, frame_index))
            alerts.extend(self._evaluate_vehicle_proximity(track, detections, frame_index))
        return alerts

    def _evaluate_missing_helmet(
        self,
        track: Track,
        detections: list[Detection],
        frame_index: int,
    ) -> list[Alert]:
        config = self.config.missing_helmet
        if not config.enabled:
            return []
        has_helmet = has_associated_ppe(
            track,
            detections,
            ppe_class_names={"helmet", "hard_hat"},
            min_overlap_ratio=config.min_overlap_ratio,
        )
        return self._update_binary_rule(
            violated=not has_helmet,
            track=track,
            frame_index=frame_index,
            alert_type=AlertType.MISSING_HELMET,
            severity=Severity.HIGH,
            message=f"Worker #{track.track_id} is missing a helmet.",
            config=config,
        )

    def _evaluate_missing_vest(
        self,
        track: Track,
        detections: list[Detection],
        frame_index: int,
    ) -> list[Alert]:
        config = self.config.missing_vest
        if not config.enabled:
            return []
        has_vest = has_associated_ppe(
            track,
            detections,
            ppe_class_names={"safety_vest", "vest", "reflective_vest"},
            min_overlap_ratio=config.min_overlap_ratio,
        )
        return self._update_binary_rule(
            violated=not has_vest,
            track=track,
            frame_index=frame_index,
            alert_type=AlertType.MISSING_VEST,
            severity=Severity.HIGH,
            message=f"Worker #{track.track_id} is missing a safety vest.",
            config=config,
        )

    def _evaluate_danger_zone(self, track: Track, frame_index: int) -> list[Alert]:
        config = self.config.danger_zone
        if not config.enabled:
            return []
        alerts: list[Alert] = []
        for zone in config.zones:
            alerts.extend(
                self._update_binary_rule(
                    violated=track_in_zone(track, zone, position=config.position),
                    track=track,
                    frame_index=frame_index,
                    alert_type=AlertType.DANGER_ZONE_VIOLATION,
                    severity=Severity.CRITICAL,
                    message=f"Worker #{track.track_id} entered danger zone '{zone.name}'.",
                    config=config,
                    scope=zone.name,
                    metadata={"zone": zone.name},
                )
            )
        return alerts

    def _evaluate_vehicle_proximity(
        self,
        track: Track,
        detections: list[Detection],
        frame_index: int,
    ) -> list[Alert]:
        config = self.config.vehicle_proximity
        if not config.enabled:
            return []
        vehicles = [d for d in detections if d.class_name in {"forklift", "vehicle", "truck"}]
        nearest_distance = min(
            (bbox_center_distance(track.bbox, vehicle.bbox) for vehicle in vehicles),
            default=float("inf"),
        )
        return self._update_binary_rule(
            violated=nearest_distance <= config.distance_threshold_px,
            track=track,
            frame_index=frame_index,
            alert_type=AlertType.VEHICLE_PROXIMITY,
            severity=Severity.CRITICAL,
            message=f"Worker #{track.track_id} is too close to a vehicle.",
            config=config,
            metadata={"nearest_vehicle_distance_px": nearest_distance},
        )

    def _update_binary_rule(
        self,
        *,
        violated: bool,
        track: Track,
        frame_index: int,
        alert_type: AlertType,
        severity: Severity,
        message: str,
        config: TemporalRuleConfig,
        scope: str = "global",
        metadata: dict[str, object] | None = None,
    ) -> list[Alert]:
        key = (track.track_id, alert_type, scope)
        if not violated:
            self._violation_counts[key] = 0
            return []

        count = self._violation_counts.get(key, 0) + 1
        self._violation_counts[key] = count
        if count < config.consecutive_frames:
            return []

        last_alert_frame = self._last_alert_frame.get(key)
        if last_alert_frame is not None and frame_index - last_alert_frame < config.cooldown_frames:
            return []

        self._last_alert_frame[key] = frame_index
        return [
            Alert(
                frame_index=frame_index,
                track_id=track.track_id,
                alert_type=alert_type,
                severity=severity,
                message=message,
                bbox=track.bbox,
                confidence=track.confidence,
                metadata=metadata or {},
            )
        ]


def load_safety_rules_config(path: str | Path) -> SafetyRulesConfig:
    """Load safety-rule config from YAML."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    rules = payload.get("rules", {})
    return SafetyRulesConfig(
        missing_helmet=_load_ppe_config(rules.get("missing_helmet", {}), default_overlap=0.02),
        missing_vest=_load_ppe_config(rules.get("missing_vest", {}), default_overlap=0.08),
        danger_zone=_load_danger_zone_config(rules.get("danger_zone", {})),
        vehicle_proximity=_load_vehicle_config(rules.get("vehicle_proximity", {})),
    )


def _load_temporal_fields(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(payload.get("enabled", True)),
        "consecutive_frames": int(payload.get("consecutive_frames", 3)),
        "cooldown_frames": int(payload.get("cooldown_frames", 50)),
    }


def _load_ppe_config(payload: dict[str, Any], *, default_overlap: float) -> PPERuleConfig:
    return PPERuleConfig(
        **_load_temporal_fields(payload),
        min_overlap_ratio=float(payload.get("min_overlap_ratio", default_overlap)),
    )


def _load_danger_zone_config(payload: dict[str, Any]) -> DangerZoneRuleConfig:
    zones = [
        DangerZone(
            name=str(zone.get("name", "zone")),
            polygon=[(float(x), float(y)) for x, y in zone.get("polygon", [])],
        )
        for zone in payload.get("zones", [])
    ]
    return DangerZoneRuleConfig(
        **_load_temporal_fields(payload),
        position=str(payload.get("position", "bottom_center")),
        zones=zones,
    )


def _load_vehicle_config(payload: dict[str, Any]) -> VehicleProximityRuleConfig:
    return VehicleProximityRuleConfig(
        **_load_temporal_fields(payload),
        distance_threshold_px=float(payload.get("distance_threshold_px", 120)),
    )
