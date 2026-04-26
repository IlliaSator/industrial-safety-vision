from industrial_safety_vision.core import BoundingBox, Detection
from industrial_safety_vision.safety.alert_types import AlertType
from industrial_safety_vision.safety.danger_zone import DangerZone
from industrial_safety_vision.safety.rules import (
    DangerZoneRuleConfig,
    PPERuleConfig,
    SafetyRulesConfig,
    SafetyRulesEngine,
    TemporalRuleConfig,
    VehicleProximityRuleConfig,
)
from industrial_safety_vision.tracking.track_types import Track


def person_track(track_id: int = 1, box: BoundingBox | None = None) -> Track:
    return Track(track_id, "person", box or BoundingBox(0, 0, 100, 200), 0.9)


def detection(class_name: str, box: BoundingBox) -> Detection:
    return Detection(0, class_name, 0.9, box)


def engine_for_tests(**overrides: object) -> SafetyRulesEngine:
    config = SafetyRulesConfig(
        missing_helmet=PPERuleConfig(consecutive_frames=2, cooldown_frames=5, min_overlap_ratio=0.01),
        missing_vest=PPERuleConfig(consecutive_frames=2, cooldown_frames=5, min_overlap_ratio=0.01),
        danger_zone=DangerZoneRuleConfig(
            consecutive_frames=2,
            cooldown_frames=5,
            zones=[DangerZone("zone-a", [(-10, 190), (110, 190), (110, 230), (-10, 230)])],
        ),
        vehicle_proximity=VehicleProximityRuleConfig(
            consecutive_frames=2,
            cooldown_frames=5,
            distance_threshold_px=100,
        ),
    )
    for name, value in overrides.items():
        config = SafetyRulesConfig(**{**config.__dict__, name: value})
    return SafetyRulesEngine(config)


def test_missing_helmet_alert_only_after_n_frames() -> None:
    engine = engine_for_tests(
        missing_vest=PPERuleConfig(enabled=False),
        danger_zone=DangerZoneRuleConfig(enabled=False),
        vehicle_proximity=VehicleProximityRuleConfig(enabled=False),
    )

    assert engine.evaluate(tracks=[person_track()], detections=[], frame_index=1) == []
    alerts = engine.evaluate(tracks=[person_track()], detections=[], frame_index=2)

    assert [alert.alert_type for alert in alerts] == [AlertType.MISSING_HELMET]


def test_no_missing_helmet_alert_when_helmet_exists() -> None:
    engine = engine_for_tests(
        missing_vest=PPERuleConfig(enabled=False),
        danger_zone=DangerZoneRuleConfig(enabled=False),
        vehicle_proximity=VehicleProximityRuleConfig(enabled=False),
    )
    helmet = detection("helmet", BoundingBox(20, 0, 80, 40))

    assert engine.evaluate(tracks=[person_track()], detections=[helmet], frame_index=1) == []
    assert engine.evaluate(tracks=[person_track()], detections=[helmet], frame_index=2) == []


def test_missing_vest_alert_only_after_n_frames() -> None:
    engine = engine_for_tests(
        missing_helmet=PPERuleConfig(enabled=False),
        danger_zone=DangerZoneRuleConfig(enabled=False),
        vehicle_proximity=VehicleProximityRuleConfig(enabled=False),
    )

    assert engine.evaluate(tracks=[person_track()], detections=[], frame_index=1) == []
    alerts = engine.evaluate(tracks=[person_track()], detections=[], frame_index=2)

    assert [alert.alert_type for alert in alerts] == [AlertType.MISSING_VEST]


def test_danger_zone_alert_only_after_n_frames_inside_polygon() -> None:
    engine = engine_for_tests(
        missing_helmet=PPERuleConfig(enabled=False),
        missing_vest=PPERuleConfig(enabled=False),
        vehicle_proximity=VehicleProximityRuleConfig(enabled=False),
    )

    assert engine.evaluate(tracks=[person_track()], detections=[], frame_index=1) == []
    alerts = engine.evaluate(tracks=[person_track()], detections=[], frame_index=2)

    assert [alert.alert_type for alert in alerts] == [AlertType.DANGER_ZONE_VIOLATION]


def test_cooldown_prevents_repeated_alerts() -> None:
    engine = engine_for_tests(
        missing_vest=PPERuleConfig(enabled=False),
        danger_zone=DangerZoneRuleConfig(enabled=False),
        vehicle_proximity=VehicleProximityRuleConfig(enabled=False),
    )

    engine.evaluate(tracks=[person_track()], detections=[], frame_index=1)
    assert engine.evaluate(tracks=[person_track()], detections=[], frame_index=2)
    assert engine.evaluate(tracks=[person_track()], detections=[], frame_index=3) == []


def test_vehicle_proximity_alert_works() -> None:
    engine = engine_for_tests(
        missing_helmet=PPERuleConfig(enabled=False),
        missing_vest=PPERuleConfig(enabled=False),
        danger_zone=DangerZoneRuleConfig(enabled=False),
    )
    forklift = detection("forklift", BoundingBox(80, 0, 180, 100))

    assert engine.evaluate(tracks=[person_track()], detections=[forklift], frame_index=1) == []
    alerts = engine.evaluate(tracks=[person_track()], detections=[forklift], frame_index=2)

    assert [alert.alert_type for alert in alerts] == [AlertType.VEHICLE_PROXIMITY]
