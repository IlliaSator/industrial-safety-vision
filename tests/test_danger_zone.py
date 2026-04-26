from industrial_safety_vision.core import BoundingBox
from industrial_safety_vision.safety.danger_zone import DangerZone, track_in_zone, track_position
from industrial_safety_vision.tracking.track_types import Track


def test_track_bottom_center_in_zone() -> None:
    zone = DangerZone("loading", [(0, 90), (100, 90), (100, 120), (0, 120)])
    track = Track(1, "person", BoundingBox(10, 10, 40, 100), 0.9)

    assert track_position(track) == (25, 100)
    assert track_in_zone(track, zone)
