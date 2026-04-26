from industrial_safety_vision.core import BoundingBox, Detection
from industrial_safety_vision.tracking.tracker import SimpleIoUTracker


def person(x1: float, y1: float, x2: float, y2: float) -> Detection:
    return Detection(0, "person", 0.9, BoundingBox(x1, y1, x2, y2))


def test_same_object_keeps_track_id_across_frames() -> None:
    tracker = SimpleIoUTracker(iou_threshold=0.3)

    first_id = tracker.update([person(0, 0, 100, 100)])[0].track_id
    second_id = tracker.update([person(5, 0, 105, 100)])[0].track_id
    third_id = tracker.update([person(10, 0, 110, 100)])[0].track_id

    assert first_id == second_id == third_id


def test_object_disappearing_for_one_frame_preserves_id() -> None:
    tracker = SimpleIoUTracker(iou_threshold=0.3, max_missed_frames=1)

    first_id = tracker.update([person(0, 0, 100, 100)])[0].track_id
    missed_track = tracker.update([])[0]
    restored_id = tracker.update([person(3, 0, 103, 100)])[0].track_id

    assert missed_track.track_id == first_id
    assert missed_track.missed_frames == 1
    assert restored_id == first_id


def test_object_disappearing_beyond_limit_is_removed() -> None:
    tracker = SimpleIoUTracker(iou_threshold=0.3, max_missed_frames=1)

    first_id = tracker.update([person(0, 0, 100, 100)])[0].track_id
    tracker.update([])
    assert tracker.update([]) == []
    new_id = tracker.update([person(0, 0, 100, 100)])[0].track_id

    assert new_id != first_id


def test_two_people_get_different_track_ids() -> None:
    tracker = SimpleIoUTracker(iou_threshold=0.3)

    tracks = tracker.update([person(0, 0, 100, 100), person(200, 0, 300, 100)])

    assert {track.track_id for track in tracks} == {1, 2}
