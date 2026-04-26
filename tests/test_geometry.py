import math

from industrial_safety_vision.core import BoundingBox, Detection
from industrial_safety_vision.utils.geometry import (
    associate_child_object_to_person,
    bbox_center_distance,
    bbox_contains_point,
    bbox_iou,
    bbox_overlap_ratio,
    point_in_polygon,
)


def test_bbox_properties() -> None:
    box = BoundingBox(10, 20, 30, 60)

    assert box.width == 20
    assert box.height == 40
    assert box.area == 800
    assert box.center == (20, 40)
    assert box.bottom_center == (20, 60)


def test_bbox_iou() -> None:
    box_a = BoundingBox(0, 0, 10, 10)
    box_b = BoundingBox(5, 5, 15, 15)

    assert math.isclose(bbox_iou(box_a, box_b), 25 / 175)


def test_bbox_center_distance() -> None:
    assert bbox_center_distance(BoundingBox(0, 0, 10, 10), BoundingBox(10, 0, 20, 10)) == 10


def test_point_in_polygon_includes_edges() -> None:
    polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]

    assert point_in_polygon((5, 5), polygon)
    assert point_in_polygon((0, 5), polygon)
    assert not point_in_polygon((15, 5), polygon)


def test_bbox_contains_point() -> None:
    box = BoundingBox(0, 0, 10, 10)

    assert bbox_contains_point(box, (10, 10))
    assert not bbox_contains_point(box, (10.1, 5))


def test_bbox_overlap_ratio_uses_child_area() -> None:
    parent = BoundingBox(0, 0, 20, 20)
    child = BoundingBox(10, 10, 30, 30)

    assert math.isclose(bbox_overlap_ratio(parent, child), 0.25)


def test_associate_child_object_to_person_by_overlap() -> None:
    person = Detection(0, "person", 0.95, BoundingBox(0, 0, 100, 200))
    outside = Detection(1, "helmet", 0.9, BoundingBox(150, 0, 180, 30))
    inside = Detection(1, "helmet", 0.9, BoundingBox(30, 5, 70, 35))

    assert associate_child_object_to_person(person, [outside, inside]) == inside
