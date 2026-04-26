"""Geometry utilities for detections, tracks, and safety zones."""

from __future__ import annotations

import math
from collections.abc import Sequence

from industrial_safety_vision.core import BoundingBox, Detection

Point = tuple[float, float]


def bbox_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Return intersection-over-union for two axis-aligned boxes."""

    inter_x1 = max(box_a.x1, box_b.x1)
    inter_y1 = max(box_a.y1, box_b.y1)
    inter_x2 = min(box_a.x2, box_b.x2)
    inter_y2 = min(box_a.y2, box_b.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union = box_a.area + box_b.area - inter_area
    return 0.0 if union <= 0 else inter_area / union


def bbox_center_distance(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Return Euclidean distance between box centers."""

    ax, ay = box_a.center
    bx, by = box_b.center
    return math.hypot(ax - bx, ay - by)


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Return whether a point is inside or on the edge of a polygon."""

    if len(polygon) < 3:
        return False

    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if _point_on_segment(point, (xi, yi), (xj, yj)):
            return True
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def bbox_contains_point(box: BoundingBox, point: Point) -> bool:
    """Return whether a point lies inside a bbox, including edges."""

    x, y = point
    return box.x1 <= x <= box.x2 and box.y1 <= y <= box.y2


def bbox_overlap_ratio(parent: BoundingBox, child: BoundingBox) -> float:
    """Return child-box area overlapped by parent box."""

    inter_x1 = max(parent.x1, child.x1)
    inter_y1 = max(parent.y1, child.y1)
    inter_x2 = min(parent.x2, child.x2)
    inter_y2 = min(parent.y2, child.y2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    return 0.0 if child.area <= 0 else inter_area / child.area


def associate_child_object_to_person(
    person: Detection,
    candidates: Sequence[Detection],
    *,
    min_overlap_ratio: float = 0.1,
) -> Detection | None:
    """Return the child object most strongly associated with a person bbox."""

    best_candidate: Detection | None = None
    best_overlap = 0.0
    for candidate in candidates:
        overlap = bbox_overlap_ratio(person.bbox, candidate.bbox)
        if overlap >= min_overlap_ratio and overlap > best_overlap:
            best_overlap = overlap
            best_candidate = candidate
    return best_candidate


def _point_on_segment(point: Point, start: Point, end: Point, *, eps: float = 1e-9) -> bool:
    px, py = point
    sx, sy = start
    ex, ey = end
    cross = (px - sx) * (ey - sy) - (py - sy) * (ex - sx)
    if abs(cross) > eps:
        return False
    return min(sx, ex) - eps <= px <= max(sx, ex) + eps and min(sy, ey) - eps <= py <= max(
        sy, ey
    ) + eps
