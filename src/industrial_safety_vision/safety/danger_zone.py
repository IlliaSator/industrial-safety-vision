"""Danger-zone polygon checks."""

from __future__ import annotations

from dataclasses import dataclass

from industrial_safety_vision.tracking.track_types import Track
from industrial_safety_vision.utils.geometry import point_in_polygon


@dataclass(frozen=True)
class DangerZone:
    name: str
    polygon: list[tuple[float, float]]


def track_position(track: Track, *, position: str = "bottom_center") -> tuple[float, float]:
    if position == "center":
        return track.bbox.center
    return track.bbox.bottom_center


def track_in_zone(track: Track, zone: DangerZone, *, position: str = "bottom_center") -> bool:
    return point_in_polygon(track_position(track, position=position), zone.polygon)
