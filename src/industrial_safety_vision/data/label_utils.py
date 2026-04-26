"""YOLO label parsing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class YOLOLabel:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def parse_yolo_label_line(line: str) -> YOLOLabel:
    parts = line.strip().split()
    if len(parts) != 5:
        msg = f"Expected 5 YOLO label fields, got {len(parts)}: {line!r}"
        raise ValueError(msg)
    class_id = int(parts[0])
    x_center, y_center, width, height = [float(value) for value in parts[1:]]
    return YOLOLabel(class_id, x_center, y_center, width, height)


def read_yolo_labels(path: str | Path) -> list[YOLOLabel]:
    path = Path(path)
    if not path.exists():
        return []
    labels: list[YOLOLabel] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            labels.append(parse_yolo_label_line(line))
    return labels


def is_valid_normalized_bbox(label: YOLOLabel) -> bool:
    values = [label.x_center, label.y_center, label.width, label.height]
    return all(0.0 <= value <= 1.0 for value in values) and label.width > 0 and label.height > 0
