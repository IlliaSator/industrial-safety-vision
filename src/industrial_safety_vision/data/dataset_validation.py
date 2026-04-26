"""Validate YOLO dataset structure and labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from industrial_safety_vision.data.label_utils import is_valid_normalized_bbox, read_yolo_labels

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def validate_yolo_dataset(
    dataset_yaml: str | Path,
    *,
    report_path: str | Path = "reports/dataset_summary.json",
) -> dict[str, Any]:
    dataset_yaml = Path(dataset_yaml)
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML does not exist: {dataset_yaml}")

    config = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")) or {}
    dataset_root = Path(config.get("path", dataset_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = (dataset_yaml.parent / dataset_root).resolve()

    names = config.get("names", [])
    class_count = len(names) if isinstance(names, list) else len(names.keys())
    summary: dict[str, Any] = {
        "dataset_yaml": str(dataset_yaml),
        "dataset_root": str(dataset_root),
        "class_count": class_count,
        "splits": {},
        "issues": [],
    }

    for split in SPLITS:
        split_summary = _validate_split(dataset_root, split, class_count, config)
        summary["splits"][split] = split_summary
        summary["issues"].extend(split_summary["issues"])

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _validate_split(
    dataset_root: Path,
    split: str,
    class_count: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    image_dir = _resolve_image_dir(dataset_root, split, config)
    label_dir = _resolve_label_dir(dataset_root, split, image_dir)
    issues: list[dict[str, str]] = []
    if not image_dir.exists():
        issues.append({"type": "missing_image_dir", "path": str(image_dir)})
        return {"image_count": 0, "label_count": 0, "class_distribution": {}, "issues": issues}
    if not label_dir.exists():
        issues.append({"type": "missing_label_dir", "path": str(label_dir)})

    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    class_distribution: Counter[int] = Counter()
    label_count = 0

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            issues.append({"type": "missing_label", "path": str(label_path)})
            continue
        raw_text = label_path.read_text(encoding="utf-8")
        if not raw_text.strip():
            issues.append({"type": "empty_label", "path": str(label_path)})
            continue
        try:
            labels = read_yolo_labels(label_path)
        except ValueError as exc:
            issues.append(
                {"type": "invalid_label_format", "path": str(label_path), "message": str(exc)}
            )
            continue
        label_count += len(labels)
        for label in labels:
            class_distribution[label.class_id] += 1
            if not is_valid_normalized_bbox(label):
                issues.append({"type": "invalid_bbox", "path": str(label_path)})
            if class_count and not 0 <= label.class_id < class_count:
                issues.append({"type": "class_id_out_of_range", "path": str(label_path)})

    return {
        "image_count": len(images),
        "label_count": label_count,
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "class_distribution": dict(class_distribution),
        "issues": issues,
    }


def _resolve_image_dir(dataset_root: Path, split: str, config: dict[str, Any]) -> Path:
    configured = config.get("val" if split == "val" else split)
    if configured:
        configured_path = Path(configured)
        if not configured_path.is_absolute():
            configured_path = dataset_root / configured_path
        return configured_path

    candidates = [
        dataset_root / "images" / split,
        dataset_root / split / "images",
        dataset_root / ("valid" if split == "val" else split) / "images",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_label_dir(dataset_root: Path, split: str, image_dir: Path) -> Path:
    if image_dir.name == "images":
        return image_dir.parent / "labels"
    candidates = [
        dataset_root / "labels" / split,
        dataset_root / split / "labels",
        dataset_root / ("valid" if split == "val" else split) / "labels",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a YOLO dataset.")
    parser.add_argument("--config", default="configs/train.yaml", help="Training config path.")
    parser.add_argument("--dataset-yaml", default=None, help="Override dataset YAML path.")
    parser.add_argument(
        "--report",
        default="reports/dataset_summary.json",
        help="Output report path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_yaml = args.dataset_yaml
    if dataset_yaml is None:
        train_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        dataset_yaml = train_config.get("dataset", {}).get("data_yaml")
    summary = validate_yolo_dataset(dataset_yaml, report_path=args.report)
    print(json.dumps({"issues": len(summary["issues"]), "report": args.report}, indent=2))


if __name__ == "__main__":
    main()
